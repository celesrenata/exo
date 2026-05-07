"""Pipeline parallelism orchestration across distributed nodes.

Coordinates pipeline-parallel inference across multiple nodes. The
PipelineCoordinator manages forward passes through the pipeline,
sending and receiving activations between adjacent stages, and
orchestrating full token generation across all stages.

Requirements: 5.4, 5.7, 5.8, 5.9
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, final

import numpy as np
import torch

from .activation_pass import CommunicatorProtocol, recv_activation, send_activation
from .stage import StageAssignment, compute_stage_assignments

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engine protocol — defines the interface the coordinator expects from the
# inference engine. The actual engine implementation lives elsewhere.
# ---------------------------------------------------------------------------


class EngineProtocol(Protocol):
    """Protocol defining the inference engine interface for pipeline use.

    The real UnifiedPyTorchEngine must satisfy this protocol. Using a
    Protocol here avoids circular imports and allows the coordinator to
    be tested independently.
    """

    def forward_layers(
        self,
        hidden_states: torch.Tensor,
        start_layer: int,
        end_layer: int,
    ) -> torch.Tensor:
        """Run forward pass through a subset of transformer layers."""
        ...

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to embeddings."""
        ...

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits."""
        ...

    def sample_token(self, logits: torch.Tensor) -> np.ndarray:
        """Sample a token from logits, returning token ID(s) as numpy array."""
        ...

    @property
    def device(self) -> str:
        """Return the device string (e.g., 'xpu:0', 'cuda:0')."""
        ...

    @property
    def hidden_size(self) -> int:
        """Return the model's hidden dimension size."""
        ...

    @property
    def dtype(self) -> torch.dtype:
        """Return the model's tensor dtype."""
        ...


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for pipeline-parallel inference.

    Attributes:
        world_size: Number of nodes participating (2, 3, or 4).
        rank: This node's rank (0-indexed).
        total_layers: Total number of transformer layers in the model.
        master_addr: IP address of rank 0 for rendezvous.
        master_port: Port for rendezvous.
        transport: Network transport type.
    """

    world_size: int
    rank: int
    total_layers: int
    master_addr: str
    master_port: int
    transport: Literal["rdma", "ethernet", "lacp"]


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class PipelineNodeUnreachableError(Exception):
    """Raised when a node in the pipeline becomes unreachable.

    Includes node identity and pipeline position for diagnostics.
    """

    def __init__(self, rank: int, world_size: int, operation: str, cause: Exception | None = None) -> None:
        self.rank = rank
        self.world_size = world_size
        self.operation = operation
        self.cause = cause
        position = _describe_position(rank, world_size)
        message = (
            f"Pipeline node unreachable: rank={rank} ({position}), "
            f"world_size={world_size}, operation={operation}"
        )
        if cause is not None:
            message += f", cause={cause}"
        super().__init__(message)


def _describe_position(rank: int, world_size: int) -> str:
    """Return a human-readable description of a rank's pipeline position."""
    if rank == 0:
        return "first stage / embedding"
    elif rank == world_size - 1:
        return "last stage / lm_head"
    else:
        return f"middle stage {rank}/{world_size - 1}"


# ---------------------------------------------------------------------------
# Pipeline coordinator
# ---------------------------------------------------------------------------


@final
class PipelineCoordinator:
    """Coordinates pipeline-parallel inference across nodes.

    Manages the execution of forward passes through the local pipeline
    stage and the transfer of activations to/from adjacent stages.
    Orchestrates full token generation across all pipeline stages.
    """

    def __init__(
        self,
        config: PipelineConfig,
        engine: EngineProtocol,
        communicator: CommunicatorProtocol,
    ) -> None:
        """Initialize the pipeline coordinator.

        Args:
            config: Pipeline configuration specifying topology.
            engine: The inference engine for local computation.
            communicator: The distributed communicator for tensor transfer.
        """
        self._config = config
        self._engine = engine
        self._communicator = communicator

        # Compute stage assignments for all ranks
        self._assignments = compute_stage_assignments(
            config.total_layers, config.world_size
        )
        self._local_assignment = self._assignments[config.rank]

        logger.info(
            "PipelineCoordinator initialized: rank=%d/%d, layers=[%d, %d), "
            "has_embedding=%s, has_lm_head=%s, transport=%s",
            config.rank,
            config.world_size,
            self._local_assignment.start_layer,
            self._local_assignment.end_layer,
            self._local_assignment.has_embedding,
            self._local_assignment.has_lm_head,
            config.transport,
        )

    @property
    def config(self) -> PipelineConfig:
        """Return the pipeline configuration."""
        return self._config

    @property
    def local_assignment(self) -> StageAssignment:
        """Return the local stage assignment."""
        return self._local_assignment

    def get_stage_assignment(self) -> tuple[int, int]:
        """Return (start_layer, end_layer) for this node's pipeline stage.

        Returns:
            Tuple of (start_layer_inclusive, end_layer_exclusive).
        """
        return (self._local_assignment.start_layer, self._local_assignment.end_layer)

    async def forward_pipeline(
        self, input_tensor: torch.Tensor, request_id: str
    ) -> torch.Tensor | None:
        """Execute one forward pass through the pipeline.

        For the first stage (rank 0): processes input through embedding
        and local layers, then sends activations to the next stage.

        For middle stages: receives activations from the previous stage,
        processes through local layers, sends to the next stage.

        For the last stage: receives activations from the previous stage,
        processes through local layers and lm_head, returns output logits.

        Args:
            input_tensor: For rank 0, the token IDs tensor. For other ranks,
                this parameter is ignored (activations come from previous stage).
            request_id: Unique identifier for this inference request.

        Returns:
            Output logits tensor on the last stage (rank == world_size - 1).
            None on all other stages.

        Raises:
            PipelineNodeUnreachableError: If an adjacent node is unreachable.
        """
        rank = self._config.rank
        world_size = self._config.world_size

        # --- First stage: embed and process local layers ---
        if rank == 0:
            hidden_states = self._engine.embed_tokens(input_tensor)
            hidden_states = self._engine.forward_layers(
                hidden_states,
                self._local_assignment.start_layer,
                self._local_assignment.end_layer,
            )

            # Send to next stage if not the only node
            if world_size > 1:
                self._send_to_next(hidden_states, request_id, sequence_position=0)

            # If this is also the last stage (world_size == 1, shouldn't happen
            # with supported sizes but handle gracefully)
            if self._local_assignment.has_lm_head:
                return self._engine.lm_head(hidden_states)

            return None

        # --- Middle or last stage: receive from previous ---
        hidden_states = self._recv_from_previous(request_id, sequence_position=0)

        # Process through local layers
        hidden_states = self._engine.forward_layers(
            hidden_states,
            self._local_assignment.start_layer,
            self._local_assignment.end_layer,
        )

        # --- Last stage: apply lm_head and return ---
        if self._local_assignment.has_lm_head:
            logits = self._engine.lm_head(hidden_states)
            return logits

        # --- Middle stage: send to next ---
        self._send_to_next(hidden_states, request_id, sequence_position=0)
        return None

    async def generate_token(self, prompt_tokens: np.ndarray) -> np.ndarray:
        """Coordinate full token generation across all pipeline stages.

        On rank 0: embeds prompt tokens and initiates the pipeline forward.
        On intermediate ranks: participates in the pipeline forward.
        On the last rank: receives final hidden states, applies lm_head,
        and samples the next token.

        Args:
            prompt_tokens: Token IDs as a numpy array (used on rank 0).

        Returns:
            The sampled token ID(s) as a numpy array. Only meaningful on
            the last rank; other ranks return an empty array.

        Raises:
            PipelineNodeUnreachableError: If a node becomes unreachable.
        """
        rank = self._config.rank
        request_id = f"gen-{id(prompt_tokens)}"

        # Convert prompt tokens to tensor on rank 0
        if rank == 0:
            token_tensor = torch.tensor(
                prompt_tokens, dtype=torch.long, device=self._engine.device
            )
            # Forward through the pipeline (rank 0 sends to next)
            result = await self.forward_pipeline(token_tensor, request_id)
        else:
            # Non-zero ranks participate by receiving and forwarding
            result = await self.forward_pipeline(
                torch.empty(0),  # Unused for non-zero ranks
                request_id,
            )

        # Last rank samples the token
        if self._local_assignment.has_lm_head and result is not None:
            sampled = self._engine.sample_token(result)
            return sampled

        # Non-last ranks return empty array
        return np.array([], dtype=np.int64)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _send_to_next(
        self, hidden_states: torch.Tensor, request_id: str, sequence_position: int
    ) -> None:
        """Send activations to the next pipeline stage.

        Raises:
            PipelineNodeUnreachableError: If the next node is unreachable.
        """
        dst_rank = self._config.rank + 1
        try:
            send_activation(
                communicator=self._communicator,
                tensor=hidden_states,
                dst_rank=dst_rank,
                request_id=request_id,
                sequence_position=sequence_position,
            )
        except (TimeoutError, OSError, RuntimeError) as exc:
            logger.error(
                "Failed to send activation to rank %d: %s", dst_rank, exc
            )
            raise PipelineNodeUnreachableError(
                rank=dst_rank,
                world_size=self._config.world_size,
                operation="send_activation",
                cause=exc,
            ) from exc

    def _recv_from_previous(
        self, request_id: str, sequence_position: int
    ) -> torch.Tensor:
        """Receive activations from the previous pipeline stage.

        Raises:
            PipelineNodeUnreachableError: If the previous node is unreachable.
        """
        src_rank = self._config.rank - 1
        # We need to know the expected shape and dtype from the engine
        # The hidden size and dtype come from the engine protocol
        hidden_size = self._engine.hidden_size
        dtype = self._engine.dtype
        device = self._engine.device

        # Shape: we expect (batch_size=1, seq_len, hidden_size) or (seq_len, hidden_size)
        # For pipeline parallelism, we use a standard shape convention.
        # The actual shape is determined by what the previous stage sent.
        # We use a placeholder shape that will be filled by the communicator.
        # In practice, the shape must be communicated out-of-band or be fixed.
        # For simplicity, we receive with a known hidden_size dimension.
        from .activation_pass import _DTYPE_TO_STR

        dtype_str = _DTYPE_TO_STR.get(dtype, "float32")

        try:
            # Use a standard shape: (1, hidden_size) for single-token generation
            # The actual implementation may need shape negotiation
            shape = (1, hidden_size)
            tensor, _message = recv_activation(
                communicator=self._communicator,
                src_rank=src_rank,
                shape=shape,
                dtype=dtype_str,
                target_device=device,
                request_id=request_id,
                sequence_position=sequence_position,
            )
            return tensor
        except (TimeoutError, OSError, RuntimeError) as exc:
            logger.error(
                "Failed to receive activation from rank %d: %s", src_rank, exc
            )
            raise PipelineNodeUnreachableError(
                rank=src_rank,
                world_size=self._config.world_size,
                operation="recv_activation",
                cause=exc,
            ) from exc

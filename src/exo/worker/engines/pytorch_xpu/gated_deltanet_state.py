"""Persistent state container for GatedDeltaNet recurrent decode optimization.

Stores fp32 recurrent state tensors, conv state, and preallocated output buffers
per request and per layer. Eliminates per-token dtype promotion from bf16 to fp32
and avoids repeated tensor allocation during steady-state decode.

The GatedDeltaNet recurrent state has shape ``(batch_size, num_heads, key_dim, value_dim)``
and must remain in fp32 for numerical stability during state accumulation. Without
persistent storage, the decode path casts from bf16 to fp32 on the first token and
allocates intermediate tensors on every token. This container preallocates all
required buffers once and reuses them across decode steps.

For Qwen3.5-4B: ``(1, 32, 128, 128)`` = 524,288 fp32 elements = 2 MB per layer.
For Qwen3.5-27B: ``(1, 64, 128, 128)`` = 1,048,576 fp32 elements = 4 MB per layer.

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.7, 5.8**
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, final

import torch

if TYPE_CHECKING:
    from exo.worker.engines.pytorch_xpu.gated_deltanet_cache import (
        GatedDeltaNetCache,
    )
    from exo.worker.engines.pytorch_xpu.instrumentation import PerformanceRecorder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shape metadata — immutable description of the recurrent state geometry
# ---------------------------------------------------------------------------


@final
@dataclass(frozen=True, slots=True)
class GatedDeltaNetStateShape:
    """Immutable shape metadata for a GatedDeltaNet recurrent state tensor.

    Captures the geometry of the recurrent state matrix and conv state buffer
    so that shape validation and buffer preallocation can be performed without
    inspecting the tensors themselves.
    """

    batch_size: int
    """Number of sequences in the batch (typically 1 for single-request decode)."""

    num_heads: int
    """Number of attention heads (value heads for GatedDeltaNet)."""

    key_dim: int
    """Dimension of the key vectors (d_k in the state matrix)."""

    value_dim: int
    """Dimension of the value vectors (d_v in the state matrix)."""

    conv_dim: int
    """Dimension of the causal conv1d input (typically num_heads * (key_dim + value_dim))."""

    conv_kernel_size: int
    """Kernel size of the causal conv1d (typically 4)."""

    @property
    def recurrent_state_shape(self) -> tuple[int, int, int, int]:
        """Shape of the recurrent state matrix: (B, H, d_k, d_v)."""
        return (self.batch_size, self.num_heads, self.key_dim, self.value_dim)

    @property
    def conv_state_shape(self) -> tuple[int, int, int]:
        """Shape of the conv state buffer: (B, conv_dim, kernel_size)."""
        return (self.batch_size, self.conv_dim, self.conv_kernel_size)

    @property
    def output_shape(self) -> tuple[int, int, int]:
        """Shape of the decode output: (B, H, d_v)."""
        return (self.batch_size, self.num_heads, self.value_dim)

    @property
    def recurrent_state_element_count(self) -> int:
        """Total number of elements in the recurrent state matrix."""
        return self.batch_size * self.num_heads * self.key_dim * self.value_dim

    @property
    def recurrent_state_bytes_fp32(self) -> int:
        """Memory footprint of the recurrent state in fp32 (4 bytes per element)."""
        return self.recurrent_state_element_count * 4


# ---------------------------------------------------------------------------
# GatedDeltaNetPersistentState — mutable state container for decode reuse
# ---------------------------------------------------------------------------


@final
@dataclass(slots=True)
class GatedDeltaNetPersistentState:
    """Persistent state container for a single GatedDeltaNet layer and request.

    Holds the fp32 recurrent state matrix, conv state buffer, and preallocated
    output buffers that are reused across decode steps without reallocation.

    Lifecycle:
        1. Created once per request per layer via ``create(...)``
        2. Used on every decode step — state is updated in place where safe
        3. ``reset()`` zeros all tensors without deallocating (for reuse within same request)
        4. ``recycle()`` marks the container as available for a new request

    Thread safety: NOT thread-safe. Designed for single-threaded per-rank decode.

    Memory layout:
        - ``recurrent_state``: fp32, shape (B, H, d_k, d_v) — the main state matrix
        - ``conv_state``: bf16, shape (B, conv_dim, kernel_size) — causal conv1d sliding window
        - ``output_buffer_fp32``: fp32, shape (B, H, d_v) — preallocated decode output
        - ``output_buffer_bf16``: bf16, shape (B, H, d_v) — preallocated cast buffer
    """

    request_identifier: str
    """Identifier of the request that owns this state. Empty string when recycled."""

    layer_index: int
    """Global layer index this state belongs to (0-indexed across all model layers)."""

    recurrent_state: torch.Tensor
    """The fp32 recurrent state matrix, shape (B, H, d_k, d_v).

    This tensor persists across decode steps and is updated in place during the
    recurrent step. Stored in fp32 to avoid per-token bf16→fp32 promotion.
    """

    conv_state: torch.Tensor
    """The causal conv1d sliding window buffer, shape (B, conv_dim, kernel_size).

    Stored in the model's compute dtype (bf16) since conv operations do not
    require fp32 accumulation.
    """

    device: torch.device
    """Device where all state tensors reside."""

    shape_metadata: GatedDeltaNetStateShape
    """Immutable shape description for validation and diagnostics."""

    output_buffer_fp32: torch.Tensor
    """Preallocated fp32 output buffer for the recurrent step, shape (B, H, d_v).

    Reused across decode steps to avoid allocating a new output tensor each time.
    """

    output_buffer_bf16: torch.Tensor
    """Preallocated bf16 buffer for casting the fp32 output back to compute dtype.

    The recurrent step produces fp32 output which must be cast to bf16 before
    returning to the model forward path. This buffer avoids that allocation.
    """

    _available_for_reuse: bool = field(default=False, init=False, repr=False)
    """Internal flag indicating this container has been recycled and is available."""

    _decode_step_count: int = field(default=0, init=False, repr=False)
    """Number of decode steps processed using this state (for diagnostics)."""

    @staticmethod
    def create(
        *,
        request_identifier: str,
        layer_index: int,
        batch_size: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        conv_dim: int,
        conv_kernel_size: int,
        device: torch.device,
    ) -> GatedDeltaNetPersistentState:
        """Create a new persistent state container with preallocated tensors.

        All tensors are allocated once on the specified device. The recurrent
        state and output buffers are fp32; the conv state is bf16.

        Args:
            request_identifier: Unique identifier for the owning request.
            layer_index: Global layer index (0-indexed).
            batch_size: Batch dimension (typically 1 for single-request decode).
            num_heads: Number of value attention heads.
            key_dim: Key vector dimension (d_k).
            value_dim: Value vector dimension (d_v).
            conv_dim: Causal conv1d input dimension.
            conv_kernel_size: Causal conv1d kernel size.
            device: Target device for tensor allocation.

        Returns:
            A fully initialized persistent state container with zeroed tensors.
        """
        shape_metadata = GatedDeltaNetStateShape(
            batch_size=batch_size,
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            conv_dim=conv_dim,
            conv_kernel_size=conv_kernel_size,
        )

        recurrent_state = torch.zeros(
            shape_metadata.recurrent_state_shape,
            dtype=torch.float32,
            device=device,
        )

        conv_state = torch.zeros(
            shape_metadata.conv_state_shape,
            dtype=torch.bfloat16,
            device=device,
        )

        output_buffer_fp32 = torch.zeros(
            shape_metadata.output_shape,
            dtype=torch.float32,
            device=device,
        )

        output_buffer_bf16 = torch.zeros(
            shape_metadata.output_shape,
            dtype=torch.bfloat16,
            device=device,
        )

        logger.debug(
            "Created GatedDeltaNetPersistentState: request=%r, layer=%d, "
            "state_shape=%s, device=%s, fp32_bytes=%d",
            request_identifier,
            layer_index,
            shape_metadata.recurrent_state_shape,
            device,
            shape_metadata.recurrent_state_bytes_fp32,
        )

        return GatedDeltaNetPersistentState(
            request_identifier=request_identifier,
            layer_index=layer_index,
            recurrent_state=recurrent_state,
            conv_state=conv_state,
            device=device,
            shape_metadata=shape_metadata,
            output_buffer_fp32=output_buffer_fp32,
            output_buffer_bf16=output_buffer_bf16,
        )

    @property
    def is_available_for_reuse(self) -> bool:
        """Whether this container has been recycled and is available for a new request."""
        return self._available_for_reuse

    @property
    def decode_step_count(self) -> int:
        """Number of decode steps processed using this state."""
        return self._decode_step_count

    def increment_decode_step(self) -> None:
        """Record that a decode step has been processed using this state."""
        self._decode_step_count += 1

    def reset(self) -> None:
        """Zero all state tensors without deallocating memory.

        Use this to restart generation within the same request (e.g., after
        a prompt change or retry) without paying the allocation cost again.
        The request identifier and layer index remain unchanged.

        After reset, the state is equivalent to a freshly created container
        but reuses the same underlying memory.
        """
        self.recurrent_state.zero_()
        self.conv_state.zero_()
        self.output_buffer_fp32.zero_()
        self.output_buffer_bf16.zero_()
        self._decode_step_count = 0
        self._available_for_reuse = False

        logger.debug(
            "Reset GatedDeltaNetPersistentState: request=%r, layer=%d",
            self.request_identifier,
            self.layer_index,
        )

    def recycle(self) -> None:
        """Mark this container as available for reuse by a new request.

        Zeros all state tensors and clears the request identifier. The container
        can then be claimed by a new request via ``claim(...)``, avoiding the
        cost of allocating new tensors.

        After recycling, the container must not be used for decode until claimed.
        """
        self.recurrent_state.zero_()
        self.conv_state.zero_()
        self.output_buffer_fp32.zero_()
        self.output_buffer_bf16.zero_()
        self.request_identifier = ""
        self._decode_step_count = 0
        self._available_for_reuse = True

        logger.debug(
            "Recycled GatedDeltaNetPersistentState: layer=%d, "
            "state_shape=%s, device=%s",
            self.layer_index,
            self.shape_metadata.recurrent_state_shape,
            self.device,
        )

    def claim(self, request_identifier: str) -> None:
        """Claim this recycled container for a new request.

        Assigns the new request identifier and marks the container as active.
        The state tensors are already zeroed from the prior ``recycle()`` call.

        Args:
            request_identifier: Identifier of the new owning request.

        Raises:
            RuntimeError: If the container is not available for reuse.
        """
        if not self._available_for_reuse:
            raise RuntimeError(
                f"Cannot claim GatedDeltaNetPersistentState that is not recycled: "
                f"current_request={self.request_identifier!r}, layer={self.layer_index}"
            )

        self.request_identifier = request_identifier
        self._available_for_reuse = False

        logger.debug(
            "Claimed GatedDeltaNetPersistentState: request=%r, layer=%d",
            request_identifier,
            self.layer_index,
        )

    def validate_shape(
        self,
        batch_size: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
    ) -> bool:
        """Validate that the recurrent state shape matches expected dimensions.

        Args:
            batch_size: Expected batch size.
            num_heads: Expected number of heads.
            key_dim: Expected key dimension.
            value_dim: Expected value dimension.

        Returns:
            True if the shape matches, False otherwise.
        """
        return (
            self.shape_metadata.batch_size == batch_size
            and self.shape_metadata.num_heads == num_heads
            and self.shape_metadata.key_dim == key_dim
            and self.shape_metadata.value_dim == value_dim
        )

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        status = "available" if self._available_for_reuse else "active"
        return (
            f"GatedDeltaNetPersistentState("
            f"request={self.request_identifier!r}, "
            f"layer={self.layer_index}, "
            f"shape={self.shape_metadata.recurrent_state_shape}, "
            f"device={self.device}, "
            f"status={status}, "
            f"steps={self._decode_step_count})"
        )


# ---------------------------------------------------------------------------
# Initialization function — creates or reuses persistent state per request/layer
# ---------------------------------------------------------------------------


def initialize_gated_deltanet_state(
    *,
    cache: GatedDeltaNetCache,
    request_identifier: str,
    layer_index: int,
    batch_size: int,
    num_heads: int,
    key_dim: int,
    value_dim: int,
    conv_dim: int,
    conv_kernel_size: int,
    device: torch.device,
    performance_recorder: PerformanceRecorder | None = None,
) -> GatedDeltaNetPersistentState:
    """Initialize persistent GatedDeltaNet state for a request and layer.

    Called once per request per layer (not per token). This function is
    idempotent: repeated calls for the same request and layer return the
    same state without reallocation.

    Initialization strategy:
        1. If state already exists for this layer and is active for the same
           request, return it (idempotent).
        2. If state exists for this layer and is recycled (available for reuse),
           claim it for the new request (avoids reallocation).
        3. Otherwise, create a new state container and register it in the cache.

    All state tensors are allocated on the specified device in fp32 for
    numerical stability during recurrent state accumulation.

    Args:
        cache: The GatedDeltaNetCache to register the state in.
        request_identifier: Unique identifier for the owning request.
        layer_index: Global layer index (0-indexed across all model layers).
        batch_size: Batch dimension (typically 1 for single-request decode).
        num_heads: Number of value attention heads.
        key_dim: Key vector dimension (d_k).
        value_dim: Value vector dimension (d_v).
        conv_dim: Causal conv1d input dimension.
        conv_kernel_size: Causal conv1d kernel size.
        device: Target device for tensor allocation (e.g., torch.device("xpu:0")).
        performance_recorder: Optional recorder for counting state allocations
            vs reuses. When provided, increments ``state_allocation_new`` or
            ``state_allocation_reused`` counters. Zero-cost when None.

    Returns:
        The initialized (or reused) persistent state container.

    **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.7, 5.8**
    """
    # Case 1: State already exists for this layer
    existing_state = cache.get_gated_deltanet_state(layer_index)

    if existing_state is not None:
        # Case 1a: Active state for the same request — idempotent return
        if (
            not existing_state.is_available_for_reuse
            and existing_state.request_identifier == request_identifier
        ):
            logger.debug(
                "initialize_gated_deltanet_state: idempotent hit, "
                "request=%r, layer=%d, steps=%d",
                request_identifier,
                layer_index,
                existing_state.decode_step_count,
            )
            return existing_state

        # Case 1b: Recycled state available for reuse — claim it
        if existing_state.is_available_for_reuse:
            # Validate shape compatibility before claiming
            if existing_state.validate_shape(
                batch_size=batch_size,
                num_heads=num_heads,
                key_dim=key_dim,
                value_dim=value_dim,
            ):
                existing_state.claim(request_identifier)
                logger.info(
                    "initialize_gated_deltanet_state: reused recycled state, "
                    "request=%r, layer=%d, device=%s, data_ptr=%d",
                    request_identifier,
                    layer_index,
                    existing_state.device,
                    existing_state.recurrent_state.data_ptr(),
                )
                if performance_recorder is not None:
                    performance_recorder.increment_counter("state_allocation_reused")
                return existing_state

            # Shape mismatch on recycled state — remove it and create fresh
            logger.debug(
                "initialize_gated_deltanet_state: recycled state shape mismatch, "
                "layer=%d, existing=%s, requested=(%d, %d, %d, %d)",
                layer_index,
                existing_state.shape_metadata.recurrent_state_shape,
                batch_size,
                num_heads,
                key_dim,
                value_dim,
            )
            cache.remove_gated_deltanet_state(layer_index)

        # Case 1c: Active state for a different request — this is unexpected
        # in single-request mode. Log a warning and create fresh state.
        elif existing_state.request_identifier != request_identifier:
            logger.warning(
                "initialize_gated_deltanet_state: layer %d has active state "
                "for request=%r but initialization requested for request=%r. "
                "Removing stale state.",
                layer_index,
                existing_state.request_identifier,
                request_identifier,
            )
            cache.remove_gated_deltanet_state(layer_index)

    # Case 2: No existing state (or removed above) — create fresh
    new_state = GatedDeltaNetPersistentState.create(
        request_identifier=request_identifier,
        layer_index=layer_index,
        batch_size=batch_size,
        num_heads=num_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        conv_dim=conv_dim,
        conv_kernel_size=conv_kernel_size,
        device=device,
    )

    cache.set_gated_deltanet_state(layer_index, new_state)

    logger.info(
        "initialize_gated_deltanet_state: created new state, "
        "request=%r, layer=%d, shape=%s, device=%s, "
        "fp32_bytes=%d, data_ptr=%d",
        request_identifier,
        layer_index,
        new_state.shape_metadata.recurrent_state_shape,
        device,
        new_state.shape_metadata.recurrent_state_bytes_fp32,
        new_state.recurrent_state.data_ptr(),
    )

    if performance_recorder is not None:
        performance_recorder.increment_counter("state_allocation_new")

    return new_state

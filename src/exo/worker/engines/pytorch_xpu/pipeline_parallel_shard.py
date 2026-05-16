"""Pipeline-parallel model sharding configuration and layer assignment.

This module provides the PipelineStageConfig dataclass for configuring pipeline-parallel
model sharding across multiple ranks, the compute_layer_assignment function for
determining which layers each rank is responsible for, and the PipelineParallelShard
class that wraps a contiguous layer range for pipeline-parallel inference.

Unlike tensor parallelism (TensorParallelShard), which shards weight matrices across
all ranks with all-reduce synchronization, pipeline parallelism assigns contiguous
layer ranges to each rank with point-to-point communication between adjacent stages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from exo.worker.engines.pytorch_xpu.instrumentation import PerformanceRecorder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineStageConfig:
    """Configuration for a single pipeline stage.

    Defines which transformer layers are assigned to this rank in the pipeline.
    Each rank holds a contiguous range [start_layer, end_layer) of layers with
    full (unsharded) weights. Communication is point-to-point between adjacent
    stages rather than all-reduce across all ranks.

    Attributes:
        rank: Pipeline stage rank (0-indexed).
        world_size: Total number of pipeline stages.
        start_layer: First layer index assigned to this stage (inclusive).
        end_layer: Last layer index assigned to this stage (exclusive).
        hidden_size: Model hidden dimension (e.g., 2560 for Qwen3.5-4B).
        vocab_size: Tokenizer vocabulary size.
        num_layers: Total number of transformer layers in the model.
        device: Target device string (e.g., "xpu:0").
    """

    rank: int
    world_size: int
    start_layer: int
    end_layer: int
    hidden_size: int
    vocab_size: int
    num_layers: int
    device: str

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.rank < 0:
            raise ValueError(
                f"rank must be non-negative, got {self.rank}"
            )
        if self.world_size < 1:
            raise ValueError(
                f"world_size must be at least 1, got {self.world_size}"
            )
        if self.rank >= self.world_size:
            raise ValueError(
                f"rank ({self.rank}) must be less than world_size ({self.world_size})"
            )
        if self.start_layer >= self.end_layer:
            raise ValueError(
                f"start_layer ({self.start_layer}) must be less than "
                f"end_layer ({self.end_layer})"
            )

    @property
    def is_first_stage(self) -> bool:
        """Whether this is the first pipeline stage (responsible for embedding)."""
        return self.rank == 0

    @property
    def is_last_stage(self) -> bool:
        """Whether this is the last pipeline stage (responsible for lm_head)."""
        return self.rank == self.world_size - 1

    @property
    def num_local_layers(self) -> int:
        """Number of transformer layers assigned to this stage."""
        return self.end_layer - self.start_layer


def compute_layer_assignment(
    n_layers: int, world_size: int, rank: int
) -> tuple[int, int]:
    """Compute the [start_layer, end_layer) range for a given rank.

    Uses balanced distribution (divmod): the first `remainder` ranks get
    (base + 1) layers, remaining ranks get `base` layers. This ensures
    all layers are covered with at most 1 layer difference between ranks,
    and every rank gets at least 1 layer when world_size <= n_layers.

    Args:
        n_layers: Total number of transformer layers in the model.
        world_size: Number of pipeline stages.
        rank: This stage's rank (0-indexed).

    Returns:
        (start_layer, end_layer) tuple, where end_layer is exclusive.

    Raises:
        ValueError: If n_layers < 1, world_size < 1, rank < 0,
            or rank >= world_size.
    """
    if n_layers < 1:
        raise ValueError(f"n_layers must be at least 1, got {n_layers}")
    if world_size < 1:
        raise ValueError(f"world_size must be at least 1, got {world_size}")
    if rank < 0:
        raise ValueError(f"rank must be non-negative, got {rank}")
    if rank >= world_size:
        raise ValueError(
            f"rank ({rank}) must be less than world_size ({world_size})"
        )

    # Balanced distribution: first `remainder` ranks get (base + 1) layers,
    # remaining ranks get `base` layers. This ensures at most 1 layer
    # difference between any two ranks and every rank gets at least 1 layer
    # (when world_size <= n_layers).
    base = n_layers // world_size
    remainder = n_layers % world_size

    if rank < remainder:
        start_layer = rank * (base + 1)
        end_layer = start_layer + (base + 1)
    else:
        start_layer = remainder * (base + 1) + (rank - remainder) * base
        end_layer = start_layer + base

    return (start_layer, end_layer)


class PipelineParallelShard:
    """Pipeline-parallel model wrapper holding a contiguous layer range.

    Unlike TensorParallelShard (which holds ALL layers with sharded weights),
    PipelineParallelShard holds ONLY the layers assigned to this stage with
    full (unsharded) weights. Communication is point-to-point between adjacent
    stages rather than all-reduce across all ranks.

    This class does NOT inherit from nn.Module. It wraps HuggingFace layer
    modules and delegates forward passes to them directly.

    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 5.1, 5.2, 6.1, 6.2
    """

    def __init__(
        self,
        layers: torch.nn.ModuleList,
        config: PipelineStageConfig,
        embed_tokens: torch.nn.Embedding | None,
        lm_head: torch.nn.Linear | None,
        final_norm: torch.nn.Module | None,
        rotary_emb: torch.nn.Module | None = None,
        text_model_config: Any | None = None,
        performance_recorder: PerformanceRecorder | None = None,
    ) -> None:
        """Initialize the pipeline-parallel shard.

        Args:
            layers: The transformer layers assigned to this stage (ModuleList).
            config: Pipeline stage configuration.
            embed_tokens: Token embedding module (only on rank 0).
            lm_head: Language model head (only on last rank).
            final_norm: Final layer norm (only on last rank).
            rotary_emb: Rotary embedding module for position encoding.
            text_model_config: The HuggingFace model config (for layer_types, etc.).
            performance_recorder: Optional recorder for performance instrumentation.
                When provided, per-stage, per-layer, final norm, and lm_head
                timings are recorded via the span context-manager API.
        """
        self.layers = layers
        self.config = config
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        self.final_norm = final_norm
        self.rotary_emb = rotary_emb
        self.text_model_config = text_model_config
        self.performance_recorder: PerformanceRecorder | None = performance_recorder

        # KV cache: one entry per local layer (None until first forward pass)
        self._kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None] = [
            None
        ] * config.num_local_layers

        # HuggingFace DynamicCache for native layer compatibility.
        # Linear attention layers (GatedDeltaNet) store conv_state and recurrent_state
        # inside this cache object. Without it, they have no memory between decode steps.
        self._hf_cache: Any | None = None
        try:
            from transformers.cache_utils import DynamicCache as _DynamicCache
            if text_model_config is not None:
                self._hf_cache = _DynamicCache(config=text_model_config)
            else:
                self._hf_cache = _DynamicCache()
            logger.info(
                f"Pipeline shard rank={config.rank}: created DynamicCache for native layer state"
            )
        except ImportError:
            logger.warning(
                "Pipeline shard: could not import DynamicCache from transformers. "
                "Linear attention layers will not maintain recurrent state between decode steps."
            )

        # Detect layer types for hybrid attention models (Qwen3.5)
        self._layer_types: list[str] = self._detect_layer_types()

        # Attempt torch.compile() for kernel fusion (fallback to eager on failure)
        self._compiled_forward = self._try_compile()

        # Decode shape stability tracking for fast-path communication.
        # The fast path relies on decode activations having a stable shape
        # (batch_size, 1, hidden_size) every step. These fields track whether
        # the shard has entered steady-state decode mode and detect shape changes.
        self._last_output_shape: tuple[int, ...] | None = None
        self._in_decode_steady_state: bool = False

        logger.info(
            f"PipelineParallelShard initialized: rank={config.rank}/{config.world_size}, "
            f"layers=[{config.start_layer}, {config.end_layer}), "
            f"device={config.device}, "
            f"embed={embed_tokens is not None}, "
            f"lm_head={lm_head is not None}, "
            f"final_norm={final_norm is not None}, "
            f"rotary_emb={rotary_emb is not None}, "
            f"layer_types={self._layer_types}, "
            f"hf_cache={self._hf_cache is not None}, "
            f"compiled={self._compiled_forward is not None}, "
            f"instrumented={self.performance_recorder is not None}"
        )

    def _detect_layer_types(self) -> list[str]:
        """Detect layer types (full_attention vs linear_attention) for local layers.

        For Qwen3.5 hybrid models, the config has a `layer_types` list indicating
        which layers use linear attention vs full attention. For other models,
        all layers are assumed to be full_attention.

        Returns:
            List of layer type strings, one per local layer.
        """
        if (
            self.text_model_config is not None
            and hasattr(self.text_model_config, "layer_types")
        ):
            layer_types_all = self.text_model_config.layer_types
            return [
                layer_types_all[i]
                if i < len(layer_types_all)
                else "full_attention"
                for i in range(self.config.start_layer, self.config.end_layer)
            ]
        return ["full_attention"] * self.config.num_local_layers

    def _try_compile(self) -> Any | None:
        """Attempt to compile the local layer forward pass with torch.compile.

        Uses backend="inductor" which supports XPU via the triton-based
        code generation path in PyTorch 2.11+. The "reduce-overhead" mode
        is chosen because the decode loop calls the same function repeatedly
        with the same shapes (batch=1, seq_len=1), making graph capture effective.

        Returns the compiled callable, or None if compilation fails.

        Requirements: 7.1, 7.2
        """
        try:
            compiled = torch.compile(
                self._eager_forward,
                backend="inductor",
                mode="reduce-overhead",
            )
            logger.info(f"torch.compile succeeded for pipeline stage rank={self.config.rank}")
            return compiled
        except Exception as e:
            logger.warning(
                f"torch.compile failed for pipeline stage rank={self.config.rank}: {e}. "
                f"Falling back to eager execution."
            )
            return None

    @property
    def decode_activation_shape(self) -> tuple[int, int, int]:
        """Return the expected decode activation shape for single-request mode.

        During decode, every step produces activations with shape
        (1, 1, hidden_size). The pipeline generator uses this to validate
        shapes before attempting fast-path send.

        Returns:
            (1, 1, hidden_size) tuple representing the expected decode shape.

        Requirements: 2.1, 2.5
        """
        return (1, 1, self.config.hidden_size)

    @property
    def is_decode_shape_stable(self) -> bool:
        """Whether the shard is in steady-state decode mode with stable shapes.

        Returns True when the shard has completed prefill and is producing
        fixed-shape decode activations (batch_size=1, seq_len=1). This signals
        to the pipeline generator that the fast path is safe to use.

        The shard enters steady-state decode after the first decode step
        produces an output matching the expected decode shape. It exits
        steady-state if the output shape changes (e.g., during a new prefill
        or batch size change).

        Requirements: 2.1, 2.2, 2.5
        """
        return self._in_decode_steady_state

    def _validate_decode_output(self, output: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """Validate and prepare decode output for fast-path communication.

        During decode (seq_len=1), validates that the output activation tensor:
        - Has the expected shape (batch_size, 1, hidden_size)
        - Is contiguous in memory (required by the fast path)
        - Has the expected dtype (bfloat16)

        If the output is not contiguous, makes it contiguous before returning.
        Logs shape changes at debug level for diagnostics.

        Args:
            output: The output activation tensor from the forward pass.
            batch_size: Expected batch size.
            seq_len: Sequence length of this step.

        Returns:
            The validated (and possibly made contiguous) output tensor.

        Requirements: 2.1, 2.5
        """
        current_shape = tuple(output.shape)

        # Track shape changes for fast-path stability detection
        if self._last_output_shape is not None and current_shape != self._last_output_shape:
            logger.debug(
                f"Pipeline stage rank={self.config.rank}: output shape changed "
                f"from {self._last_output_shape} to {current_shape}. "
                f"Fast path needs fallback to generic."
            )
            self._in_decode_steady_state = False

        self._last_output_shape = current_shape

        # Only validate and enforce stability for decode steps (seq_len=1)
        if seq_len == 1:
            expected_shape = (batch_size, 1, self.config.hidden_size)

            if current_shape == expected_shape:
                # Shape matches expected decode shape — enter steady state
                self._in_decode_steady_state = True
            else:
                # Shape does not match expected decode shape — not stable
                self._in_decode_steady_state = False
                logger.debug(
                    f"Pipeline stage rank={self.config.rank}: decode output shape "
                    f"{current_shape} does not match expected {expected_shape}. "
                    f"Fast path unavailable."
                )

            # Validate dtype (expected bfloat16 for decode activations)
            if output.dtype != torch.bfloat16:
                logger.debug(
                    f"Pipeline stage rank={self.config.rank}: decode output dtype "
                    f"{output.dtype} is not bfloat16. Fast path requires bfloat16."
                )

            # Ensure contiguity (required by fast-path Gloo send)
            if not output.is_contiguous():
                output = output.contiguous()
                logger.debug(
                    f"Pipeline stage rank={self.config.rank}: made decode output "
                    f"contiguous for fast-path communication."
                )
        else:
            # Prefill step — not in decode steady state
            self._in_decode_steady_state = False

        return output

    def _eager_forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor] | None]]:
        """Eager forward pass through local layers (compilable inner loop).

        This method extracts the layer iteration loop into a standalone callable
        that can be wrapped by torch.compile() for kernel fusion.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            position_ids: Position IDs [batch, seq_len].
            position_embeddings: Precomputed rotary embeddings (cos, sin) or None.
            kv_cache: KV cache list, one entry per local layer.

        Returns:
            (output_hidden_states, updated_kv_cache) tuple.

        Requirements: 7.1
        """
        updated_kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None] = []

        for i, layer in enumerate(self.layers):
            layer_past = kv_cache[i] if i < len(kv_cache) else None

            layer_outputs, new_kv = self._forward_layer(
                layer=layer,
                hidden_states=hidden_states,
                layer_index=i,
                layer_past=layer_past,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs
            updated_kv_cache.append(new_kv)

        return hidden_states, updated_kv_cache

    def forward(
        self,
        input_data: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor] | None]]:
        """Execute local forward pass through assigned layers.

        Args:
            input_data: Token IDs [batch, seq_len] if first stage,
                       hidden_state [batch, seq_len, hidden_size] if not.
            past_key_values: KV cache for local layers only. If None, uses
                internal cache state.

        Returns:
            (output, updated_kv_cache) where output is:
            - logits [batch, seq_len, vocab_size] if last stage
            - hidden_state [batch, seq_len, hidden_size] if not last stage

        Requirements: 2.1, 2.2, 2.3, 2.4, 5.1, 5.2, 6.1, 6.2
        """
        with torch.no_grad():
            # Use provided KV cache or internal state
            if past_key_values is not None:
                kv_cache = past_key_values
            else:
                kv_cache = self._kv_cache

            # Embedding: only on first stage (rank 0)
            if self.config.is_first_stage and self.embed_tokens is not None:
                hidden_states = self.embed_tokens(input_data)
            else:
                hidden_states = input_data

            batch_size, seq_len, _ = hidden_states.shape

            # Determine mode for instrumentation: prefill vs decode
            mode = "prefill" if seq_len > 1 else "decode"

            # Compute position IDs based on past KV cache length
            past_seq_len = self._get_past_seq_len(kv_cache)
            position_ids = torch.arange(
                past_seq_len,
                past_seq_len + seq_len,
                device=hidden_states.device,
            ).unsqueeze(0).expand(batch_size, -1)

            # Compute rotary position embeddings if available
            position_embeddings = None
            if self.rotary_emb is not None:
                position_embeddings = self.rotary_emb(hidden_states, position_ids)

            # Wrap the entire stage compute in an instrumentation span
            recorder = self.performance_recorder
            if recorder is not None:
                with recorder.span(
                    "stage_compute",
                    mode=mode,
                    metadata={"rank": self.config.rank, "seq_len": seq_len},
                ):
                    hidden_states, updated_kv_cache = self._execute_layers(
                        hidden_states, position_ids, position_embeddings, kv_cache, mode
                    )

                    # Final norm + lm_head: only on last stage
                    if self.config.is_last_stage:
                        if self.final_norm is not None:
                            with recorder.span(
                                "final_norm",
                                mode=mode,
                                metadata={"rank": self.config.rank},
                            ):
                                hidden_states = self.final_norm(hidden_states)
                        if self.lm_head is not None:
                            with recorder.span(
                                "lm_head",
                                mode=mode,
                                metadata={"rank": self.config.rank},
                            ):
                                hidden_states = self.lm_head(hidden_states)
            else:
                hidden_states, updated_kv_cache = self._execute_layers(
                    hidden_states, position_ids, position_embeddings, kv_cache, mode
                )

                # Final norm + lm_head: only on last stage
                if self.config.is_last_stage:
                    if self.final_norm is not None:
                        hidden_states = self.final_norm(hidden_states)
                    if self.lm_head is not None:
                        hidden_states = self.lm_head(hidden_states)

            # Update internal KV cache state
            self._kv_cache = updated_kv_cache

            # Validate decode output shape stability for fast-path communication.
            # Only applies to non-last-stage shards (last stage returns logits).
            if not self.config.is_last_stage:
                hidden_states = self._validate_decode_output(
                    hidden_states, batch_size, seq_len
                )

            return hidden_states, updated_kv_cache

    def _execute_layers(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None],
        mode: str,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor] | None]]:
        """Execute layers through compiled or eager path.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            position_ids: Position IDs [batch, seq_len].
            position_embeddings: Precomputed rotary embeddings (cos, sin) or None.
            kv_cache: KV cache list, one entry per local layer.
            mode: "prefill" or "decode" for instrumentation classification.

        Returns:
            (output_hidden_states, updated_kv_cache) tuple.
        """
        if self._compiled_forward is not None:
            try:
                hidden_states, updated_kv_cache = self._compiled_forward(
                    hidden_states, position_ids, position_embeddings, kv_cache
                )
            except Exception as compile_err:
                # torch.compile wraps lazily — actual compilation happens on
                # first invocation and can fail if triton/inductor is missing.
                # Fall back to eager permanently.
                logger.warning(
                    f"torch.compile runtime failure on rank={self.config.rank}: "
                    f"{compile_err}. Disabling compiled path permanently."
                )
                self._compiled_forward = None
                hidden_states, updated_kv_cache = self._eager_forward(
                    hidden_states, position_ids, position_embeddings, kv_cache
                )
        else:
            hidden_states, updated_kv_cache = self._eager_forward(
                hidden_states, position_ids, position_embeddings, kv_cache
            )
        return hidden_states, updated_kv_cache

    def _forward_layer(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        layer_index: int,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None,
        position_ids: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Forward pass through a single transformer layer.

        Handles both full attention and linear attention layers transparently.
        Each HuggingFace layer module expects:
            layer(hidden_states, position_embeddings=None, attention_mask=None,
                  position_ids=None, past_key_values=None, use_cache=True)

        For linear attention layers (Qwen3.5 hybrid), the layer uses DynamicCache
        internally and the KV cache entry is None.

        Args:
            layer: The transformer layer module.
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            layer_index: Local layer index (0-based within this shard).
            layer_past: Past KV cache for this layer, or None.
            position_ids: Position IDs [batch, seq_len].
            position_embeddings: Precomputed rotary embeddings (cos, sin) or None.

        Returns:
            (output_hidden_states, new_kv_entry) tuple.
        """
        layer_type = self._layer_types[layer_index]
        global_layer_index = self.config.start_layer + layer_index

        # Build kwargs for the layer call
        layer_kwargs: dict[str, Any] = {
            "attention_mask": None,
            "position_ids": position_ids,
            "use_cache": True,
        }

        if position_embeddings is not None:
            layer_kwargs["position_embeddings"] = position_embeddings

        # Pass the HuggingFace DynamicCache to ALL layers (both full_attention
        # and linear_attention). Linear attention layers (GatedDeltaNet) store
        # conv_state and recurrent_state inside this cache object. Without it,
        # they lose memory between decode steps and produce garbage.
        #
        # CRITICAL: The kwarg name MUST be "past_key_values" (plural) to match
        # Qwen3_5DecoderLayer.forward(). Using singular "past_key_value" causes
        # the argument to be swallowed by **kwargs and silently ignored.
        if self._hf_cache is not None:
            layer_kwargs["past_key_values"] = self._hf_cache
        elif layer_type == "full_attention" and layer_past is not None:
            layer_kwargs["past_key_values"] = layer_past
        else:
            layer_kwargs["past_key_values"] = None

        # Determine mode from sequence length for per-layer instrumentation
        seq_len = hidden_states.shape[1]
        mode = "prefill" if seq_len > 1 else "decode"

        # Call the layer with optional instrumentation
        recorder = self.performance_recorder
        if recorder is not None:
            with recorder.span(
                "layer_compute",
                mode=mode,
                metadata={
                    "layer_index": global_layer_index,
                    "layer_type": layer_type,
                    "rank": self.config.rank,
                },
            ):
                layer_outputs = layer(hidden_states, **layer_kwargs)
        else:
            layer_outputs = layer(hidden_states, **layer_kwargs)

        # Extract hidden states (always first element)
        if isinstance(layer_outputs, tuple):
            output_hidden_states = layer_outputs[0]
        else:
            output_hidden_states = layer_outputs

        # Extract KV cache update
        new_kv: tuple[torch.Tensor, torch.Tensor] | None = None
        if self._hf_cache is not None:
            # When using HuggingFace DynamicCache, the cache is updated in-place
            # by the layer. We don't need to extract KV entries manually.
            pass
        elif layer_type == "full_attention":
            if isinstance(layer_outputs, tuple) and len(layer_outputs) > 1:
                cache_entry = layer_outputs[1]
                if cache_entry is not None:
                    # HuggingFace layers return (key, value) tuples or Cache objects
                    if isinstance(cache_entry, tuple):
                        new_kv = cache_entry
                    elif hasattr(cache_entry, "key_cache") and hasattr(
                        cache_entry, "value_cache"
                    ):
                        # DynamicCache-style object — extract tensors
                        new_kv = (cache_entry.key_cache, cache_entry.value_cache)
        # Linear attention layers manage their own recurrent state internally;
        # we store None for their KV cache entry.

        return output_hidden_states, new_kv

    def _get_past_seq_len(
        self, kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None]
    ) -> int:
        """Get the past sequence length from the KV cache.

        Looks through the cache entries to find the first non-None entry
        and returns its sequence length dimension. Also checks the HF
        DynamicCache if available.

        Args:
            kv_cache: List of (key, value) tuples or None entries.

        Returns:
            Past sequence length (0 if no cache entries exist).
        """
        # Check HF DynamicCache first (it tracks seq_length internally)
        if self._hf_cache is not None and hasattr(self._hf_cache, "get_seq_length"):
            seq_len = self._hf_cache.get_seq_length()
            if seq_len > 0:
                return seq_len

        for entry in kv_cache:
            if entry is not None:
                # key shape: [batch, num_kv_heads, seq_len, head_dim]
                return entry[0].shape[2]
        return 0

    def reset_state(self) -> None:
        """Reset KV cache and linear attention recurrent state for new sequence.

        Requirements: 5.3, 6.3
        """
        self._kv_cache = [None] * self.config.num_local_layers
        # Reset decode shape stability tracking
        self._last_output_shape = None
        self._in_decode_steady_state = False
        # Reset the HuggingFace DynamicCache (clears linear attention recurrent state)
        if self._hf_cache is not None:
            try:
                from transformers.cache_utils import DynamicCache as _DynamicCache
                if self.text_model_config is not None:
                    self._hf_cache = _DynamicCache(config=self.text_model_config)
                else:
                    self._hf_cache = _DynamicCache()
            except ImportError:
                self._hf_cache = None
        logger.debug(
            f"Pipeline stage rank={self.config.rank}: state reset "
            f"({self.config.num_local_layers} cache entries cleared, hf_cache recreated, "
            f"decode shape stability reset)"
        )

    def __call__(
        self,
        input_data: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor] | None]]:
        """Make the shard callable like a model."""
        return self.forward(input_data, past_key_values)

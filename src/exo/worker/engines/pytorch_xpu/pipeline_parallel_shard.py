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
    from exo.worker.engines.pytorch_xpu.compiled_decode_path import (
        CompiledDecodePath,
    )
    from exo.worker.engines.pytorch_xpu.continuous_batching import (
        DecodeMicrobatch,
        PerRequestCacheManager,
        TokenResultBatch,
    )
    from exo.worker.engines.pytorch_xpu.decode_output_buffer_pool import (
        DecodeOutputBufferPool,
    )
    from exo.worker.engines.pytorch_xpu.gated_deltanet_cache import (
        GatedDeltaNetCache,
    )
    from exo.worker.engines.pytorch_xpu.instrumentation import PerformanceRecorder
    from exo.worker.engines.pytorch_xpu.local_shard_loader import (
        LocalShardModules,
    )
    from exo.worker.engines.pytorch_xpu.pipeline_config import (
        PipelineStageAssignment,
        PytorchXpuOptimizationConfiguration,
    )

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
        optimization_config: PytorchXpuOptimizationConfiguration | None = None,
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
            optimization_config: Optional optimization configuration. When provided,
                the ``enable_gated_deltanet_persistent_state`` flag controls whether
                GatedDeltaNet persistent state, GatedDeltaNetCache wrapper, and
                DecodeOutputBufferPool are created. When None, defaults to enabled
                (all optimizations active).
        """
        self.layers = layers
        self.config = config

        # Validate that only the assigned layers are present — no extra layers loaded.
        assert len(layers) == config.num_local_layers, (
            f"Expected {config.num_local_layers} layers for rank {config.rank}, "
            f"got {len(layers)}"
        )

        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        self.final_norm = final_norm
        self.rotary_emb = rotary_emb
        self.text_model_config = text_model_config
        self.performance_recorder: PerformanceRecorder | None = performance_recorder
        self._optimization_config: PytorchXpuOptimizationConfiguration | None = optimization_config

        # Determine whether GatedDeltaNet persistent state optimization is enabled.
        # When no config is provided, default to enabled (optimization active).
        self._gated_deltanet_persistent_state_enabled: bool = (
            optimization_config.enable_gated_deltanet_persistent_state
            if optimization_config is not None
            else True
        )

        # Determine whether sync removal optimization is enabled.
        # When enabled, the decode hot path avoids .item(), .cpu(), .numpy()
        # calls on XPU tensors (Req 1.1, 1.2, 1.3).
        # When no config is provided, default to disabled (safe default).
        self._sync_removal_enabled: bool = (
            optimization_config.enable_sync_removal
            if optimization_config is not None
            else False
        )

        # KV cache: one entry per local layer (None until first forward pass)
        self._kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None] = [
            None
        ] * config.num_local_layers

        # HuggingFace DynamicCache wrapped in GatedDeltaNetCache for native layer
        # compatibility. Linear attention layers (GatedDeltaNet) store conv_state and
        # recurrent_state inside this cache object. The GatedDeltaNetCache wrapper
        # additionally stores persistent fp32 recurrent state for the optimized decode
        # path. Without it, layers have no memory between decode steps.
        self._hf_cache: Any | None = None
        self._gated_deltanet_cache: GatedDeltaNetCache | None = None
        try:
            from transformers.cache_utils import DynamicCache as _DynamicCache

            if text_model_config is not None:
                dynamic_cache = _DynamicCache(config=text_model_config)
            else:
                dynamic_cache = _DynamicCache()

            if self._gated_deltanet_persistent_state_enabled:
                # Optimization enabled: wrap DynamicCache in GatedDeltaNetCache
                # for persistent fp32 state storage
                from exo.worker.engines.pytorch_xpu.gated_deltanet_cache import (
                    GatedDeltaNetCache as _GatedDeltaNetCache,
                )
                self._gated_deltanet_cache = _GatedDeltaNetCache(dynamic_cache=dynamic_cache)
                # _hf_cache points to the wrapper — it delegates DynamicCache operations
                # transparently, so existing layer code continues to work unchanged.
                self._hf_cache = self._gated_deltanet_cache
                logger.info(
                    f"Pipeline shard rank={config.rank}: created GatedDeltaNetCache "
                    f"(wrapping DynamicCache) for native layer state"
                )
            else:
                # Optimization disabled: use plain DynamicCache (existing behavior)
                self._hf_cache = dynamic_cache
                logger.info(
                    f"Pipeline shard rank={config.rank}: created plain DynamicCache "
                    f"(GatedDeltaNet persistent state disabled)"
                )
        except ImportError:
            logger.warning(
                "Pipeline shard: could not import DynamicCache or GatedDeltaNetCache. "
                "Linear attention layers will not maintain recurrent state between decode steps."
            )

        # Decode output buffer pool for reusable intermediate tensors
        # (only created when GatedDeltaNet persistent state optimization is enabled)
        self._decode_output_buffer_pool: DecodeOutputBufferPool | None = None
        if self._gated_deltanet_persistent_state_enabled:
            try:
                from exo.worker.engines.pytorch_xpu.decode_output_buffer_pool import (
                    DecodeOutputBufferPool as _DecodeOutputBufferPool,
                )
                self._decode_output_buffer_pool = _DecodeOutputBufferPool()
                logger.info(
                    f"Pipeline shard rank={config.rank}: created DecodeOutputBufferPool"
                )
            except ImportError:
                logger.debug(
                    "Pipeline shard: could not import DecodeOutputBufferPool. "
                    "Decode output buffer reuse unavailable."
                )
        else:
            logger.debug(
                f"Pipeline shard rank={config.rank}: DecodeOutputBufferPool skipped "
                f"(GatedDeltaNet persistent state disabled)"
            )

        # Current request identifier for GatedDeltaNet persistent state tracking.
        # Set via initialize_request() before decode begins, cleared on reset_state().
        self._current_request_id: str | None = None

        # Detect layer types for hybrid attention models (Qwen3.5)
        self._layer_types: list[str] = self._detect_layer_types()

        # Detect the correct kwarg name for passing KV cache to layers.
        # Phi-3/Phi-4 uses "past_key_value" (singular), Qwen3.5/3.6 uses "past_key_values" (plural).
        import inspect
        _first_layer_params = inspect.signature(layers[0].forward).parameters
        if "past_key_value" in _first_layer_params and "past_key_values" not in _first_layer_params:
            self._cache_kwarg_name: str = "past_key_value"
        else:
            self._cache_kwarg_name = "past_key_values"
        logger.info(f"Pipeline shard rank={config.rank}: cache kwarg = '{self._cache_kwarg_name}'")

        # Attempt torch.compile() for kernel fusion (fallback to eager on failure)
        self._compiled_forward = self._try_compile()

        # CompiledDecodePath: full-graph compiled decode for single-token steps.
        # Created when enable_torch_compile is True AND this shard has all required
        # components (embed_tokens, lm_head, final_norm — i.e., a full model or
        # the last stage with all components). Gate behind enable_torch_compile flag.
        # Req 4.1, 11.1, 11.3
        self._compiled_decode_path: CompiledDecodePath | None = None
        self._torch_compile_enabled: bool = (
            optimization_config.enable_torch_compile
            if optimization_config is not None
            else False
        )
        if self._torch_compile_enabled:
            self._compiled_decode_path = self._try_create_compiled_decode_path()

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
            f"gated_deltanet_cache={self._gated_deltanet_cache is not None}, "
            f"gated_deltanet_persistent_state={self._gated_deltanet_persistent_state_enabled}, "
            f"sync_removal={self._sync_removal_enabled}, "
            f"decode_buffer_pool={self._decode_output_buffer_pool is not None}, "
            f"compiled={self._compiled_forward is not None}, "
            f"compiled_decode_path={self._compiled_decode_path is not None}, "
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

    def _try_create_compiled_decode_path(self) -> CompiledDecodePath | None:
        """Attempt to create a CompiledDecodePath for optimized single-token decode.

        The CompiledDecodePath requires embed_tokens, lm_head, and final_norm to be
        present (full model on this shard, or a single-stage pipeline). If any are
        missing, or if creation fails, returns None and logs at WARNING level.

        The caller falls back to the existing _forward_layer loop when this returns None.

        Returns:
            A CompiledDecodePath instance, or None if creation failed.

        Requirements: 4.1, 11.1, 11.3
        """
        # CompiledDecodePath requires all model components on this shard
        if self.embed_tokens is None or self.lm_head is None or self.final_norm is None:
            logger.info(
                f"Pipeline stage rank={self.config.rank}: CompiledDecodePath not created "
                f"(requires embed_tokens, lm_head, and final_norm on this shard). "
                f"embed={self.embed_tokens is not None}, lm_head={self.lm_head is not None}, "
                f"final_norm={self.final_norm is not None}"
            )
            return None

        try:
            from exo.worker.engines.pytorch_xpu.compiled_decode_path import (
                CompiledDecodePath as _CompiledDecodePath,
            )

            compiled_path = _CompiledDecodePath(
                layers=self.layers,
                embed_tokens=self.embed_tokens,
                lm_head=self.lm_head,
                final_norm=self.final_norm,
                config=self.config,
                optimization_config=self._optimization_config,  # type: ignore[arg-type]
            )

            if compiled_path.is_compiled:
                logger.info(
                    f"Pipeline stage rank={self.config.rank}: CompiledDecodePath created "
                    f"and compiled (max_seq_len={compiled_path.max_seq_len})"
                )
            else:
                logger.warning(
                    f"Pipeline stage rank={self.config.rank}: CompiledDecodePath created "
                    f"but compilation failed. Will use as eager fallback."
                )

            return compiled_path

        except Exception as exc:
            logger.warning(
                f"Pipeline stage rank={self.config.rank}: CompiledDecodePath creation "
                f"failed: {exc}. Falling back to standard _forward_layer loop."
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

        Requirements: 2.1, 2.2, 2.3, 2.4, 4.1, 5.1, 5.2, 6.1, 6.2, 11.1, 11.3
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

            # --- CompiledDecodePath dispatch (Req 4.1, 11.1, 11.3) ---
            # For single-token decode steps, use the compiled decode path if
            # available. This bypasses the Python _forward_layer loop entirely
            # and executes the full layer stack as a compiled graph.
            if (
                seq_len == 1
                and self._compiled_decode_path is not None
                and self._compiled_decode_path.is_compiled
            ):
                try:
                    # Extract the token ID from the input for the compiled path.
                    # If we already embedded (first stage), we need the original
                    # token ID. If not first stage, the compiled path handles
                    # the full pipeline including embedding, so this path only
                    # applies when we have all components.
                    token_id = input_data[0, 0] if input_data.dim() == 2 else input_data[0, 0, 0].to(torch.int64)

                    # Use the compiled path's internal position tracking
                    position = self._compiled_decode_path.position

                    # Ensure token_id is a 1D tensor of shape [1]
                    if token_id.dim() == 0:
                        token_id = token_id.unsqueeze(0)

                    logits = self._compiled_decode_path.forward(token_id, position)

                    # The compiled path returns logits [1, vocab_size].
                    # Reshape to [batch, 1, vocab_size] for consistency.
                    if logits.dim() == 2:
                        logits = logits.unsqueeze(1)

                    # Return logits with empty KV cache update (compiled path
                    # manages its own static cache internally).
                    return logits, self._kv_cache

                except Exception as exc:
                    # Runtime failure in compiled path — fall back to eager
                    # and disable the compiled path permanently (Req 11.3).
                    logger.warning(
                        f"Pipeline stage rank={self.config.rank}: CompiledDecodePath "
                        f"runtime failure: {exc}. Falling back to eager execution "
                        f"permanently."
                    )
                    self._compiled_decode_path = None
                    # Fall through to the standard eager path below

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
        # CRITICAL: The kwarg name MUST match the layer's forward() signature.
        # Qwen3.5/3.6 uses "past_key_values" (plural), Phi-3/4 uses "past_key_value" (singular).
        # Using the wrong name causes the argument to be swallowed by **kwargs.
        if self._hf_cache is not None:
            layer_kwargs[self._cache_kwarg_name] = self._hf_cache
        elif layer_type == "full_attention" and layer_past is not None:
            layer_kwargs[self._cache_kwarg_name] = layer_past
        else:
            layer_kwargs[self._cache_kwarg_name] = None

        # Determine mode from sequence length for per-layer instrumentation
        seq_len = hidden_states.shape[1]
        mode = "prefill" if seq_len > 1 else "decode"

        # Instrumentation: count optimized vs baseline path usage for
        # GatedDeltaNet layers during decode. The optimized path avoids the
        # bf16→fp32 state cast because persistent state is already fp32.
        recorder = self.performance_recorder
        if recorder is not None and mode == "decode" and layer_type == "linear_attention":
            if self._gated_deltanet_persistent_state_enabled:
                recorder.increment_counter("per_token_casts_avoided")
            else:
                recorder.increment_counter("per_token_casts_baseline")

        # Call the layer with optional instrumentation
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

        Clears all cache state including:
        - Per-layer KV cache entries
        - HuggingFace DynamicCache (via GatedDeltaNetCache wrapper)
        - GatedDeltaNet persistent fp32 recurrent states
        - Decode output buffer pool
        - Decode shape stability tracking
        - Current request identifier
        - CompiledDecodePath static cache (if active)

        Requirements: 5.3, 5.7, 5.8, 6.3
        """
        self._kv_cache = [None] * self.config.num_local_layers
        # Reset decode shape stability tracking
        self._last_output_shape = None
        self._in_decode_steady_state = False
        # Clear current request identifier
        self._current_request_id = None

        # Reset the CompiledDecodePath static cache for new generation
        self.reset_compiled_path()

        # Recycle GatedDeltaNet persistent states (zeros tensors, keeps containers
        # available for reuse by the next request — avoids reallocation cost)
        if (
            self._gated_deltanet_persistent_state_enabled
            and self._gated_deltanet_cache is not None
        ):
            self._gated_deltanet_cache.recycle_all_gated_deltanet_states()

        # Clear decode output buffer pool (releases intermediate tensor memory)
        if (
            self._gated_deltanet_persistent_state_enabled
            and self._decode_output_buffer_pool is not None
        ):
            self._decode_output_buffer_pool.clear()

        # Reset the HuggingFace DynamicCache (clears linear attention recurrent state).
        # We recreate the DynamicCache inside the GatedDeltaNetCache wrapper to ensure
        # the HF layer state (conv_state, recurrent_state stored by the layer itself)
        # is fully cleared. The GatedDeltaNet persistent states are separate and handled
        # above via recycle_all_gated_deltanet_states().
        if self._gated_deltanet_cache is not None:
            try:
                from transformers.cache_utils import DynamicCache as _DynamicCache
                if self.text_model_config is not None:
                    new_dynamic_cache = _DynamicCache(config=self.text_model_config)
                else:
                    new_dynamic_cache = _DynamicCache()
                # Replace the underlying DynamicCache in the wrapper
                self._gated_deltanet_cache._dynamic_cache = new_dynamic_cache
            except ImportError:
                pass
        elif self._hf_cache is not None:
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
            f"({self.config.num_local_layers} cache entries cleared, "
            f"hf_cache recreated, gated_deltanet states recycled, "
            f"decode buffer pool cleared, decode shape stability reset)"
        )

    def reset_compiled_path(self) -> None:
        """Reset the CompiledDecodePath static cache for a new generation.

        Calls ``CompiledDecodePath.reset()`` to zero the static KV cache position
        and reuse the pre-allocated tensor storage. This must be called at the start
        of each new generation request.

        If no CompiledDecodePath is active, this is a no-op.

        Requirements: 2.4, 4.1
        """
        if self._compiled_decode_path is not None:
            self._compiled_decode_path.reset()
            logger.debug(
                f"Pipeline stage rank={self.config.rank}: CompiledDecodePath cache reset"
            )

    def initialize_request(self, request_id: str) -> None:
        """Initialize per-request state before decode begins.

        Sets the current request identifier and initializes GatedDeltaNet persistent
        fp32 state for all linear_attention layers in this shard. This must be called
        after prefill completes and before the first decode step.

        The persistent state eliminates per-token bf16→fp32 promotion and per-token
        tensor allocation during the recurrent decode step.

        When ``enable_gated_deltanet_persistent_state`` is disabled, only the request
        identifier is set — no persistent state initialization occurs.

        Args:
            request_id: Unique identifier for the generation request. Used to track
                which request owns which persistent state containers.

        Requirements: 5.1, 5.2, 5.7, 5.8
        """
        self._current_request_id = request_id

        # Initialize GatedDeltaNet persistent state for linear_attention layers
        # (only when the optimization is enabled)
        if (
            self._gated_deltanet_persistent_state_enabled
            and self._gated_deltanet_cache is not None
        ):
            self._initialize_gated_deltanet_persistent_states(request_id)

        logger.debug(
            f"Pipeline stage rank={self.config.rank}: initialized request {request_id!r}, "
            f"persistent_state_enabled={self._gated_deltanet_persistent_state_enabled}, "
            f"gated_deltanet_states={self._gated_deltanet_cache.gated_deltanet_state_count if self._gated_deltanet_cache else 0}"
        )

    def _initialize_gated_deltanet_persistent_states(self, request_id: str) -> None:
        """Initialize persistent fp32 state for all GatedDeltaNet layers in this shard.

        For each linear_attention layer, creates (or reuses) a
        GatedDeltaNetPersistentState container with preallocated fp32 tensors.
        This is called once per request, not per token.

        The model dimensions are extracted from the text_model_config. If the config
        is unavailable or dimensions cannot be determined, initialization is skipped
        with a warning (the shard falls back to the standard decode path).

        Args:
            request_id: Unique identifier for the owning request.
        """
        if self._gated_deltanet_cache is None:
            return

        # Extract model dimensions from text_model_config
        config = self.text_model_config
        if config is None:
            logger.debug(
                f"Pipeline stage rank={self.config.rank}: no text_model_config, "
                f"skipping GatedDeltaNet persistent state initialization"
            )
            return

        # Qwen3.5 model dimensions (from HuggingFace config attributes)
        # num_attention_heads: total query heads (for GatedDeltaNet, this is num_v_heads)
        # For linear attention layers in Qwen3.5:
        #   key_dim = head_dim (hidden_size / num_attention_heads)
        #   value_dim = head_dim
        #   conv_dim = num_heads * (key_dim + value_dim) typically
        #   conv_kernel_size = 4 (default for Qwen3.5)
        num_heads = getattr(config, "num_attention_heads", None)
        hidden_size = getattr(config, "hidden_size", None)
        conv_kernel_size = getattr(config, "conv_kernel_size", 4)

        if num_heads is None or hidden_size is None:
            logger.debug(
                f"Pipeline stage rank={self.config.rank}: cannot determine model "
                f"dimensions (num_heads={num_heads}, hidden_size={hidden_size}), "
                f"skipping GatedDeltaNet persistent state initialization"
            )
            return

        head_dim = hidden_size // num_heads
        # For GatedDeltaNet layers, key_dim and value_dim are both head_dim
        key_dim = head_dim
        value_dim = head_dim
        # conv_dim is typically the projection dimension for the linear attention
        # In Qwen3.5, this is num_heads * (key_dim + value_dim) = hidden_size * 2
        # But the actual conv_dim depends on the layer's internal projection.
        # Use a safe default: num_heads * (key_dim + value_dim)
        conv_dim = num_heads * (key_dim + value_dim)

        device = torch.device(self.config.device)

        try:
            from exo.worker.engines.pytorch_xpu.gated_deltanet_state import (
                initialize_gated_deltanet_state,
            )
        except ImportError:
            logger.warning(
                f"Pipeline stage rank={self.config.rank}: could not import "
                f"initialize_gated_deltanet_state, skipping persistent state init"
            )
            return

        initialized_count = 0
        for local_idx, layer_type in enumerate(self._layer_types):
            if layer_type == "linear_attention":
                global_layer_index = self.config.start_layer + local_idx
                initialize_gated_deltanet_state(
                    cache=self._gated_deltanet_cache,
                    request_identifier=request_id,
                    layer_index=global_layer_index,
                    batch_size=1,  # Single-request decode
                    num_heads=num_heads,
                    key_dim=key_dim,
                    value_dim=value_dim,
                    conv_dim=conv_dim,
                    conv_kernel_size=conv_kernel_size,
                    device=device,
                )
                initialized_count += 1

        if initialized_count > 0:
            logger.info(
                f"Pipeline stage rank={self.config.rank}: initialized {initialized_count} "
                f"GatedDeltaNet persistent states for request {request_id!r} "
                f"(num_heads={num_heads}, key_dim={key_dim}, value_dim={value_dim}, "
                f"conv_dim={conv_dim}, device={device})"
            )

    # -------------------------------------------------------------------
    # Continuous Batching: Microbatch Interface
    # -------------------------------------------------------------------

    def forward_microbatch(
        self,
        microbatch: DecodeMicrobatch,
        cache_manager: PerRequestCacheManager,
    ) -> TokenResultBatch:
        """Execute a batched decode step for multiple concurrent requests.

        Gathers per-request cache entries from the cache_manager, builds a
        batched input tensor from the microbatch's input_token_ids, runs the
        forward pass, and returns token results for each active slot.

        For now, this iterates over active slots and calls the existing
        forward() method once per request (single-request-at-a-time). True
        batched tensor operations will replace this when the full continuous
        batching tensor path is implemented.

        Args:
            microbatch: The decode microbatch describing active slots and
                their input tokens.
            cache_manager: Per-request cache manager for gathering/updating
                per-request state.

        Returns:
            TokenResultBatch with generated tokens for each active slot.

        Raises:
            ValueError: If microbatch has no active slots.

        Requirements: 3.1, 3.2, 3.3, 3.9
        """
        from exo.worker.engines.pytorch_xpu.continuous_batching import (
            TokenResultBatch as _TokenResultBatch,
        )

        if microbatch.active_slot_count <= 0:
            raise ValueError(
                "Cannot forward_microbatch with zero active slots"
            )

        token_ids: list[int] = []
        slot_indices: list[int] = []
        slot_generations: list[int] = []

        for slot_state in microbatch.slot_states:
            if not slot_state.is_active:
                continue
            if slot_state.request_id is None:
                continue

            request_id = slot_state.request_id

            # Gather per-request cache entry
            cache_entry = cache_manager.get_cache(request_id)
            if cache_entry is None:
                logger.warning(
                    "forward_microbatch: no cache for request %r at slot %d, skipping",
                    request_id,
                    slot_state.slot_index,
                )
                continue

            # Find the input token for this slot
            # input_token_ids is ordered by active slot position
            active_idx = slot_indices.__len__()  # current position in active list
            if active_idx >= len(microbatch.input_token_ids):
                logger.warning(
                    "forward_microbatch: input_token_ids exhausted at active_idx=%d",
                    active_idx,
                )
                break

            input_token_id = microbatch.input_token_ids[active_idx]

            # Build single-token input tensor for this request
            input_tensor = torch.tensor(
                [[input_token_id]], dtype=torch.long, device=self.config.device
            )

            # Set current request context for GatedDeltaNet state tracking
            self._current_request_id = request_id

            # Execute forward pass for this single request
            # (true batching will replace this loop with a single batched call)
            output, _ = self.forward(input_tensor)

            # For the last stage, output is logits [1, 1, vocab_size]
            # Take argmax as the generated token (greedy placeholder)
            if self.config.is_last_stage and output.dim() == 3:
                # When enable_sync_removal is active, avoid .item() on XPU
                # tensors during the decode hot path. The actual token
                # extraction is deferred to the pipeline generator /
                # OnDeviceSampler which handles on-device sampling and
                # async CPU transfer. Use 0 as a placeholder here.
                # (Req 1.1: no .item() on XPU tensors during decode)
                if self._sync_removal_enabled:
                    generated_token = 0
                else:
                    generated_token = int(torch.argmax(output[0, -1, :]).item())
            else:
                # Non-last stages produce hidden states, not tokens.
                # Use a placeholder token_id of 0 (the pipeline generator
                # handles actual token extraction from the final rank).
                generated_token = 0

            token_ids.append(generated_token)
            slot_indices.append(slot_state.slot_index)
            slot_generations.append(slot_state.slot_generation)

        return _TokenResultBatch(
            token_ids=tuple(token_ids),
            slot_indices=tuple(slot_indices),
            slot_generations=tuple(slot_generations),
        )

    def validate_microbatch_compatibility(
        self,
        microbatch: DecodeMicrobatch,
        cache_manager: PerRequestCacheManager,
    ) -> list[str]:
        """Validate that all requests in the microbatch have valid caches.

        Checks each active slot's request_id against the cache_manager to
        ensure a cache entry exists. Returns a list of request IDs that are
        missing or have invalid caches.

        Args:
            microbatch: The decode microbatch to validate.
            cache_manager: Per-request cache manager to check against.

        Returns:
            List of request IDs with missing or invalid caches. Empty list
            means all requests are valid and ready for decode.

        Requirements: 3.2, 3.9
        """
        missing_cache_requests: list[str] = []

        for slot_state in microbatch.slot_states:
            if not slot_state.is_active:
                continue
            if slot_state.request_id is None:
                continue

            request_id = slot_state.request_id
            if not cache_manager.has_cache(request_id):
                missing_cache_requests.append(request_id)

        return missing_cache_requests

    @property
    def current_request_id(self) -> str | None:
        """The current request identifier, or None if no request is active."""
        return self._current_request_id

    @property
    def gated_deltanet_persistent_state_enabled(self) -> bool:
        """Whether GatedDeltaNet persistent state optimization is active."""
        return self._gated_deltanet_persistent_state_enabled

    @property
    def sync_removal_enabled(self) -> bool:
        """Whether implicit synchronization removal is active on the decode hot path.

        When True, the shard avoids .item(), .cpu(), .numpy() calls on XPU
        tensors during decode. All output tensors remain on XPU device.

        Requirements: 1.1, 1.2, 1.3
        """
        return self._sync_removal_enabled

    @property
    def compiled_decode_path(self) -> CompiledDecodePath | None:
        """Access the CompiledDecodePath instance (None if not created or disabled).

        Requirements: 4.1, 11.1
        """
        return self._compiled_decode_path

    @property
    def gated_deltanet_cache(self) -> GatedDeltaNetCache | None:
        """Access the GatedDeltaNetCache wrapper (for external state queries)."""
        return self._gated_deltanet_cache

    @property
    def decode_output_buffer_pool(self) -> DecodeOutputBufferPool | None:
        """Access the decode output buffer pool (for external diagnostics)."""
        return self._decode_output_buffer_pool

    @property
    def stage_assignment(self) -> PipelineStageAssignment:
        """Return the PipelineStageAssignment for this shard.

        Derives a PipelineStageAssignment from the existing PipelineStageConfig,
        bridging the dataclass-based config to the Pydantic model used by the
        local shard loader.
        """
        from exo.worker.engines.pytorch_xpu.pipeline_config import (
            PipelineStageAssignment as _PipelineStageAssignment,
        )

        return _PipelineStageAssignment(
            rank=self.config.rank,
            start_layer=self.config.start_layer,
            end_layer=self.config.end_layer,
            owns_embedding=self.config.is_first_stage,
            owns_lm_head=self.config.is_last_stage,
        )

    @property
    def loaded_layer_range(self) -> tuple[int, int]:
        """Return (start_layer, end_layer) for the layers loaded on this rank."""
        return (self.config.start_layer, self.config.end_layer)

    @classmethod
    def from_local_shard_modules(
        cls,
        modules: LocalShardModules,
        config: PipelineStageConfig,
        text_model_config: Any | None = None,
        performance_recorder: PerformanceRecorder | None = None,
        optimization_config: PytorchXpuOptimizationConfiguration | None = None,
    ) -> PipelineParallelShard:
        """Create a PipelineParallelShard from pre-built local shard modules.

        This factory method provides a cleaner path from the local shard loader
        output (LocalShardModules) to a fully initialized PipelineParallelShard.
        It extracts layers, embedding, lm_head, final_norm, and rotary_emb from
        the LocalShardModules and passes them to the standard constructor.

        Args:
            modules: Pre-built local shard modules from build_local_shard_modules().
            config: Pipeline stage configuration (PipelineStageConfig).
            text_model_config: Optional HuggingFace model config for layer types.
            performance_recorder: Optional performance instrumentation recorder.
            optimization_config: Optional optimization configuration flags.

        Returns:
            A fully initialized PipelineParallelShard.
        """
        return cls(
            layers=modules.layers,
            config=config,
            embed_tokens=modules.embed_tokens,
            lm_head=modules.lm_head,
            final_norm=modules.final_norm,
            rotary_emb=modules.rotary_emb,
            text_model_config=text_model_config,
            performance_recorder=performance_recorder,
            optimization_config=optimization_config,
        )

    def __call__(
        self,
        input_data: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor] | None]]:
        """Make the shard callable like a model."""
        return self.forward(input_data, past_key_values)

"""Compiled decode path wrapping torch.compile() for the XPU Inductor backend.

Orchestrates compilation of the tensor-only decode function and provides the
entry point for the optimized decode path. Extracts weights from HuggingFace
model layers, packs projections, creates a static KV cache, precomputes rotary
embeddings, and compiles the decode function with torch.compile().

On compilation failure, falls back to None (caller uses eager
``PipelineParallelShard.forward()`` instead) and logs at ERROR level.

**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
"""

from __future__ import annotations

from typing import Any, Callable, final

import torch
from loguru import logger

from exo.worker.engines.pytorch_xpu.graph_break_detector import (
    detect_graph_breaks,
    snapshot_graph_break_counters,
)
from exo.worker.engines.pytorch_xpu.packed_projections import (
    pack_gate_up_weights,
    pack_qkv_weights,
)
from exo.worker.engines.pytorch_xpu.pipeline_config import (
    PytorchXpuOptimizationConfiguration,
)
from exo.worker.engines.pytorch_xpu.pipeline_parallel_shard import (
    PipelineStageConfig,
)
from exo.worker.engines.pytorch_xpu.static_kv_cache import (
    GatedDeltaNetLayerConfig,
    StaticKVCache,
)
from exo.worker.engines.pytorch_xpu.tensor_only_decode import decode_one_token

_logger = logger.bind(module="pytorch_xpu.compiled_decode_path")

# ---------------------------------------------------------------------------
# Constants for Qwen3.5-4B architecture
# ---------------------------------------------------------------------------

_NUM_HEADS: int = 32
_NUM_KV_HEADS: int = 4
_HEAD_DIM: int = 80
_HIDDEN_SIZE: int = 2560
_INTERMEDIATE_SIZE: int = 9728
_GDN_CONV_SIZE: int = 4

# Type alias for the decode function signature
_DecodeFnType = Callable[
    ..., tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]


# ---------------------------------------------------------------------------
# Helper functions (defined before the class that uses them)
# ---------------------------------------------------------------------------


def _precompute_rotary_embeddings(
    max_seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute rotary embedding cos/sin tables for all positions.

    Args:
        max_seq_len: Maximum sequence length to precompute for.
        head_dim: Dimension of each attention head.
        device: Target device for the tables.
        dtype: Data type for the output tables.
        base: Base frequency for rotary embeddings.

    Returns:
        Tuple of (cos_table, sin_table), each of shape [max_seq_len, head_dim].
    """
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos_table = emb.cos().to(dtype)
    sin_table = emb.sin().to(dtype)
    return cos_table, sin_table


def _detect_layer_type(layer: torch.nn.Module) -> int:
    """Detect whether a layer is full-attention (0) or GatedDeltaNet (1).

    Inspects the layer's class name and attributes to determine its type.

    Args:
        layer: A transformer decoder layer module.

    Returns:
        0 for full-attention, 1 for GatedDeltaNet.
    """
    class_name = type(layer).__name__.lower()
    if "deltanet" in class_name or "linear" in class_name:
        return 1

    if hasattr(layer, "self_attn"):
        attn_module: Any = layer.self_attn
        attn_class_name: str = type(attn_module).__name__.lower()  # pyright: ignore[reportAny]
        if "deltanet" in attn_class_name or "linear" in attn_class_name:
            return 1
        if hasattr(attn_module, "a_log") or hasattr(attn_module, "A_log"):  # pyright: ignore[reportAny]
            return 1

    return 0


def _extract_gdn_params(
    layer: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract GatedDeltaNet-specific parameters from a layer.

    Args:
        layer: A GatedDeltaNet decoder layer.
        device: Target device.
        dtype: Target dtype for conv_weight.

    Returns:
        Tuple of (conv_weight, a_log, dt_bias) tensors on the target device.
    """
    attn: Any = layer.self_attn

    conv_weight: torch.Tensor
    if hasattr(attn, "conv1d"):  # pyright: ignore[reportAny]
        conv_weight = attn.conv1d.weight.data.squeeze()  # pyright: ignore[reportAny]
    elif hasattr(attn, "conv"):  # pyright: ignore[reportAny]
        conv_weight = attn.conv.weight.data.squeeze()  # pyright: ignore[reportAny]
    else:
        conv_weight = torch.zeros(_HIDDEN_SIZE, _GDN_CONV_SIZE, device=device, dtype=dtype)

    a_log: torch.Tensor
    if hasattr(attn, "a_log"):  # pyright: ignore[reportAny]
        a_log = attn.a_log.data  # pyright: ignore[reportAny]
    elif hasattr(attn, "A_log"):  # pyright: ignore[reportAny]
        a_log = attn.A_log.data  # pyright: ignore[reportAny]
    else:
        a_log = torch.zeros(_NUM_HEADS, device=device, dtype=torch.float32)

    dt_bias: torch.Tensor
    if hasattr(attn, "dt_bias"):  # pyright: ignore[reportAny]
        dt_bias = attn.dt_bias.data  # pyright: ignore[reportAny]
    elif hasattr(attn, "dt_proj") and hasattr(attn.dt_proj, "bias"):  # pyright: ignore[reportAny]
        dt_bias = attn.dt_proj.bias.data  # pyright: ignore[reportAny]
    else:
        dt_bias = torch.zeros(_NUM_HEADS, device=device, dtype=torch.float32)

    return (
        conv_weight.to(device=device, dtype=dtype),
        a_log.to(device=device, dtype=torch.float32),
        dt_bias.to(device=device, dtype=torch.float32),
    )


def _extract_layernorm_weight(
    layer: torch.nn.Module,
    attr_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Extract a layernorm weight from a layer, with fallback to ones.

    Args:
        layer: The transformer decoder layer.
        attr_name: Name of the layernorm attribute (e.g., "input_layernorm").
        device: Target device.
        dtype: Target dtype.

    Returns:
        The layernorm weight tensor on the target device.
    """
    norm_module: Any = getattr(layer, attr_name, None)
    if norm_module is not None and hasattr(norm_module, "weight"):  # pyright: ignore[reportAny]
        weight: torch.Tensor = norm_module.weight.data  # pyright: ignore[reportAny]
        return weight.to(device=device, dtype=dtype)
    return torch.ones(_HIDDEN_SIZE, device=device, dtype=dtype)


def _extract_qkv_weights(
    layer: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract and pack QKV weights and output projection from a layer.

    Args:
        layer: The transformer decoder layer.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Tuple of (packed_qkv_weight, o_proj_weight).
    """
    q_dim = _NUM_HEADS * _HEAD_DIM
    kv_dim = _NUM_KV_HEADS * _HEAD_DIM

    self_attn: Any = getattr(layer, "self_attn", None)
    if self_attn is None:
        return (
            torch.zeros(q_dim + 2 * kv_dim, _HIDDEN_SIZE, device=device, dtype=dtype),
            torch.zeros(_HIDDEN_SIZE, _HIDDEN_SIZE, device=device, dtype=dtype),
        )

    q_proj: Any = getattr(self_attn, "q_proj", None)  # pyright: ignore[reportAny]
    k_proj: Any = getattr(self_attn, "k_proj", None)  # pyright: ignore[reportAny]
    v_proj: Any = getattr(self_attn, "v_proj", None)  # pyright: ignore[reportAny]

    if (
        q_proj is not None
        and k_proj is not None
        and v_proj is not None
        and isinstance(q_proj, torch.nn.Linear)
        and isinstance(k_proj, torch.nn.Linear)
        and isinstance(v_proj, torch.nn.Linear)
    ):
        packed_qkv = pack_qkv_weights(
            q_proj.weight.data,
            k_proj.weight.data,
            v_proj.weight.data,
        ).to(device=device, dtype=dtype)
    else:
        packed_qkv = torch.zeros(q_dim + 2 * kv_dim, _HIDDEN_SIZE, device=device, dtype=dtype)

    o_proj: Any = getattr(self_attn, "o_proj", None)  # pyright: ignore[reportAny]
    if o_proj is not None and isinstance(o_proj, torch.nn.Linear):
        o_proj_weight = o_proj.weight.data.to(device=device, dtype=dtype)
    else:
        o_proj_weight = torch.zeros(_HIDDEN_SIZE, _HIDDEN_SIZE, device=device, dtype=dtype)

    return packed_qkv, o_proj_weight


def _extract_mlp_weights(
    layer: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract and pack gate/up weights and down projection from a layer.

    Args:
        layer: The transformer decoder layer.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Tuple of (packed_gate_up_weight, down_proj_weight).
    """
    mlp: Any = getattr(layer, "mlp", None)
    if mlp is None:
        return (
            torch.zeros(2 * _INTERMEDIATE_SIZE, _HIDDEN_SIZE, device=device, dtype=dtype),
            torch.zeros(_HIDDEN_SIZE, _INTERMEDIATE_SIZE, device=device, dtype=dtype),
        )

    gate_proj: Any = getattr(mlp, "gate_proj", None)  # pyright: ignore[reportAny]
    up_proj: Any = getattr(mlp, "up_proj", None)  # pyright: ignore[reportAny]

    if (
        gate_proj is not None
        and up_proj is not None
        and isinstance(gate_proj, torch.nn.Linear)
        and isinstance(up_proj, torch.nn.Linear)
    ):
        packed_gate_up = pack_gate_up_weights(
            gate_proj.weight.data,
            up_proj.weight.data,
        ).to(device=device, dtype=dtype)
    else:
        packed_gate_up = torch.zeros(
            2 * _INTERMEDIATE_SIZE, _HIDDEN_SIZE, device=device, dtype=dtype
        )

    down_proj: Any = getattr(mlp, "down_proj", None)  # pyright: ignore[reportAny]
    if down_proj is not None and isinstance(down_proj, torch.nn.Linear):
        down_proj_weight = down_proj.weight.data.to(device=device, dtype=dtype)
    else:
        down_proj_weight = torch.zeros(_HIDDEN_SIZE, _INTERMEDIATE_SIZE, device=device, dtype=dtype)

    return packed_gate_up, down_proj_weight


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


@final
class CompiledDecodePath:
    """Compiled decode path wrapping torch.compile() for the XPU Inductor backend.

    Extracts weights from HuggingFace model layers at construction time, packs
    QKV and gate/up projections, creates a static KV cache, precomputes rotary
    embedding tables, and compiles the ``decode_one_token`` function using
    ``torch.compile(backend="inductor", mode="max-autotune")``.

    If compilation fails, the instance records the failure and ``is_compiled``
    returns False. The caller (``PipelineParallelShard``) checks this flag and
    falls back to the eager ``_forward_layer`` loop.

    **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
    """

    __slots__ = (
        "_compiled_fn",
        "_is_compiled",
        "_device",
        "_dtype",
        "_cache",
        "_embed_weight",
        "_final_norm_weight",
        "_lm_head_weight",
        "_rotary_cos",
        "_rotary_sin",
        "_input_layernorm_weights",
        "_post_attention_layernorm_weights",
        "_packed_qkv_weights",
        "_o_proj_weights",
        "_packed_gate_up_weights",
        "_down_proj_weights",
        "_layer_types",
        "_conv_weights",
        "_a_log_params",
        "_dt_bias_params",
        "_num_layers",
        "_max_seq_len",
    )

    def __init__(
        self,
        layers: torch.nn.ModuleList,
        embed_tokens: torch.nn.Embedding,
        lm_head: torch.nn.Linear,
        final_norm: torch.nn.Module,
        config: PipelineStageConfig,
        optimization_config: PytorchXpuOptimizationConfiguration,
    ) -> None:
        """Extract weights, pack projections, create static cache, compile decode function.

        Args:
            layers: The transformer layers assigned to this pipeline stage.
            embed_tokens: Token embedding module.
            lm_head: Language model head (linear projection to vocabulary).
            final_norm: Final RMSNorm before the LM head.
            config: Pipeline stage configuration with device and dimension info.
            optimization_config: Optimization flags including compile mode and
                static cache max sequence length.

        **Validates: Requirements 4.1, 4.5**
        """
        device = torch.device(config.device)
        dtype = torch.bfloat16
        self._device = device
        self._dtype = dtype
        self._num_layers = len(layers)
        self._max_seq_len = optimization_config.static_cache_max_seq_len

        # --- Extract model weights (frozen, moved to target device) ---
        self._embed_weight: torch.Tensor = embed_tokens.weight.data.to(
            device=device, dtype=dtype
        )
        self._lm_head_weight: torch.Tensor = lm_head.weight.data.to(
            device=device, dtype=dtype
        )

        # Extract final norm weight (RMSNorm has a 'weight' parameter)
        norm_weight: Any = getattr(final_norm, "weight", None)
        if norm_weight is not None and isinstance(norm_weight, torch.Tensor):
            self._final_norm_weight: torch.Tensor = norm_weight.data.to(
                device=device, dtype=dtype
            )
        else:
            self._final_norm_weight = torch.ones(
                _HIDDEN_SIZE, device=device, dtype=dtype
            )

        # --- Detect layer types and extract per-layer weights ---
        self._layer_types: list[int] = []
        self._input_layernorm_weights: list[torch.Tensor] = []
        self._post_attention_layernorm_weights: list[torch.Tensor] = []
        self._packed_qkv_weights: list[torch.Tensor] = []
        self._o_proj_weights: list[torch.Tensor] = []
        self._packed_gate_up_weights: list[torch.Tensor] = []
        self._down_proj_weights: list[torch.Tensor] = []
        self._conv_weights: list[torch.Tensor] = []
        self._a_log_params: list[torch.Tensor] = []
        self._dt_bias_params: list[torch.Tensor] = []

        num_attention_layers = 0
        gdn_configs: list[GatedDeltaNetLayerConfig] = []

        for layer in layers:
            layer_type = _detect_layer_type(layer)
            self._layer_types.append(layer_type)

            # Extract layernorm weights
            self._input_layernorm_weights.append(
                _extract_layernorm_weight(layer, "input_layernorm", device, dtype)
            )
            self._post_attention_layernorm_weights.append(
                _extract_layernorm_weight(layer, "post_attention_layernorm", device, dtype)
            )

            # Extract and pack QKV + output projection weights
            packed_qkv, o_proj_weight = _extract_qkv_weights(layer, device, dtype)
            self._packed_qkv_weights.append(packed_qkv)
            self._o_proj_weights.append(o_proj_weight)

            # Extract and pack gate/up + down projection weights
            packed_gate_up, down_proj_weight = _extract_mlp_weights(layer, device, dtype)
            self._packed_gate_up_weights.append(packed_gate_up)
            self._down_proj_weights.append(down_proj_weight)

            # Extract GatedDeltaNet-specific parameters
            if layer_type == 1:
                conv_weight, a_log, dt_bias = _extract_gdn_params(layer, device, dtype)
                self._conv_weights.append(conv_weight)
                self._a_log_params.append(a_log)
                self._dt_bias_params.append(dt_bias)
                gdn_configs.append(
                    GatedDeltaNetLayerConfig(
                        conv_size=_GDN_CONV_SIZE,
                        hidden_size=_HIDDEN_SIZE,
                        num_heads=_NUM_HEADS,
                        head_dim=_HEAD_DIM,
                    )
                )
            else:
                num_attention_layers += 1

        # --- Create static KV cache ---
        self._cache = StaticKVCache(
            num_attention_layers=num_attention_layers,
            num_kv_heads=_NUM_KV_HEADS,
            head_dim=_HEAD_DIM,
            max_seq_len=self._max_seq_len,
            device=device,
            dtype=dtype,
            gated_deltanet_configs=gdn_configs if gdn_configs else None,
        )

        # --- Precompute rotary embedding tables ---
        self._rotary_cos, self._rotary_sin = _precompute_rotary_embeddings(
            max_seq_len=self._max_seq_len,
            head_dim=_HEAD_DIM,
            device=device,
            dtype=dtype,
        )

        # --- Compile decode_one_token with torch.compile ---
        self._compiled_fn: _DecodeFnType | None = None
        self._is_compiled: bool = False

        if optimization_config.enable_torch_compile:
            self._try_compile(optimization_config.torch_compile_mode)

        _logger.info(
            "CompiledDecodePath initialized: layers={layers}, "
            "attention_layers={attn}, gdn_layers={gdn}, "
            "max_seq_len={max_seq}, device={device}, "
            "compiled={compiled}",
            layers=self._num_layers,
            attn=num_attention_layers,
            gdn=len(gdn_configs),
            max_seq=self._max_seq_len,
            device=str(device),
            compiled=self._is_compiled,
        )

    def _try_compile(self, mode: str) -> None:
        """Attempt to compile decode_one_token with torch.compile.

        On success, sets ``_compiled_fn`` and ``_is_compiled = True``.
        On failure, logs at ERROR level and leaves ``_is_compiled = False``.

        Args:
            mode: torch.compile optimization mode (e.g., "max-autotune").

        **Validates: Requirements 4.1, 4.4**
        """
        try:
            before_snapshot = snapshot_graph_break_counters()

            compiled: _DecodeFnType = torch.compile(  # type: ignore[assignment]
                decode_one_token,
                backend="inductor",
                mode=mode,
            )
            self._compiled_fn = compiled
            self._is_compiled = True

            report = detect_graph_breaks(before_snapshot)
            if report.total_count > 0:
                _logger.warning(
                    "Graph breaks detected during torch.compile setup: "
                    "{count} breaks. The compiled graph is split.",
                    count=report.total_count,
                )

            _logger.info(
                "torch.compile(backend='inductor', mode='{mode}') succeeded. "
                "First invocation will trigger JIT compilation (warmup).",
                mode=mode,
            )

        except Exception as exc:
            self._compiled_fn = None
            self._is_compiled = False
            _logger.error(
                "torch.compile() failed for XPU Inductor backend. "
                "Falling back to eager execution. Error: {error}",
                error=str(exc),
            )

    @property
    def is_compiled(self) -> bool:
        """Whether the decode function was compiled.

        When False, the caller should fall back to eager
        ``PipelineParallelShard.forward()``.
        """
        return self._is_compiled

    def forward(self, token_id: torch.Tensor, position: int) -> torch.Tensor:
        """Execute one decode step through the compiled graph.

        Converts the position int to a tensor, invokes the compiled (or eager
        fallback) decode function with all extracted weight tensors and cache
        state, and returns the logits tensor on the XPU device.

        Args:
            token_id: Token ID tensor of shape [1], int64, on the target device.
            position: Current sequence position as a Python int.

        Returns:
            Logits tensor of shape [1, vocab_size] on the XPU device.

        Raises:
            RuntimeError: If compilation failed and no fallback is available.
                (In practice, the caller checks ``is_compiled`` first.)

        **Validates: Requirements 4.2, 4.3**
        """
        if token_id.device != self._device:
            token_id = token_id.to(self._device)

        position_tensor = torch.tensor(
            [position], dtype=torch.int64, device=self._device
        )

        num_attn_layers = self._cache.num_attention_layers
        num_gdn_layers = self._cache.num_gated_deltanet_layers

        # Full-attention KV cache: [num_attn_layers, max_seq, num_kv_heads, head_dim]
        cache_keys: torch.Tensor
        cache_values: torch.Tensor
        if num_attn_layers > 0:
            cache_keys = torch.stack(
                [self._cache.get_key_cache(i).squeeze(0) for i in range(num_attn_layers)]
            )
            cache_values = torch.stack(
                [self._cache.get_value_cache(i).squeeze(0) for i in range(num_attn_layers)]
            )
        else:
            cache_keys = torch.zeros(
                0, self._max_seq_len, _NUM_KV_HEADS, _HEAD_DIM,
                device=self._device, dtype=self._dtype,
            )
            cache_values = torch.zeros(
                0, self._max_seq_len, _NUM_KV_HEADS, _HEAD_DIM,
                device=self._device, dtype=self._dtype,
            )

        # GatedDeltaNet state tensors
        conv_states: torch.Tensor
        recurrent_states: torch.Tensor
        if num_gdn_layers > 0:
            conv_states = torch.stack(
                [
                    self._cache.get_gated_deltanet_slot(i).conv_state.squeeze(0)
                    for i in range(num_gdn_layers)
                ]
            )
            recurrent_states = torch.stack(
                [
                    self._cache.get_gated_deltanet_slot(i).recurrent_state.squeeze(0)
                    for i in range(num_gdn_layers)
                ]
            )
        else:
            conv_states = torch.zeros(
                0, _GDN_CONV_SIZE, _HIDDEN_SIZE,
                device=self._device, dtype=self._dtype,
            )
            recurrent_states = torch.zeros(
                0, _NUM_HEADS, _HEAD_DIM, _HEAD_DIM,
                device=self._device, dtype=torch.float32,
            )

        # Select the function to call (compiled or eager fallback)
        fn: _DecodeFnType = (
            self._compiled_fn if self._compiled_fn is not None else decode_one_token
        )

        # Invoke the decode function
        logits: torch.Tensor
        logits, _, _, _, _ = fn(
            token_id=token_id,
            position=position_tensor,
            cache_keys=cache_keys,
            cache_values=cache_values,
            conv_states=conv_states,
            recurrent_states=recurrent_states,
            embed_weight=self._embed_weight,
            final_norm_weight=self._final_norm_weight,
            lm_head_weight=self._lm_head_weight,
            rotary_cos=self._rotary_cos,
            rotary_sin=self._rotary_sin,
            input_layernorm_weights=self._input_layernorm_weights,
            post_attention_layernorm_weights=self._post_attention_layernorm_weights,
            packed_qkv_weights=self._packed_qkv_weights,
            o_proj_weights=self._o_proj_weights,
            packed_gate_up_weights=self._packed_gate_up_weights,
            down_proj_weights=self._down_proj_weights,
            layer_types=self._layer_types,
            conv_weights=self._conv_weights,
            a_log_params=self._a_log_params,
            dt_bias_params=self._dt_bias_params,
        )

        # Advance the cache position
        self._cache.advance_position()

        # Write back updated cache state from the returned tensors
        # (The decode function modifies cache tensors in-place via slice assignment,
        # but since we stacked them, we need to copy back to the individual caches)
        if num_attn_layers > 0:
            for i in range(num_attn_layers):
                self._cache.get_key_cache(i).squeeze(0).copy_(cache_keys[i])
                self._cache.get_value_cache(i).squeeze(0).copy_(cache_values[i])

        if num_gdn_layers > 0:
            for i in range(num_gdn_layers):
                self._cache.get_gated_deltanet_slot(i).conv_state.squeeze(0).copy_(
                    conv_states[i]
                )
                self._cache.get_gated_deltanet_slot(i).recurrent_state.squeeze(0).copy_(
                    recurrent_states[i]
                )

        return logits

    def reset(self) -> None:
        """Reset cache for new generation request.

        Zeros all cache tensors and resets the position counter. The underlying
        tensor storage is reused without reallocation.

        **Validates: Requirement 2.4**
        """
        self._cache.reset()
        _logger.debug("CompiledDecodePath cache reset")

    @property
    def position(self) -> int:
        """Current sequence position in the static cache."""
        return self._cache.position

    @property
    def max_seq_len(self) -> int:
        """Maximum sequence length the cache was pre-allocated for."""
        return self._max_seq_len

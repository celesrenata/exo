"""Tensor-parallel weight sharding configuration and model wrapper.

This module provides the TPShardConfig dataclass for configuring tensor-parallel
weight sharding across multiple ranks connected via Thunderbolt 4, and the
TensorParallelShard class that holds sharded weights and executes the
tensor-parallel forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

from exo.worker.engines.pytorch_xpu.distributed import get_tensor_parallel_group


class ModelArchitecture(str, Enum):
    """Detected model architecture based on state_dict key patterns."""

    QWEN_LLAMA = "qwen_llama"  # Separate Q, K, V projections, RMSNorm
    PHI = "phi"  # Fused QKV or separate with LayerNorm + biases


@dataclass(frozen=True)
class TPShardConfig:
    """Configuration for tensor-parallel weight sharding.

    Defines how weight matrices are split across tensor-parallel ranks.
    Each rank holds 1/world_size of the attention heads and MLP intermediate
    dimension, with all-reduce synchronization after row-parallel layers.

    Validates that attention heads, key-value heads, and intermediate size
    are evenly divisible by world_size at construction time.
    """

    rank: int
    world_size: int
    hidden_size: int  # 2560 for Qwen3.5-4B
    num_attention_heads: int  # 32 for Qwen3.5-4B
    head_dim: int  # 256 for Qwen3.5-4B (hidden_size * 2 / num_heads for GQA)
    intermediate_size: int  # MLP intermediate dimension
    num_key_value_heads: int  # For GQA models
    allreduce_timeout_seconds: int = 30
    # RoPE parameters — must be read from model config, not hardcoded.
    # Qwen3.5-4B uses rope_theta=1000000.0 and YaRN rope_scaling.
    rope_theta: float = 1000000.0
    # rope_scaling dict from HuggingFace config (e.g. {"type": "yarn", "factor": 4.0, ...})
    # None means standard RoPE with no scaling.
    rope_scaling: dict[str, object] | None = None

    def __post_init__(self) -> None:
        """Validate that model dimensions are compatible with the world size."""
        if self.num_attention_heads % self.world_size != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be evenly "
                f"divisible by world_size ({self.world_size}), but "
                f"{self.num_attention_heads} % {self.world_size} = "
                f"{self.num_attention_heads % self.world_size}"
            )
        if self.intermediate_size % self.world_size != 0:
            raise ValueError(
                f"intermediate_size ({self.intermediate_size}) must be evenly "
                f"divisible by world_size ({self.world_size}), but "
                f"{self.intermediate_size} % {self.world_size} = "
                f"{self.intermediate_size % self.world_size}"
            )
        if self.num_key_value_heads % self.world_size != 0:
            raise ValueError(
                f"num_key_value_heads ({self.num_key_value_heads}) must be evenly "
                f"divisible by world_size ({self.world_size}), but "
                f"{self.num_key_value_heads} % {self.world_size} = "
                f"{self.num_key_value_heads % self.world_size}"
            )

    @property
    def heads_per_rank(self) -> int:
        """Number of attention heads assigned to each rank."""
        return self.num_attention_heads // self.world_size

    @property
    def kv_heads_per_rank(self) -> int:
        """Number of key-value heads assigned to each rank (for GQA models)."""
        return self.num_key_value_heads // self.world_size

    @property
    def intermediate_per_rank(self) -> int:
        """MLP intermediate dimension assigned to each rank."""
        return self.intermediate_size // self.world_size


# Weight parameter name patterns for HuggingFace Qwen convention
_LAYER_PREFIX = "model.layers.{layer_idx}"
_ATTN_PREFIX = _LAYER_PREFIX + ".self_attn"
_MLP_PREFIX = _LAYER_PREFIX + ".mlp"

# Column-parallel projections (split along output dim = dim 0)
_COLUMN_PARALLEL_ATTN_KEYS = ("q_proj.weight", "k_proj.weight", "v_proj.weight")
_COLUMN_PARALLEL_MLP_KEYS = ("gate_proj.weight", "up_proj.weight")

# Row-parallel projections (split along input dim = dim 1)
_ROW_PARALLEL_ATTN_KEYS = ("o_proj.weight",)
_ROW_PARALLEL_MLP_KEYS = ("down_proj.weight",)

# Redundant (not sharded) parameter name patterns
# These use suffix matching to handle both "model.layers." and "model.language_model.layers." prefixes
_REDUNDANT_PATTERNS = (
    "embed_tokens.weight",
    "norm.weight",
    "norm.bias",
    "lm_head.weight",
    "input_layernorm.weight",
    "input_layernorm.bias",
    "post_attention_layernorm.weight",
    "post_attention_layernorm.bias",
    "q_norm.weight",
    "k_norm.weight",
)

# Linear attention (Gated DeltaNet) weights are kept redundant on all ranks.
# They don't use tensor parallelism — only the MLP layers are sharded.
_LINEAR_ATTN_PATTERN = "linear_attn."


class TensorParallelShard:
    """Tensor-parallel model wrapper with sharded weights.

    Unlike TransformerShard (which holds a subset of layers),
    TensorParallelShard holds ALL layers but with sharded weights
    within each layer. Each rank holds:
    - Full embedding table (redundant)
    - 1/world_size of each attention head group
    - 1/world_size of each MLP intermediate dimension
    - Full layer norms (redundant)
    - Full lm_head (redundant)

    Weight sharding strategy:
    - QKV projections: column-parallel (split output dim by heads_per_rank)
    - Attention output: row-parallel (split input dim by head_dim * heads_per_rank)
    - MLP gate/up: column-parallel (split output dim by intermediate_per_rank)
    - MLP down: row-parallel (split input dim by intermediate_per_rank)
    """

    def __init__(
        self,
        model: dict[str, torch.Tensor] | Any,
        config: TPShardConfig,
        device: str,
    ) -> None:
        """Initialize with sharded weights extracted from the full model.

        The model is loaded fully on CPU first, then only this rank's
        weight slices are moved to the target device. The rest is discarded.

        Args:
            model: A state dict (dict[str, Tensor]) or a module with a
                   state_dict() method. Weights should be on CPU.
            config: TPShardConfig defining rank, world_size, and model dimensions.
            device: Target device string (e.g., "xpu:0", "cpu", "cuda:0").
        """
        self.config = config
        self.device = device
        self.sharded_state_dict: dict[str, torch.Tensor] = {}
        self._native_linear_attn_layers: dict[int, Any] = {}

        # Extract state dict from model if it's a module
        if isinstance(model, dict):
            state_dict = model
        elif hasattr(model, "state_dict"):
            # Before extracting state dict, save references to native linear_attn layers
            # for hybrid models (Qwen3.5/3.6) that use Gated DeltaNet
            self._extract_native_linear_attn_layers(model)
            state_dict = model.state_dict()
        else:
            raise TypeError(
                f"model must be a state dict (dict) or have a state_dict() method, "
                f"got {type(model).__name__}"
            )

        self.shard_weights(state_dict)

        # Move native linear attention layers to target device and set eval mode
        for _layer_idx, native_layer in self._native_linear_attn_layers.items():
            native_layer.to(self.device).eval()

        # Detect architecture from stored keys
        self.architecture = self._detect_architecture()

        # Detect the layer key prefix (e.g., "model.layers" or "model.language_model.layers")
        self._layer_prefix = self._detect_layer_prefix()

        # Detect layer types (linear_attention vs full_attention) for hybrid models
        self._layer_types = self._detect_layer_types()

        # Recurrent state for linear attention (Gated DeltaNet) layers
        self._linear_attn_states: dict[str, torch.Tensor] = {}

        logger.info(
            f"TensorParallelShard initialized: rank={config.rank}/{config.world_size}, "
            f"device={device}, "
            f"architecture={self.architecture.value}, "
            f"layer_prefix={self._layer_prefix}, "
            f"sharded_params={len(self.sharded_state_dict)}, "
            f"heads_per_rank={config.heads_per_rank}, "
            f"kv_heads_per_rank={config.kv_heads_per_rank}, "
            f"intermediate_per_rank={config.intermediate_per_rank}"
        )

    def _extract_native_linear_attn_layers(self, model: Any) -> None:  # pyright: ignore[reportAny]
        """Extract native linear attention layer modules from the HuggingFace model.

        Supports multiple model layouts:
        - model.model.layers (standard Qwen/Llama)
        - model.model.language_model.layers (VL models: Qwen3.5-VL)
        - model.language_model.layers (older VL layout)

        Within each layer, the linear attention sub-module may be exposed as:
        - layer.linear_attn  (Qwen3.5 Gated DeltaNet layers)
        - layer.self_attn when layer_types[i] == "linear_attention" (fallback)

        If zero native layers are found for a hybrid model (i.e. the model config
        declares linear_attention layer_types but no native modules were extracted),
        a warning is emitted so the caller knows the fallback implementation will run.
        """
        layers: Any = None  # pyright: ignore[reportAny]

        # Resolve the transformer layer list, trying the most common layouts first.
        for candidate in (
            lambda m: m.model.language_model.layers if hasattr(m, 'model') and hasattr(m.model, 'language_model') and hasattr(m.model.language_model, 'layers') else None,  # noqa: E731
            lambda m: m.model.layers if hasattr(m, 'model') and hasattr(m.model, 'layers') else None,  # noqa: E731
            lambda m: m.language_model.layers if hasattr(m, 'language_model') and hasattr(m.language_model, 'layers') else None,  # noqa: E731
        ):
            result = candidate(model)  # pyright: ignore[reportAny]
            if result is not None:
                layers = result
                break

        if layers is None:
            logger.warning(
                "TensorParallelShard: could not locate transformer layer list in model "
                "for native linear_attn extraction. Custom fallback will be used for "
                "all linear attention layers."
            )
            return

        for idx, layer in enumerate(layers):  # pyright: ignore[reportAny]
            # Primary attribute name: linear_attn (Qwen3.5 Gated DeltaNet)
            native: Any = getattr(layer, 'linear_attn', None)  # pyright: ignore[reportAny]
            if native is not None:
                self._native_linear_attn_layers[idx] = native
                continue
            # Secondary: some builds expose the recurrent layer as 'mamba' or 'ssm'
            for alt_attr in ('mamba', 'ssm', 'recurrent_layer'):
                alt: Any = getattr(layer, alt_attr, None)  # pyright: ignore[reportAny]
                if alt is not None:
                    self._native_linear_attn_layers[idx] = alt
                    break

        num_found = len(self._native_linear_attn_layers)
        logger.info(f"Extracted {num_found} native linear_attn layers")

        # Warn if the model config declares linear_attention layers but we found none,
        # because the custom fallback is a best-effort reimplementation and may differ
        # numerically from the reference.
        if num_found == 0:
            # Try to detect whether the model is actually hybrid by checking model config.
            model_config: Any = getattr(model, 'config', None)  # pyright: ignore[reportAny]
            if model_config is not None:
                text_cfg: Any = getattr(model_config, 'text_config', model_config)  # pyright: ignore[reportAny]
                layer_types: Any = getattr(text_cfg, 'layer_types', None)  # pyright: ignore[reportAny]
                if layer_types is not None and 'linear_attention' in layer_types:
                    logger.warning(
                        "TensorParallelShard: model config declares 'linear_attention' "
                        "layer_types but no native linear_attn modules were found. "
                        "The custom Gated DeltaNet fallback will run for these layers. "
                        "Output quality may be degraded if the fallback diverges from "
                        "the reference implementation."
                    )

    def _create_native_cache(self) -> Any:
        """Create a HuggingFace Cache object for the native linear attention layers."""
        try:
            from transformers.cache_utils import Cache, LinearAttentionLayer, DynamicLayer
        except ImportError:
            logger.warning("Could not import Cache/LinearAttentionLayer/DynamicLayer from transformers.cache_utils")
            return None

        num_layers = self._detect_num_layers()
        cache_layers = []
        for idx in range(num_layers):
            if self._layer_types[idx] == "linear_attention":
                cache_layers.append(LinearAttentionLayer())
            else:
                cache_layers.append(DynamicLayer())

        cache = Cache(layers=cache_layers)
        logger.info(f"Created native cache with {num_layers} layers")
        return cache

    def shard_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Extract this rank's portion of each weight matrix.

        For each transformer layer:
        - QKV weights: slice along output dim (heads_per_rank heads)
        - Attention output: slice along input dim (head_dim * heads_per_rank cols)
        - MLP gate/up: slice along output dim (intermediate_per_rank cols)
        - MLP down: slice along input dim (intermediate_per_rank cols)

        Biases are sliced correspondingly where present.

        After sharding, moves weights to the target device and discards
        the full state dict.

        Fused QKV weights (qkv_proj.weight) are detected and split into
        separate q_proj.weight, k_proj.weight, v_proj.weight entries so
        that forward() can use uniform key patterns for all architectures.
        """
        rank = self.config.rank
        world_size = self.config.world_size
        head_dim = self.config.head_dim
        heads_per_rank = self.config.heads_per_rank
        kv_heads_per_rank = self.config.kv_heads_per_rank
        intermediate_per_rank = self.config.intermediate_per_rank

        for param_name, param_tensor in state_dict.items():
            # Detect fused QKV and split into separate Q, K, V entries
            if self._matches_attn_key(param_name, "qkv_proj.weight"):
                self._split_fused_qkv(
                    param_name, param_tensor, is_bias=False,
                    rank=rank, world_size=world_size, head_dim=head_dim,
                    heads_per_rank=heads_per_rank, kv_heads_per_rank=kv_heads_per_rank,
                )
                continue

            if self._matches_attn_key(param_name, "qkv_proj.bias"):
                self._split_fused_qkv(
                    param_name, param_tensor, is_bias=True,
                    rank=rank, world_size=world_size, head_dim=head_dim,
                    heads_per_rank=heads_per_rank, kv_heads_per_rank=kv_heads_per_rank,
                )
                continue

            # Detect fused gate_up_proj and split into separate gate_proj, up_proj
            if self._matches_mlp_key(param_name, "gate_up_proj.weight"):
                self._split_fused_gate_up(
                    param_name, param_tensor, is_bias=False,
                    rank=rank, intermediate_per_rank=intermediate_per_rank,
                )
                continue

            if self._matches_mlp_key(param_name, "gate_up_proj.bias"):
                self._split_fused_gate_up(
                    param_name, param_tensor, is_bias=True,
                    rank=rank, intermediate_per_rank=intermediate_per_rank,
                )
                continue

            shard = self._shard_parameter(
                param_name,
                param_tensor,
                rank=rank,
                world_size=world_size,
                head_dim=head_dim,
                heads_per_rank=heads_per_rank,
                kv_heads_per_rank=kv_heads_per_rank,
                intermediate_per_rank=intermediate_per_rank,
            )
            # Move to target device
            self.sharded_state_dict[param_name] = shard.to(self.device)

        logger.debug(
            f"Rank {rank}: sharded {len(self.sharded_state_dict)} parameters "
            f"to device {self.device}"
        )

    def _split_fused_qkv(
        self,
        param_name: str,
        param_tensor: torch.Tensor,
        *,
        is_bias: bool,
        rank: int,
        world_size: int,
        head_dim: int,
        heads_per_rank: int,
        kv_heads_per_rank: int,
    ) -> None:
        """Split a fused QKV weight/bias into separate Q, K, V entries.

        Fused QKV layout (Phi-style):
        - Weight shape: [(num_heads + 2 * num_kv_heads) * head_dim, hidden_size]
        - Bias shape: [(num_heads + 2 * num_kv_heads) * head_dim]

        The fused tensor is ordered as [Q, K, V] along dimension 0.
        After splitting, each part is sharded per-rank and stored with
        the canonical key names (q_proj.weight, k_proj.weight, v_proj.weight).
        """
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads

        q_size = num_heads * head_dim
        k_size = num_kv_heads * head_dim
        v_size = num_kv_heads * head_dim

        # Split along dimension 0
        q_full = param_tensor.narrow(0, 0, q_size)
        k_full = param_tensor.narrow(0, q_size, k_size)
        v_full = param_tensor.narrow(0, q_size + k_size, v_size)

        # Shard each part for this rank
        q_shard_size = heads_per_rank * head_dim
        k_shard_size = kv_heads_per_rank * head_dim
        v_shard_size = kv_heads_per_rank * head_dim

        q_shard = q_full.narrow(0, rank * q_shard_size, q_shard_size).clone()
        k_shard = k_full.narrow(0, rank * k_shard_size, k_shard_size).clone()
        v_shard = v_full.narrow(0, rank * v_shard_size, v_shard_size).clone()

        # Construct canonical key names by replacing qkv_proj with q/k/v_proj
        suffix = "bias" if is_bias else "weight"
        q_key = param_name.replace(f"qkv_proj.{suffix}", f"q_proj.{suffix}")
        k_key = param_name.replace(f"qkv_proj.{suffix}", f"k_proj.{suffix}")
        v_key = param_name.replace(f"qkv_proj.{suffix}", f"v_proj.{suffix}")

        self.sharded_state_dict[q_key] = q_shard.to(self.device)
        self.sharded_state_dict[k_key] = k_shard.to(self.device)
        self.sharded_state_dict[v_key] = v_shard.to(self.device)

        logger.debug(
            f"Rank {rank}: split fused QKV '{param_name}' into "
            f"q={q_key} ({tuple(q_shard.shape)}), "
            f"k={k_key} ({tuple(k_shard.shape)}), "
            f"v={v_key} ({tuple(v_shard.shape)})"
        )

    def _shard_parameter(
        self,
        param_name: str,
        param_tensor: torch.Tensor,
        *,
        rank: int,
        world_size: int,
        head_dim: int,
        heads_per_rank: int,
        kv_heads_per_rank: int,
        intermediate_per_rank: int,
    ) -> torch.Tensor:
        """Determine how to shard a single parameter and return this rank's slice.

        Returns the full tensor for redundant parameters, or the appropriate
        slice for sharded parameters.
        """
        # Check if this is a redundant (non-sharded) parameter
        if self._is_redundant(param_name):
            return param_tensor.clone()

        # Attention QKV projections: column-parallel (split output dim)
        if self._matches_attn_key(param_name, "q_proj.weight"):
            # q_proj: shape (num_heads * head_dim, hidden_size)
            # Each rank gets heads_per_rank * head_dim rows
            shard_size = heads_per_rank * head_dim
            start = rank * shard_size
            return param_tensor.narrow(0, start, shard_size).clone()

        if self._matches_attn_key(param_name, "k_proj.weight"):
            # k_proj: shape (num_kv_heads * head_dim, hidden_size)
            # Each rank gets kv_heads_per_rank * head_dim rows
            shard_size = kv_heads_per_rank * head_dim
            start = rank * shard_size
            return param_tensor.narrow(0, start, shard_size).clone()

        if self._matches_attn_key(param_name, "v_proj.weight"):
            # v_proj: shape (num_kv_heads * head_dim, hidden_size)
            # Each rank gets kv_heads_per_rank * head_dim rows
            shard_size = kv_heads_per_rank * head_dim
            start = rank * shard_size
            return param_tensor.narrow(0, start, shard_size).clone()

        # Attention output projection: row-parallel (split input dim)
        if self._matches_attn_key(param_name, "o_proj.weight"):
            # o_proj: shape (hidden_size, num_heads * head_dim)
            # Each rank gets head_dim * heads_per_rank columns
            shard_size = heads_per_rank * head_dim
            start = rank * shard_size
            return param_tensor.narrow(1, start, shard_size).clone()

        # MLP gate/up projections: column-parallel (split output dim)
        if self._matches_mlp_key(param_name, "gate_proj.weight"):
            # gate_proj: shape (intermediate_size, hidden_size)
            # Each rank gets intermediate_per_rank rows
            start = rank * intermediate_per_rank
            return param_tensor.narrow(0, start, intermediate_per_rank).clone()

        if self._matches_mlp_key(param_name, "up_proj.weight"):
            # up_proj: shape (intermediate_size, hidden_size)
            # Each rank gets intermediate_per_rank rows
            start = rank * intermediate_per_rank
            return param_tensor.narrow(0, start, intermediate_per_rank).clone()

        # MLP down projection: row-parallel (split input dim)
        if self._matches_mlp_key(param_name, "down_proj.weight"):
            # down_proj: shape (hidden_size, intermediate_size)
            # Each rank gets intermediate_per_rank columns
            start = rank * intermediate_per_rank
            return param_tensor.narrow(1, start, intermediate_per_rank).clone()

        # Handle bias terms with the same sharding as their corresponding weights
        if self._matches_attn_key(param_name, "q_proj.bias"):
            shard_size = heads_per_rank * head_dim
            start = rank * shard_size
            return param_tensor.narrow(0, start, shard_size).clone()

        if self._matches_attn_key(param_name, "k_proj.bias"):
            shard_size = kv_heads_per_rank * head_dim
            start = rank * shard_size
            return param_tensor.narrow(0, start, shard_size).clone()

        if self._matches_attn_key(param_name, "v_proj.bias"):
            shard_size = kv_heads_per_rank * head_dim
            start = rank * shard_size
            return param_tensor.narrow(0, start, shard_size).clone()

        if self._matches_attn_key(param_name, "o_proj.bias"):
            # o_proj bias is NOT sharded — it's added after all-reduce
            return param_tensor.clone()

        if self._matches_mlp_key(param_name, "gate_proj.bias"):
            start = rank * intermediate_per_rank
            return param_tensor.narrow(0, start, intermediate_per_rank).clone()

        if self._matches_mlp_key(param_name, "up_proj.bias"):
            start = rank * intermediate_per_rank
            return param_tensor.narrow(0, start, intermediate_per_rank).clone()

        if self._matches_mlp_key(param_name, "down_proj.bias"):
            # down_proj bias is NOT sharded — it's added after all-reduce
            return param_tensor.clone()

        # Unknown parameter — keep redundant (safe default)
        logger.debug(f"Rank {rank}: keeping parameter '{param_name}' redundant (unrecognized)")
        return param_tensor.clone()

    @staticmethod
    def _is_redundant(param_name: str) -> bool:
        """Check if a parameter should be kept redundant (not sharded)."""
        for pattern in _REDUNDANT_PATTERNS:
            if param_name.endswith(pattern) or param_name == pattern:
                return True
        # Linear attention (Gated DeltaNet) weights are redundant on all ranks
        if _LINEAR_ATTN_PATTERN in param_name:
            return True
        return False

    @staticmethod
    def _matches_attn_key(param_name: str, suffix: str) -> bool:
        """Check if param_name matches an attention layer parameter."""
        return ".self_attn." + suffix in param_name

    @staticmethod
    def _matches_mlp_key(param_name: str, suffix: str) -> bool:
        """Check if param_name matches an MLP layer parameter."""
        return ".mlp." + suffix in param_name

    def _split_fused_gate_up(
        self,
        param_name: str,
        param_tensor: torch.Tensor,
        *,
        is_bias: bool,
        rank: int,
        intermediate_per_rank: int,
    ) -> None:
        """Split a fused gate_up_proj weight/bias into separate gate_proj, up_proj entries.

        Fused gate_up_proj layout (Phi-style):
        - Weight shape: [2 * intermediate_size, hidden_size]
        - Bias shape: [2 * intermediate_size]

        The fused tensor is ordered as [gate, up] along dimension 0.
        After splitting, each part is sharded per-rank and stored with
        the canonical key names (gate_proj.weight, up_proj.weight).
        """
        total_intermediate = param_tensor.shape[0] // 2
        gate_full = param_tensor.narrow(0, 0, total_intermediate)
        up_full = param_tensor.narrow(0, total_intermediate, total_intermediate)

        # Shard each part for this rank (column-parallel: slice output dim)
        gate_shard = gate_full.narrow(0, rank * intermediate_per_rank, intermediate_per_rank).clone()
        up_shard = up_full.narrow(0, rank * intermediate_per_rank, intermediate_per_rank).clone()

        # Construct canonical key names
        suffix = "bias" if is_bias else "weight"
        gate_key = param_name.replace(f"gate_up_proj.{suffix}", f"gate_proj.{suffix}")
        up_key = param_name.replace(f"gate_up_proj.{suffix}", f"up_proj.{suffix}")

        self.sharded_state_dict[gate_key] = gate_shard.to(self.device)
        self.sharded_state_dict[up_key] = up_shard.to(self.device)

        logger.debug(
            f"Rank {rank}: split fused gate_up '{param_name}' into "
            f"gate={gate_key} ({tuple(gate_shard.shape)}), "
            f"up={up_key} ({tuple(up_shard.shape)})"
        )

    @staticmethod
    def _matches_mlp_key(param_name: str, suffix: str) -> bool:
        """Check if param_name matches an MLP layer parameter."""
        return ".mlp." + suffix in param_name

    def _detect_architecture(self) -> ModelArchitecture:
        """Detect the model architecture from sharded_state_dict key patterns.

        Inspects the stored keys to determine the model family:
        - QWEN_LLAMA: Has separate self_attn.q_proj.weight, k_proj.weight, v_proj.weight
          and uses RMSNorm (no input_layernorm.bias)
        - PHI: Has fused self_attn.qkv_proj.weight (before splitting) or
          has input_layernorm.bias (LayerNorm instead of RMSNorm)

        Returns:
            ModelArchitecture enum value.
        """
        has_layernorm_bias = False
        has_separate_qkv = False

        for key in self.sharded_state_dict:
            if "input_layernorm.bias" in key:
                has_layernorm_bias = True
            if ".self_attn.q_proj.weight" in key:
                has_separate_qkv = True

        # Phi uses LayerNorm (has bias on layernorms)
        if has_layernorm_bias:
            return ModelArchitecture.PHI

        # If we have separate QKV (either native or from fused split), check
        # for other Phi indicators. At this point, if fused QKV was split,
        # the original qkv_proj keys won't exist, but we can check for
        # Phi-specific MLP naming or other patterns.
        if has_separate_qkv:
            return ModelArchitecture.QWEN_LLAMA

        # Default to QWEN_LLAMA for unknown patterns
        return ModelArchitecture.QWEN_LLAMA

    def _column_parallel_linear(
        self, input: Any, weight: Any, bias: Any | None = None
    ) -> Any:
        """F.linear with column-parallel weight shard. No communication needed.

        Column-parallel layers split the output dimension across ranks. Each rank
        computes its slice of the output independently — no all-reduce is required
        because the partial result is consumed locally (e.g., by attention heads
        assigned to this rank, or by the activation function in MLP).

        Requirements: 4.2, 4.3, 5.1, 5.3
        """
        return F.linear(input, weight, bias)

    def _row_parallel_linear(
        self, input: Any, weight: Any, bias: Any | None = None, layer_index: int = -1
    ) -> Any:
        """F.linear with row-parallel weight shard, followed by all_reduce.

        Row-parallel layers split the input dimension across ranks. Each rank
        computes a partial output (with its slice of the input), and the partial
        outputs are summed via all-reduce to produce the full result on all ranks.

        Args:
            input: Input tensor.
            weight: This rank's row-parallel weight shard.
            bias: Optional bias (applied after all-reduce if present).
            layer_index: Transformer layer index for error context in all-reduce.

        Requirements: 4.2, 4.3, 5.1, 5.3
        """
        # Compute partial output with this rank's weight shard
        partial_output = F.linear(input, weight, None)

        # Sum partial outputs across all ranks
        result = self._all_reduce(partial_output, layer_index=layer_index)

        # Bias is added after all-reduce (it's not sharded for row-parallel layers)
        if bias is not None:
            result = result + bias

        return result

    def _detect_num_layers(self) -> int:
        """Detect the number of transformer layers from sharded_state_dict keys."""
        max_layer_idx = -1
        for key in self.sharded_state_dict:
            if ".layers." in key:
                # Extract layer index from "...layers.{idx}...."
                parts = key.split(".")
                try:
                    layer_idx_pos = parts.index("layers") + 1
                    layer_idx = int(parts[layer_idx_pos])
                    max_layer_idx = max(max_layer_idx, layer_idx)
                except (ValueError, IndexError):
                    continue
        return max_layer_idx + 1 if max_layer_idx >= 0 else 0

    def _detect_layer_prefix(self) -> str:
        """Detect the key prefix for transformer layers.

        Returns the prefix before '.layers.N.' in the state dict keys.
        Examples:
        - "model" for keys like "model.layers.0.self_attn.q_proj.weight"
        - "model.language_model" for keys like "model.language_model.layers.0.self_attn.q_proj.weight"
        """
        for key in self.sharded_state_dict:
            if ".layers." in key and ("self_attn" in key or "linear_attn" in key):
                # Extract everything before ".layers."
                idx = key.index(".layers.")
                return key[:idx]
        # Fallback
        return "model"

    def _detect_layer_types(self) -> list[str]:
        """Detect whether each layer uses linear_attention or full_attention.

        Qwen3.5/3.6 hybrid models use ~75% linear attention (Gated DeltaNet)
        and ~25% full attention (standard transformer). This method inspects
        the sharded_state_dict keys to determine each layer's type.

        A layer has linear_attn if keys like
        '{prefix}.layers.{idx}.linear_attn.in_proj_qkv.weight' exist.
        A layer has self_attn if keys like
        '{prefix}.layers.{idx}.self_attn.q_proj.weight' exist.

        Returns:
            List of layer type strings, one per layer. Each is either
            "linear_attention" or "full_attention".
        """
        num_layers = self._detect_num_layers()
        layer_types: list[str] = []

        for idx in range(num_layers):
            linear_key = f"{self._layer_prefix}.layers.{idx}.linear_attn.in_proj_qkv.weight"
            full_key = f"{self._layer_prefix}.layers.{idx}.self_attn.q_proj.weight"

            if linear_key in self.sharded_state_dict:
                layer_types.append("linear_attention")
            elif full_key in self.sharded_state_dict:
                layer_types.append("full_attention")
            else:
                # Default to full_attention for unknown layer structures
                logger.warning(
                    f"Layer {idx}: could not detect type from keys, "
                    f"defaulting to full_attention"
                )
                layer_types.append("full_attention")

        linear_count = sum(1 for t in layer_types if t == "linear_attention")
        full_count = sum(1 for t in layer_types if t == "full_attention")
        logger.info(
            f"Layer type detection: {linear_count} linear_attention, "
            f"{full_count} full_attention out of {num_layers} total layers"
        )

        return layer_types

    def _get_weight(self, key: str) -> torch.Tensor:
        """Get a weight tensor from the sharded state dict.

        Raises:
            KeyError: If the key is not found, with a descriptive message
                including the missing key and available keys with the same
                layer prefix.
        """
        try:
            return self.sharded_state_dict[key]
        except KeyError:
            # Extract prefix (everything up to the last '.')
            prefix = key.rsplit(".", 1)[0] if "." in key else ""
            similar_keys = sorted(
                k for k in self.sharded_state_dict if k.startswith(prefix)
            )
            raise KeyError(
                f"Weight key '{key}' not found in sharded_state_dict. "
                f"Available keys with prefix '{prefix}': {similar_keys}"
            )

    def _get_weight_optional(self, key: str) -> torch.Tensor | None:
        """Get a weight tensor if it exists, otherwise return None."""
        return self.sharded_state_dict.get(key)

    def _rms_norm(self, hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Apply RMS normalization (used by Qwen models instead of LayerNorm)."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        return (weight * hidden_states).to(input_dtype)

    def _apply_norm(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Apply the appropriate normalization based on bias presence.

        If bias is present, applies F.layer_norm (LayerNorm, used by Phi).
        If bias is absent, applies RMSNorm (used by Qwen/Llama).

        Args:
            hidden_states: Input tensor of shape [..., hidden_size].
            weight: Normalization weight of shape [hidden_size].
            bias: Optional normalization bias of shape [hidden_size].
                  When present, indicates LayerNorm should be used.
            eps: Epsilon for numerical stability.

        Returns:
            Normalized tensor with same shape as input.
        """
        if bias is not None:
            # LayerNorm (Phi-style): uses both weight and bias
            normalized_shape = [weight.shape[0]]
            return F.layer_norm(hidden_states, normalized_shape, weight, bias, eps)
        else:
            # RMSNorm (Qwen/Llama-style): weight only, no bias
            return self._rms_norm(hidden_states, weight, eps)

    def _apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary positional embeddings (RoPE) to Q and K tensors.

        Uses rope_theta and rope_scaling from TPShardConfig (which must be
        populated from the model's HuggingFace config). Supports standard RoPE
        and YaRN-scaled RoPE (used by Qwen3.5-4B).

        For YaRN scaling (rope_scaling["type"] == "yarn"):
        - The frequency table is rescaled by the long-factor / short-factor arrays
          when present; otherwise a uniform scaling_factor is applied.
        - This mirrors the HuggingFace Qwen3_5RotaryEmbedding implementation.

        Args:
            q: Query tensor of shape [batch, heads, seq_len, head_dim]
            k: Key tensor of shape [batch, kv_heads, seq_len, head_dim]
            position_ids: Position indices of shape [batch, seq_len]

        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes as inputs.
        """
        head_dim = q.shape[-1]
        rope_theta: float = self.config.rope_theta
        rope_scaling: dict[str, object] | None = self.config.rope_scaling

        # --- Base inverse-frequency table ---
        inv_freq = 1.0 / (
            rope_theta ** (
                torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim
            )
        )

        # --- YaRN / linear scaling ---
        if rope_scaling is not None:
            scaling_type = str(rope_scaling.get("type", ""))
            # Helper: safely extract a float from the rope_scaling dict.
            # dict.get() returns object when the dict is dict[str, object], so
            # we must guard the cast.
            _raw_factor = rope_scaling.get("factor", 1.0)
            scaling_factor_scalar: float = float(_raw_factor) if isinstance(_raw_factor, (int, float)) else 1.0

            if scaling_type in ("yarn", "longrope"):
                # YaRN may provide per-dimension long_factor / short_factor arrays or a
                # single scalar scaling_factor. Apply the reciprocal so that effectively
                # inv_freq[i] = base_inv_freq[i] / factor[i].
                long_factor = rope_scaling.get("long_factor")
                if long_factor is not None and isinstance(long_factor, (list, tuple)):
                    # Per-dimension factors (list of floats, one per frequency slot)
                    factor_tensor = torch.tensor(
                        list(long_factor),
                        device=q.device,
                        dtype=torch.float32,
                    )
                    # factor_tensor has length head_dim//2 — same as inv_freq
                    inv_freq = inv_freq / factor_tensor
                elif scaling_factor_scalar != 1.0:
                    inv_freq = inv_freq / scaling_factor_scalar
            elif scaling_type in ("linear", "dynamic"):
                # Linear: uniform rescaling.  Dynamic NTK: approximated by
                # the configured factor for inference (no runtime adaptation).
                if scaling_factor_scalar != 1.0:
                    inv_freq = inv_freq / scaling_factor_scalar

        # --- Build cos/sin tables ---
        # position_ids: [batch, seq_len] → [batch, seq_len, 1]
        pos = position_ids.unsqueeze(-1).float()
        # inv_freq: [head_dim/2] → [1, 1, head_dim/2]
        inv_freq = inv_freq.unsqueeze(0).unsqueeze(0)
        # freqs: [batch, seq_len, head_dim/2]
        freqs = pos * inv_freq
        # emb: [batch, seq_len, head_dim]  (cat of freqs with itself = standard RoPE)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(1)  # [batch, 1, seq_len, head_dim]
        sin = emb.sin().unsqueeze(1)  # [batch, 1, seq_len, head_dim]

        # --- Apply rotation ---
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed.to(q.dtype), k_embed.to(k.dtype)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half of the hidden dims of the input for RoPE."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _forward_linear_attn_layer(
        self, hidden_states: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """Forward pass for a linear attention (Gated DeltaNet) layer.

        If a native HuggingFace linear attention layer is available (extracted
        during __init__), delegates to it for correct computation. Otherwise
        falls back to the custom implementation.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size).
            layer_idx: Index of the current transformer layer.

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size).
        """
        # Use native HuggingFace layer if available (guaranteed correct)
        if layer_idx in self._native_linear_attn_layers:
            native_layer = self._native_linear_attn_layers[layer_idx]
            # Get or create the cache for this layer
            if not hasattr(self, '_native_cache'):
                self._native_cache = self._create_native_cache()
            if self._native_cache is not None:
                with torch.no_grad():
                    output = native_layer(
                        hidden_states,
                        cache_params=self._native_cache,
                        attention_mask=None,
                    )
                return output

        # Fallback to custom implementation
        from exo.worker.engines.pytorch_xpu.gated_deltanet import (
            causal_conv1d_prefill,
            causal_conv1d_update,
            gated_deltanet_chunk_prefill,
            gated_deltanet_recurrent_step,
        )

        batch_size, seq_len, _hidden_size = hidden_states.shape
        prefix = f"{self._layer_prefix}.layers.{layer_idx}.linear_attn"

        # 1. Get weights
        in_proj_qkv = self._get_weight(f"{prefix}.in_proj_qkv.weight")
        in_proj_a = self._get_weight(f"{prefix}.in_proj_a.weight")
        in_proj_b = self._get_weight(f"{prefix}.in_proj_b.weight")
        in_proj_z = self._get_weight(f"{prefix}.in_proj_z.weight")
        conv_weight = self._get_weight(f"{prefix}.conv1d.weight")
        a_log = self._get_weight(f"{prefix}.A_log")
        dt_bias = self._get_weight(f"{prefix}.dt_bias")
        norm_weight = self._get_weight(f"{prefix}.norm.weight")
        out_proj = self._get_weight(f"{prefix}.out_proj.weight")

        # 2. Infer dimensions from weight shapes
        total_qkv_dim = in_proj_qkv.shape[0]
        v_dim = in_proj_z.shape[0]
        k_dim = (total_qkv_dim - v_dim) // 2
        q_dim = k_dim
        conv_dim = q_dim + k_dim + v_dim

        # Infer head structure from A_log (num_v_heads) and dimensions
        num_v_heads = a_log.shape[0]  # 32 for Qwen3.5-4B
        value_head_dim = v_dim // num_v_heads  # 4096/32 = 128
        # key_head_dim = value_head_dim in Gated DeltaNet (both 128)
        key_head_dim = value_head_dim
        num_k_heads = q_dim // key_head_dim  # 2048/128 = 16

        # Prepare conv weight
        if conv_weight.dim() == 3:
            w_conv_3d = conv_weight  # (conv_dim, 1, kernel_size) for F.conv1d
            w_conv_2d = conv_weight.squeeze(1)  # (conv_dim, kernel_size) for manual
        else:
            w_conv_2d = conv_weight
            w_conv_3d = conv_weight.unsqueeze(1)
        kernel_size = w_conv_2d.shape[-1]

        # State keys
        conv_state_key = f"conv_{layer_idx}"
        rec_state_key = f"state_{layer_idx}"

        # 3. Project ALL tokens in parallel
        qkv_all = F.linear(hidden_states, in_proj_qkv)   # (B, T, qkv_dim)
        a_all = F.linear(hidden_states, in_proj_a)        # (B, T, num_v_heads)
        b_all = F.linear(hidden_states, in_proj_b)        # (B, T, num_v_heads)
        z_all = F.linear(hidden_states, in_proj_z)        # (B, T, v_dim)

        # Split QKV
        q_all = qkv_all[..., :q_dim]                      # (B, T, q_dim)
        k_all = qkv_all[..., q_dim:q_dim + k_dim]         # (B, T, k_dim)
        v_all = qkv_all[..., q_dim + k_dim:]              # (B, T, v_dim)

        if seq_len == 1:
            # === DECODE PATH: single token, use recurrent step ===
            conv_state = self._linear_attn_states.get(conv_state_key)
            if conv_state is None:
                conv_state = torch.zeros(
                    batch_size, conv_dim, kernel_size,
                    device=hidden_states.device, dtype=hidden_states.dtype,
                )

            rec_state = self._linear_attn_states.get(rec_state_key)
            if rec_state is None:
                rec_state = torch.zeros(
                    batch_size, num_v_heads, key_head_dim, value_head_dim,
                    device=hidden_states.device, dtype=hidden_states.dtype,
                )

            # Conv1d update (single token)
            conv_input = torch.cat([q_all, k_all, v_all], dim=-1).squeeze(1)  # (B, conv_dim)
            conv_out, conv_state = causal_conv1d_update(conv_input, conv_state, w_conv_2d)

            # Split post-conv
            q_t = conv_out[:, :q_dim]
            k_t = conv_out[:, q_dim:q_dim + k_dim]
            v_t = conv_out[:, q_dim + k_dim:]

            # Reshape to heads
            if num_k_heads != num_v_heads:
                q_t = q_t.view(batch_size, num_k_heads, key_head_dim)
                k_t = k_t.view(batch_size, num_k_heads, key_head_dim)
                repeat_factor = num_v_heads // num_k_heads
                q_t = q_t.repeat_interleave(repeat_factor, dim=1)
                k_t = k_t.repeat_interleave(repeat_factor, dim=1)
            else:
                q_t = q_t.view(batch_size, num_v_heads, key_head_dim)
                k_t = k_t.view(batch_size, num_v_heads, key_head_dim)
            v_t = v_t.view(batch_size, num_v_heads, value_head_dim)

            # L2 normalize
            q_t = F.normalize(q_t, p=2, dim=-1)
            k_t = F.normalize(k_t, p=2, dim=-1)

            # Compute gates (float32 for numerical stability, matching HF reference)
            a = a_all.squeeze(1)  # (B, num_v_heads)
            alpha = -a_log.float().exp() * F.softplus(a.float() + dt_bias.float())
            alpha = alpha.to(hidden_states.dtype)
            beta = torch.sigmoid(b_all.squeeze(1))

            # Recurrent step
            output_t, rec_state = gated_deltanet_recurrent_step(
                q_t, k_t, v_t, alpha, beta, rec_state
            )

            # Reshape and apply gated RMSNorm (per-head)
            output_t = output_t.reshape(batch_size, num_v_heads, value_head_dim)
            rms = output_t.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
            output_t = output_t * rms * norm_weight
            output_t = output_t.reshape(batch_size, v_dim)
            z_t = z_all.squeeze(1)
            output_t = output_t * F.silu(z_t)

            # Output projection
            output_t = F.linear(output_t, out_proj)

            # Store states
            self._linear_attn_states[conv_state_key] = conv_state
            self._linear_attn_states[rec_state_key] = rec_state

            return output_t.unsqueeze(1)  # (B, 1, hidden_size)

        else:
            # === PREFILL PATH: multiple tokens, use chunk-parallel ===

            # 4. Apply causal conv1d in parallel (full sequence)
            # Concatenate q, k, v for conv: (B, T, conv_dim) -> (B, conv_dim, T)
            conv_input = torch.cat([q_all, k_all, v_all], dim=-1).transpose(1, 2)
            conv_out, conv_state = causal_conv1d_prefill(conv_input, w_conv_3d)
            # conv_out: (B, conv_dim, T) -> (B, T, conv_dim)
            conv_out = conv_out.transpose(1, 2)

            # Split post-conv
            q_conv = conv_out[..., :q_dim]                    # (B, T, q_dim)
            k_conv = conv_out[..., q_dim:q_dim + k_dim]       # (B, T, k_dim)
            v_conv = conv_out[..., q_dim + k_dim:]            # (B, T, v_dim)

            # 5. Reshape to heads: (B, T, dim) -> (B, T, H, head_dim)
            if num_k_heads != num_v_heads:
                q_heads = q_conv.view(batch_size, seq_len, num_k_heads, key_head_dim)
                k_heads = k_conv.view(batch_size, seq_len, num_k_heads, key_head_dim)
                repeat_factor = num_v_heads // num_k_heads
                q_heads = q_heads.repeat_interleave(repeat_factor, dim=2)
                k_heads = k_heads.repeat_interleave(repeat_factor, dim=2)
            else:
                q_heads = q_conv.view(batch_size, seq_len, num_v_heads, key_head_dim)
                k_heads = k_conv.view(batch_size, seq_len, num_v_heads, key_head_dim)
            v_heads = v_conv.view(batch_size, seq_len, num_v_heads, value_head_dim)

            # 6. L2 normalize Q and K
            q_heads = F.normalize(q_heads, p=2, dim=-1)
            k_heads = F.normalize(k_heads, p=2, dim=-1)

            # 7. Compute gates for all tokens (float32 for numerical stability)
            alpha_all = -a_log.float().exp().unsqueeze(0).unsqueeze(0) * F.softplus(a_all.float() + dt_bias.float().unsqueeze(0).unsqueeze(0))
            alpha_all = alpha_all.to(hidden_states.dtype)
            # alpha_all: (B, T, num_v_heads)
            beta_all = torch.sigmoid(b_all)  # (B, T, num_v_heads)

            # 8. Chunk-parallel Gated DeltaNet
            rec_state = self._linear_attn_states.get(rec_state_key)
            attn_output, rec_state = gated_deltanet_chunk_prefill(
                q_heads, k_heads, v_heads, alpha_all, beta_all,
                initial_state=rec_state,
                chunk_size=64,
            )
            # attn_output: (B, T, num_v_heads, value_head_dim)

            # 9. Reshape: (B, T, H, d_v) -> (B, T, v_dim)
            attn_output = attn_output.reshape(batch_size, seq_len, v_dim)

            # 10. Gated RMSNorm + output projection (vectorized over T)
            # norm_weight is per-head (shape: value_head_dim), apply per head
            # Reshape to (B, T, H, d_v) for per-head norm, then back
            attn_reshaped = attn_output.view(batch_size, seq_len, num_v_heads, value_head_dim)
            rms = attn_reshaped.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
            attn_normed = attn_reshaped * rms * norm_weight
            attn_normed = attn_normed.view(batch_size, seq_len, v_dim)
            attn_normed = attn_normed * F.silu(z_all)

            # Output projection
            output = F.linear(attn_normed, out_proj)  # (B, T, hidden_size)

            # Store states
            self._linear_attn_states[conv_state_key] = conv_state
            self._linear_attn_states[rec_state_key] = rec_state

            return output

    # Track whether the first-forward diagnostic has been logged
    _forward_diag_logged: bool = False

    def forward(
        self,
        input_data: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor] | None]]:
        """Tensor-parallel forward pass through all layers.

        For each layer:
        1. LayerNorm (redundant, all ranks compute same result)
        2. Dispatch to either:
           - Full attention: QKV → RoPE → SDPA → output projection (row-parallel)
           - Linear attention: Gated DeltaNet recurrent step (redundant)
        3. Residual + LayerNorm (redundant)
        4. MLP gate/up with column-parallel weights → activation
        5. MLP down with row-parallel weights → all_reduce
        6. Residual

        Args:
            input_data: Token IDs of shape [batch, seq_len] (long tensor).
            attention_mask: Optional attention mask of shape [batch, seq_len]
                or [batch, 1, seq_len, total_seq_len]. If None, causal masking
                is applied automatically.
            past_key_values: Optional list of (key, value) tuples per layer,
                each of shape [batch, kv_heads_per_rank, past_seq_len, head_dim].
                Entries are None for linear attention layers.

        Returns:
            Tuple of (logits, new_kv_cache) where:
            - logits: shape [batch, seq_len, vocab_size]
            - new_kv_cache: list of (key, value) tuples per layer (None for
              linear attention layers which use internal recurrent state)

        Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 6.5
        """
        num_layers = self._detect_num_layers()
        heads_per_rank = self.config.heads_per_rank
        kv_heads_per_rank = self.config.kv_heads_per_rank
        head_dim = self.config.head_dim

        # --- ONE-SHOT FORWARD DIAGNOSTIC (first call only) ---
        # Emits a single INFO log per rank with shard configuration and
        # key weight shapes so we can verify tensor-parallel sharding is correct.
        if not self._forward_diag_logged:
            self._forward_diag_logged = True
            import torch.distributed as _dist_diag
            pg_initialized = _dist_diag.is_initialized()
            try:
                _rank_from_pg = _dist_diag.get_rank() if pg_initialized else -1
            except Exception:
                _rank_from_pg = -1

            # Sample the actual shape of layer 0 q_proj.weight to confirm sharding
            _q_key = f"{self._layer_prefix}.layers.0.self_attn.q_proj.weight"
            _q_shape = tuple(self.sharded_state_dict[_q_key].shape) if _q_key in self.sharded_state_dict else "NOT_FOUND"
            _o_key = f"{self._layer_prefix}.layers.0.self_attn.o_proj.weight"
            _o_shape = tuple(self.sharded_state_dict[_o_key].shape) if _o_key in self.sharded_state_dict else "NOT_FOUND"

            logger.info(
                f"[FORWARD_DIAG] rank={self.config.rank}/{self.config.world_size} "
                f"pg_initialized={pg_initialized} rank_from_pg={_rank_from_pg} "
                f"tp_group={get_tensor_parallel_group()!r} "
                f"num_layers={num_layers} heads_per_rank={heads_per_rank} "
                f"kv_heads_per_rank={kv_heads_per_rank} head_dim={head_dim} "
                f"rope_theta={self.config.rope_theta} "
                f"rope_scaling_type={self.config.rope_scaling.get('type') if self.config.rope_scaling else None} "
                f"q_proj.weight[0] shape={_q_shape} "
                f"o_proj.weight[0] shape={_o_shape} "
                f"(q expected ({heads_per_rank * head_dim}, {self.config.hidden_size}), "
                f"o expected ({self.config.hidden_size}, {heads_per_rank * head_dim}))"
            )

        # GQA repeat factor: how many Q head groups share each KV head
        gqa_groups = heads_per_rank // kv_heads_per_rank

        # Initialize KV cache if not provided
        if past_key_values is None:
            past_key_values = [None] * num_layers  # type: ignore[list-item]

        new_kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None] = []

        # --- Embedding lookup (redundant on all ranks) ---
        embed_weight = self._get_weight(f"{self._layer_prefix}.embed_tokens.weight")
        hidden_states = F.embedding(input_data, embed_weight)

        batch_size, seq_len, _ = hidden_states.shape

        # Compute position IDs based on past KV cache length
        past_seq_len = 0
        if past_key_values is not None:
            for pv in past_key_values:
                if pv is not None:
                    past_seq_len = pv[0].shape[2]
                    break
        position_ids = torch.arange(
            past_seq_len, past_seq_len + seq_len, device=hidden_states.device
        ).unsqueeze(0).expand(batch_size, -1)

        # --- Transformer layers ---
        for layer_idx in range(num_layers):
            residual = hidden_states

            # 1. Input LayerNorm (redundant — all ranks compute same result)
            ln_weight = self._get_weight(
                f"{self._layer_prefix}.layers.{layer_idx}.input_layernorm.weight"
            )
            ln_bias = self._get_weight_optional(
                f"{self._layer_prefix}.layers.{layer_idx}.input_layernorm.bias"
            )
            hidden_states = self._apply_norm(hidden_states, ln_weight, ln_bias)

            # Dispatch based on layer type (linear_attention vs full_attention)
            if self._layer_types[layer_idx] == "linear_attention":
                # Linear attention (Gated DeltaNet) — no KV cache needed
                attn_output = self._forward_linear_attn_layer(hidden_states, layer_idx)
                new_kv_cache.append(None)  # type: ignore[arg-type]
            else:
                # Full attention (standard transformer with KV cache)
                # 2. QKV projections (column-parallel)
                q_weight = self._get_weight(
                    f"{self._layer_prefix}.layers.{layer_idx}.self_attn.q_proj.weight"
                )
                k_weight = self._get_weight(
                    f"{self._layer_prefix}.layers.{layer_idx}.self_attn.k_proj.weight"
                )
                v_weight = self._get_weight(
                    f"{self._layer_prefix}.layers.{layer_idx}.self_attn.v_proj.weight"
                )

                q_bias = self._get_weight_optional(
                    f"{self._layer_prefix}.layers.{layer_idx}.self_attn.q_proj.bias"
                )
                k_bias = self._get_weight_optional(
                    f"{self._layer_prefix}.layers.{layer_idx}.self_attn.k_proj.bias"
                )
                v_bias = self._get_weight_optional(
                    f"{self._layer_prefix}.layers.{layer_idx}.self_attn.v_proj.bias"
                )

                # Column-parallel QKV: each rank computes its assigned heads
                q = self._column_parallel_linear(hidden_states, q_weight, q_bias)
                k = self._column_parallel_linear(hidden_states, k_weight, k_bias)
                v = self._column_parallel_linear(hidden_states, v_weight, v_bias)

                # Reshape for multi-head attention
                # q: [batch, seq_len, heads_per_rank * head_dim] -> [batch, heads_per_rank, seq_len, head_dim]
                q = q.view(batch_size, seq_len, heads_per_rank, head_dim).transpose(1, 2)
                # k, v: [batch, seq_len, kv_heads_per_rank * head_dim] -> [batch, kv_heads_per_rank, seq_len, head_dim]
                k = k.view(batch_size, seq_len, kv_heads_per_rank, head_dim).transpose(1, 2)
                v = v.view(batch_size, seq_len, kv_heads_per_rank, head_dim).transpose(1, 2)

                # Apply QK normalization if present (Qwen3.5/3.6)
                q_norm_weight = self._get_weight_optional(
                    f"{self._layer_prefix}.layers.{layer_idx}.self_attn.q_norm.weight"
                )
                k_norm_weight = self._get_weight_optional(
                    f"{self._layer_prefix}.layers.{layer_idx}.self_attn.k_norm.weight"
                )
                if q_norm_weight is not None:
                    q = self._rms_norm(q, q_norm_weight)
                if k_norm_weight is not None:
                    k = self._rms_norm(k, k_norm_weight)

                # Apply rotary positional embeddings
                q, k = self._apply_rotary_pos_emb(q, k, position_ids)

                # KV cache: append new K, V to past cache (head-parallel)
                layer_past = past_key_values[layer_idx]
                if layer_past is not None:
                    past_k, past_v = layer_past
                    k = torch.cat([past_k, k], dim=2)
                    v = torch.cat([past_v, v], dim=2)

                # Store updated KV cache for this layer
                new_kv_cache.append((k, v))

                # GQA: expand K, V to match Q head count if needed
                if gqa_groups > 1:
                    # k: [batch, kv_heads_per_rank, total_seq, head_dim]
                    # -> [batch, heads_per_rank, total_seq, head_dim]
                    k_expanded = k.unsqueeze(2).expand(
                        batch_size, kv_heads_per_rank, gqa_groups, k.shape[2], head_dim
                    ).reshape(batch_size, heads_per_rank, k.shape[2], head_dim)
                    v_expanded = v.unsqueeze(2).expand(
                        batch_size, kv_heads_per_rank, gqa_groups, v.shape[2], head_dim
                    ).reshape(batch_size, heads_per_rank, v.shape[2], head_dim)
                else:
                    k_expanded = k
                    v_expanded = v

                # Scaled dot-product attention (local — only this rank's heads)
                attn_output = F.scaled_dot_product_attention(
                    q, k_expanded, v_expanded,
                    attn_mask=attention_mask,
                    is_causal=(attention_mask is None and seq_len > 1),
                )

                # Reshape attention output: [batch, heads_per_rank, seq_len, head_dim]
                # -> [batch, seq_len, heads_per_rank * head_dim]
                attn_output = attn_output.transpose(1, 2).contiguous().view(
                    batch_size, seq_len, heads_per_rank * head_dim
                )

                # 3. Output projection (row-parallel) → all_reduce
                o_weight = self._get_weight(
                    f"{self._layer_prefix}.layers.{layer_idx}.self_attn.o_proj.weight"
                )
                o_bias = self._get_weight_optional(
                    f"{self._layer_prefix}.layers.{layer_idx}.self_attn.o_proj.bias"
                )
                attn_output = self._row_parallel_linear(
                    attn_output, o_weight, o_bias, layer_index=layer_idx
                )

            # 4. Residual connection
            hidden_states = residual + attn_output

            # --- MLP block ---
            residual = hidden_states

            # Post-attention LayerNorm (redundant)
            post_ln_weight = self._get_weight(
                f"{self._layer_prefix}.layers.{layer_idx}.post_attention_layernorm.weight"
            )
            post_ln_bias = self._get_weight_optional(
                f"{self._layer_prefix}.layers.{layer_idx}.post_attention_layernorm.bias"
            )
            hidden_states = self._apply_norm(hidden_states, post_ln_weight, post_ln_bias)

            # 5. MLP gate/up projections (column-parallel)
            gate_weight = self._get_weight(
                f"{self._layer_prefix}.layers.{layer_idx}.mlp.gate_proj.weight"
            )
            up_weight = self._get_weight(
                f"{self._layer_prefix}.layers.{layer_idx}.mlp.up_proj.weight"
            )
            gate_bias = self._get_weight_optional(
                f"{self._layer_prefix}.layers.{layer_idx}.mlp.gate_proj.bias"
            )
            up_bias = self._get_weight_optional(
                f"{self._layer_prefix}.layers.{layer_idx}.mlp.up_proj.bias"
            )

            gate = self._column_parallel_linear(hidden_states, gate_weight, gate_bias)
            up = self._column_parallel_linear(hidden_states, up_weight, up_bias)

            # SiLU activation (SwiGLU): hidden = silu(gate) * up
            mlp_hidden = F.silu(gate) * up

            # 6. MLP down projection (row-parallel) → all_reduce
            down_weight = self._get_weight(
                f"{self._layer_prefix}.layers.{layer_idx}.mlp.down_proj.weight"
            )
            down_bias = self._get_weight_optional(
                f"{self._layer_prefix}.layers.{layer_idx}.mlp.down_proj.bias"
            )
            mlp_output = self._row_parallel_linear(
                mlp_hidden, down_weight, down_bias, layer_index=layer_idx
            )

            # 7. Residual connection
            hidden_states = residual + mlp_output

        # --- Final LayerNorm (redundant on all ranks) ---
        final_ln_weight = self._get_weight(f"{self._layer_prefix}.norm.weight")
        final_ln_bias = self._get_weight_optional(f"{self._layer_prefix}.norm.bias")
        hidden_states = self._apply_norm(hidden_states, final_ln_weight, final_ln_bias)

        # --- lm_head projection (redundant on all ranks) ---
        # Some models tie lm_head weights with embed_tokens
        lm_head_weight = self._get_weight_optional("lm_head.weight")
        if lm_head_weight is None:
            # Tied embeddings — reuse embed_tokens weight
            lm_head_weight = self._get_weight(f"{self._layer_prefix}.embed_tokens.weight")
        logits = F.linear(hidden_states, lm_head_weight)

        return logits, new_kv_cache

    # Track whether we have already logged the first all-reduce summary (once per rank)
    _allreduce_diag_logged: bool = False

    def _all_reduce(self, tensor: Any, layer_index: int = -1) -> Any:
        """All-reduce (sum) over the tensor-parallel process group with CPU staging.

        Gloo backend does not support XPU tensors for collective operations.
        Tensors are staged to CPU before all_reduce, then moved back to the
        original device. On Intel iGPUs with shared memory, the CPU↔XPU copy
        is near-zero cost.

        The Gloo backend also does not reliably support bfloat16 all-reduce in
        all PyTorch builds — it may silently return the local partial sum without
        performing the reduction. We therefore cast to float32 on CPU, perform the
        all-reduce, then cast back to the original dtype before moving to device.
        This is numerically safe: float32 has higher precision than bfloat16, so
        the round-trip bf16→f32→bf16 introduces no additional error beyond what
        the original bf16 computation already has.

        Args:
            tensor: Tensor to all-reduce (on any device).
            layer_index: Transformer layer index for error context on timeout.

        Returns:
            The tensor after all-reduce, on the original device.

        Raises:
            RuntimeError: If the all-reduce times out or fails.

        Requirements: 4.2, 4.3, 4.5, 11.1
        """
        tp_group = get_tensor_parallel_group()
        original_device = tensor.device
        original_dtype = tensor.dtype

        try:
            # Stage to CPU as float32. Gloo does not reliably support bfloat16
            # collectives; an explicit cast ensures the all-reduce is actually
            # performed across all ranks rather than silently returning the local
            # partial result.
            cpu_f32 = tensor.detach().to(dtype=torch.float32, device="cpu")

            # --- DIAGNOSTIC: capture pre-reduce norm on layer 0 o_proj (first call) ---
            _do_diag = (not self._allreduce_diag_logged and layer_index == 0)
            _pre_norm: float = 0.0
            if _do_diag:
                _pre_norm = float(cpu_f32.norm().item())

            dist.all_reduce(
                cpu_f32,
                op=dist.ReduceOp.SUM,
                group=tp_group,
                async_op=False,
            )

            if _do_diag:
                _post_norm = float(cpu_f32.norm().item())
                _ratio = _post_norm / (_pre_norm + 1e-10)
                logger.info(
                    f"[ALLREDUCE_DIAG] rank={self.config.rank}/{self.config.world_size} "
                    f"layer={layer_index} "
                    f"tp_group={tp_group!r} "
                    f"pre_norm={_pre_norm:.4f} post_norm={_post_norm:.4f} "
                    f"ratio={_ratio:.4f} "
                    f"(expected ~{self.config.world_size}.0 if reduction is correct, "
                    f"~1.0 if reduction is no-op)"
                )
                self._allreduce_diag_logged = True

            # Cast back to original dtype and move to original device in one step
            result = cpu_f32.to(dtype=original_dtype, device=original_device)
            tensor.copy_(result)
        except Exception as exc:
            rank = self.config.rank
            timeout = self.config.allreduce_timeout_seconds
            raise RuntimeError(
                f"Tensor-parallel all-reduce failed: "
                f"layer_index={layer_index}, "
                f"tensor_shape={tuple(tensor.shape)}, "
                f"tensor_dtype={original_dtype}, "
                f"timeout={timeout}s, "
                f"device={original_device}, "
                f"rank={rank}"
            ) from exc

        return tensor

"""Tensor-parallel weight sharding configuration and model wrapper.

This module provides the TPShardConfig dataclass for configuring tensor-parallel
weight sharding across multiple ranks connected via Thunderbolt 4, and the
TensorParallelShard class that holds sharded weights and executes the
tensor-parallel forward pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from exo.worker.engines.pytorch_xpu.distributed import get_tensor_parallel_group

logger = logging.getLogger(__name__)


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
_REDUNDANT_PATTERNS = (
    "model.embed_tokens.weight",
    "model.norm.weight",
    "lm_head.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)


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

        # Extract state dict from model if it's a module
        if isinstance(model, dict):
            state_dict = model
        elif hasattr(model, "state_dict"):
            state_dict = model.state_dict()
        else:
            raise TypeError(
                f"model must be a state dict (dict) or have a state_dict() method, "
                f"got {type(model).__name__}"
            )

        self.shard_weights(state_dict)

        logger.info(
            f"TensorParallelShard initialized: rank={config.rank}/{config.world_size}, "
            f"device={device}, "
            f"sharded_params={len(self.sharded_state_dict)}, "
            f"heads_per_rank={config.heads_per_rank}, "
            f"kv_heads_per_rank={config.kv_heads_per_rank}, "
            f"intermediate_per_rank={config.intermediate_per_rank}"
        )

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
        """
        rank = self.config.rank
        world_size = self.config.world_size
        head_dim = self.config.head_dim
        heads_per_rank = self.config.heads_per_rank
        kv_heads_per_rank = self.config.kv_heads_per_rank
        intermediate_per_rank = self.config.intermediate_per_rank

        for param_name, param_tensor in state_dict.items():
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
        return False

    @staticmethod
    def _matches_attn_key(param_name: str, suffix: str) -> bool:
        """Check if param_name matches an attention layer parameter."""
        return ".self_attn." + suffix in param_name

    @staticmethod
    def _matches_mlp_key(param_name: str, suffix: str) -> bool:
        """Check if param_name matches an MLP layer parameter."""
        return ".mlp." + suffix in param_name

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
            if "model.layers." in key:
                # Extract layer index from "model.layers.{idx}...."
                parts = key.split(".")
                try:
                    layer_idx_pos = parts.index("layers") + 1
                    layer_idx = int(parts[layer_idx_pos])
                    max_layer_idx = max(max_layer_idx, layer_idx)
                except (ValueError, IndexError):
                    continue
        return max_layer_idx + 1 if max_layer_idx >= 0 else 0

    def _get_weight(self, key: str) -> torch.Tensor:
        """Get a weight tensor from the sharded state dict."""
        return self.sharded_state_dict[key]

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

    def _apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary positional embeddings (RoPE) to Q and K tensors.

        Args:
            q: Query tensor of shape [batch, heads, seq_len, head_dim]
            k: Key tensor of shape [batch, kv_heads, seq_len, head_dim]
            position_ids: Position indices of shape [batch, seq_len]

        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes as inputs.
        """
        head_dim = q.shape[-1]
        # Compute inverse frequencies for RoPE
        # Use base 1000000.0 (Qwen3.5 uses a large RoPE base)
        inv_freq = 1.0 / (
            1000000.0 ** (torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim)
        )
        # position_ids: [batch, seq_len] -> [batch, seq_len, 1]
        pos = position_ids.unsqueeze(-1).float()
        # inv_freq: [head_dim/2] -> [1, 1, head_dim/2]
        inv_freq = inv_freq.unsqueeze(0).unsqueeze(0)
        # freqs: [batch, seq_len, head_dim/2]
        freqs = pos * inv_freq
        # emb: [batch, seq_len, head_dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(1)  # [batch, 1, seq_len, head_dim]
        sin = emb.sin().unsqueeze(1)  # [batch, 1, seq_len, head_dim]

        # Apply rotation
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed.to(q.dtype), k_embed.to(k.dtype)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half of the hidden dims of the input for RoPE."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        input_data: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Tensor-parallel forward pass through all layers.

        For each layer:
        1. LayerNorm (redundant, all ranks compute same result)
        2. QKV projection with column-parallel weights → local attention
        3. Output projection with row-parallel weights → all_reduce
        4. Residual + LayerNorm (redundant)
        5. MLP gate/up with column-parallel weights → activation
        6. MLP down with row-parallel weights → all_reduce
        7. Residual

        Args:
            input_data: Token IDs of shape [batch, seq_len] (long tensor).
            attention_mask: Optional attention mask of shape [batch, seq_len]
                or [batch, 1, seq_len, total_seq_len]. If None, causal masking
                is applied automatically.
            past_key_values: Optional list of (key, value) tuples per layer,
                each of shape [batch, kv_heads_per_rank, past_seq_len, head_dim].

        Returns:
            Tuple of (logits, new_kv_cache) where:
            - logits: shape [batch, seq_len, vocab_size]
            - new_kv_cache: list of (key, value) tuples per layer

        Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 6.5
        """
        num_layers = self._detect_num_layers()
        heads_per_rank = self.config.heads_per_rank
        kv_heads_per_rank = self.config.kv_heads_per_rank
        head_dim = self.config.head_dim

        # GQA repeat factor: how many Q head groups share each KV head
        gqa_groups = heads_per_rank // kv_heads_per_rank

        # Initialize KV cache if not provided
        if past_key_values is None:
            past_key_values = [None] * num_layers  # type: ignore[list-item]

        new_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = []

        # --- Embedding lookup (redundant on all ranks) ---
        embed_weight = self._get_weight("model.embed_tokens.weight")
        hidden_states = F.embedding(input_data, embed_weight)

        batch_size, seq_len, _ = hidden_states.shape

        # Compute position IDs based on past KV cache length
        past_seq_len = 0
        if past_key_values[0] is not None:
            past_seq_len = past_key_values[0][0].shape[2]
        position_ids = torch.arange(
            past_seq_len, past_seq_len + seq_len, device=hidden_states.device
        ).unsqueeze(0).expand(batch_size, -1)

        # --- Transformer layers ---
        for layer_idx in range(num_layers):
            residual = hidden_states

            # 1. Input LayerNorm (redundant — all ranks compute same result)
            ln_weight = self._get_weight(
                f"model.layers.{layer_idx}.input_layernorm.weight"
            )
            hidden_states = self._rms_norm(hidden_states, ln_weight)

            # 2. QKV projections (column-parallel)
            q_weight = self._get_weight(
                f"model.layers.{layer_idx}.self_attn.q_proj.weight"
            )
            k_weight = self._get_weight(
                f"model.layers.{layer_idx}.self_attn.k_proj.weight"
            )
            v_weight = self._get_weight(
                f"model.layers.{layer_idx}.self_attn.v_proj.weight"
            )

            q_bias = self._get_weight_optional(
                f"model.layers.{layer_idx}.self_attn.q_proj.bias"
            )
            k_bias = self._get_weight_optional(
                f"model.layers.{layer_idx}.self_attn.k_proj.bias"
            )
            v_bias = self._get_weight_optional(
                f"model.layers.{layer_idx}.self_attn.v_proj.bias"
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
                f"model.layers.{layer_idx}.self_attn.o_proj.weight"
            )
            o_bias = self._get_weight_optional(
                f"model.layers.{layer_idx}.self_attn.o_proj.bias"
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
                f"model.layers.{layer_idx}.post_attention_layernorm.weight"
            )
            hidden_states = self._rms_norm(hidden_states, post_ln_weight)

            # 5. MLP gate/up projections (column-parallel)
            gate_weight = self._get_weight(
                f"model.layers.{layer_idx}.mlp.gate_proj.weight"
            )
            up_weight = self._get_weight(
                f"model.layers.{layer_idx}.mlp.up_proj.weight"
            )
            gate_bias = self._get_weight_optional(
                f"model.layers.{layer_idx}.mlp.gate_proj.bias"
            )
            up_bias = self._get_weight_optional(
                f"model.layers.{layer_idx}.mlp.up_proj.bias"
            )

            gate = self._column_parallel_linear(hidden_states, gate_weight, gate_bias)
            up = self._column_parallel_linear(hidden_states, up_weight, up_bias)

            # SiLU activation (SwiGLU): hidden = silu(gate) * up
            mlp_hidden = F.silu(gate) * up

            # 6. MLP down projection (row-parallel) → all_reduce
            down_weight = self._get_weight(
                f"model.layers.{layer_idx}.mlp.down_proj.weight"
            )
            down_bias = self._get_weight_optional(
                f"model.layers.{layer_idx}.mlp.down_proj.bias"
            )
            mlp_output = self._row_parallel_linear(
                mlp_hidden, down_weight, down_bias, layer_index=layer_idx
            )

            # 7. Residual connection
            hidden_states = residual + mlp_output

        # --- Final LayerNorm (redundant on all ranks) ---
        final_ln_weight = self._get_weight("model.norm.weight")
        hidden_states = self._rms_norm(hidden_states, final_ln_weight)

        # --- lm_head projection (redundant on all ranks) ---
        lm_head_weight = self._get_weight("lm_head.weight")
        logits = F.linear(hidden_states, lm_head_weight)

        return logits, new_kv_cache

    def _all_reduce(self, tensor: Any, layer_index: int = -1) -> Any:
        """All-reduce (sum) over the tensor-parallel process group with CPU staging.

        Gloo backend does not support XPU tensors for collective operations.
        Tensors are staged to CPU before all_reduce, then moved back to the
        original device. On Intel iGPUs with shared memory, the CPU↔XPU copy
        is near-zero cost.

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

        try:
            # Stage to CPU for Gloo all_reduce
            cpu_tensor = tensor.to("cpu")
            dist.all_reduce(
                cpu_tensor,
                op=dist.ReduceOp.SUM,
                group=tp_group,
                async_op=False,
            )
            # Move back to original device
            tensor.copy_(cpu_tensor.to(original_device))
        except Exception as exc:
            rank = self.config.rank
            raise RuntimeError(
                f"Tensor-parallel all-reduce failed: "
                f"layer_index={layer_index}, "
                f"tensor_shape={tuple(tensor.shape)}, "
                f"device={original_device}, "
                f"rank={rank}"
            ) from exc

        return tensor

"""Fused pointwise kernel reference implementations for XPU Inductor auto-fusion.

These are reference implementations written in a fusion-friendly style that
``torch.compile()`` with the Inductor backend will auto-fuse into single GPU
kernels. They serve as:

1. Correctness reference for testing
2. Fallback if Inductor fails to fuse
3. Documentation of the intended fused operations

No custom SYCL kernels — pure PyTorch ops. The Inductor backend generates
optimized SYCL code for XPU targets when these functions are compiled.

Written in fusion-friendly style:
- No intermediate tensor names that prevent fusion
- No control flow
- All operations are standard PyTorch ops

**Validates: Requirements 6.1, 6.2, 6.3, 6.5, 7.1, 7.2, 7.3, 7.4**
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional

# ---------------------------------------------------------------------------
# Qwen3.5-4B architecture constants for fused decode attention defaults
# ---------------------------------------------------------------------------

_NUM_HEADS: int = 32
_NUM_KV_HEADS: int = 4
_HEAD_DIM: int = 80


def fused_rmsnorm_residual(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm(hidden_states + residual), returns (normed, new_residual).

    Computes the residual addition and RMSNorm in a single fused pass.
    The new residual is ``hidden_states + residual`` (pre-norm architecture),
    and the normed output is ``RMSNorm(new_residual)``.

    When compiled with ``torch.compile(backend="inductor")``, Inductor fuses
    the residual add, variance computation, rsqrt, and scale into a single
    GPU kernel launch.

    Args:
        hidden_states: Input tensor, shape ``[..., hidden_size]``.
        residual: Residual tensor to add, shape ``[..., hidden_size]``.
        weight: Learnable RMSNorm scale parameter, shape ``[hidden_size]``.
        eps: Epsilon for numerical stability in rsqrt.

    Returns:
        Tuple of (normed_output, new_residual) where:
            - normed_output: RMSNorm applied to the sum, same shape as input.
            - new_residual: ``hidden_states + residual``, for the next residual
              connection.

    **Validates: Requirement 6.1**
    """
    # Fuse residual addition with norm input — single memory pass
    new_residual = hidden_states + residual
    # RMSNorm in float32 for numerical stability, then cast back
    normed = new_residual.to(torch.float32)
    normed = normed * torch.rsqrt(normed.pow(2).mean(-1, keepdim=True) + eps)
    normed = (weight * normed).to(hidden_states.dtype)
    return normed, new_residual


def fused_silu_gate(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """SiLU(gate) * up in a single fused pass.

    Computes the gated activation used in the MLP block of transformer layers.
    When compiled with ``torch.compile(backend="inductor")``, Inductor fuses
    the SiLU activation and element-wise multiply into a single GPU kernel.

    Args:
        gate: Gate projection output, shape ``[..., intermediate_size]``.
        up: Up projection output, shape ``[..., intermediate_size]``.

    Returns:
        Gated activation output, same shape as inputs.

    **Validates: Requirement 6.2**
    """
    return functional.silu(gate) * up



# ---------------------------------------------------------------------------
# Fused decode attention — single-token attention as one fusible operation
# ---------------------------------------------------------------------------


def fused_decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_heads: int = _NUM_HEADS,
    num_kv_heads: int = _NUM_KV_HEADS,
    head_dim: int = _HEAD_DIM,
) -> torch.Tensor:
    """Fused single-token decode attention: Q×K^T scaling, softmax, V multiply.

    Computes the complete scaled dot-product attention for a single query token
    against the full KV cache in a single expression chain. Written in a
    fusion-friendly style so that ``torch.compile(backend="inductor")`` fuses
    Q×K^T, scaling, softmax, and V multiply into a single GPU kernel.

    Reads K and V directly from the StaticKVCache tensors without intermediate
    buffer copies. Handles Group Query Attention (GQA) by expanding KV heads
    to match the number of query heads.

    No causal mask is needed for decode: the single query token attends to all
    cached positions (positions 0..seq_len-1 are all valid past tokens).

    Supports Qwen3.5-4B dimensions:
        - hidden_size=2560, num_heads=32, num_kv_heads=4, head_dim=80
        - GQA group size: 32 / 4 = 8 (each KV head serves 8 query heads)

    The attention computation in a single expression chain:
        1. Expand KV heads for GQA (4 -> 32 via expand, no copy)
        2. Q x K^T (batched matmul across heads)
        3. Scale by 1/sqrt(head_dim)
        4. Softmax over sequence dimension (in float32 for stability)
        5. Attention weights x V (batched matmul)

    Args:
        query: Query tensor from current decode step.
            Shape: ``[1, 1, num_heads, head_dim]``.
        key_cache: Key cache read directly from StaticKVCache.
            Shape: ``[seq_len, num_kv_heads, head_dim]``.
            Contains valid entries for positions 0..seq_len-1.
        value_cache: Value cache read directly from StaticKVCache.
            Shape: ``[seq_len, num_kv_heads, head_dim]``.
            Contains valid entries for positions 0..seq_len-1.
        num_heads: Number of query attention heads (default: 32 for Qwen3.5-4B).
        num_kv_heads: Number of key/value attention heads (default: 4 for Qwen3.5-4B).
        head_dim: Dimension per attention head (default: 80 for Qwen3.5-4B).

    Returns:
        Attention output tensor, shape ``[1, 1, num_heads, head_dim]``.

    **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
    """
    # GQA group size and scale factor derived from parameters
    gqa_group_size = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    # --- GQA expansion: repeat KV heads to match Q heads ---
    # key_cache: [seq_len, num_kv_heads, head_dim]
    #         -> [seq_len, num_kv_heads, 1, head_dim]
    #         -> [seq_len, num_kv_heads, gqa_group_size, head_dim]
    #         -> [seq_len, num_heads, head_dim]
    # expand() is a view (no copy), reshape collapses the GQA dimension
    keys_expanded = (
        key_cache
        .unsqueeze(2)
        .expand(-1, -1, gqa_group_size, -1)
        .reshape(-1, num_heads, head_dim)
    )
    values_expanded = (
        value_cache
        .unsqueeze(2)
        .expand(-1, -1, gqa_group_size, -1)
        .reshape(-1, num_heads, head_dim)
    )

    # --- Reshape for batched matmul (heads as batch dimension) ---
    # query: [1, 1, num_heads, head_dim] -> query[0]: [1, num_heads, head_dim]
    #      -> permute(1, 0, 2): [num_heads, 1, head_dim]
    q_attn = query[0].permute(1, 0, 2)
    # keys_expanded: [seq_len, num_heads, head_dim] -> [num_heads, head_dim, seq_len]
    k_attn = keys_expanded.permute(1, 2, 0)
    # values_expanded: [seq_len, num_heads, head_dim] -> [num_heads, seq_len, head_dim]
    v_attn = values_expanded.permute(1, 0, 2)

    # --- Fused attention: Q x K^T * scale -> softmax -> x V ---
    # Single expression chain for Inductor fusion.
    # Softmax in float32 for numerical stability, cast back for V multiply.
    attn_output = torch.matmul(
        functional.softmax(
            torch.matmul(q_attn, k_attn) * scale,
            dim=-1,
            dtype=torch.float32,
        ).to(query.dtype),
        v_attn,
    )  # [num_heads, 1, head_dim]

    # --- Reshape output: [num_heads, 1, head_dim] -> [1, 1, num_heads, head_dim] ---
    return attn_output.permute(1, 0, 2).unsqueeze(0)


def fused_rotary_embedding(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to Q and K without intermediate tensors.

    Applies the RoPE rotation to both query and key tensors in a single fused
    pass. Written to avoid named intermediate tensors that would prevent
    Inductor from fusing the operations.

    When compiled with ``torch.compile(backend="inductor")``, Inductor fuses
    the split, negate, concatenate, multiply, and add operations for both Q
    and K into a single GPU kernel launch.

    The rotation formula for each head dimension pair (x1, x2):
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x2 * cos + x1 * sin

    Args:
        query: Query tensor, shape ``[batch, seq_len, num_heads, head_dim]``.
        key: Key tensor, shape ``[batch, seq_len, num_kv_heads, head_dim]``.
        cos: Cosine component, broadcastable to query/key shape.
            Typical shape: ``[1, seq_len, 1, head_dim]``.
        sin: Sine component, broadcastable to query/key shape.
            Typical shape: ``[1, seq_len, 1, head_dim]``.

    Returns:
        Tuple of (rotated_query, rotated_key) with rotary embeddings applied,
        same shapes as inputs.

    **Validates: Requirement 6.3**
    """
    # Apply rotation to query: split halves, rotate, recombine
    rotated_query = (
        query * cos
        + torch.cat((-query[..., query.shape[-1] // 2 :], query[..., : query.shape[-1] // 2]), dim=-1) * sin
    )
    # Apply rotation to key: same operation, fused with query rotation by Inductor
    rotated_key = (
        key * cos
        + torch.cat((-key[..., key.shape[-1] // 2 :], key[..., : key.shape[-1] // 2]), dim=-1) * sin
    )
    return rotated_query, rotated_key

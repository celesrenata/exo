"""Gated DeltaNet linear attention implementation for Qwen3.5/3.6 hybrid models.

Implements the recurrent (decode) and chunk-parallel (prefill) forward passes
for the Gated DeltaNet linear attention mechanism used in Qwen3.5 and Qwen3.6.

Reference: Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule"
(arXiv:2412.06464, ICLR 2025)

The state update rule:
    S_t = g_t * S_{t-1} + k_t ⊗ β_t * (v_t - g_t * S_{t-1}^T k_t)
    o_t = q_t^T * S_t / sqrt(d_k)

Where:
    g_t = exp(-exp(A_log) * softplus(α_t + dt_bias))  ∈ (0, 1]
    β_t = sigmoid(β_raw_t)  ∈ (0, 1)
    q_t, k_t are L2-normalized
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """L2 normalize along the given dimension."""
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def gated_deltanet_recurrent_step(
    q: torch.Tensor,       # (batch, num_v_heads, key_head_dim)
    k: torch.Tensor,       # (batch, num_v_heads, key_head_dim)
    v: torch.Tensor,       # (batch, num_v_heads, value_head_dim)
    gate: torch.Tensor,    # (batch, num_v_heads) — log-space decay
    beta: torch.Tensor,    # (batch, num_v_heads) — update rate [0,1]
    state: torch.Tensor,   # (batch, num_v_heads, key_head_dim, value_head_dim)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-step recurrent Gated DeltaNet update.

    Args:
        q: L2-normalized query, shape (B, H, d_k)
        k: L2-normalized key, shape (B, H, d_k)
        v: value, shape (B, H, d_v)
        gate: log-space decay gate, shape (B, H)
        beta: sigmoid update rate, shape (B, H)
        state: recurrent state matrix, shape (B, H, d_k, d_v)

    Returns:
        output: shape (B, H, d_v)
        new_state: shape (B, H, d_k, d_v)
    """
    # Step 1: Decay old state
    # gate is in log-space, exp(gate) ∈ (0, 1]
    g = gate.exp().unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
    state = state * g  # (B, H, d_k, d_v)

    # Step 2: Retrieve what state predicts for this key
    # retrieved = einsum('bhkv,bhk->bhv', state, k)
    k_expanded = k.unsqueeze(-1)  # (B, H, d_k, 1)
    retrieved = (state * k_expanded).sum(dim=-2)  # (B, H, d_v)

    # Step 3: Compute delta (error correction)
    beta_expanded = beta.unsqueeze(-1)  # (B, H, 1)
    delta = beta_expanded * (v - retrieved)  # (B, H, d_v)

    # Step 4: Write correction into state (outer product update)
    # state += k ⊗ delta
    delta_expanded = delta.unsqueeze(-2)  # (B, H, 1, d_v)
    state = state + k_expanded * delta_expanded  # (B, H, d_k, d_v)

    # Step 5: Read output
    # output = einsum('bhkv,bhk->bhv', state, q)
    q_expanded = q.unsqueeze(-1)  # (B, H, d_k, 1)
    output = (state * q_expanded).sum(dim=-2)  # (B, H, d_v)

    # Scale by 1/sqrt(d_k)
    d_k = q.shape[-1]
    output = output / (d_k ** 0.5)

    return output, state


def gated_deltanet_prefill(
    q: torch.Tensor,       # (batch, seq_len, num_v_heads, key_head_dim)
    k: torch.Tensor,       # (batch, seq_len, num_v_heads, key_head_dim)
    v: torch.Tensor,       # (batch, seq_len, num_v_heads, value_head_dim)
    gate: torch.Tensor,    # (batch, seq_len, num_v_heads) — log-space decay
    beta: torch.Tensor,    # (batch, seq_len, num_v_heads) — update rate [0,1]
    initial_state: torch.Tensor | None = None,  # (batch, num_v_heads, key_head_dim, value_head_dim)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefill forward pass using sequential recurrence.

    For simplicity, this uses the sequential recurrent form rather than the
    chunk-parallel algorithm. This is O(T * d²) but correct. For short
    sequences (typical prefill), this is acceptable.

    Args:
        q: L2-normalized queries, shape (B, T, H, d_k)
        k: L2-normalized keys, shape (B, T, H, d_k)
        v: values, shape (B, T, H, d_v)
        gate: log-space decay gates, shape (B, T, H)
        beta: sigmoid update rates, shape (B, T, H)
        initial_state: optional initial state, shape (B, H, d_k, d_v)

    Returns:
        output: shape (B, T, H, d_v)
        final_state: shape (B, H, d_k, d_v)
    """
    batch, seq_len, num_heads, d_k = q.shape
    d_v = v.shape[-1]

    if initial_state is None:
        state = torch.zeros(
            batch, num_heads, d_k, d_v,
            device=q.device, dtype=q.dtype
        )
    else:
        state = initial_state.clone()

    outputs = []
    for t in range(seq_len):
        q_t = q[:, t]      # (B, H, d_k)
        k_t = k[:, t]      # (B, H, d_k)
        v_t = v[:, t]      # (B, H, d_v)
        g_t = gate[:, t]   # (B, H)
        b_t = beta[:, t]   # (B, H)

        out_t, state = gated_deltanet_recurrent_step(q_t, k_t, v_t, g_t, b_t, state)
        outputs.append(out_t)

    output = torch.stack(outputs, dim=1)  # (B, T, H, d_v)
    return output, state


def causal_conv1d_update(
    x: torch.Tensor,           # (batch, dim) — single token input
    conv_state: torch.Tensor,  # (batch, dim, kernel_size) — sliding window
    weight: torch.Tensor,      # (dim, kernel_size) — depthwise conv weights
) -> tuple[torch.Tensor, torch.Tensor]:
    """Incremental causal conv1d for single-token decode.

    Shifts the conv state left, appends the new input, and computes
    the convolution output.

    Args:
        x: input for current timestep, shape (B, D)
        conv_state: sliding window buffer, shape (B, D, K)
        weight: depthwise conv weights, shape (D, K)

    Returns:
        output: conv output with SiLU activation, shape (B, D)
        new_conv_state: updated sliding window, shape (B, D, K)
    """
    # Shift state left and append new input
    new_state = torch.roll(conv_state, shifts=-1, dims=-1)
    new_state[:, :, -1] = x

    # Compute depthwise convolution: sum(state * weight, dim=-1)
    output = (new_state * weight.unsqueeze(0)).sum(dim=-1)  # (B, D)

    # Apply SiLU activation
    output = F.silu(output)

    return output, new_state


def causal_conv1d_prefill(
    x: torch.Tensor,       # (batch, dim, seq_len) — input sequence
    weight: torch.Tensor,  # (dim, 1, kernel_size) — depthwise conv weights
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal conv1d for prefill (full sequence).

    Args:
        x: input sequence, shape (B, D, T)
        weight: depthwise conv weights, shape (D, 1, K)

    Returns:
        output: conv output with SiLU activation, shape (B, D, T)
        conv_state: final sliding window state, shape (B, D, K)
    """
    kernel_size = weight.shape[-1]

    # Pad left for causal convolution
    x_padded = F.pad(x, (kernel_size - 1, 0))

    # Depthwise convolution
    output = F.conv1d(x_padded, weight, groups=x.shape[1])

    # SiLU activation
    output = F.silu(output)

    # Extract final conv state (last kernel_size elements of input)
    conv_state = x[:, :, -(kernel_size):].clone() if x.shape[-1] >= kernel_size else F.pad(x, (kernel_size - x.shape[-1], 0))

    return output, conv_state

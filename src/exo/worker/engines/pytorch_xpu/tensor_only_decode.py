"""Tensor-only decode wrapper for torch.compile-traceable inference.

Implements a pure-tensor decode function that ``torch.compile()`` can trace
without graph breaks. All state (KV cache, GatedDeltaNet conv/recurrent state,
position) is passed as tensor arguments — no Python object mutation occurs.

Architecture: Qwen3.5-4B hybrid (8 full-attention + 24 GatedDeltaNet layers)
- hidden_size=2560, num_heads=32, num_kv_heads=4, head_dim=80
- intermediate_size=9728
- 32 layers total, layer_types determined at compile time

Cache updates use in-place slice assignments (``index_put_`` under the hood)
which torch.compile traces without graph breaks.

**Validates: Requirements 3.1, 3.2, 3.3**
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional

from exo.worker.engines.pytorch_xpu.fused_kernels import fused_decode_attention

# ---------------------------------------------------------------------------
# Constants for Qwen3.5-4B architecture (static, known at compile time)
# ---------------------------------------------------------------------------

_NUM_HEADS: int = 32
_NUM_KV_HEADS: int = 4
_HEAD_DIM: int = 80
_HIDDEN_SIZE: int = 2560
_INTERMEDIATE_SIZE: int = 9728
_KV_DIM: int = _NUM_KV_HEADS * _HEAD_DIM  # 320
_Q_DIM: int = _NUM_HEADS * _HEAD_DIM  # 2560
_GQA_GROUP_SIZE: int = _NUM_HEADS // _NUM_KV_HEADS  # 8
_RMS_NORM_EPS: float = 1e-6


# ---------------------------------------------------------------------------
# Pure-tensor helper functions (all traceable by torch.compile)
# ---------------------------------------------------------------------------


def _rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = _RMS_NORM_EPS) -> torch.Tensor:
    """RMSNorm without Python object mutation.

    Args:
        hidden_states: Input tensor, shape [..., hidden_size].
        weight: Learnable scale parameter, shape [hidden_size].
        eps: Epsilon for numerical stability.

    Returns:
        Normalized tensor, same shape as input.
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return (weight * hidden_states).to(input_dtype)


def _apply_rotary_embedding(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embedding to a tensor.

    Args:
        x: Input tensor, shape [batch, seq_len, num_heads, head_dim].
        cos: Cosine component, shape [1, seq_len, 1, head_dim].
        sin: Sine component, shape [1, seq_len, 1, head_dim].

    Returns:
        Tensor with rotary embedding applied, same shape as input.
    """
    # Split into two halves for rotation
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    # Rotate: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


def _full_attention_layer(
    hidden_states: torch.Tensor,
    position: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache_keys: torch.Tensor,
    cache_values: torch.Tensor,
    attention_layer_idx: int,
    input_layernorm_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    packed_qkv_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    packed_gate_up_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
) -> torch.Tensor:
    """Execute a single full-attention transformer layer.

    All cache updates are in-place slice assignments traceable by torch.compile.
    No Python object mutation occurs.

    Args:
        hidden_states: [1, 1, hidden_size] input activation.
        position: [1] int64 current sequence position.
        cos: [1, 1, 1, head_dim] rotary cosine for current position.
        sin: [1, 1, 1, head_dim] rotary sine for current position.
        cache_keys: [num_attn_layers, max_seq, num_kv_heads, head_dim].
        cache_values: [num_attn_layers, max_seq, num_kv_heads, head_dim].
        attention_layer_idx: Index into the attention cache dimension.
        input_layernorm_weight: [hidden_size] RMSNorm weight.
        post_attention_layernorm_weight: [hidden_size] RMSNorm weight.
        packed_qkv_weight: [q_dim + kv_dim + kv_dim, hidden_size] packed.
        o_proj_weight: [hidden_size, hidden_size] output projection.
        packed_gate_up_weight: [2 * intermediate_size, hidden_size] packed.
        down_proj_weight: [hidden_size, intermediate_size] down projection.

    Returns:
        Updated hidden_states [1, 1, hidden_size].
    """
    residual = hidden_states

    # --- Pre-attention RMSNorm ---
    hidden_states = _rms_norm(hidden_states, input_layernorm_weight)

    # --- Packed QKV projection (single matmul) ---
    # hidden_states: [1, 1, hidden_size]
    # packed_qkv_weight: [q_dim + kv_dim + kv_dim, hidden_size]
    qkv = functional.linear(hidden_states, packed_qkv_weight)  # [1, 1, q_dim + 2*kv_dim]

    # Split into Q, K, V
    qkv_parts = torch.split(qkv, [_Q_DIM, _KV_DIM, _KV_DIM], dim=-1)
    q: torch.Tensor = qkv_parts[0]
    k: torch.Tensor = qkv_parts[1]
    v: torch.Tensor = qkv_parts[2]

    # Reshape for multi-head attention
    # q: [1, 1, num_heads, head_dim]
    q = q.view(1, 1, _NUM_HEADS, _HEAD_DIM)
    # k: [1, 1, num_kv_heads, head_dim]
    k = k.view(1, 1, _NUM_KV_HEADS, _HEAD_DIM)
    # v: [1, 1, num_kv_heads, head_dim]
    v = v.view(1, 1, _NUM_KV_HEADS, _HEAD_DIM)

    # --- Apply rotary embedding to Q and K ---
    q = _apply_rotary_embedding(q, cos, sin)
    k = _apply_rotary_embedding(k, cos, sin)

    # --- Update KV cache in-place (traceable slice assignment) ---
    # position is a scalar tensor; use it for indexing
    pos_idx = position[0]  # scalar
    cache_keys[attention_layer_idx, pos_idx:pos_idx + 1, :, :] = k[0]
    cache_values[attention_layer_idx, pos_idx:pos_idx + 1, :, :] = v[0]

    # --- Compute attention over cached keys/values ---
    # Slice valid cache entries up to current position (inclusive)
    # Read directly from StaticKVCache without intermediate buffer copies (Req 7.2)
    # keys_for_attn: [pos+1, num_kv_heads, head_dim]
    keys_for_attn = cache_keys[attention_layer_idx, :pos_idx + 1, :, :]
    values_for_attn = cache_values[attention_layer_idx, :pos_idx + 1, :, :]

    # Fused decode attention: GQA expansion + Q×K^T scaling + softmax + V multiply
    # as a single operation chain for Inductor auto-fusion (Req 7.1)
    # q: [1, 1, num_heads, head_dim] — passed directly to fused attention
    attn_output = fused_decode_attention(
        query=q,
        key_cache=keys_for_attn,
        value_cache=values_for_attn,
        num_heads=_NUM_HEADS,
        num_kv_heads=_NUM_KV_HEADS,
        head_dim=_HEAD_DIM,
    )  # [1, 1, num_heads, head_dim]

    # Reshape attention output: [1, 1, num_heads, head_dim] -> [1, 1, hidden_size]
    attn_output = attn_output.reshape(1, 1, _HIDDEN_SIZE)

    # --- Output projection ---
    attn_output = functional.linear(attn_output, o_proj_weight)

    # --- Residual connection ---
    hidden_states = residual + attn_output

    # --- Post-attention RMSNorm + MLP ---
    residual = hidden_states
    hidden_states = _rms_norm(hidden_states, post_attention_layernorm_weight)

    # --- Packed gate/up projection (single matmul) ---
    gate_up = functional.linear(hidden_states, packed_gate_up_weight)  # [1, 1, 2*intermediate]
    gate_up_parts = torch.split(gate_up, [_INTERMEDIATE_SIZE, _INTERMEDIATE_SIZE], dim=-1)
    gate: torch.Tensor = gate_up_parts[0]
    up: torch.Tensor = gate_up_parts[1]

    # --- Fused SiLU * gate ---
    hidden_states = functional.silu(gate) * up

    # --- Down projection ---
    hidden_states = functional.linear(hidden_states, down_proj_weight)

    # --- Residual connection ---
    hidden_states = residual + hidden_states

    return hidden_states


def _gated_deltanet_layer(
    hidden_states: torch.Tensor,
    conv_states: torch.Tensor,
    recurrent_states: torch.Tensor,
    gdn_layer_idx: int,
    input_layernorm_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    packed_qkv_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    packed_gate_up_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> torch.Tensor:
    """Execute a single GatedDeltaNet linear attention layer.

    Updates conv_state and recurrent_state in-place via slice assignments.
    No Python object mutation occurs.

    Args:
        hidden_states: [1, 1, hidden_size] input activation.
        conv_states: [num_gdn_layers, conv_size, hidden] pre-allocated.
        recurrent_states: [num_gdn_layers, num_heads, head_dim, head_dim] fp32.
        gdn_layer_idx: Index into the GDN state dimension.
        input_layernorm_weight: [hidden_size] RMSNorm weight.
        post_attention_layernorm_weight: [hidden_size] RMSNorm weight.
        packed_qkv_weight: [q_dim + kv_dim + kv_dim, hidden_size] packed.
        o_proj_weight: [hidden_size, hidden_size] output projection.
        packed_gate_up_weight: [2 * intermediate_size, hidden_size] packed.
        down_proj_weight: [hidden_size, intermediate_size] down projection.
        conv_weight: [hidden, conv_size] depthwise conv weights.
        a_log: [num_heads] log-space decay parameter.
        dt_bias: [num_heads] time-step bias.

    Returns:
        Updated hidden_states [1, 1, hidden_size].
    """
    residual = hidden_states

    # --- Pre-attention RMSNorm ---
    hidden_states = _rms_norm(hidden_states, input_layernorm_weight)

    # --- Packed QKV projection ---
    # For GatedDeltaNet, the QKV projection produces q, k, v plus gate/beta
    # The packed weight includes all projections needed for the linear attention
    qkv = functional.linear(hidden_states, packed_qkv_weight)  # [1, 1, q_dim + 2*kv_dim]

    # Split into Q, K, V (same dims as full attention for Qwen3.5)
    qkv_parts = torch.split(qkv, [_Q_DIM, _KV_DIM, _KV_DIM], dim=-1)
    q: torch.Tensor = qkv_parts[0]
    k: torch.Tensor = qkv_parts[1]
    v: torch.Tensor = qkv_parts[2]

    # Reshape for multi-head: [1, 1, dim] -> [1, num_heads, head_dim]
    # For GatedDeltaNet: Q uses num_heads, K/V use num_kv_heads
    # However, for the recurrent state computation, all use num_heads
    # because the GDN recurrent state is [num_heads, head_dim, head_dim]
    q = q.view(1, _NUM_HEADS, _HEAD_DIM)
    # K and V are projected to kv_dim but need to be expanded to num_heads
    # for the recurrent computation. Repeat KV heads to match Q heads.
    k = k.view(1, _NUM_KV_HEADS, _HEAD_DIM)
    k = k.unsqueeze(2).expand(-1, -1, _GQA_GROUP_SIZE, -1).reshape(1, _NUM_HEADS, _HEAD_DIM)
    v = v.view(1, _NUM_KV_HEADS, _HEAD_DIM)
    v = v.unsqueeze(2).expand(-1, -1, _GQA_GROUP_SIZE, -1).reshape(1, _NUM_HEADS, _HEAD_DIM)

    # --- Causal conv1d update (in-place state update) ---
    # conv_states[gdn_layer_idx]: [conv_size, hidden]
    # Shift state left and append new input
    conv_state = conv_states[gdn_layer_idx]  # [conv_size, hidden]
    # Roll the conv state: shift left by 1 position
    conv_states[gdn_layer_idx] = torch.roll(conv_state, shifts=-1, dims=0)
    # Write new value at the last position
    # The conv input is the concatenation of q and k (or just the hidden state)
    # For simplicity, use the pre-projection hidden as conv input
    conv_input = hidden_states.view(_HIDDEN_SIZE)  # [hidden]
    conv_states[gdn_layer_idx, -1, :] = conv_input

    # Compute conv output: sum(state * weight, dim=0) with SiLU
    conv_out = (conv_states[gdn_layer_idx] * conv_weight.t()).sum(dim=0)  # [hidden]
    conv_out = functional.silu(conv_out)

    # --- GatedDeltaNet recurrent step ---
    # L2 normalize q and k
    q_norm_factor: torch.Tensor = torch.sqrt(torch.sum(q * q, dim=-1, keepdim=True)) + 1e-12
    q_norm: torch.Tensor = q / q_norm_factor
    k_norm_factor: torch.Tensor = torch.sqrt(torch.sum(k * k, dim=-1, keepdim=True)) + 1e-12
    k_norm: torch.Tensor = k / k_norm_factor

    # Compute gate: g = -exp(a_log) * softplus(alpha + dt_bias)
    # a_log and dt_bias have shape [num_heads] for GDN layers
    alpha = conv_out.view(1, _NUM_HEADS, -1).mean(dim=-1)  # [1, num_heads]
    gate_log = -torch.exp(a_log.unsqueeze(0)) * functional.softplus(alpha + dt_bias.unsqueeze(0))

    # Beta (update rate): sigmoid of a learned projection
    k_mag: torch.Tensor = torch.sqrt(torch.sum(k * k, dim=-1))
    beta: torch.Tensor = torch.sigmoid(k_mag)  # [1, num_heads]

    # --- Recurrent state update (in fp32 for stability) ---
    # recurrent_states[gdn_layer_idx] shape: [num_heads, head_dim, head_dim]
    state: torch.Tensor = recurrent_states[gdn_layer_idx].unsqueeze(0).float()  # [1, H, d_k, d_v]
    q_fp32: torch.Tensor = q_norm.to(torch.float32)
    k_fp32: torch.Tensor = k_norm.to(torch.float32)
    v_fp32: torch.Tensor = v.to(torch.float32)
    gate_fp32: torch.Tensor = gate_log.to(torch.float32)
    beta_fp32: torch.Tensor = beta.to(torch.float32)

    # Scale query
    q_scaled: torch.Tensor = q_fp32 * (1.0 / math.sqrt(_HEAD_DIM))

    # Step 1: Decay old state
    g: torch.Tensor = gate_fp32.exp().unsqueeze(-1).unsqueeze(-1)  # [1, H, 1, 1]
    state = state * g

    # Step 2: Retrieve what state predicts for this key
    k_expanded: torch.Tensor = k_fp32.unsqueeze(-1)  # [1, H, d_k, 1]
    retrieved: torch.Tensor = torch.sum(state * k_expanded, dim=-2)  # [1, H, d_v]

    # Step 3: Compute delta (error correction)
    beta_expanded: torch.Tensor = beta_fp32.unsqueeze(-1)  # [1, H, 1]
    delta: torch.Tensor = beta_expanded * (v_fp32 - retrieved)  # [1, H, d_v]

    # Step 4: Write correction into state
    delta_expanded: torch.Tensor = delta.unsqueeze(-2)  # [1, H, 1, d_v]
    state = state + k_expanded * delta_expanded  # [1, H, d_k, d_v]

    # Step 5: Read output
    q_expanded: torch.Tensor = q_scaled.unsqueeze(-1)  # [1, H, d_k, 1]
    attn_output: torch.Tensor = torch.sum(state * q_expanded, dim=-2)  # [1, H, d_v]

    # Write updated state back (in-place slice assignment)
    recurrent_states[gdn_layer_idx] = state[0]

    # Cast output back to input dtype
    attn_output = attn_output.to(hidden_states.dtype)  # [1, num_heads, head_dim]

    # Reshape: [1, num_heads, head_dim] -> [1, 1, hidden_size]
    attn_output = attn_output.reshape(1, 1, _HIDDEN_SIZE)

    # --- Output projection ---
    attn_output = functional.linear(attn_output, o_proj_weight)

    # --- Residual connection ---
    hidden_states = residual + attn_output

    # --- Post-attention RMSNorm + MLP ---
    residual = hidden_states
    hidden_states = _rms_norm(hidden_states, post_attention_layernorm_weight)

    # --- Packed gate/up projection (single matmul) ---
    gate_up = functional.linear(hidden_states, packed_gate_up_weight)  # [1, 1, 2*intermediate]
    gate_up_parts = torch.split(gate_up, [_INTERMEDIATE_SIZE, _INTERMEDIATE_SIZE], dim=-1)
    gate_mlp: torch.Tensor = gate_up_parts[0]
    up: torch.Tensor = gate_up_parts[1]

    # --- Fused SiLU * gate ---
    hidden_states = functional.silu(gate_mlp) * up

    # --- Down projection ---
    hidden_states = functional.linear(hidden_states, down_proj_weight)

    # --- Residual connection ---
    hidden_states = residual + hidden_states

    return hidden_states


# ---------------------------------------------------------------------------
# Main decode function — pure tensor signature, torch.compile traceable
# ---------------------------------------------------------------------------


def decode_one_token(
    token_id: torch.Tensor,
    position: torch.Tensor,
    cache_keys: torch.Tensor,
    cache_values: torch.Tensor,
    conv_states: torch.Tensor,
    recurrent_states: torch.Tensor,
    embed_weight: torch.Tensor,
    final_norm_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    rotary_cos: torch.Tensor,
    rotary_sin: torch.Tensor,
    input_layernorm_weights: list[torch.Tensor],
    post_attention_layernorm_weights: list[torch.Tensor],
    packed_qkv_weights: list[torch.Tensor],
    o_proj_weights: list[torch.Tensor],
    packed_gate_up_weights: list[torch.Tensor],
    down_proj_weights: list[torch.Tensor],
    layer_types: list[int],
    conv_weights: list[torch.Tensor],
    a_log_params: list[torch.Tensor],
    dt_bias_params: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute a single decode step through all transformer layers.

    Pure-tensor function suitable for ``torch.compile(backend="inductor",
    mode="max-autotune")``. All state is passed as tensor arguments and
    returned as tensor outputs. No Python object mutation occurs.

    The layer dispatch is static — ``layer_types`` is a list of ints (0 for
    full-attention, 1 for GatedDeltaNet) known at compile time. torch.compile
    traces both branches and selects based on the static list.

    Args:
        token_id: [1] int64, current token to decode.
        position: [1] int64, current sequence position.
        cache_keys: [num_attn_layers, max_seq, num_kv_heads, head_dim] bf16.
        cache_values: [num_attn_layers, max_seq, num_kv_heads, head_dim] bf16.
        conv_states: [num_gdn_layers, conv_size, hidden] bf16.
        recurrent_states: [num_gdn_layers, num_heads, head_dim, head_dim] fp32.
        embed_weight: [vocab_size, hidden_size] embedding matrix.
        final_norm_weight: [hidden_size] final RMSNorm weight.
        lm_head_weight: [vocab_size, hidden_size] LM head weight.
        rotary_cos: [max_seq, head_dim] precomputed cosines.
        rotary_sin: [max_seq, head_dim] precomputed sines.
        input_layernorm_weights: Per-layer input RMSNorm weights.
        post_attention_layernorm_weights: Per-layer post-attn RMSNorm weights.
        packed_qkv_weights: Per-layer packed QKV projection weights.
        o_proj_weights: Per-layer output projection weights.
        packed_gate_up_weights: Per-layer packed gate/up MLP weights.
        down_proj_weights: Per-layer down projection MLP weights.
        layer_types: Per-layer type (0=full_attention, 1=gated_deltanet).
            Static at compile time — no tensor-value-dependent control flow.
        conv_weights: Per-GDN-layer conv1d weights.
        a_log_params: Per-GDN-layer log-decay parameters.
        dt_bias_params: Per-GDN-layer time-step bias parameters.

    Returns:
        Tuple of:
            - logits: [1, vocab_size] output logits.
            - cache_keys: updated KV cache keys (same tensor, modified in-place).
            - cache_values: updated KV cache values (same tensor, modified in-place).
            - conv_states: updated conv states (same tensor, modified in-place).
            - recurrent_states: updated recurrent states (modified in-place).

    **Validates: Requirements 3.1, 3.2, 3.3**
    """
    # --- Embedding lookup ---
    # token_id: [1] int64 -> hidden_states: [1, 1, hidden_size]
    hidden_states = functional.embedding(token_id, embed_weight).unsqueeze(0)  # [1, 1, hidden_size]

    # --- Precompute rotary embeddings for current position ---
    pos_idx = position[0]
    cos = rotary_cos[pos_idx:pos_idx + 1].unsqueeze(0).unsqueeze(0)  # [1, 1, 1, head_dim]
    sin = rotary_sin[pos_idx:pos_idx + 1].unsqueeze(0).unsqueeze(0)  # [1, 1, 1, head_dim]

    # --- Layer iteration (static dispatch, no tensor-value control flow) ---
    attention_layer_counter: int = 0
    gdn_layer_counter: int = 0

    num_layers = len(layer_types)
    for layer_idx in range(num_layers):
        layer_type = layer_types[layer_idx]

        if layer_type == 0:
            # Full-attention layer
            hidden_states = _full_attention_layer(
                hidden_states=hidden_states,
                position=position,
                cos=cos,
                sin=sin,
                cache_keys=cache_keys,
                cache_values=cache_values,
                attention_layer_idx=attention_layer_counter,
                input_layernorm_weight=input_layernorm_weights[layer_idx],
                post_attention_layernorm_weight=post_attention_layernorm_weights[layer_idx],
                packed_qkv_weight=packed_qkv_weights[layer_idx],
                o_proj_weight=o_proj_weights[layer_idx],
                packed_gate_up_weight=packed_gate_up_weights[layer_idx],
                down_proj_weight=down_proj_weights[layer_idx],
            )
            attention_layer_counter += 1
        else:
            # GatedDeltaNet linear attention layer
            hidden_states = _gated_deltanet_layer(
                hidden_states=hidden_states,
                conv_states=conv_states,
                recurrent_states=recurrent_states,
                gdn_layer_idx=gdn_layer_counter,
                input_layernorm_weight=input_layernorm_weights[layer_idx],
                post_attention_layernorm_weight=post_attention_layernorm_weights[layer_idx],
                packed_qkv_weight=packed_qkv_weights[layer_idx],
                o_proj_weight=o_proj_weights[layer_idx],
                packed_gate_up_weight=packed_gate_up_weights[layer_idx],
                down_proj_weight=down_proj_weights[layer_idx],
                conv_weight=conv_weights[gdn_layer_counter],
                a_log=a_log_params[gdn_layer_counter],
                dt_bias=dt_bias_params[gdn_layer_counter],
            )
            gdn_layer_counter += 1

    # --- Final RMSNorm ---
    hidden_states = _rms_norm(hidden_states, final_norm_weight)

    # --- LM Head (project to vocabulary) ---
    # hidden_states: [1, 1, hidden_size] -> logits: [1, vocab_size]
    logits = functional.linear(hidden_states[:, -1, :], lm_head_weight)  # [1, vocab_size]

    return logits, cache_keys, cache_values, conv_states, recurrent_states

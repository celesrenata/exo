# GatedDeltaNet Decode Path Analysis

Analysis of performance hotspots in `gated_deltanet.py` and its caller
(`tensor_parallel_shard.py`) for Task 4: GatedDeltaNet State Optimization.

---

## 1. Per-Token Casts from bf16 to fp32

### Location: `_gated_deltanet_recurrent_step_impl()` (lines 88–95)

Every single decode token triggers six `.float()` casts:

```python
q = q.float()       # bf16 → fp32, shape (B, H, d_k)
k = k.float()       # bf16 → fp32, shape (B, H, d_k)
v = v.float()       # bf16 → fp32, shape (B, H, d_v)
gate = gate.float() # bf16 → fp32, shape (B, H)
beta = beta.float() # bf16 → fp32, shape (B, H)
state = state.float()  # bf16 → fp32, shape (B, H, d_k, d_v)
```

**Important nuance on the state cast**: The recurrent state matrix has shape
`(B, H, d_k, d_v)` = `(1, 32, 128, 128)` = 524,288 elements for Qwen3.5-4B.

The function returns `state` in fp32 (no cast-back), and `tensor_parallel_shard.py`
stores it as-is in `self._linear_attn_states[rec_state_key]`. So on the **second
and subsequent tokens**, the state is already fp32 and `state.float()` is a no-op
(returns self). However:

1. On the **first token**, the state is allocated in bf16 (`dtype=hidden_states.dtype`)
   and gets cast to fp32 — a real 524,288-element cast.
2. The `.float()` call still executes a dtype check on every token (minor overhead).
3. The q, k, v, gate, beta casts are real on every token since they come from
   bf16 linear projections.
4. The `output.to(initial_dtype)` on return casts the fp32 output back to bf16
   on every token.

### Location: `tensor_parallel_shard.py` decode path (around line 1127)

The gate computation casts to float32 and then casts back:

```python
alpha = -a_log.float().exp() * F.softplus(a.float() + dt_bias.float())
alpha = alpha.to(hidden_states.dtype)  # cast back to bf16
```

This is then passed into `gated_deltanet_recurrent_step` which immediately
casts it back to fp32. Two unnecessary round-trips per token.

### Summary of per-token casts

| Tensor | Shape (Qwen3.5-4B) | Elements | Cast direction | Per-token? |
|--------|---------------------|----------|----------------|------------|
| q | (1, 32, 128) | 4,096 | bf16 → fp32 | Yes (every token) |
| k | (1, 32, 128) | 4,096 | bf16 → fp32 | Yes (every token) |
| v | (1, 32, 128) | 4,096 | bf16 → fp32 | Yes (every token) |
| gate | (1, 32) | 32 | bf16 → fp32 | Yes (every token) |
| beta | (1, 32) | 32 | bf16 → fp32 | Yes (every token) |
| **state** | **(1, 32, 128, 128)** | **524,288** | **bf16 → fp32** | **First token only** |
| output (return) | (1, 32, 128) | 4,096 | fp32 → bf16 | Yes (every token) |
| alpha (caller) | (1, 32) | 32 | fp32 → bf16 → fp32 | Yes (round-trip waste) |

The state cast is a first-token cost only. The per-token casts are the q/k/v/gate/beta
inputs (small tensors, ~16 KB total) and the output cast back to bf16.
The alpha round-trip in the caller is wasteful but small.

---

## 2. Per-Token Recurrent State Allocations

### Location: `tensor_parallel_shard.py` decode path (lines ~1088–1095)

On the first decode token (or if state is missing), a new state tensor is allocated:

```python
rec_state = self._linear_attn_states.get(rec_state_key)
if rec_state is None:
    rec_state = torch.zeros(
        batch_size, num_v_heads, key_head_dim, value_head_dim,
        device=hidden_states.device, dtype=hidden_states.dtype,
    )
```

This allocates in **bf16** (`hidden_states.dtype`), which forces the cast
inside `_gated_deltanet_recurrent_step_impl`.

### Location: `_gated_deltanet_recurrent_step_impl()` — implicit allocations

The function does NOT allocate a new state tensor explicitly, but the
arithmetic operations create new tensors on every token:

1. **Line 98**: `state = state * g` — allocates a new `(B, H, d_k, d_v)` tensor
2. **Line 101**: `k_expanded = k.unsqueeze(-1)` — allocates `(B, H, d_k, 1)`
3. **Line 102**: `retrieved = (state * k_expanded).sum(dim=-2)` — allocates intermediate `(B, H, d_k, d_v)` + result `(B, H, d_v)`
4. **Line 105**: `beta_expanded = beta.unsqueeze(-1)` — allocates `(B, H, 1)`
5. **Line 106**: `delta = beta_expanded * (v - retrieved)` — allocates `(B, H, d_v)` intermediate + result
6. **Line 109**: `delta_expanded = delta.unsqueeze(-2)` — allocates `(B, H, 1, d_v)`
7. **Line 110**: `state = state + k_expanded * delta_expanded` — allocates intermediate `(B, H, d_k, d_v)` + new state `(B, H, d_k, d_v)`
8. **Line 113**: `q_expanded = q.unsqueeze(-1)` — allocates `(B, H, d_k, 1)`
9. **Line 114**: `output = (state * q_expanded).sum(dim=-2)` — allocates intermediate `(B, H, d_k, d_v)` + result `(B, H, d_v)`

**Per-token state-related allocations**: At least 3 tensors of shape `(B, H, d_k, d_v)` are
allocated per token (the decayed state, the outer-product intermediate, and the updated state).
Each is 524,288 × 4 bytes = ~2 MB for Qwen3.5-4B per layer.

### Location: `tensor_parallel_shard.py` — conv state allocation

```python
conv_state = self._linear_attn_states.get(conv_state_key)
if conv_state is None:
    conv_state = torch.zeros(
        batch_size, conv_dim, kernel_size,
        device=hidden_states.device, dtype=hidden_states.dtype,
    )
```

This is a one-time allocation (not per-token) since the state is stored back.
Not a hotspot.

---

## 3. Output Tensor Allocations in Decode Path

### Location: `_gated_deltanet_recurrent_step_impl()` return (line 116)

```python
return output.to(initial_dtype), state
```

The `output.to(initial_dtype)` allocates a new tensor (bf16 copy of the fp32 output).
The `state` is already a new tensor from the arithmetic above.

### Location: `tensor_parallel_shard.py` decode path — post-processing

```python
output_t = output_t.reshape(batch_size, num_v_heads, value_head_dim)  # view, no alloc
rms = output_t.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()   # allocates rms tensor
output_t = output_t * rms * norm_weight                                # allocates 2 intermediates
output_t = output_t.reshape(batch_size, v_dim)                         # view, no alloc
z_t = z_all.squeeze(1)                                                 # view, no alloc
output_t = output_t * F.silu(z_t)                                      # allocates silu + product
output_t = F.linear(output_t, out_proj)                                # allocates output
return output_t.unsqueeze(1)                                           # view, no alloc
```

### Location: `causal_conv1d_update()` — per-token allocations

```python
new_state = torch.roll(conv_state, shifts=-1, dims=-1)  # allocates new tensor
output = (new_state * weight.unsqueeze(0)).sum(dim=-1)  # allocates intermediate + result
output = F.silu(output)                                  # allocates new tensor
```

`torch.roll` always allocates a new tensor. This happens per token per layer.

---

## 4. Summary of Optimization Opportunities

### High Impact (allocation reduction via in-place operations)

- **Use in-place state update operations**: `state.mul_(g)` instead of
  `state = state * g`, and `state.add_(k_expanded * delta_expanded)` instead
  of `state = state + k_expanded * delta_expanded`. This eliminates 2 of the 3
  large intermediate allocations per token per layer.
- **Preallocate output buffers** for the recurrent step. The output shape
  `(B, H, d_v)` is stable across decode tokens — reuse the same tensor.
- The state is already stored in fp32 between tokens (good). The initial
  allocation should also be fp32 to avoid the first-token cast.

### Medium Impact (conv and small tensor optimizations)

- **Replace `torch.roll`** in `causal_conv1d_update` with an in-place shift
  or circular buffer index to avoid per-token allocation.
- **Eliminate the alpha round-trip cast** in the caller: compute alpha in fp32
  and pass it directly without the intermediate `.to(hidden_states.dtype)`.
- **Preallocate the output cast buffer**: instead of `output.to(initial_dtype)`
  creating a new tensor, copy into a preallocated bf16 buffer.

### Lower Impact (already partially addressed)

- The state bf16→fp32 cast is already a no-op on steady-state tokens (state
  stays fp32 between calls). Only the first-token allocation needs fixing
  (allocate in fp32 from the start).

---

## 5. Quantified Per-Token Overhead (Qwen3.5-4B, single layer, steady-state decode)

| Category | Estimated bytes allocated per token |
|----------|-------------------------------------|
| State cast (no-op after first token) | 0 |
| q/k/v/gate/beta fp32 casts (new tensors) | ~65,536 |
| State arithmetic intermediates (3 × 524K × 4B) | 6,291,456 |
| Output cast back to bf16 | ~16,384 |
| Conv roll allocation | ~131,072 |
| RMSNorm + gating intermediates | ~65,536 |
| **Total per token per layer** | **~6.6 MB** |

For 48 GatedDeltaNet layers (Qwen3.5-27B): **~317 MB allocated per token**.

The dominant cost is **state arithmetic intermediates** — three full-size
`(B, H, d_k, d_v)` fp32 tensors created by the decay, outer-product, and
state-update operations. These are the primary optimization target for
in-place operations or preallocated output buffers.

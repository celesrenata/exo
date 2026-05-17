# Chunked GatedDeltaNet Prefill: Mathematical Analysis

## 1. Recurrence Equations

The sequential prefill in `_gated_deltanet_prefill_sequential_impl()` iterates over each token position `t` and calls `_gated_deltanet_recurrent_step_impl()`. The recurrent step implements the following equations:

### State Update

Given inputs at position `t`:
- `S_{t-1}` — recurrent state matrix, shape `(B, H, d_k, d_v)`, in fp32
- `g_t` — log-space decay gate, shape `(B, H)`, where `exp(g_t) ∈ (0, 1]`
- `k_t` — L2-normalized key, shape `(B, H, d_k)`
- `v_t` — value vector, shape `(B, H, d_v)`
- `β_t` — sigmoid update rate, shape `(B, H)`, where `β_t ∈ (0, 1)`
- `q_t` — L2-normalized query, shape `(B, H, d_k)`

The state update proceeds in two stages:

**Stage 1: Decay**
```
S'_t = exp(g_t) · S_{t-1}
```

**Stage 2: Delta rule correction and write**
```
retrieved_t = S'_t^T · k_t          (what the state predicts for key k_t)
δ_t = β_t · (v_t - retrieved_t)     (error-corrected update)
S_t = S'_t + k_t ⊗ δ_t             (outer product write)
```

Expanding `retrieved_t`:
```
S_t = exp(g_t) · S_{t-1} + k_t ⊗ β_t · (v_t - exp(g_t) · S_{t-1}^T · k_t)
```

### Output Computation
```
o_t = (1/√d_k) · S_t^T · q_t
```

The query scaling `1/√d_k` is applied to `q_t` before the dot product in the implementation.

### Compact Form

Substituting and rearranging the state update:
```
S_t = exp(g_t) · S_{t-1} + β_t · k_t ⊗ v_t - β_t · exp(g_t) · k_t ⊗ (S_{t-1}^T · k_t)
```

Which factors as:
```
S_t = exp(g_t) · (I - β_t · k_t · k_t^T) · S_{t-1} + β_t · k_t ⊗ v_t
```

Here `k_t · k_t^T` is the outer product of `k_t` with itself (a `d_k × d_k` matrix), and the expression `(I - β_t · k_t · k_t^T)` acts on `S_{t-1}` from the left (operating on the key dimension).

---

## 2. State Transition Form

The recurrence has an **affine** structure:

```
S_t = A_t · S_{t-1} + B_t
```

where:

### A_t — State Decay and Correction Matrix

```
A_t = exp(g_t) · (I_{d_k} - β_t · k_t · k_t^T)
```

- Shape: conceptually `(d_k, d_k)` acting on the key dimension of `S` (which is `d_k × d_v`)
- `exp(g_t) ∈ (0, 1]` provides exponential decay
- `(I - β_t · k_t · k_t^T)` is a rank-1 perturbation of identity that "erases" the component of the old state along `k_t` proportional to `β_t`
- When `β_t = 1` and `||k_t|| = 1` (L2-normalized), this is a projection that fully replaces the `k_t` direction
- When `β_t = 0`, `A_t = exp(g_t) · I` (pure decay, no correction)

### B_t — New Information Write

```
B_t = β_t · k_t ⊗ v_t
```

- Shape: `(d_k, d_v)` — an outer product (rank-1 matrix)
- Writes the new value `v_t` into the state at the key direction `k_t`, scaled by `β_t`

### Verification

Expanding `A_t · S_{t-1} + B_t`:
```
= exp(g_t) · (I - β_t · k_t · k_t^T) · S_{t-1} + β_t · k_t ⊗ v_t
= exp(g_t) · S_{t-1} - exp(g_t) · β_t · k_t · (k_t^T · S_{t-1}) + β_t · k_t ⊗ v_t
= exp(g_t) · S_{t-1} + β_t · k_t ⊗ (v_t - exp(g_t) · S_{t-1}^T · k_t)
```

This matches the implementation exactly. ✓

---

## 3. Numerical Stability Requirements

### fp32 Required for State Accumulation

The state matrix `S` accumulates information across the entire sequence. In bf16:
- Mantissa is 7 bits (vs 23 bits in fp32)
- Repeated multiply-accumulate operations lose precision exponentially
- After ~100 tokens, accumulated rounding error dominates the state
- The delta rule correction `(v_t - S^T k_t)` amplifies errors: if `S` is imprecise, the correction overshoots or undershoots

The implementation casts all inputs to fp32 at the start of `_gated_deltanet_recurrent_step_impl()` and keeps the state in fp32 throughout. The output is cast back to the initial dtype (bf16) only at the end.

### Gate in Log-Space

The gate `g_t` is stored in log-space (negative values). The actual decay factor is `exp(g_t)`:
- Since `g_t ≤ 0`, we have `exp(g_t) ∈ (0, 1]`
- This provides natural exponential decay without risk of values exceeding 1
- Log-space representation avoids underflow for very small decay factors
- The cumulative decay over a chunk is `exp(Σ g_t)` = product of individual decays, computed stably via cumulative sum in log-space

### Beta is Sigmoid-Bounded

The update rate `β_t = sigmoid(β_raw_t) ∈ (0, 1)`:
- Bounds the magnitude of the state update
- Prevents the rank-1 correction from overshooting
- When `β_t` is small, the state changes slowly (conservative updates)
- When `β_t` is close to 1, the state aggressively replaces the `k_t` direction

### Delta Rule Correction Magnitude

The correction term `(v_t - S'^T_t · k_t)` can have large magnitude when:
- The state `S` is poorly conditioned (accumulated errors)
- The state has not seen key `k_t` before (retrieval returns near-zero, correction ≈ `v_t`)
- The state has stale information for `k_t` (retrieval returns wrong value)

Mitigations in the current design:
- fp32 state prevents error accumulation
- L2-normalized keys bound `||k_t|| = 1`, so `S^T k_t` is bounded by the spectral norm of `S`
- `β_t ∈ (0, 1)` scales the correction
- The exponential decay `exp(g_t)` continuously shrinks old state, preventing unbounded growth

### Chunked Computation Stability Considerations

For the chunked algorithm:
- Cumulative log-decay `G[i] = Σ_{j=0}^{i} g_j` must be computed in fp32 to avoid log-space underflow
- The decay matrix `L[i,j] = exp(G[i] - G[j])` is numerically stable because `G[i] - G[j] ≤ 0` for `i ≥ j`
- Transform composition must use fp32 to preserve the affine structure across chunks
- The final state materialization must match the sequential fp32 result

---

## 4. Chunk-Parallel Opportunity

### Why the Affine Form Enables Parallelism

The affine recurrence `S_t = A_t · S_{t-1} + B_t` is **associatively composable**. Given two consecutive transforms:

```
S_1 = A_1 · S_0 + B_1
S_2 = A_2 · S_1 + B_2
```

Substituting:
```
S_2 = A_2 · (A_1 · S_0 + B_1) + B_2
    = (A_2 · A_1) · S_0 + (A_2 · B_1 + B_2)
```

So the composition of two affine transforms `(A_2, B_2) ∘ (A_1, B_1)` is:
```
A_{composed} = A_2 · A_1
B_{composed} = A_2 · B_1 + B_2
```

This composition is **associative**: `(T_3 ∘ T_2) ∘ T_1 = T_3 ∘ (T_2 ∘ T_1)`.

### Chunk-Parallel Algorithm

Given a sequence of length `T` and chunk size `C`:

**Phase 1: Intra-chunk computation (parallel across chunks)**

For each chunk `c` containing tokens `[c·C, (c+1)·C - 1]`:
1. Compute all `A_t` and `B_t` for tokens in the chunk
2. Compose the per-token transforms within the chunk into a single chunk transform:
   ```
   (A_chunk, B_chunk) = (A_{cC+C-1}, B_{cC+C-1}) ∘ ... ∘ (A_{cC+1}, B_{cC+1}) ∘ (A_{cC}, B_{cC})
   ```
3. Compute intra-chunk outputs (tokens within the chunk attending to each other)

**Phase 2: Inter-chunk state propagation (sequential across T/C chunks)**

For each chunk `c` in order:
1. Receive the final state from the previous chunk: `S_{prev}`
2. Apply the chunk's composed transform: `S_{final} = A_chunk · S_{prev} + B_chunk`
3. Propagate `S_{final}` to the next chunk

**Phase 3: Output materialization (parallel across chunks)**

For each chunk `c`:
1. Given the incoming state `S_{prev}` for this chunk, compute the inter-chunk contribution to each token's output
2. Combine with the intra-chunk output from Phase 1

### Complexity Comparison

| Approach | Sequential Steps | Parallel Work per Step |
|----------|-----------------|----------------------|
| Sequential | O(T) | O(d_k · d_v) per token |
| Chunked | O(T/C) | O(C² · d_k + C · d_k · d_v) per chunk |

For `T = 2048`, `C = 64`: sequential needs 2048 steps, chunked needs 32 sequential steps with 64× more parallel work per step. On hardware with sufficient parallelism (Intel Arc iGPUs with many execution units), the chunked approach is faster.

### Practical Considerations for Implementation

1. **A_t is rank-1 structured**: `A_t = exp(g_t) · (I - β_t · k_t · k_t^T)`. Storing the full `d_k × d_k` matrix is wasteful. Instead, represent `A_t` implicitly via `(g_t, β_t, k_t)` and use the Woodbury identity or WY decomposition for efficient composition.

2. **Chunk transform composition**: Rather than materializing `d_k × d_k` matrices, use the WY representation where the composed transform is stored as `(decay_product, W_matrix, Y_matrix)` enabling O(C · d_k²) composition instead of O(C · d_k³).

3. **Intra-chunk attention**: Within a chunk, the decay-weighted causal attention matrix `L[i,j] = exp(G[i] - G[j])` for `i ≥ j` enables vectorized computation of intra-chunk outputs using matrix multiplication.

4. **Partial chunks**: The final chunk may have fewer than `C` tokens. Handle by padding or by processing the remainder with the sequential algorithm.

5. **Memory**: The chunked algorithm requires storing intermediate chunk transforms and intra-chunk attention matrices. For `C = 64`, the attention matrix is `64 × 64` per head — small enough to fit in registers/L1 cache.

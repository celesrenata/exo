# Qwen3 Linear Attention Cache Bugfix Design

## Overview

Qwen3.5-4B and Qwen3.6-27B produce whitespace/garbage during tensor-parallel inference because the native HuggingFace `Qwen3_5GatedDeltaNet` layers cannot execute correctly — their `nn.Module` parameters remain on CPU while the forward pass receives XPU tensors. The fix moves the extracted native layers to the target device after weight sharding, ensuring the Cache-backed recurrent state persists across autoregressive decode steps.

## Glossary

- **Bug_Condition (C)**: A forward call through a native linear attention layer where the layer's parameters are on a different device than the input hidden_states, causing device mismatch or silent failure
- **Property (P)**: After the fix, native linear attention layers execute on the correct device with a properly-initialized Cache, and recurrent state persists across decode steps producing coherent output
- **Preservation**: Full attention layers (25% of Qwen3.5/3.6), non-hybrid models (Phi-4, Qwen2.5), MLP sharding, and the custom fallback implementation remain unchanged
- **`Qwen3_5GatedDeltaNet`**: The native HuggingFace linear attention module in `transformers-5.7.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py` implementing Gated DeltaNet with conv1d + recurrent state
- **`LinearAttentionLayer`**: Cache layer class in `transformers.cache_utils` that stores `conv_states` and `recurrent_states` tensors, with lazy initialization on first `update_conv_state()` call
- **`Cache`**: Container class in `transformers.cache_utils` holding a list of `CacheLayerMixin` / `LinearAttentionCacheLayerMixin` instances indexed by layer position
- **`has_previous_state`**: Boolean attribute on `LinearAttentionLayer` that starts `False` and becomes `True` after the first `update_conv_state()` call — controls whether the native layer reads from cache or initializes fresh

## Bug Details

### Bug Condition

The bug manifests when `_forward_linear_attn_layer()` delegates to a native `Qwen3_5GatedDeltaNet` module whose `nn.Linear` parameters (in_proj_qkv, in_proj_a, in_proj_b, in_proj_z, conv1d, out_proj) reside on CPU while the input `hidden_states` tensor is on XPU. The native layer's forward pass either raises a device mismatch error (caught silently) or produces garbage output because the computation mixes CPU and XPU tensors.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type ForwardCall
  OUTPUT: boolean
  
  RETURN input.layer_idx IN native_linear_attn_layers
     AND native_linear_attn_layers[input.layer_idx].parameters_device != input.hidden_states.device
     AND input.decode_step >= 0
END FUNCTION
```

### Examples

- **Prefill (seq_len > 1)**: `native_layer(hidden_states_xpu, cache_params=cache, attention_mask=mask)` → `self.in_proj_qkv(hidden_states_xpu)` fails because `in_proj_qkv.weight` is on CPU. Expected: projection computed on XPU, conv_state and recurrent_state initialized in cache.
- **Decode step 1 (seq_len = 1)**: Same device mismatch. Even if prefill somehow succeeded, the cache's `has_previous_state` would be True but the conv1d update would fail on device mismatch.
- **Decode step N (seq_len = 1)**: Recurrent state from step N-1 should be read from cache and updated. Instead, device mismatch prevents any computation.
- **Edge case — fallback path**: When `layer_idx NOT IN _native_linear_attn_layers`, the custom implementation uses `self._get_weight()` which returns XPU tensors from `sharded_state_dict`. This path works correctly (no bug).

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Full attention layers (self_attn with QKV → RoPE → SDPA → o_proj) continue using the existing KV cache mechanism with row-parallel all-reduce
- Models without linear attention (Phi-4, Qwen2.5-7B-Instruct) continue producing coherent output via the full attention path
- MLP gate/up/down projections continue to be column/row-parallel sharded with all-reduce
- The custom fallback implementation in `_forward_linear_attn_layer()` (when native layers are unavailable) continues maintaining its own `_linear_attn_states` dict
- Linear attention weights remain redundant (not sharded) on all ranks per `_LINEAR_ATTN_PATTERN`
- The `_all_reduce` CPU-staging mechanism for Gloo backend remains unchanged

**Scope:**
All inputs that do NOT flow through native linear attention layers are completely unaffected by this fix. This includes:
- All forward calls for Phi-4 and Qwen2.5-7B-Instruct
- Full attention layers within Qwen3.5/3.6 (every 4th layer)
- MLP blocks in all layers
- Embedding lookup and final lm_head projection

## Hypothesized Root Cause

Based on code analysis of `tensor_parallel_shard.py` and `transformers-5.7.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py`:

1. **Native layers never moved to target device (PRIMARY)**: `_extract_native_linear_attn_layers()` saves references to `nn.Module` instances at line 237 while the model is on CPU. After `shard_weights()` moves tensor slices to XPU via `.to(self.device)`, the native layer modules' parameters (`self.in_proj_qkv.weight`, `self.conv1d.weight`, `self.out_proj.weight`, etc.) remain on CPU. When `_forward_linear_attn_layer()` passes XPU `hidden_states` to the native layer, `self.in_proj_qkv(hidden_states)` triggers a device mismatch between CPU weight and XPU input.

2. **Cache created with wrong layer type for hybrid models (SECONDARY)**: The current `_create_native_cache()` creates `LinearAttentionLayer()` for ALL layers (both linear and full attention). While this doesn't cause a crash (full attention layers never access their cache entry), it diverges from the canonical `DynamicCache(config=model.config)` which creates `LinearAttentionAndFullAttentionLayer()` for hybrid layers or `DynamicLayer()` for full attention layers. Using `LinearAttentionLayer()` for full attention layers means `has_previous_state(layer_idx)` would return the wrong type for full attention layers if ever queried.

3. **No device placement for Cache tensors**: When `LinearAttentionLayer.lazy_initialization()` is called, it creates `torch.zeros_like(conv_states)` — inheriting the device from the first conv_states tensor passed in. If the native layer's computation happens on CPU (due to root cause 1), the cache tensors will also be on CPU. After fixing root cause 1, the cache tensors will correctly be on XPU.

4. **`attention_mask` handling is correct**: The current code passes `attention_mask=None` which matches the behavior of `_update_linear_attn_mask()` in the HF model — it sets the mask to None when `has_previous_state()` is True or when all mask values are 1. For decode steps (where the cache has state), None is correct.

## Correctness Properties

Property 1: Bug Condition - Native Layer Device Alignment

_For any_ forward call where a native linear attention layer is invoked, the fixed code SHALL ensure all native layer parameters reside on the same device as the input hidden_states, and the layer SHALL produce a non-zero output tensor of shape `(batch, seq_len, hidden_size)` with the recurrent state updated in the Cache.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4**

Property 2: Preservation - Non-Linear-Attention Path Unchanged

_For any_ forward call that does NOT flow through a native linear attention layer (full attention layers, non-hybrid models, MLP blocks), the fixed code SHALL produce exactly the same output as the original code, preserving all existing tensor-parallel sharding, all-reduce synchronization, and KV cache behavior.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `src/exo/worker/engines/pytorch_xpu/tensor_parallel_shard.py`

**Function**: `__init__` (after `shard_weights`)

**Specific Changes**:

1. **Move native layers to target device**: After `shard_weights()` completes, iterate over `self._native_linear_attn_layers` and call `.to(self.device)` on each native module. This moves all `nn.Linear` weights, `nn.Conv1d` weights, `nn.Parameter` tensors (A_log, dt_bias), and norm weights to XPU.

   ```python
   # After shard_weights(state_dict)
   for layer_idx, native_layer in self._native_linear_attn_layers.items():
       native_layer.to(self.device)
   ```

2. **Replace `_create_native_cache()` with proper `DynamicCache`-style initialization**: Instead of creating a bare `Cache(layers=...)` with `LinearAttentionLayer()` for all layers, create the cache using the same logic as `DynamicCache(config=...)` — `LinearAttentionLayer()` for linear attention layers and `DynamicLayer()` for full attention layers. This ensures correct behavior if `has_previous_state()` or `get_seq_length()` is ever called at the Cache level.

   ```python
   def _create_native_cache(self) -> Any:
       from transformers.cache_utils import Cache, LinearAttentionLayer, DynamicLayer
       
       num_layers = self._detect_num_layers()
       cache_layers = []
       for idx in range(num_layers):
           if self._layer_types[idx] == "linear_attention":
               cache_layers.append(LinearAttentionLayer())
           else:
               cache_layers.append(DynamicLayer())
       return Cache(layers=cache_layers)
   ```

3. **Move cache creation after `_layer_types` detection**: The current code creates the cache lazily on first use (`if not hasattr(self, '_native_cache')`). This is fine, but the cache creation depends on `self._layer_types` which is set during `__init__`. Ensure the lazy creation happens after `_detect_layer_types()` has run (it already does since `_forward_linear_attn_layer` is only called during `forward()` which is after `__init__`).

4. **Set native layers to eval mode**: After moving to device, call `.eval()` on each native layer to disable dropout and ensure deterministic inference behavior.

   ```python
   for layer_idx, native_layer in self._native_linear_attn_layers.items():
       native_layer.to(self.device).eval()
   ```

5. **Remove debug print statements**: Clean up the `import sys` and `print(..., file=sys.stderr)` debug statements in `_extract_native_linear_attn_layers()` and `_create_native_cache()` that were added during development.

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code, then verify the fix works correctly and preserves existing behavior.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Write tests that create a `TensorParallelShard` with a mock Qwen3.5-style model containing native linear attention layers, then verify that the native layers' parameters are on the wrong device. Run these tests on the UNFIXED code to observe failures.

**Test Cases**:
1. **Device Mismatch Test**: Create TPS with device="xpu:0" (or "cpu" for unit test), verify native layer parameters are NOT on the target device (will demonstrate bug on unfixed code)
2. **Forward Pass Failure Test**: Call `_forward_linear_attn_layer()` with hidden_states on target device, observe RuntimeError or garbage output (will fail on unfixed code)
3. **Cache State Persistence Test**: Execute two sequential decode steps through a native layer, verify `has_previous_state` is True after first step and recurrent_state is non-zero after second step (will fail on unfixed code)
4. **Multi-Step Coherence Test**: Execute 5+ decode steps, verify output is not all-zeros or whitespace tokens (will fail on unfixed code)

**Expected Counterexamples**:
- Native layer parameters on CPU while hidden_states on XPU → RuntimeError: Expected all tensors to be on the same device
- If error is caught silently: output tensor is all zeros or NaN
- Possible causes: `.to(device)` never called on native modules after extraction

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function produces the expected behavior.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  result := forward_linear_attn_layer_fixed(input)
  ASSERT result.device == input.hidden_states.device
  ASSERT result.shape == (batch, seq_len, hidden_size)
  ASSERT NOT torch.all(result == 0)
  ASSERT cache.has_previous_state(input.layer_idx) == True
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT forward_original(input) == forward_fixed(input)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many random tensor shapes, sequence lengths, and layer configurations
- It catches edge cases in the full attention path that manual tests miss
- It provides strong guarantees that MLP sharding and all-reduce behavior is unchanged

**Test Plan**: Observe behavior on UNFIXED code first for full attention layers and non-hybrid models, then write property-based tests capturing that behavior.

**Test Cases**:
1. **Full Attention Preservation**: Verify that full attention layers (QKV → RoPE → SDPA → o_proj → all_reduce) produce identical output before and after the fix
2. **MLP Sharding Preservation**: Verify that MLP gate/up (column-parallel) and down (row-parallel) produce identical output
3. **Non-Hybrid Model Preservation**: Verify that Phi-4 and Qwen2.5 model architectures produce identical output (no linear attention layers exist)
4. **Custom Fallback Preservation**: Verify that when native layers are NOT available, the custom `_linear_attn_states` dict path produces identical output

### Unit Tests

- Test that native layers are on the correct device after `__init__` completes
- Test that `_create_native_cache()` creates correct layer types (LinearAttentionLayer for linear, DynamicLayer for full)
- Test that `has_previous_state()` returns False before first forward, True after
- Test that conv_states and recurrent_states have correct shapes after lazy initialization
- Test single-token decode path reads from cache correctly
- Test multi-token prefill path initializes cache correctly

### Property-Based Tests

- Generate random batch sizes, sequence lengths, and hidden dimensions; verify native layer forward produces non-zero output on correct device
- Generate random sequences of prefill + N decode steps; verify recurrent_state is updated monotonically (never resets to zero after initialization)
- Generate random inputs for full attention layers; verify output is identical with and without the native layer device fix applied

### Integration Tests

- End-to-end generation test on gremlin-1: load Qwen3.5-4B, generate 50 tokens, verify output is coherent English text (not whitespace)
- Multi-step decode test: verify that token N's output depends on tokens 0..N-1 (recurrent state accumulates)
- Hybrid layer interleaving test: verify that alternating linear_attention and full_attention layers produce correct output when the linear layers use native modules and full layers use KV cache

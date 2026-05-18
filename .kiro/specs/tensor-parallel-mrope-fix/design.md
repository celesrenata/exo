# Tensor Parallel MRoPE Fix — Bugfix Design

## Overview

The streaming loader path creates `TensorParallelShard(model=sharded_state_dict, pre_sharded=True)` where `model` is a plain dict. The `_extract_native_rotary_emb()` method only searches HuggingFace model objects (via `getattr` traversal), so it finds nothing and sets `_native_rotary_emb = None`. The manual 1D RoPE fallback then runs for all 16 full attention layers in Qwen3.5-27B, producing incorrect positional encodings that corrupt the model's output into degenerate whitespace tokens.

The fix instantiates `Qwen3_5RotaryEmbedding` from the model config in `streaming_loader.py` (following the existing `Qwen3_5GatedDeltaNet` pattern), returns it alongside the sharded state dict and native layers, and passes it to `TensorParallelShard` via a new `native_rotary_emb` constructor parameter. A secondary fix derives `q_proj.bias` shard size from actual tensor shape (matching the weight sharding logic) instead of hardcoding `heads_per_rank * head_dim`. A fail-fast validation raises `RuntimeError` when an MRoPE model lacks native rotary embedding.

## Glossary

- **Bug_Condition (C)**: The condition that triggers the bug — streaming-loaded MRoPE model has `_native_rotary_emb = None`
- **Property (P)**: The desired behavior — MRoPE models get a native `Qwen3_5RotaryEmbedding` instance attached to the shard, producing correct 3D positional encodings
- **Preservation**: Non-streaming path extraction, standard 1D RoPE fallback for non-MRoPE models, native linear attention layer loading, and existing weight sharding must remain unchanged
- **`_extract_native_rotary_emb()`**: Method on `TensorParallelShard` that traverses a HuggingFace model object to find `model.model.rotary_emb`
- **`_apply_rotary_pos_emb()`**: Method on `TensorParallelShard` that applies RoPE to Q/K tensors — delegates to native rotary_emb when available, falls back to manual 1D RoPE otherwise
- **MRoPE**: Multi-Resolution Rotary Position Embedding — 3D positional encoding with `mrope_section=[11, 11, 10]` for text/height/width dimensions
- **`streaming_loader.py`**: Module that loads safetensors files directly, sharding each tensor as it's read without materializing the full model
- **`model_loader.py`**: Orchestrator that decides between streaming vs full HF model loading and constructs `TensorParallelShard`
- **`Qwen3_5RotaryEmbedding`**: HuggingFace's native rotary embedding class that handles MRoPE section splitting, partial rotation, and interleaved layout

## Bug Details

### Bug Condition

The bug manifests when a Qwen3.5 model with MRoPE is loaded via the streaming path. The `TensorParallelShard.__init__()` receives a dict (not a model object), so `_extract_native_rotary_emb()` is never called. The `_native_rotary_emb` field stays `None`, and `_apply_rotary_pos_emb()` falls back to manual 1D RoPE which is incorrect for 3D MRoPE.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type ModelLoadContext
  OUTPUT: boolean

  RETURN input.model_config.rope_scaling.type = "mrope"
         AND input.load_path = "streaming"
         AND input.tensor_parallel_shard._native_rotary_emb = None
END FUNCTION
```

### Examples

- **Qwen3.5-27B on 4 gremlin nodes (streaming path)**: Expected coherent text output. Actual: degenerate logits `top5_ids=[220, 198, 13, 16, 271]`, constant hidden state norm 113.5, whitespace-only output.
- **Qwen3.5-27B q_proj.bias sharding**: Expected shard size `heads_per_rank * head_dim * 2 = 6 * 256 * 2 = 3072`. Actual: shard size `heads_per_rank * head_dim = 6 * 256 = 1536`, corrupting gate bias values.
- **Qwen3.5-4B on pipeline parallel (non-streaming path)**: Works correctly because `_extract_native_rotary_emb()` finds `model.model.rotary_emb` on the HF model object.
- **Llama-3 on streaming path (no MRoPE)**: Works correctly because manual 1D RoPE fallback is correct for standard RoPE models.

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Non-streaming path (full HF model object) continues to extract native rotary_emb via `_extract_native_rotary_emb()` traversal
- Standard 1D RoPE models (Llama, Mistral, Phi) continue to use the manual fallback correctly
- Native linear attention layers (`Qwen3_5GatedDeltaNet`) continue to be instantiated and attached via the streaming loader
- `q_proj.weight` sharding for doubled Q projections continues to use shape-based detection (`actual_out_dim == expected_standard * 2`)
- MLP weight sharding for hybrid models with varying intermediate sizes continues to use actual tensor dimensions
- Pipeline parallel inference on Qwen3.5-4B continues to produce coherent output

**Scope:**
All inputs that do NOT involve MRoPE models loaded via the streaming path are completely unaffected by this fix. This includes:
- All non-streaming (full HF model) loads
- All standard RoPE models regardless of load path
- All MLP and linear attention weight sharding
- All non-attention forward pass logic

## Hypothesized Root Cause

Based on the bug description, the root causes are:

1. **Missing rotary_emb instantiation in streaming path**: `streaming_loader.py` instantiates `Qwen3_5GatedDeltaNet` modules from config for linear attention layers, but has no equivalent logic for `Qwen3_5RotaryEmbedding`. The streaming path returns `(sharded_state_dict, native_layers, tokenizer)` with no rotary embedding.

2. **`TensorParallelShard.__init__` only extracts rotary_emb from model objects**: The `isinstance(model, dict)` branch skips `_extract_native_rotary_emb()` entirely. There is no constructor parameter to inject a pre-built rotary embedding.

3. **`model_loader.py` does not pass rotary_emb to the shard**: After `load_sharded_from_safetensors()` returns, `model_loader.py` creates `TensorParallelShard(model=sharded_state_dict, ...)` and attaches `_native_linear_attn_layers` post-construction, but has no equivalent for rotary_emb.

4. **`q_proj.bias` sharding ignores doubled layout**: The `_shard_parameter` method checks actual tensor shape for `q_proj.weight` to detect doubled Q projections, but the `q_proj.bias` branch hardcodes `heads_per_rank * head_dim` without checking the actual bias tensor size.

5. **No fail-fast validation**: When `_native_rotary_emb` is `None` and the model config indicates MRoPE, the system logs a warning but proceeds with incorrect 1D RoPE instead of raising an error.

## Correctness Properties

Property 1: Bug Condition - MRoPE Models Get Native Rotary Embedding via Streaming Path

_For any_ model load context where the model config has `rope_scaling.type = "mrope"` and the load path is "streaming", the fixed system SHALL instantiate `Qwen3_5RotaryEmbedding` from the model config and attach it to the `TensorParallelShard` such that `shard._native_rotary_emb` is not None and is a valid rotary embedding module.

**Validates: Requirements 2.1, 2.2, 2.3**

Property 2: Preservation - Non-MRoPE Models Use Manual RoPE Fallback

_For any_ model load context where the model config does NOT have `rope_scaling.type = "mrope"` (standard 1D RoPE models), the fixed system SHALL produce the same result as the original system, preserving the manual partial-RoPE fallback behavior and all existing weight sharding logic.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

Property 3: Bug Condition - q_proj.bias Sharding Matches Weight Sharding

_For any_ model with doubled Q projections (Q + gate) where `q_proj.bias` exists, the fixed system SHALL derive the bias shard size from the actual tensor shape (matching the `q_proj.weight` logic) instead of hardcoding `heads_per_rank * head_dim`.

**Validates: Requirements 2.5**

Property 4: Bug Condition - Fail-Fast on Missing MRoPE Rotary Embedding

_For any_ model load context where the model config has `rope_scaling.type = "mrope"` but native rotary embedding cannot be instantiated, the fixed system SHALL raise a `RuntimeError` with a descriptive message instead of silently falling back to incorrect 1D RoPE.

**Validates: Requirements 2.4**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `src/exo/worker/engines/pytorch_xpu/streaming_loader.py`

**Function**: `load_sharded_from_safetensors` (return value) + new `_load_native_rotary_emb` function

**Specific Changes**:
1. **Add `_load_native_rotary_emb()` function**: Following the exact pattern of `_load_native_linear_attn_layers()` — load `AutoConfig`, check for `rope_scaling.type == "mrope"`, import `Qwen3_5RotaryEmbedding`, instantiate from `text_config`, move to device. Return `None` for non-MRoPE models.

2. **Update `load_sharded_from_safetensors()` return type**: Change from `tuple[dict, dict, Any]` to `tuple[dict, dict, Any, Any]` — adding the native rotary embedding as the 4th element (or `None` for non-MRoPE models).

3. **Call `_load_native_rotary_emb()` before returning**: After `_load_native_linear_attn_layers()`, call the new function and include the result in the return tuple.

---

**File**: `src/exo/worker/engines/pytorch_xpu/tensor_parallel_shard.py`

**Function**: `TensorParallelShard.__init__` + `_shard_parameter`

**Specific Changes**:
4. **Add `native_rotary_emb` constructor parameter**: Add an optional keyword argument `native_rotary_emb: Any = None` to `__init__`. When provided, assign it to `self._native_rotary_emb` and move to device. This runs BEFORE the `isinstance(model, dict)` branch so it works for both paths.

5. **Fix `q_proj.bias` sharding**: In `_shard_parameter`, change the `q_proj.bias` branch to derive shard size from actual tensor shape (matching the `q_proj.weight` logic): check if `param_tensor.shape[0] == heads_per_rank * head_dim * 2 * world_size`, and if so use `heads_per_rank * head_dim * 2` as shard size.

6. **Add fail-fast validation**: After `__init__` completes, if `self._native_rotary_emb is None` and the config indicates MRoPE (`self.config.mrope_interleaved` is True or `rope_scaling` has type "mrope"), raise `RuntimeError` with a descriptive message.

---

**File**: `src/exo/worker/engines/pytorch_xpu/model_loader.py`

**Function**: `_load_model_streaming`

**Specific Changes**:
7. **Unpack 4th return value**: Update the destructuring of `load_sharded_from_safetensors()` to capture the native rotary embedding: `sharded_state_dict, native_layers, tokenizer, native_rotary_emb = ...`

8. **Pass `native_rotary_emb` to `TensorParallelShard` constructor**: Add `native_rotary_emb=native_rotary_emb` to the constructor call.

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code, then verify the fix works correctly and preserves existing behavior.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Write tests that mock the streaming loader path for a Qwen3.5 MRoPE config and verify that `TensorParallelShard._native_rotary_emb` is `None` after construction. Run these tests on the UNFIXED code to observe the failure mode.

**Test Cases**:
1. **Streaming path MRoPE test**: Create a `TensorParallelShard` with `model={}` (dict) and `pre_sharded=True` for a config with `mrope_interleaved=True` — verify `_native_rotary_emb` is None (will demonstrate bug on unfixed code)
2. **q_proj.bias doubled test**: Create a bias tensor of size `num_heads * head_dim * 2` and shard it — verify the shard size is incorrect at `heads_per_rank * head_dim` (will demonstrate bug on unfixed code)
3. **Silent fallback test**: Verify that no RuntimeError is raised when MRoPE model has no native rotary_emb (will demonstrate bug on unfixed code — silent corruption)

**Expected Counterexamples**:
- `_native_rotary_emb` is None for streaming-loaded MRoPE models
- `q_proj.bias` shard is half the correct size for doubled Q projections
- No error raised when MRoPE model silently falls back to 1D RoPE

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function produces the expected behavior.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  native_rotary_emb := _load_native_rotary_emb(model_path, model_id, device)
  shard := TensorParallelShard(model=state_dict, config=tp_config, device=device,
                               pre_sharded=True, native_rotary_emb=native_rotary_emb)
  ASSERT shard._native_rotary_emb IS NOT None
  ASSERT type(shard._native_rotary_emb).__name__ CONTAINS "RotaryEmbedding"
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT TensorParallelShard_original(input) = TensorParallelShard_fixed(input)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many random tensor shapes and configs to verify sharding correctness
- It catches edge cases in bias sharding that manual unit tests miss
- It provides strong guarantees that non-MRoPE models are unaffected

**Test Plan**: Observe behavior on UNFIXED code first for non-MRoPE configs and standard sharding, then write property-based tests capturing that behavior.

**Test Cases**:
1. **Non-MRoPE sharding preservation**: Generate random TPShardConfigs without MRoPE and verify all weight sharding produces identical results before and after fix
2. **HF model path preservation**: Verify that `_extract_native_rotary_emb()` still works when model is a HF model object (not a dict)
3. **Linear attention layer preservation**: Verify that `_load_native_linear_attn_layers()` continues to work correctly after streaming_loader changes
4. **MLP sharding preservation**: Verify gate_proj, up_proj, down_proj sharding is unchanged for hybrid models with varying intermediate sizes

### Unit Tests

- Test `_load_native_rotary_emb()` returns a valid rotary embedding for Qwen3.5 config
- Test `_load_native_rotary_emb()` returns None for non-MRoPE configs (Llama, Mistral)
- Test `TensorParallelShard.__init__` accepts and stores `native_rotary_emb` parameter
- Test `q_proj.bias` sharding uses actual tensor shape for doubled Q projections
- Test fail-fast RuntimeError when MRoPE config has no native rotary_emb
- Test that `load_sharded_from_safetensors` returns 4-tuple with rotary_emb

### Property-Based Tests

- Generate random `TPShardConfig` instances (varying heads, head_dim, world_size) and verify `q_proj.bias` shard size always matches `q_proj.weight` shard size for the same config
- Generate random non-MRoPE configs and verify manual RoPE fallback produces identical cos/sin tensors before and after fix
- Generate random tensor shapes for all shardable parameters and verify sharding is deterministic and produces correct output dimensions

### Integration Tests

- End-to-end streaming load of Qwen3.5-27B config with mocked safetensors — verify `_native_rotary_emb` is attached and is the correct type
- End-to-end streaming load of Llama config — verify `_native_rotary_emb` is None and manual fallback is used
- Forward pass through `_apply_rotary_pos_emb()` with native rotary_emb — verify output shape and dtype are correct
- Full `_load_model_streaming` flow with mocked model path — verify shard has both native_linear_attn_layers and native_rotary_emb

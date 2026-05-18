# Implementation Plan

## Overview

Fix tensor parallel inference for Qwen3.5-27B by adding native MRoPE rotary embedding support to the streaming loader path, fixing q_proj.bias sharding for doubled Q projections, and adding fail-fast validation for MRoPE models.

## Tasks

- [x] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - Streaming MRoPE Models Get No Native Rotary Embedding
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the bug exists
  - **Scoped PBT Approach**: Scope the property to concrete failing cases: MRoPE configs (mrope_interleaved=True, rope_scaling.type="mrope") loaded via streaming path (model=dict, pre_sharded=True)
  - Test that `TensorParallelShard(model={}, config=mrope_config, device="cpu", pre_sharded=True)` results in `shard._native_rotary_emb is None` (from Bug Condition in design: input.model_config.rope_scaling.type = "mrope" AND input.load_path = "streaming" AND input.tensor_parallel_shard._native_rotary_emb = None)
  - Test that `q_proj.bias` sharding for a doubled Q projection (bias tensor size = num_heads * head_dim * 2) produces shard size of `heads_per_rank * head_dim` instead of the correct `heads_per_rank * head_dim * 2` (demonstrates the bias sharding bug)
  - Test that no RuntimeError is raised when MRoPE config has `_native_rotary_emb = None` (demonstrates silent fallback bug)
  - The test assertions should match the Expected Behavior Properties from design: shard._native_rotary_emb IS NOT None, type contains "RotaryEmbedding", q_proj.bias shard matches weight shard size
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (this is correct - it proves the bug exists)
  - Document counterexamples found: `_native_rotary_emb` is None for streaming MRoPE models, q_proj.bias shard is half correct size, no RuntimeError raised
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 1.2, 1.4, 1.5_

- [x] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Non-MRoPE Sharding and HF Model Path Unchanged
  - **IMPORTANT**: Follow observation-first methodology
  - Observe: `TensorParallelShard` with a HF model object (has `state_dict()` method) calls `_extract_native_rotary_emb()` and finds `model.model.rotary_emb` on unfixed code
  - Observe: Non-MRoPE configs (no rope_scaling or rope_scaling.type != "mrope") use manual 1D RoPE fallback correctly on unfixed code
  - Observe: `q_proj.weight` sharding for doubled Q projections uses shape-based detection (`actual_out_dim == expected_standard * 2`) and produces correct shard size on unfixed code
  - Observe: MLP weight sharding (gate_proj, up_proj, down_proj) uses actual tensor dimensions for hybrid models on unfixed code
  - Observe: `k_proj.weight`, `v_proj.weight`, `o_proj.weight` sharding produces correct shard sizes on unfixed code
  - Write property-based test: for all TPShardConfig instances WITHOUT MRoPE (mrope_interleaved=False, no mrope rope_scaling), `_shard_parameter` produces shards with correct output dimensions matching `heads_per_rank * head_dim` for Q, `kv_heads_per_rank * head_dim` for K/V, and `intermediate_per_rank` for MLP (from Preservation Requirements in design)
  - Write property-based test: for all random tensor shapes for shardable parameters, sharding is deterministic and shard dimensions equal `full_dim / world_size`
  - Verify tests pass on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [x] 3. Fix for streaming MRoPE native rotary embedding and q_proj.bias sharding

  - [x] 3.1 Add `_load_native_rotary_emb()` to streaming_loader.py
    - Follow the existing `_load_native_linear_attn_layers()` pattern
    - Load `AutoConfig` from model_id or model_path
    - Check for `rope_scaling.type == "mrope"` in text_config
    - Import `Qwen3_5RotaryEmbedding` from `transformers.models.qwen3_5.modeling_qwen3_5`
    - Instantiate from text_config, move to device with dtype=torch.bfloat16
    - Return `None` for non-MRoPE models
    - _Bug_Condition: isBugCondition(input) where input.model_config.rope_scaling.type = "mrope" AND input.load_path = "streaming"_
    - _Expected_Behavior: native_rotary_emb IS NOT None AND type contains "RotaryEmbedding"_
    - _Preservation: Non-MRoPE models return None, no change to existing _load_native_linear_attn_layers_
    - _Requirements: 2.1_

  - [x] 3.2 Update `load_sharded_from_safetensors()` return type and call
    - Change return type from `tuple[dict, dict, Any]` to `tuple[dict, dict, Any, Any]`
    - Call `_load_native_rotary_emb(model_path, model_id, device)` after `_load_native_linear_attn_layers()`
    - Include native_rotary_emb as 4th element in return tuple (or None for non-MRoPE)
    - _Bug_Condition: streaming path returns no rotary embedding currently_
    - _Expected_Behavior: 4-tuple returned with rotary_emb for MRoPE models_
    - _Preservation: Existing 3 return values unchanged in position and semantics_
    - _Requirements: 2.1_

  - [x] 3.3 Add `native_rotary_emb` constructor parameter to TensorParallelShard
    - Add optional keyword argument `native_rotary_emb: Any = None` to `__init__`
    - When provided and not None, assign to `self._native_rotary_emb` and move to device BEFORE the `isinstance(model, dict)` branch
    - This allows the streaming path to inject the rotary embedding without needing a HF model object
    - _Bug_Condition: isinstance(model, dict) branch skips _extract_native_rotary_emb()_
    - _Expected_Behavior: native_rotary_emb parameter is stored and moved to device_
    - _Preservation: When native_rotary_emb=None (default), behavior is unchanged_
    - _Requirements: 2.1, 2.2_

  - [x] 3.4 Fix `q_proj.bias` sharding for doubled Q projections
    - In `_shard_parameter`, change the `q_proj.bias` branch to derive shard size from actual tensor shape
    - Check if `param_tensor.shape[0] == heads_per_rank * head_dim * 2 * world_size` (doubled layout)
    - If doubled: use `heads_per_rank * head_dim * 2` as shard size
    - If standard: use `heads_per_rank * head_dim` as shard size (existing behavior)
    - Match the logic used for `q_proj.weight` sharding
    - _Bug_Condition: q_proj.bias uses hardcoded heads_per_rank * head_dim ignoring doubled layout_
    - _Expected_Behavior: shard size derived from actual tensor shape, matching weight logic_
    - _Preservation: Standard (non-doubled) q_proj.bias sharding unchanged_
    - _Requirements: 2.5, 1.5_

  - [x] 3.5 Add fail-fast validation for MRoPE models without native rotary_emb
    - After `__init__` completes (after all extraction/assignment), check if `self._native_rotary_emb is None` AND config indicates MRoPE (`self.config.mrope_interleaved is True`)
    - If so, raise `RuntimeError` with descriptive message: "MRoPE model detected (mrope_interleaved=True) but no native rotary embedding available. The streaming loader must pass native_rotary_emb to TensorParallelShard."
    - This prevents silent corruption from incorrect 1D RoPE fallback
    - _Bug_Condition: MRoPE model silently falls back to 1D RoPE_
    - _Expected_Behavior: RuntimeError raised with descriptive message_
    - _Preservation: Non-MRoPE models (mrope_interleaved=False) unaffected_
    - _Requirements: 2.4, 1.4_

  - [x] 3.6 Update model_loader.py to pass native_rotary_emb
    - Update destructuring of `load_sharded_from_safetensors()` to capture 4th value: `sharded_state_dict, native_layers, tokenizer, native_rotary_emb = ...`
    - Pass `native_rotary_emb=native_rotary_emb` to `TensorParallelShard` constructor call
    - _Bug_Condition: model_loader creates shard without rotary_emb_
    - _Expected_Behavior: native_rotary_emb passed through from streaming loader to shard_
    - _Preservation: native_layers attachment unchanged_
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 3.7 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Streaming MRoPE Models Get Native Rotary Embedding
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior
    - When this test passes, it confirms: `_native_rotary_emb` is not None for streaming MRoPE models, q_proj.bias shard size matches weight shard size, RuntimeError raised for missing rotary_emb
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed)
    - _Requirements: 2.1, 2.2, 2.4, 2.5_

  - [x] 3.8 Verify preservation tests still pass
    - **Property 2: Preservation** - Non-MRoPE Sharding and HF Model Path Unchanged
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm all tests still pass after fix (no regressions)

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Task Dependencies

```yaml
1: []
2: []
3.1: [1, 2]
3.2: [3.1]
3.3: [1, 2]
3.4: [1, 2]
3.5: [1, 2]
3.6: [3.2, 3.3]
3.7: [3.1, 3.2, 3.3, 3.4, 3.5, 3.6]
3.8: [3.1, 3.2, 3.3, 3.4, 3.5, 3.6]
4: [3.7, 3.8]
```

## Notes

- Tests in tasks 1 and 2 use Hypothesis (property-based testing framework already in use in this project)
- Test file location: `src/exo/worker/engines/pytorch_xpu/tests/test_tensor_parallel_mrope_fix.py`
- The exploration test (task 1) is expected to FAIL on unfixed code — this confirms the bug exists
- The preservation test (task 2) is expected to PASS on unfixed code — this captures baseline behavior
- After the fix (tasks 3.1–3.6), the exploration test should PASS and preservation tests should still PASS

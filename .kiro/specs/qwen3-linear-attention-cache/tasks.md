# Implementation Plan

- [x] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - Native Linear Attention Layer Device Mismatch
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate native linear attention layers remain on CPU after TensorParallelShard initialization
  - **Scoped PBT Approach**: Scope the property to concrete failing cases: any layer_idx present in `_native_linear_attn_layers` where the layer's parameters are not on `self.device`
  - Test setup:
    - Create a mock HuggingFace model with `model.model.language_model.layers` containing modules with `linear_attn` attributes (mock `Qwen3_5GatedDeltaNet` modules with `nn.Linear` submodules)
    - Instantiate `TensorParallelShard(model, config, device="cpu")` (use CPU as target device for unit testing without XPU hardware)
    - Use Hypothesis to generate layer indices from the set of linear attention layers
  - Property assertion: For all `layer_idx` in `self._native_linear_attn_layers`, ALL parameters of `native_linear_attn_layers[layer_idx]` SHALL be on `self.device`
  - Secondary assertion: Calling `_forward_linear_attn_layer(hidden_states_on_device, layer_idx)` SHALL produce output of shape `(batch, seq_len, hidden_size)` on the correct device without raising RuntimeError
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS because native layers remain on CPU after extraction (confirms bug exists per design Bug Condition: `native_linear_attn_layers[layer_idx].parameters_device != input.hidden_states.device`)
  - Document counterexamples found (e.g., "layer 0 in_proj_qkv.weight is on cpu, expected cpu target device but layer was never moved")
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 2.1, 2.2, 2.3_

- [x] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Non-Linear-Attention Path Unchanged
  - **IMPORTANT**: Follow observation-first methodology
  - **GOAL**: Verify that full attention layers, MLP sharding, and non-hybrid model paths produce correct output on UNFIXED code (these paths are not affected by the bug)
  - Test setup:
    - Create a `TensorParallelShard` with a mock state_dict containing full attention layer weights (q_proj, k_proj, v_proj, o_proj) and MLP weights (gate_proj, up_proj, down_proj)
    - Use Hypothesis to generate random batch sizes (1-4), sequence lengths (1-32), and hidden dimensions compatible with the TPShardConfig
  - Observe on UNFIXED code:
    - Full attention forward path (QKV → RoPE → SDPA → o_proj) produces deterministic output for given input
    - MLP forward path (gate/up column-parallel → SiLU → down row-parallel) produces deterministic output
    - `_is_redundant()` correctly identifies linear_attn weights as redundant
    - `_detect_layer_types()` correctly classifies layers from state_dict keys
  - Write property-based tests:
    - For all valid inputs to full attention layers, output shape is `(batch, seq_len, hidden_size)` and values are finite (no NaN/Inf)
    - For all valid inputs to MLP blocks, output shape is `(batch, seq_len, hidden_size)` and values are finite
    - For all parameter names matching `_LINEAR_ATTN_PATTERN`, `_is_redundant()` returns True
    - For all layer indices, `_detect_layer_types()` returns "linear_attention" iff `linear_attn.in_proj_qkv.weight` key exists for that layer
  - Verify tests pass on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (confirms baseline behavior to preserve — these paths work correctly today)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 3. Fix for native linear attention layer device mismatch and cache initialization

  - [x] 3.1 Implement the fix
    - Move native layers to target device after `shard_weights()` completes in `__init__`:
      ```python
      for layer_idx, native_layer in self._native_linear_attn_layers.items():
          native_layer.to(self.device).eval()
      ```
    - Fix `_create_native_cache()` to use correct layer types based on `self._layer_types`:
      - `LinearAttentionLayer()` for layers where `_layer_types[idx] == "linear_attention"`
      - `DynamicLayer()` for layers where `_layer_types[idx] == "full_attention"`
    - Remove debug print statements: delete all `import sys` and `print(..., file=sys.stderr)` lines from `_extract_native_linear_attn_layers()` and `_create_native_cache()`
    - Set native layers to eval mode (already done via `.eval()` in the `.to()` call above) to disable dropout during inference
    - _Bug_Condition: isBugCondition(input) where input.layer_idx IN native_linear_attn_layers AND native_layer.parameters_device != self.device_
    - _Expected_Behavior: All native layer parameters on self.device after __init__, forward produces non-zero output on correct device with cache state persisting_
    - _Preservation: Full attention layers, MLP sharding, non-hybrid models, custom fallback path, linear_attn weight redundancy all unchanged_
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 3.2 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Native Linear Attention Layer Device Alignment
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior (native layers on correct device, forward produces valid output)
    - When this test passes, it confirms the expected behavior is satisfied
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms native layers are now on correct device and forward works)
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 3.3 Verify preservation tests still pass
    - **Property 2: Preservation** - Non-Linear-Attention Path Unchanged
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions in full attention, MLP, or non-hybrid paths)
    - Confirm all tests still pass after fix (no regressions)

- [x] 4. Checkpoint - Ensure all tests pass
  - Run the full test suite for this module: both exploration and preservation tests
  - Verify no other tests in the pytorch_xpu test directory are broken
  - Ensure all tests pass, ask the user if questions arise

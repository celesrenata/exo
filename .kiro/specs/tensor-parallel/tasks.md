# Implementation Plan

- [x] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - Tensor-Parallel Dispatch Falls Through to Pipeline Path
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the runner dispatches to pipeline-parallel instead of tensor-parallel when shard metadata indicates tensor parallelism
  - **Scoped PBT Approach**: Scope the property to concrete failing cases: `PyTorchXPURingInstance` with `start_layer=0, end_layer=n_layers, world_size>1`
  - **Test file**: `src/exo/worker/runner/tests/test_tensor_parallel_dispatch.py`
  - **Bug Condition**: `isBugCondition(X)` where `X.instance IS PyTorchXPURingInstance AND X.shard_metadata.start_layer = 0 AND X.shard_metadata.end_layer = X.shard_metadata.n_layers AND X.world_size > 1`
  - **Test approach**: Use Hypothesis to generate `(n_layers, world_size)` tuples where `world_size > 1`, create `PyTorchXPURingInstance` with tensor-parallel shard metadata (`start_layer=0, end_layer=n_layers`), and assert the dispatch logic routes to `tensor_parallel_generate`/`tensor_parallel_worker_loop`
  - **Concrete cases to include**: `n_layers=32, world_size=4` (the actual gremlin cluster config), `n_layers=24, world_size=2`
  - **Expected assertion**: The dispatch path chosen is "tensor_parallel" (not "pipeline_parallel")
  - **Also verify**: `isinstance(PyTorchXPURingInstance(...), TensorParallelInstance)` is False (confirms root cause of type mismatch)
  - Run test: `LD_LIBRARY_PATH="/nix/store/cf1a53iqg6ncnygl698c4v0l8qam5a2q-gcc-14.3.0-lib/lib:$LD_LIBRARY_PATH" uv run pytest src/exo/worker/runner/tests/test_tensor_parallel_dispatch.py -v --tb=short`
  - Run on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (this is correct - it proves the bug exists: dispatch goes to pipeline path instead of tensor-parallel path)
  - Document counterexamples found (e.g., "with n_layers=32, world_size=4, dispatch returns 'pipeline_parallel' instead of 'tensor_parallel'")
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 1.2, 2.1, 2.2_

- [x] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Pipeline-Parallel and Single-Node Dispatch Unchanged
  - **IMPORTANT**: Follow observation-first methodology
  - **Test file**: `src/exo/worker/runner/tests/test_tensor_parallel_dispatch.py` (same file, separate test class)
  - **Non-bug condition**: Cases where `start_layer != 0` OR `end_layer != n_layers` OR `world_size == 1`
  - **Observation-first approach**:
    - Observe: Pipeline-parallel instance (`start_layer=0, end_layer=8, n_layers=32, world_size=4`) dispatches to `distributed_generate` on unfixed code
    - Observe: Single-node instance (`world_size=1, start_layer=0, end_layer=32, n_layers=32`) dispatches to single-node path on unfixed code
    - Observe: Non-first pipeline stage (`start_layer=8, end_layer=16, n_layers=32, world_size=4`) dispatches to `distributed_worker_loop` on unfixed code
  - **Property-based tests**:
    - Generate random `(start_layer, end_layer, n_layers, world_size)` where `start_layer != 0 OR end_layer != n_layers` and `world_size > 1` → assert dispatch goes to pipeline-parallel path
    - Generate random `(start_layer, end_layer, n_layers)` with `world_size=1` → assert dispatch goes to single-node path
    - Generate `MlxRingInstance` / `TinygradRingInstance` inputs → assert they are NOT affected by PyTorch dispatch logic
  - **Edge cases**: `world_size=1` with `start_layer=0, end_layer=n_layers` (single-node full model, NOT tensor-parallel), `start_layer=0` with `end_layer < n_layers` (first pipeline stage, NOT tensor-parallel)
  - Run test: `LD_LIBRARY_PATH="/nix/store/cf1a53iqg6ncnygl698c4v0l8qam5a2q-gcc-14.3.0-lib/lib:$LD_LIBRARY_PATH" uv run pytest src/exo/worker/runner/tests/test_tensor_parallel_dispatch.py -v --tb=short -k "preservation"`
  - Run on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.6_

- [x] 3. Fix tensor-parallel dispatch and model loading

  - [x] 3.1 Implement metadata-based tensor-parallel detection in runner.py
    - Replace `isinstance(instance, _TensorParallelInstance)` check with metadata-based detection
    - Add detection logic BEFORE the existing `elif world_size > 1` branch:
      ```python
      is_tensor_parallel = (
          world_size > 1
          and shard_metadata.start_layer == 0
          and shard_metadata.end_layer == shard_metadata.n_layers
      )
      ```
    - Change dispatch: `if is_tensor_parallel:` → tensor_parallel_generate/tensor_parallel_worker_loop
    - Change existing `elif world_size > 1:` to `elif world_size > 1 and not is_tensor_parallel:` (pipeline-parallel)
    - Remove dead `isinstance(instance, _TensorParallelInstance)` check and unused import of `TensorParallelInstance`
    - File: `src/exo/worker/runner/runner.py` (dispatch logic around line 736)
    - _Bug_Condition: isBugCondition(input) where instance IS PyTorchXPURingInstance AND start_layer=0 AND end_layer=n_layers AND world_size>1_
    - _Expected_Behavior: dispatch_path = "tensor_parallel" for all inputs satisfying bug condition_
    - _Preservation: Pipeline-parallel (different layer ranges) and single-node (world_size=1) paths unchanged_
    - _Requirements: 1.1, 1.2, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3_

  - [x] 3.2 Implement TensorParallelShard creation in model_loader.py
    - In the sharding decision logic (~line 205-212), detect tensor-parallel configuration BEFORE creating TransformerShard
    - When `start_layer == 0 AND end_layer == n_layers AND world_size > 1`: create `TensorParallelShard` instead of `TransformerShard`
    - Import `TensorParallelShard` from `src/exo/worker/engines/pytorch_xpu/tensor_parallel_shard.py`
    - Add `_create_tensor_parallel_shard(model, shard_metadata)` method that:
      - Extracts model config: `hidden_size`, `num_attention_heads`, `head_dim`, `intermediate_size`, `num_key_value_heads`
      - Constructs `TPShardConfig` with rank=`shard_metadata.device_rank`, world_size=`shard_metadata.world_size`
      - Returns `TensorParallelShard(model, tp_config)`
    - File: `src/exo/worker/engines/pytorch_xpu/model_loader.py`
    - _Bug_Condition: model loading with start_layer=0, end_layer=n_layers, world_size>1 creates TransformerShard_
    - _Expected_Behavior: creates TensorParallelShard with all-reduce inside forward_
    - _Preservation: Pipeline-parallel model loading (different layer ranges) still creates TransformerShard_
    - _Requirements: 1.5, 2.4, 3.1_

  - [x] 3.3 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Tensor-Parallel Dispatch Detection
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior (dispatch to tensor-parallel path)
    - When this test passes, it confirms the expected behavior is satisfied
    - Run: `LD_LIBRARY_PATH="/nix/store/cf1a53iqg6ncnygl698c4v0l8qam5a2q-gcc-14.3.0-lib/lib:$LD_LIBRARY_PATH" uv run pytest src/exo/worker/runner/tests/test_tensor_parallel_dispatch.py -v --tb=short -k "bug_condition"`
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed — dispatch now routes to tensor-parallel path)
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 3.4 Verify preservation tests still pass
    - **Property 2: Preservation** - Pipeline-Parallel and Single-Node Unchanged
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run: `LD_LIBRARY_PATH="/nix/store/cf1a53iqg6ncnygl698c4v0l8qam5a2q-gcc-14.3.0-lib/lib:$LD_LIBRARY_PATH" uv run pytest src/exo/worker/runner/tests/test_tensor_parallel_dispatch.py -v --tb=short -k "preservation"`
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions — pipeline-parallel and single-node paths unchanged)
    - Confirm all preservation tests still pass after fix (no regressions)
    - _Requirements: 3.1, 3.2, 3.3, 3.6_

- [x] 4. Checkpoint - Ensure all tests pass
  - Run full test suite for the affected files:
    ```bash
    LD_LIBRARY_PATH="/nix/store/cf1a53iqg6ncnygl698c4v0l8qam5a2q-gcc-14.3.0-lib/lib:$LD_LIBRARY_PATH" uv run pytest src/exo/worker/runner/tests/test_tensor_parallel_dispatch.py src/exo/worker/runner/tests/test_runner_dispatch.py -v --tb=short
    ```
  - Verify existing `test_runner_dispatch.py` tests still pass (no regressions in ConnectToGroup, LoadModel, StartWarmup, Shutdown, BackendDispatch)
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Integration test - Deploy and verify end-to-end tensor-parallel generation
  - Deploy to gremlin cluster: `bash deploy_cluster.sh`
  - Wait for model to load and warm up (check dashboard at http://10.1.1.12:52415)
  - Verify all 4 nodes are connected and show `RunnerReady` status on dashboard
  - Submit "hello world" prompt via the API:
    ```bash
    curl -X POST http://10.1.1.12:52415/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "Qwen/Qwen3.5-4B", "messages": [{"role": "user", "content": "hello world"}], "max_tokens": 50}'
    ```
  - Verify tokens are generated within 30 seconds (not 120s timeout)
  - Check logs for no "Timed out waiting" errors: `ssh root@10.1.1.12 "journalctl -u exo -n 50 --no-pager | grep -i timeout"`
  - Verify all ranks participate in tensor-parallel generation (check logs for "tensor_parallel_generate" and "tensor_parallel_worker_loop" on respective nodes)
  - _Requirements: 2.5_

# Tensor-Parallel Dispatch Bugfix Design

## Overview

The 4-node gremlin cluster hangs for 120 seconds and times out when tensor-parallel sharding is configured. The root cause is a type dispatch mismatch in `runner.py`: the runner checks `isinstance(instance, TensorParallelInstance)` but the placement module creates `PyTorchXPURingInstance` for all PyTorchXPURing cases regardless of sharding mode. The `TensorParallelInstance` dataclass (a plain frozen dataclass) is never instantiated. This causes the runner to fall through to the pipeline-parallel code path, which assumes sequential layer ranges per node, resulting in a shape mismatch and Gloo timeout.

The fix detects tensor parallelism from shard metadata (all nodes have `start_layer=0, end_layer=n_layers`) rather than relying on instance type, and routes to the correct generation path. Additionally, the model loader must create a `TensorParallelShard` (with all-reduce inside forward) instead of a `TransformerShard` (sequential layer execution) when tensor parallelism is detected.

## Glossary

- **Bug_Condition (C)**: The condition that triggers the bug — when a multi-node `PyTorchXPURingInstance` has tensor-parallel shard metadata (all layers on every node) but the runner dispatches to the pipeline-parallel path
- **Property (P)**: The desired behavior — the runner dispatches to `tensor_parallel_generate`/`tensor_parallel_worker_loop` and the model is loaded as `TensorParallelShard`
- **Preservation**: Existing pipeline-parallel behavior, single-node generation, and non-PyTorchXPURing backends must remain unchanged
- **`runner.py`**: The file at `src/exo/worker/runner/runner.py` that dispatches generation to the correct code path based on instance type and shard metadata
- **`model_loader.py`**: The file at `src/exo/worker/engines/pytorch_xpu/model_loader.py` that loads HuggingFace models and wraps them in shard wrappers
- **`TensorParallelShard`**: The class at `src/exo/worker/engines/pytorch_xpu/tensor_parallel_shard.py` that holds sharded weights and executes all-reduce inside forward()
- **`TransformerShard`**: The existing shard wrapper that executes a sequential subset of layers (used for pipeline parallelism)
- **`TensorParallelInstance`**: A frozen dataclass at `src/exo/worker/engines/pytorch_xpu/tensor_parallel_instance.py` — never instantiated by placement
- **`PyTorchXPURingInstance`**: The Pydantic model at `src/exo/shared/types/worker/instances.py` that placement actually creates for all PyTorchXPURing cases
- **Shard metadata**: `ShardMetadata` containing `start_layer`, `end_layer`, `n_layers`, `world_size`, `device_rank`

## Bug Details

### Bug Condition

The bug manifests when the placement module creates a `PyTorchXPURingInstance` with `Sharding.Tensor` configuration. The shard metadata assigns ALL layers to ALL nodes (`start_layer=0, end_layer=n_layers`), but the runner's type-based dispatch (`isinstance(instance, TensorParallelInstance)`) fails because the instance is a `PyTorchXPURingInstance`, not a `TensorParallelInstance`. The runner falls through to the pipeline-parallel path which expects sequential layer ranges.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type RunnerDispatchContext (instance, shard_metadata, world_size, model)
  OUTPUT: boolean

  RETURN input.instance IS PyTorchXPURingInstance
     AND input.shard_metadata.start_layer = 0
     AND input.shard_metadata.end_layer = input.shard_metadata.n_layers
     AND input.world_size > 1
END FUNCTION
```

### Examples

- **4-node Qwen3.5-4B tensor-parallel**: `shard_metadata = {start_layer=0, end_layer=32, n_layers=32, world_size=4}` on all 4 nodes. Runner checks `isinstance(PyTorchXPURingInstance, TensorParallelInstance)` → False. Falls to pipeline path. Rank 0 runs all 32 layers + lm_head, produces logits shape `(1, seq_len, 248320)`, sends to rank 1. Rank 1 expects hidden_states shape `(1, seq_len, 2560)`. Gloo times out after 120s.

- **4-node Qwen3.5-4B pipeline-parallel (NOT buggy)**: `shard_metadata = {start_layer=0, end_layer=8, n_layers=32, world_size=4}` on rank 0, `{start_layer=8, end_layer=16, ...}` on rank 1, etc. Runner correctly takes pipeline path. Each rank processes its layer range and passes activations.

- **Single-node (NOT buggy)**: `world_size=1`. Runner takes single-node path regardless of instance type.

- **Model loading with tensor-parallel metadata**: `_create_model_shard()` creates `TransformerShard` which executes layers sequentially without all-reduce. Should create `TensorParallelShard` which shards weights and uses all-reduce inside forward().

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Pipeline-parallel instances (different `start_layer`/`end_layer` per node) must continue to use `distributed_generate` / `distributed_worker_loop` with sequential activation passing
- Single-node instances (`world_size=1`) must continue to use the single-node generation path
- `MlxRing`, `MlxJaccl`, and `TinygradRing` instance types must be completely unaffected
- Gloo process group initialization must continue to work for both tensor-parallel and pipeline-parallel
- Warmup with dummy tensors must continue to succeed
- The `PyTorchXPURingInstance` Pydantic model and its serialization must remain unchanged

**Scope:**
All inputs where `shard_metadata.start_layer != 0` OR `shard_metadata.end_layer != shard_metadata.n_layers` OR `world_size == 1` should be completely unaffected by this fix. This includes:
- Pipeline-parallel multi-node generation
- Single-node generation
- Non-PyTorchXPURing backends
- Model loading for pipeline-parallel configurations

## Hypothesized Root Cause

Based on the bug analysis, the issues are:

1. **Type Dispatch Mismatch (runner.py:736)**: The runner imports `TensorParallelInstance` from `tensor_parallel_instance.py` — a plain frozen dataclass. The actual instance type is `PyTorchXPURingInstance` (a Pydantic `TaggedModel` from `instances.py`). These are completely different class hierarchies. `isinstance(instance, TensorParallelInstance)` will NEVER be True because placement never creates a `TensorParallelInstance`.

2. **Placement Creates Wrong Type (placement.py)**: The `PyTorchXPURing` case in placement always creates `PyTorchXPURingInstance` regardless of whether `Sharding.Tensor` or `Sharding.Pipeline` was requested. The `TensorParallelInstance` dataclass is never instantiated anywhere in the codebase.

3. **Model Loader Creates Wrong Shard Type (model_loader.py:212)**: When `world_size > 1` and all layers are assigned (tensor-parallel metadata), `_create_model_shard()` creates a `TransformerShard`. This wrapper executes layers sequentially without all-reduce synchronization. Tensor parallelism requires `TensorParallelShard` which shards weights across ranks and performs all-reduce after row-parallel layers.

4. **Information IS Available**: The shard metadata correctly distinguishes tensor-parallel from pipeline-parallel: tensor-parallel has `start_layer=0, end_layer=n_layers` on ALL nodes, while pipeline-parallel has different layer ranges per node. This metadata is available at both dispatch time and model loading time.

## Correctness Properties

Property 1: Bug Condition - Tensor-Parallel Dispatch Detection

_For any_ multi-node `PyTorchXPURingInstance` where shard metadata indicates tensor parallelism (`start_layer=0` AND `end_layer=n_layers` AND `world_size > 1`), the fixed runner SHALL dispatch to `tensor_parallel_generate` (rank 0) or `tensor_parallel_worker_loop` (non-rank-0) instead of the pipeline-parallel path.

**Validates: Requirements 2.1, 2.2, 2.3**

Property 2: Bug Condition - Model Loading Creates TensorParallelShard

_For any_ model loading request where shard metadata indicates tensor parallelism (`start_layer=0` AND `end_layer=n_layers` AND `world_size > 1`), the fixed model loader SHALL create a `TensorParallelShard` (with all-reduce inside forward) instead of a `TransformerShard`.

**Validates: Requirements 2.4**

Property 3: Preservation - Pipeline-Parallel Dispatch Unchanged

_For any_ multi-node `PyTorchXPURingInstance` where shard metadata indicates pipeline parallelism (`start_layer != 0` OR `end_layer != n_layers`), the fixed runner SHALL produce the same dispatch result as the original runner, routing to `distributed_generate` / `distributed_worker_loop`.

**Validates: Requirements 3.1, 3.2**

Property 4: Preservation - Single-Node and Other Backends Unchanged

_For any_ input where `world_size == 1` OR the instance is not `PyTorchXPURingInstance`, the fixed code SHALL produce exactly the same behavior as the original code, preserving single-node generation and non-PyTorchXPURing backend handling.

**Validates: Requirements 3.3, 3.6**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `src/exo/worker/runner/runner.py`

**Function**: Dispatch logic at line ~736

**Specific Changes**:
1. **Replace type-based dispatch with metadata-based dispatch**: Instead of `isinstance(instance, _TensorParallelInstance)`, detect tensor parallelism from shard metadata:
   ```python
   is_tensor_parallel = (
       world_size > 1
       and shard_metadata.start_layer == 0
       and shard_metadata.end_layer == shard_metadata.n_layers
   )

   if is_tensor_parallel:
       # tensor_parallel_generate / tensor_parallel_worker_loop
   elif world_size > 1:
       # distributed_generate / distributed_worker_loop (pipeline)
   else:
       # Single-node
   ```

2. **Remove dead TensorParallelInstance import**: The import of `TensorParallelInstance` from `tensor_parallel_instance.py` is no longer needed for dispatch.

---

**File**: `src/exo/worker/engines/pytorch_xpu/model_loader.py`

**Function**: `_create_model_shard` and the sharding decision logic (~line 205-212)

**Specific Changes**:
3. **Detect tensor-parallel at model loading time**: When `start_layer == 0` AND `end_layer == n_layers` AND `world_size > 1`, create a `TensorParallelShard` instead of `TransformerShard`:
   ```python
   if (
       shard_metadata.start_layer == 0
       and shard_metadata.end_layer == shard_metadata.n_layers
       and shard_metadata.world_size > 1
   ):
       # Tensor-parallel: shard weights, all-reduce inside forward
       model = self._create_tensor_parallel_shard(model, shard_metadata)
   elif shard_metadata.world_size > 1 or not (
       shard_metadata.start_layer == 0
       and shard_metadata.end_layer == shard_metadata.n_layers
   ):
       # Pipeline-parallel: sequential layer subset
       model = self._create_model_shard(model, shard_metadata)
   ```

4. **Add `_create_tensor_parallel_shard` method**: New method that creates a `TensorParallelShard` with the appropriate `TPShardConfig` derived from the model's config (hidden_size, num_attention_heads, head_dim, intermediate_size, num_key_value_heads) and the shard metadata (rank, world_size).

5. **Extract model config for TPShardConfig**: Read `hidden_size`, `num_attention_heads`, `head_dim`, `intermediate_size`, and `num_key_value_heads` from the HuggingFace model config to construct `TPShardConfig`.

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code, then verify the fix works correctly and preserves existing behavior.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Write unit tests that simulate the dispatch decision logic with tensor-parallel shard metadata and verify that the current code incorrectly routes to the pipeline-parallel path. Run these tests on the UNFIXED code to observe failures.

**Test Cases**:
1. **Dispatch with TP metadata**: Create a `PyTorchXPURingInstance` with `start_layer=0, end_layer=32, n_layers=32, world_size=4`. Assert dispatch goes to tensor-parallel path (will fail on unfixed code — goes to pipeline path instead)
2. **Model loader with TP metadata**: Call model loading with `start_layer=0, end_layer=32, n_layers=32, world_size=4`. Assert model is `TensorParallelShard` (will fail on unfixed code — creates `TransformerShard`)
3. **isinstance check**: Verify `isinstance(PyTorchXPURingInstance(...), TensorParallelInstance)` is False (confirms root cause)

**Expected Counterexamples**:
- Dispatch returns "pipeline_parallel" when shard metadata indicates tensor parallelism
- Model loader creates `TransformerShard` when `TensorParallelShard` is needed
- Root cause confirmed: `PyTorchXPURingInstance` is not an instance of `TensorParallelInstance`

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function produces the expected behavior.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  dispatch_result := runner_dispatch_fixed(input)
  ASSERT dispatch_result = "tensor_parallel"
  
  model_result := model_loader_fixed(input.shard_metadata)
  ASSERT model_result IS TensorParallelShard
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT runner_dispatch_original(input) = runner_dispatch_fixed(input)
  ASSERT model_loader_original(input) = model_loader_fixed(input)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many combinations of shard metadata (varying start_layer, end_layer, n_layers, world_size)
- It catches edge cases like `world_size=1` with full layer range, or `start_layer=0` with `end_layer < n_layers`
- It provides strong guarantees that pipeline-parallel behavior is unchanged

**Test Plan**: Observe dispatch behavior on UNFIXED code for pipeline-parallel and single-node inputs, then write property-based tests capturing that behavior.

**Test Cases**:
1. **Pipeline dispatch preservation**: Generate random shard metadata where `start_layer != 0` OR `end_layer != n_layers` with `world_size > 1`. Verify dispatch goes to pipeline-parallel path on both original and fixed code.
2. **Single-node preservation**: Generate random shard metadata with `world_size=1`. Verify dispatch goes to single-node path on both original and fixed code.
3. **Model loader preservation**: Generate random pipeline-parallel shard metadata. Verify `TransformerShard` is created on both original and fixed code.

### Unit Tests

- Test `is_tensor_parallel` detection logic with various shard metadata combinations
- Test that `_create_tensor_parallel_shard` produces a `TensorParallelShard` with correct `TPShardConfig`
- Test edge cases: `world_size=1` with all layers (single-node, NOT tensor-parallel), `start_layer=0` but `end_layer < n_layers` (first pipeline stage, NOT tensor-parallel)

### Property-Based Tests

- Generate random `(start_layer, end_layer, n_layers, world_size)` tuples and verify the dispatch detection function correctly classifies tensor-parallel vs pipeline-parallel vs single-node
- Generate random pipeline-parallel shard metadata and verify preservation of `TransformerShard` creation
- Generate random tensor-parallel shard metadata and verify `TensorParallelShard` creation with valid `TPShardConfig`

### Integration Tests

- Deploy to 4-node gremlin cluster with `Sharding.Tensor` and verify "hello world" generates tokens without timeout
- Deploy to 4-node gremlin cluster with `Sharding.Pipeline` and verify existing pipeline-parallel generation still works
- Verify single-node generation is unaffected by the changes

# Bugfix Requirements Document

## Introduction

The 4-node gremlin cluster's distributed inference hangs for 120 seconds and times out when tensor-parallel sharding is configured. The placement module creates `PyTorchXPURingInstance` with tensor-parallel shard metadata (all layers on every node), but the runner's `isinstance(instance, TensorParallelInstance)` check fails because the instance is the wrong type. This causes the runner to fall through to the pipeline-parallel code path, which assumes each node has different layers. The resulting shape mismatch between logits (1, seq_len, 248320) and expected hidden_states (1, seq_len, 2560) causes Gloo to time out waiting for a recv that never completes.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN placement creates an instance with `Sharding.Tensor` for the `PyTorchXPURing` backend THEN the system creates a `PyTorchXPURingInstance` instead of a `TensorParallelInstance`, losing the tensor-parallel type signal

1.2 WHEN the runner dispatches generation for a multi-node instance with tensor-parallel shard metadata (start_layer=0, end_layer=n_layers on all nodes) THEN the `isinstance(instance, TensorParallelInstance)` check fails and the system falls through to the pipeline-parallel path (`distributed_generate` / `distributed_worker_loop`)

1.3 WHEN the pipeline-parallel path executes with tensor-parallel shard metadata (all layers on every node) THEN rank 0 runs ALL layers including lm_head, producing logits of shape (1, seq_len, vocab_size) and attempts to send them via `send_activation()` to rank 1

1.4 WHEN rank 1 receives an activation in the pipeline-parallel path THEN it expects hidden_states of shape (1, seq_len, hidden_size) but receives logits of shape (1, seq_len, vocab_size), causing a shape mismatch and Gloo timeout after 120 seconds

1.5 WHEN the model is loaded for a tensor-parallel configuration THEN the system creates a `TransformerShard` (sequential layer execution) instead of a `TensorParallelShard` (all-reduce inside forward)

### Expected Behavior (Correct)

2.1 WHEN placement creates an instance with `Sharding.Tensor` for the `PyTorchXPURing` backend THEN the system SHALL create a `TensorParallelInstance` (or the runner SHALL detect tensor-parallel sharding from shard metadata) so the correct dispatch path is taken

2.2 WHEN the runner dispatches generation for a multi-node instance with tensor-parallel shard metadata (start_layer=0, end_layer=n_layers on all nodes) THEN the system SHALL dispatch to `tensor_parallel_generate` (rank 0) and `tensor_parallel_worker_loop` (non-rank-0) instead of the pipeline-parallel path

2.3 WHEN tensor-parallel generation executes THEN rank 0 SHALL broadcast token IDs to all ranks, all ranks SHALL forward simultaneously through all layers with all-reduce synchronization, and rank 0 SHALL sample the next token from the resulting logits

2.4 WHEN the model is loaded for a tensor-parallel configuration THEN the system SHALL create a `TensorParallelShard` (with all-reduce inside forward) instead of a `TransformerShard`

2.5 WHEN a "hello world" prompt is submitted to the 4-node cluster with tensor-parallel sharding THEN the system SHALL generate tokens without hanging or timing out

### Unchanged Behavior (Regression Prevention)

3.1 WHEN placement creates an instance with `Sharding.Pipeline` for the `PyTorchXPURing` backend THEN the system SHALL CONTINUE TO create a `PyTorchXPURingInstance` with sequential layer ranges (different start_layer/end_layer per node)

3.2 WHEN the runner dispatches generation for a pipeline-parallel instance (different layer ranges per node) THEN the system SHALL CONTINUE TO use `distributed_generate` / `distributed_worker_loop` with sequential activation passing

3.3 WHEN a single-node instance is used (world_size=1) THEN the system SHALL CONTINUE TO use the single-node generation path without distributed communication

3.4 WHEN warmup runs with dummy tensors THEN the system SHALL CONTINUE TO succeed without errors

3.5 WHEN Gloo process group initialization occurs THEN the system SHALL CONTINUE TO connect all nodes successfully

3.6 WHEN placement creates instances for `MlxRing`, `MlxJaccl`, or `TinygradRing` backends THEN the system SHALL CONTINUE TO create the appropriate instance types unchanged

---

## Bug Condition

### Bug Condition Function

```pascal
FUNCTION isBugCondition(X)
  INPUT: X of type RunnerDispatchInput (instance, shard_metadata, world_size)
  OUTPUT: boolean

  // The bug triggers when:
  // 1. The instance is PyTorchXPURingInstance (not TensorParallelInstance)
  // 2. The shard metadata indicates tensor parallelism (all nodes have all layers)
  // 3. world_size > 1 (multi-node)
  RETURN X.instance IS PyTorchXPURingInstance
     AND X.shard_metadata.start_layer = 0
     AND X.shard_metadata.end_layer = X.shard_metadata.n_layers
     AND X.world_size > 1
END FUNCTION
```

### Property Specification — Fix Checking

```pascal
// Property: Fix Checking — Tensor-parallel dispatch
FOR ALL X WHERE isBugCondition(X) DO
  dispatch_path ← runner_dispatch(X)
  ASSERT dispatch_path = "tensor_parallel"
  ASSERT dispatch_path ≠ "pipeline_parallel"
  // Rank 0 uses tensor_parallel_generate
  // Non-rank-0 uses tensor_parallel_worker_loop
  // Model is loaded as TensorParallelShard (not TransformerShard)
END FOR
```

### Preservation Goal

```pascal
// Property: Preservation Checking — Pipeline-parallel still works
FOR ALL X WHERE NOT isBugCondition(X) DO
  ASSERT F(X) = F'(X)
  // Pipeline-parallel instances (different layer ranges) still use distributed_generate
  // Single-node instances still use single-node generation
  // Non-PyTorchXPURing backends are unaffected
END FOR
```

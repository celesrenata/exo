# Implementation Plan: XPU Kernel Launch Optimization

## Overview

Eliminate the CPU-side dispatch bottleneck causing 48% GPU idle time during autoregressive decode on Intel Arc Meteor Lake-P iGPUs. Implementation follows a phased approach: benchmark harness first, then synchronization removal, static KV cache, tensor-only decode wrapper, torch.compile integration, packed projections, fused kernels, on-device sampling, async output streaming, and final integration validation. Each phase builds on the previous, with property-based tests validating correctness at each step.

## Tasks

- [x] 1. Benchmark harness and configuration flags
  - [x] 1.1 Add kernel launch optimization flags to PytorchXpuOptimizationConfiguration
    - Add `enable_static_kv_cache`, `enable_torch_compile`, `enable_packed_projections`, `enable_fused_kernels`, `enable_on_device_sampling`, `enable_async_output`, `enable_sync_removal` boolean flags (all default False)
    - Add `torch_compile_mode` (Literal["default", "reduce-overhead", "max-autotune"], default "max-autotune"), `static_cache_max_seq_len` (int, default 2048), `async_output_max_pending` (int, default 32), `async_output_resume_threshold` (int, default 16)
    - File: `src/exo/worker/engines/pytorch_xpu/pipeline_config.py`
    - _Requirements: 11.1, 11.4_

  - [x] 1.2 Create benchmark harness for decode throughput measurement
    - Create `src/exo/worker/engines/pytorch_xpu/benchmark_decode.py` with functions to measure tokens/sec, GPU idle percentage, and per-token latency coefficient of variation
    - Instrument the existing decode loop in `pipeline_generator.py` with timing hooks gated by `enable_performance_instrumentation`
    - _Requirements: 10.1, 10.2, 10.4_

  - [x]* 1.3 Write unit tests for configuration flags
    - Verify all new flags default to False
    - Verify frozen immutability
    - Verify validators for numeric parameters
    - File: `src/exo/worker/engines/pytorch_xpu/tests/test_config_flags.py`
    - _Requirements: 11.1, 11.4_

- [x] 2. Remove implicit synchronization from decode hot path
  - [x] 2.1 Audit and remove synchronization points in PipelineParallelShard
    - Identify all `.item()`, `.cpu()`, `.numpy()`, and boolean tensor evaluations in the decode path of `pipeline_parallel_shard.py`
    - Replace with on-device alternatives (e.g., keep tensors on XPU, defer CPU transfers)
    - Gate changes behind `enable_sync_removal` flag
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 2.2 Defer tokenizer decode in pipeline_generator.py
    - Modify the decode loop to launch the next forward pass before performing tokenizer decode on the previous token
    - Ensure token output ordering is preserved
    - Gate behind `enable_sync_removal` flag
    - _Requirements: 1.4_

  - [x]* 2.3 Write property test for zero-synchronization decode step
    - **Property 1: Zero-Synchronization Decode Step**
    - Verify that executing a decode step produces zero implicit synchronization events between embedding lookup and logits output
    - Use hypothesis to generate random token IDs and sequence positions
    - File: `src/exo/worker/engines/pytorch_xpu/tests/test_sync_removal_properties.py`
    - **Validates: Requirements 1.1, 1.2, 1.3, 10.3**

- [x] 3. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Static KV cache implementation
  - [x] 4.1 Implement StaticKVCache class
    - Create `src/exo/worker/engines/pytorch_xpu/static_kv_cache.py`
    - Pre-allocate fixed-size tensors for key/value storage per layer: `[1, max_seq_len, num_kv_heads, head_dim]`
    - Implement `update()` for in-place slice assignment, `reset()` for position zeroing, `position` property
    - Support GatedDeltaNet conv_state and recurrent_state in pre-allocated slots
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x]* 4.2 Write property test for static cache in-place invariant
    - **Property 3: Static Cache In-Place Invariant**
    - For any sequence of N decode steps (1 ≤ N ≤ max_seq_len), verify `data_ptr()` remains constant and position equals N
    - After `reset()`, verify position equals 0 and same tensor storage is reused
    - File: `src/exo/worker/engines/pytorch_xpu/tests/test_static_kv_cache_properties.py`
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

  - [x]* 4.3 Write property test for minimal device-to-host transfer
    - **Property 7: Minimal Device-to-Host Transfer**
    - Verify that total data transferred from XPU to CPU is exactly 8 bytes (one int64 token ID) per decode step
    - File: `src/exo/worker/engines/pytorch_xpu/tests/test_static_kv_cache_properties.py`
    - **Validates: Requirements 8.4**

- [x] 5. Tensor-only decode wrapper
  - [x] 5.1 Implement TensorOnlyDecodeWrapper
    - Create `src/exo/worker/engines/pytorch_xpu/tensor_only_decode.py`
    - Implement `decode_one_token()` function with pure-tensor signature (token_id, position, cache_keys, cache_values, conv_states, recurrent_states, weights)
    - Express cache updates as in-place slice assignments traceable by torch.compile
    - No Python object mutation, no dataclass fields, no dict mutations
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 5.2 Add graph break detection and logging
    - Implement graph break detection using `torch._dynamo.utils.counters["graph_break"]`
    - Log break location and reason at WARNING level when detected
    - _Requirements: 3.4_

- [x] 6. torch.compile with XPU Inductor backend
  - [x] 6.1 Implement CompiledDecodePath class
    - Create `src/exo/worker/engines/pytorch_xpu/compiled_decode_path.py`
    - Extract weights from HuggingFace model layers, pack projections, create static cache
    - Compile `decode_one_token` with `torch.compile(backend="inductor", mode="max-autotune")`
    - Implement fallback to eager `PipelineParallelShard.forward()` on compilation failure (log at ERROR level)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [x] 6.2 Integrate CompiledDecodePath into PipelineParallelShard
    - Add conditional dispatch in `PipelineParallelShard.forward()`: if optimized path enabled, use `CompiledDecodePath.forward()`; otherwise use existing `_forward_layer` loop
    - Gate behind `enable_torch_compile` flag
    - _Requirements: 4.1, 11.1, 11.3_

- [x] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Packed projections
  - [x] 8.1 Implement packed projection utilities
    - Create `src/exo/worker/engines/pytorch_xpu/packed_projections.py`
    - Implement `pack_qkv_weights()`: concatenate Q, K, V into `[q_dim + k_dim + v_dim, hidden_size]`
    - Implement `unpack_qkv_output()`: split packed output using `tensor.split()`
    - Implement `pack_gate_up_weights()`: concatenate gate and up into `[2 * intermediate_size, hidden_size]`
    - Detect and pack weights during model loading in CompiledDecodePath.__init__
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [x]* 8.2 Write property test for packed projection numerical equivalence
    - **Property 4: Packed Projection Numerical Equivalence**
    - For any valid input tensor of shape `[1, 1, hidden_size]` with bf16 values, verify packed QKV output matches separate Q, K, V projections within 1e-2 relative error
    - Same for packed gate/up projections
    - Use hypothesis to generate random input tensors and weight matrices
    - File: `src/exo/worker/engines/pytorch_xpu/tests/test_packed_projections_properties.py`
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**

- [x] 9. Fused pointwise kernels
  - [x] 9.1 Implement fused kernel reference implementations
    - Create `src/exo/worker/engines/pytorch_xpu/fused_kernels.py`
    - Implement `fused_rmsnorm_residual()`: RMSNorm + residual addition in single pass
    - Implement `fused_silu_gate()`: `SiLU(gate) * up` in single pass
    - Implement `fused_rotary_embedding()`: apply rotation in-place without intermediate tensors
    - These serve as reference implementations; torch.compile Inductor will auto-fuse them
    - _Requirements: 6.1, 6.2, 6.3, 6.5_

  - [x]* 9.2 Write property test for fused kernel numerical equivalence
    - **Property 5: Fused Kernel Numerical Equivalence**
    - For any valid input tensors with bf16 values, verify fused outputs match sequential unfused operations within 1e-2 relative error
    - Test: (a) fused RMSNorm + residual, (b) fused SiLU * gate, (c) fused rotary embedding
    - Use hypothesis to generate random tensors of varying shapes
    - File: `src/exo/worker/engines/pytorch_xpu/tests/test_fused_kernels_properties.py`
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.4**

- [x] 10. Fused decode attention kernel
  - [x] 10.1 Implement fused decode attention in the compiled decode path
    - Add fused attention computation (Q×K^T scaling, softmax, V multiply) as a single operation within the compiled graph
    - Read K and V directly from StaticKVCache without intermediate buffer copies
    - Support Qwen3.5-4B dimensions: hidden_size=2560, num_heads=32, num_kv_heads=4, head_dim=80
    - Rely on torch.compile Inductor to fuse the attention operations into a single kernel
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [x]* 10.2 Write property test for fused decode attention numerical equivalence
    - **Property 6: Fused Decode Attention Numerical Equivalence**
    - For any valid query `[1, 1, num_heads, head_dim]` and key/value cache `[1, seq_len, num_kv_heads, head_dim]` where 1 ≤ seq_len ≤ max_seq_len, verify fused output matches unfused multi-step attention within 1e-2 relative error
    - Use hypothesis to generate random queries and cache contents
    - File: `src/exo/worker/engines/pytorch_xpu/tests/test_fused_attention_properties.py`
    - **Validates: Requirements 7.1, 7.3**

- [x] 11. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. On-device sampling
  - [x] 12.1 Implement OnDeviceSampler class
    - Create `src/exo/worker/engines/pytorch_xpu/on_device_sampling.py`
    - Implement `sample()`: argmax, top-k, and top-p sampling entirely on XPU device
    - Implement `transfer_token_to_cpu_async()`: non-blocking copy of scalar token ID to pinned CPU memory
    - Use `torch.argmax`, `torch.topk`, `torch.sort`, `torch.cumsum`, `torch.multinomial` all on XPU
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x]* 12.2 Write property test for on-device tensor retention
    - **Property 2: On-Device Tensor Retention**
    - For any decode step with any valid input token, verify all intermediate tensors and output logits remain on XPU device
    - Verify only a single int64 scalar crosses the device boundary
    - File: `src/exo/worker/engines/pytorch_xpu/tests/test_on_device_sampling_properties.py`
    - **Validates: Requirements 1.3, 8.1, 8.2, 8.3, 8.4**

- [x] 13. Async output streaming
  - [x] 13.1 Implement AsyncOutputStreamer class
    - Create `src/exo/worker/engines/pytorch_xpu/async_output_streamer.py`
    - Implement asyncio queue with `put_token()`, `get_token()`, `should_pause`, `should_resume` properties
    - Implement backpressure: pause when queue exceeds `max_pending` (32), resume when below `resume_threshold` (16)
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

  - [x] 13.2 Integrate AsyncOutputStreamer into pipeline_generator.py
    - Modify decode loop to launch next forward pass before tokenizer decode
    - Yield tokens to API via async queue without blocking the decode loop
    - Gate behind `enable_async_output` flag
    - _Requirements: 9.1, 9.2, 9.3_

  - [x]* 13.3 Write property test for backpressure queue bounds
    - **Property 8: Backpressure Queue Bounds**
    - For any sequence of token productions, verify pause activates when queue > max_pending, resume when queue < resume_threshold
    - Verify queue depth never exceeds max_pending + 1
    - Use hypothesis to generate random production/consumption patterns
    - File: `src/exo/worker/engines/pytorch_xpu/tests/test_backpressure_properties.py`
    - **Validates: Requirements 9.4**

- [x] 14. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 15. Integration and wiring
  - [x] 15.1 Wire all components into the optimized decode path
    - Connect CompiledDecodePath with StaticKVCache, PackedProjections, FusedKernels, OnDeviceSampler, and AsyncOutputStreamer
    - Ensure the full optimized path executes end-to-end when all flags are enabled
    - Implement per-optimization fallback chain (each optimization falls back independently)
    - _Requirements: 11.1, 11.3_

  - [x] 15.2 Implement disabled-optimization output equivalence
    - Verify that when all optimization flags are False, the system produces identical output to the existing unoptimized pipeline at the same temperature and seed
    - Add integration test for this invariant
    - _Requirements: 11.2_

  - [x]* 15.3 Write property test for disabled-optimization output equivalence
    - **Property 9: Disabled-Optimization Output Equivalence**
    - For any prompt string and fixed random seed, verify byte-identical output tokens when all flags are False vs existing unoptimized path
    - File: `src/exo/worker/engines/pytorch_xpu/tests/test_output_equivalence_properties.py`
    - **Validates: Requirements 11.2**

  - [x]* 15.4 Write integration tests for compiled decode end-to-end
    - Run 10 decode steps through the compiled path, verify coherent output
    - Test GatedDeltaNet static state (no allocation during decode)
    - Test tokenizer decode ordering (forward launches before decode)
    - File: `src/exo/worker/engines/pytorch_xpu/tests/test_compiled_decode_integration.py`
    - _Requirements: 4.2, 2.5, 1.4, 9.1_

- [x] 16. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- All fused kernels rely on `torch.compile()` Inductor auto-fusion rather than hand-written SYCL kernels
- The implementation language is Python (matching the existing codebase and design document)
- Tests run on CPU (mocking XPU device) for CI; hardware-gated tests use `@pytest.mark.xpu`
- The `hypothesis` library with `hypothesis[numpy]` is used for property-based test generation

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3"] },
    { "id": 2, "tasks": ["2.1", "4.1"] },
    { "id": 3, "tasks": ["2.2", "2.3", "4.2", "4.3"] },
    { "id": 4, "tasks": ["5.1"] },
    { "id": 5, "tasks": ["5.2", "8.1"] },
    { "id": 6, "tasks": ["6.1", "8.2", "9.1"] },
    { "id": 7, "tasks": ["6.2", "9.2", "10.1"] },
    { "id": 8, "tasks": ["10.2", "12.1"] },
    { "id": 9, "tasks": ["12.2", "13.1"] },
    { "id": 10, "tasks": ["13.2", "13.3"] },
    { "id": 11, "tasks": ["15.1"] },
    { "id": 12, "tasks": ["15.2", "15.3", "15.4"] }
  ]
}
```

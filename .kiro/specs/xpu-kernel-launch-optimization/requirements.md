# Requirements Document

## Introduction

Optimize GPU kernel utilization on Intel Arc Meteor Lake-P iGPUs to increase LLM inference throughput from ~2 tok/s to 10+ tok/s on a single node running Qwen3.5-4B at bf16. Profiling shows the GPU is idle 48% of the time due to kernel launch gaps — Python interpreter overhead between GPU kernel dispatches starves the GPU. This spec addresses the CPU-side dispatch bottleneck through synchronization removal, kernel fusion, graph compilation, and on-device execution of the hot path.

## Glossary

- **Decode_Loop**: The autoregressive token generation loop in `pipeline_generator.py` that iterates once per output token, calling `model.forward()` and sampling.
- **Kernel_Launch_Gap**: The idle time between consecutive GPU kernel executions caused by CPU-side Python interpreter overhead (object allocation, GC, dynamic dispatch).
- **Implicit_Synchronization**: Any operation that forces the CPU to wait for GPU completion before proceeding (e.g., `.item()`, `.cpu()`, `.numpy()`, tensor truthiness checks).
- **Static_KV_Cache**: A pre-allocated, fixed-size tensor for key/value cache storage that avoids per-token dynamic memory allocation and the associated GPU synchronization.
- **Dynamic_Cache**: The HuggingFace `DynamicCache` class that grows its internal tensors on each decode step, triggering memory allocation and potential GPU stalls.
- **Torch_Compile**: PyTorch's `torch.compile()` API that traces Python code into a fused GPU kernel graph, eliminating per-op Python dispatch overhead.
- **XPU_Inductor_Backend**: The PyTorch Inductor code generation backend targeting Intel XPU devices via SYCL/Level Zero.
- **Graph_Break**: A point where `torch.compile()` cannot continue tracing and falls back to eager Python execution, splitting the compiled graph into multiple smaller kernels.
- **Fused_Kernel**: A single GPU kernel that combines multiple pointwise operations (e.g., RMSNorm + residual, SiLU * gate) to reduce launch overhead and memory traffic.
- **Packed_QKV_Projection**: A single matrix multiplication that computes Q, K, and V projections simultaneously from a concatenated weight matrix, replacing three separate matmuls.
- **On_Device_Sampling**: Performing argmax or multinomial sampling on the XPU device without transferring logits to CPU.
- **Pipeline_Parallel_Shard**: The `PipelineParallelShard` class that wraps a contiguous range of transformer layers for pipeline-parallel inference.
- **Hot_Path**: The per-token decode path executed once per generated token — the critical performance path where zero CPU-GPU synchronization is required.
- **Decode_Throughput**: Tokens generated per second during the autoregressive decode phase (excludes prefill).
- **GPU_Idle_Percentage**: The fraction of time samples where the GPU reports zero utilization, measured via `intel_gpu_top` or sysfs frequency monitoring.

## Requirements

### Requirement 1: Remove Implicit Synchronization from Decode Hot Path

**User Story:** As a system operator, I want the decode loop to execute without any CPU-GPU synchronization points, so that the GPU never stalls waiting for the CPU between kernel launches.

#### Acceptance Criteria

1. WHILE the Decode_Loop is executing, THE Pipeline_Parallel_Shard SHALL NOT call `.item()`, `.cpu()`, `.numpy()`, or any other Implicit_Synchronization operation on XPU tensors.
2. WHILE the Decode_Loop is executing, THE Pipeline_Parallel_Shard SHALL NOT evaluate tensor values in Python boolean contexts (e.g., `if tensor:`, `len(tensor)`).
3. WHEN a decode step completes, THE Pipeline_Parallel_Shard SHALL return the output tensor on the XPU device without transferring it to CPU.
4. WHEN the Decode_Loop produces a token, THE Decode_Loop SHALL defer tokenizer decode operations until after the next forward pass has been launched.

### Requirement 2: Static KV Cache Allocation

**User Story:** As a system operator, I want the KV cache to be pre-allocated at generation start, so that no dynamic memory allocation occurs during the decode hot path.

#### Acceptance Criteria

1. WHEN a generation request begins, THE Static_KV_Cache SHALL pre-allocate storage for the maximum sequence length on the XPU device.
2. WHILE the Decode_Loop is executing, THE Static_KV_Cache SHALL update cache entries in-place without allocating new tensors.
3. THE Static_KV_Cache SHALL maintain a position index that advances by one per decode step without requiring tensor reallocation.
4. WHEN a generation request completes, THE Static_KV_Cache SHALL reset its position index to zero for reuse by the next request.
5. THE Static_KV_Cache SHALL support the GatedDeltaNet linear attention layers' conv_state and recurrent_state storage without dynamic allocation.

### Requirement 3: Tensor-Only Decode Wrapper

**User Story:** As a system operator, I want the decode forward pass to operate exclusively on tensors without Python object mutation, so that `torch.compile()` can trace the entire decode path without Graph_Breaks.

#### Acceptance Criteria

1. THE Decode_Loop SHALL pass all per-step state (position index, cache tensors, token ID) as tensor arguments rather than Python object attributes.
2. THE Decode_Loop SHALL NOT mutate Python objects (lists, dicts, dataclass fields) between the forward call and the next token embedding lookup.
3. WHEN `torch.compile()` traces the decode forward pass, THE compiled graph SHALL contain zero Graph_Breaks for the transformer layer stack.
4. IF a Graph_Break is detected during compilation, THEN THE system SHALL log the break location and reason at WARNING level.

### Requirement 4: torch.compile() with XPU Inductor Backend

**User Story:** As a system operator, I want the decode forward pass compiled into a single fused kernel graph via `torch.compile()`, so that Python dispatch overhead is eliminated from the hot path.

#### Acceptance Criteria

1. WHEN the model is loaded, THE system SHALL compile the decode forward function using `torch.compile(backend="inductor")` targeting the XPU device.
2. THE compiled decode function SHALL execute the full transformer layer stack (32 layers for Qwen3.5-4B) in a single compiled graph invocation.
3. WHILE the compiled decode function executes, THE XPU_Inductor_Backend SHALL NOT fall back to eager execution for any operation within the layer stack.
4. IF `torch.compile()` fails for the XPU_Inductor_Backend, THEN THE system SHALL fall back to eager execution and log the failure reason at ERROR level.
5. WHEN the compiled function is first invoked, THE system SHALL accept a one-time compilation warmup latency without counting it toward Decode_Throughput measurements.

### Requirement 5: Packed QKV and Gate/Up Projections

**User Story:** As a system operator, I want Q/K/V projections and gate/up projections fused into single matrix multiplications, so that the number of kernel launches per layer is reduced.

#### Acceptance Criteria

1. WHEN the model weights are loaded, THE system SHALL detect separate Q, K, V projection weights and pack them into a single concatenated weight matrix.
2. THE packed QKV projection SHALL compute Q, K, and V outputs from a single `torch.matmul` call, splitting the result tensor afterward.
3. WHEN the model weights are loaded, THE system SHALL detect separate gate and up projection weights and pack them into a single concatenated weight matrix.
4. THE packed gate/up projection SHALL compute both outputs from a single `torch.matmul` call, splitting the result tensor afterward.
5. FOR ALL valid input tensors, THE packed projection outputs SHALL be numerically equivalent to the separate projection outputs (within bf16 precision tolerance of 1e-2 relative error).

### Requirement 6: Fused Pointwise Kernels

**User Story:** As a system operator, I want RMSNorm, SiLU activation, and rotary position embedding fused into single kernels, so that memory bandwidth and launch overhead are reduced.

#### Acceptance Criteria

1. THE system SHALL provide a fused RMSNorm + residual addition kernel that computes both operations in a single GPU launch.
2. THE system SHALL provide a fused SiLU-gating kernel that computes `SiLU(gate) * up` in a single GPU launch.
3. THE system SHALL provide a fused rotary position embedding kernel that applies rotation in-place without allocating intermediate tensors.
4. FOR ALL valid input tensors, THE fused kernel outputs SHALL be numerically equivalent to the sequential unfused operations (within bf16 precision tolerance of 1e-2 relative error).
5. IF `torch.compile()` successfully fuses these operations automatically, THEN THE system SHALL use the compiler-generated fusions instead of hand-written kernels.

### Requirement 7: Fused Decode Attention Kernel

**User Story:** As a system operator, I want a single fused kernel for the decode attention computation (Q×K^T scaling, masking, softmax, V multiply), so that the 8 full-attention layers each execute as one kernel instead of multiple.

#### Acceptance Criteria

1. WHEN a single-token decode step executes a full-attention layer, THE Fused_Kernel SHALL compute the complete attention operation (score, scale, mask, softmax, value multiply) in a single GPU kernel launch.
2. THE fused decode attention kernel SHALL read K and V directly from the Static_KV_Cache without copying to intermediate buffers.
3. FOR ALL valid query/key/value inputs, THE fused attention output SHALL be numerically equivalent to the unfused multi-step attention (within bf16 precision tolerance of 1e-2 relative error).
4. THE fused decode attention kernel SHALL support the head dimensions and group counts used by Qwen3.5-4B (hidden_size=2560, num_heads=32, num_kv_heads=4).

### Requirement 8: On-Device Sampling

**User Story:** As a system operator, I want token sampling (argmax/top-k/top-p) to execute on the XPU device, so that logits are never transferred to CPU during the decode hot path.

#### Acceptance Criteria

1. WHEN the Decode_Loop samples the next token, THE On_Device_Sampling SHALL perform argmax on the XPU device and return the token ID as an XPU tensor.
2. WHEN top-k sampling is requested, THE On_Device_Sampling SHALL perform the top-k filtering and multinomial sampling entirely on the XPU device.
3. WHEN top-p sampling is requested, THE On_Device_Sampling SHALL perform the nucleus filtering and multinomial sampling entirely on the XPU device.
4. THE On_Device_Sampling SHALL transfer only the single integer token ID to CPU (for EOS checking and inter-rank communication), not the full logits tensor.
5. THE token ID transfer to CPU SHALL occur asynchronously, overlapped with the launch of the next decode step's embedding lookup.

### Requirement 9: Async Output Streaming

**User Story:** As a system operator, I want token output (decode to text, yield to API) to happen asynchronously without blocking the next decode step, so that the GPU is never idle waiting for output processing.

#### Acceptance Criteria

1. WHEN a token is sampled, THE Decode_Loop SHALL launch the next forward pass before performing tokenizer decode on the previous token.
2. THE Decode_Loop SHALL NOT block on tokenizer string decoding between consecutive forward passes.
3. WHEN the generation response is yielded to the API layer, THE Decode_Loop SHALL NOT wait for the API to consume the response before continuing.
4. IF the output queue exceeds 32 pending tokens, THEN THE Decode_Loop SHALL apply backpressure by pausing generation until the queue drains below 16 tokens.

### Requirement 10: Performance Targets

**User Story:** As a system operator, I want the optimized decode path to achieve 10+ tok/s on a single Meteor Lake-P node with Qwen3.5-4B at bf16, so that the system delivers usable interactive inference speed.

#### Acceptance Criteria

1. WHEN running Qwen3.5-4B at bf16 on a single Intel Core Ultra 9 185H node, THE optimized Decode_Loop SHALL achieve a sustained Decode_Throughput of at least 10 tokens per second.
2. WHILE the Decode_Loop is executing at steady state, THE GPU_Idle_Percentage SHALL be less than 15%.
3. THE optimized Decode_Loop SHALL achieve zero per-token CPU-GPU synchronization points on the Hot_Path (measured by absence of `xpu.synchronize()` calls between forward passes).
4. WHEN measured over a 50-token generation, THE Decode_Throughput SHALL remain stable (coefficient of variation less than 20% across token latencies after warmup).

### Requirement 11: Backward Compatibility

**User Story:** As a system operator, I want the optimizations to be opt-in and not break existing pipeline-parallel inference, so that I can enable them incrementally and fall back if needed.

#### Acceptance Criteria

1. THE system SHALL provide a configuration flag to enable or disable each optimization independently (sync removal, static cache, torch.compile, packed projections, fused kernels, on-device sampling, async output).
2. WHEN all optimization flags are disabled, THE system SHALL produce identical output to the current unoptimized pipeline at the same temperature and seed.
3. IF an optimization fails at runtime (e.g., torch.compile graph break, kernel launch failure), THEN THE system SHALL fall back to the unoptimized code path and log the fallback at WARNING level.
4. THE optimization configuration SHALL be specified in the `PytorchXpuOptimizationConfiguration` dataclass alongside existing optimization flags.

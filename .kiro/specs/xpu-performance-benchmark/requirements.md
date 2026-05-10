# Requirements Document

## Introduction

Performance testing and optimization feature for Qwen3.5-4B inference on Intel Arc iGPUs (Meteor Lake-P). This feature provides a standalone benchmark script for measuring single-node inference performance on gremlin-1, identifies and removes decode loop inefficiencies, and explores advanced optimizations (torch.compile, dtype selection, SDPA backend). The goal is to achieve interactive chat speeds (>5 tok/s decode) on the Intel Arc iGPU with shared DDR5 memory.

## Glossary

- **Benchmark_Script**: A standalone Python script (`bench_xpu.py`) that loads a model, runs warmup iterations, and measures inference timing with proper device synchronization.
- **TTFT**: Time To First Token — the wall-clock time from prompt submission to the first generated token being available, including model prefill.
- **Decode_TPS**: Decode Tokens Per Second — the sustained rate of token generation during the autoregressive decode phase, excluding the prefill step.
- **Prefill**: The forward pass over the full input prompt that populates the KV cache before autoregressive decoding begins.
- **XPU_Device**: An Intel Arc Graphics accelerator accessed via `torch.xpu`, specifically the Meteor Lake-P iGPU on gremlin nodes.
- **Device_Sync**: A call to `torch.xpu.synchronize()` that blocks until all previously submitted XPU operations complete, required for accurate timing on accelerators.
- **Greedy_Decoding**: Token selection via argmax (temperature=0, no sampling) for deterministic, reproducible benchmark results.
- **DynamicCache**: HuggingFace's `transformers.DynamicCache` object used by Qwen3.5 models to store key-value state across autoregressive steps.
- **Decode_Loop**: The autoregressive generation loop that produces one token per iteration by running a forward pass on the previous token and sampling the next.
- **Torch_Compile**: PyTorch's `torch.compile()` API that applies graph-level optimizations (kernel fusion, memory planning) via the inductor backend.
- **SDPA**: Scaled Dot-Product Attention — PyTorch's `torch.nn.functional.scaled_dot_product_attention` which dispatches to optimized kernels when available.

## Requirements

### Requirement 1: Standalone Benchmark Script

**User Story:** As a developer, I want a standalone benchmark script that measures Qwen3.5-4B inference performance on XPU, so that I can establish baseline numbers and track optimization progress.

#### Acceptance Criteria

1. THE Benchmark_Script SHALL accept command-line arguments for model_id, device, dtype, prompt_tokens, gen_tokens, and warmup count.
2. WHEN invoked, THE Benchmark_Script SHALL load the specified model onto the XPU_Device in the specified dtype (bf16 by default).
3. WHEN the model is loaded, THE Benchmark_Script SHALL run the specified number of warmup generations before timing measurements begin.
4. WHEN measuring prefill time, THE Benchmark_Script SHALL call Device_Sync before and after the prefill forward pass to ensure accurate wall-clock timing.
5. WHEN measuring decode throughput, THE Benchmark_Script SHALL call Device_Sync before the decode loop starts and after the decode loop completes, then compute Decode_TPS as gen_tokens divided by elapsed wall-clock seconds.
6. THE Benchmark_Script SHALL use Greedy_Decoding (argmax, no sampling) for all timed measurements to ensure deterministic results.
7. THE Benchmark_Script SHALL construct a fixed-length prompt of approximately the specified prompt_tokens count for reproducibility across runs.
8. WHEN all measurements complete, THE Benchmark_Script SHALL print a structured report containing: TTFT in seconds, prefill tokens per second, Decode_TPS, total generation time, and device/dtype metadata.
9. THE Benchmark_Script SHALL use `torch.inference_mode()` context for all forward passes to disable gradient tracking and autograd overhead.
10. THE Benchmark_Script SHALL be deployable to gremlin-1 via the existing NixOS deployment and runnable via SSH without additional setup.

### Requirement 2: Decode Loop Optimization

**User Story:** As a developer, I want the decode loop to avoid unnecessary overhead, so that token generation throughput is maximized on the XPU_Device.

#### Acceptance Criteria

1. THE Decode_Loop SHALL NOT use `torch.cat` to grow the input_ids tensor on each step when a KV cache is in use, because only the last token ID is needed as input.
2. THE Decode_Loop SHALL keep generated token IDs in a CPU-side Python list and only move the single current token to the XPU_Device for each forward pass.
3. THE Decode_Loop SHALL NOT call `tokenizer.decode()` on every generated token during timed benchmark runs; decoding SHALL occur only after the full generation completes or when yielding for streaming output.
4. WHEN the Decode_Loop detects an EOS token via `.item()`, THE Decode_Loop SHALL terminate generation immediately.
5. THE Decode_Loop SHALL reuse the DynamicCache object across decode steps without recreating it each iteration.

### Requirement 3: Performance Targets

**User Story:** As a developer, I want defined performance targets for single-node inference, so that I can evaluate whether optimizations are sufficient for interactive use.

#### Acceptance Criteria

1. THE Benchmark_Script SHALL report whether Decode_TPS meets the minimum target of 5 tokens per second for interactive chat viability.
2. THE Benchmark_Script SHALL report whether TTFT meets the target of less than 5 seconds for a 256-token prompt.
3. THE Benchmark_Script SHALL report whether Decode_TPS meets the stretch target of 10 tokens per second.
4. IF Decode_TPS is below 2 tokens per second, THEN THE Benchmark_Script SHALL flag the result as "critically slow — investigate kernel fallbacks or memory bandwidth issues."

### Requirement 4: Advanced Optimization Investigation

**User Story:** As a developer, I want to investigate advanced optimizations (torch.compile, dtype, SDPA backends), so that I can determine which techniques yield measurable speedups on Intel Arc iGPU.

#### Acceptance Criteria

1. THE Benchmark_Script SHALL support a `--compile` flag that applies `torch.compile(backend="inductor", mode="reduce-overhead")` to the model before benchmarking.
2. WHEN `--compile` is enabled, THE Benchmark_Script SHALL report whether compilation succeeded or fell back to eager mode, and include this in the output report.
3. THE Benchmark_Script SHALL support a `--dtype` argument accepting `bf16` and `fp16` values to compare numeric precision performance.
4. THE Benchmark_Script SHALL support a `--report-sdpa` flag that queries and reports which SDPA backend is active (flash, math, or efficient) for the current device and model configuration.
5. WHEN `--report-sdpa` is enabled, THE Benchmark_Script SHALL run a diagnostic attention operation and report the kernel backend used by `scaled_dot_product_attention` on the XPU_Device.

### Requirement 5: Benchmark Reproducibility

**User Story:** As a developer, I want benchmark results to be reproducible across runs, so that I can reliably compare before/after optimization measurements.

#### Acceptance Criteria

1. THE Benchmark_Script SHALL set a fixed random seed (default: 42) before model loading and generation.
2. THE Benchmark_Script SHALL use Greedy_Decoding to eliminate sampling variance between runs.
3. THE Benchmark_Script SHALL report the PyTorch version, XPU device name, and driver version in the output for environment traceability.
4. THE Benchmark_Script SHALL run multiple timed iterations (default: 3) and report mean and standard deviation for both TTFT and Decode_TPS.
5. WHEN variance between iterations exceeds 20% of the mean, THE Benchmark_Script SHALL emit a warning indicating unstable measurements and suggest increasing warmup count.

### Requirement 6: Integration with Existing Deployment

**User Story:** As a developer, I want the benchmark script to integrate with the existing exo deployment on gremlin-1, so that I can run benchmarks without modifying the production service.

#### Acceptance Criteria

1. THE Benchmark_Script SHALL be located in the exo source tree and included in the NixOS deployment package.
2. THE Benchmark_Script SHALL import model loading utilities from the existing `model_loader` module to ensure consistent model initialization.
3. THE Benchmark_Script SHALL operate independently of the exo service (no dependency on the running exo systemd unit).
4. THE Benchmark_Script SHALL detect and report the available XPU devices before running, and exit with a clear error message if no XPU_Device is available.
5. IF the specified model is not cached locally, THEN THE Benchmark_Script SHALL download it using the same HuggingFace cache path used by the exo service.

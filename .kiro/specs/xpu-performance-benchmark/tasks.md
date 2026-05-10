# Implementation Plan: XPU Performance Benchmark

## Overview

Implement a standalone benchmark script for measuring Qwen3.5-4B inference performance on Intel Arc iGPUs, with property-based tests for pure logic, an optimized decode loop, and structured performance reporting. The implementation follows a phased approach: local testable logic first, then deployment and measurement on gremlin-1.

## Tasks

- [x] 1. Implement benchmark pure logic and data models
  - [x] 1.1 Create `src/exo/worker/engines/pytorch_xpu/bench_xpu.py` with dataclasses and pure functions
    - Define `BenchmarkConfig`, `BenchmarkResult`, `BenchmarkStats`, `PerformanceClassification`, `EnvironmentInfo` dataclasses
    - Implement `parse_args()` using argparse with all CLI arguments from design
    - Implement `build_prompt(tokenizer, target_tokens)` that constructs fixed-length prompts
    - Implement `compute_stats(results)` for mean/stddev calculation with instability detection
    - Implement `classify_performance(decode_tps, ttft)` for threshold classification
    - Implement `format_report(stats, config, env)` for structured output
    - _Requirements: 1.1, 1.7, 1.8, 3.1, 3.2, 3.3, 3.4, 5.3, 5.4, 5.5_

  - [x] 1.2 Write property tests for CLI parsing (Property 1)
    - **Property 1: CLI argument parsing accepts valid configurations**
    - **Validates: Requirements 1.1**
    - Test file: `src/exo/worker/engines/pytorch_xpu/tests/test_bench_xpu_properties.py`
    - Generate valid arg combinations via Hypothesis, verify BenchmarkConfig fields match

  - [x] 1.3 Write property tests for TPS calculation (Property 2)
    - **Property 2: Decode TPS calculation**
    - **Validates: Requirements 1.5**
    - For any positive gen_tokens and elapsed_seconds, verify decode_tps == gen_tokens / elapsed_seconds

  - [x] 1.4 Write property tests for greedy argmax (Property 3)
    - **Property 3: Greedy decoding selects argmax**
    - **Validates: Requirements 1.6, 5.2**
    - Generate random logits tensors with unique max, verify argmax selection

  - [x] 1.5 Write property tests for prompt construction (Property 4)
    - **Property 4: Prompt construction produces target token count**
    - **Validates: Requirements 1.7**
    - Generate target counts 1–4096, verify token tensor length equals target

  - [x] 1.6 Write property tests for report completeness (Property 5)
    - **Property 5: Report contains all required fields**
    - **Validates: Requirements 1.8, 5.3**
    - Generate valid stats/config/env, verify report contains all required substrings

  - [x] 1.7 Write property tests for performance classification (Properties 8, 9)
    - **Property 8: Performance threshold classification**
    - **Property 9: TTFT threshold classification**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
    - Generate TPS/TTFT values, verify threshold flags

  - [x] 1.8 Write property tests for statistics computation (Properties 10, 11)
    - **Property 10: Statistical computation correctness**
    - **Property 11: Instability warning fires on high variance**
    - **Validates: Requirements 5.4, 5.5**
    - Generate measurement lists, verify mean/stddev and CV threshold

- [x] 2. Implement optimized decode loop and generation logic
  - [x] 2.1 Implement `greedy_decode()` function in `bench_xpu.py`
    - Single-token input with KV cache reuse (no torch.cat)
    - Token IDs accumulated in CPU-side Python list
    - No tokenizer.decode() during timed loop
    - EOS detection via .item() with immediate termination
    - torch.xpu.synchronize() bracketing for accurate timing
    - torch.inference_mode() context
    - _Requirements: 1.4, 1.5, 1.6, 1.9, 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 2.2 Write property tests for decode loop equivalence (Property 6)
    - **Property 6: Optimized decode loop produces identical tokens to naive implementation**
    - **Validates: Requirements 2.2**
    - Mock model outputs, verify optimized loop produces same token sequence as naive

  - [x] 2.3 Write property tests for EOS termination (Property 7)
    - **Property 7: EOS terminates decode at correct position**
    - **Validates: Requirements 2.4**
    - Mock model producing EOS at step N, verify exactly N tokens returned

- [x] 3. Implement benchmark orchestration and entry point
  - [x] 3.1 Implement `run_benchmark_iteration()` in `bench_xpu.py`
    - Load model onto XPU device in specified dtype
    - Run warmup iterations (untimed)
    - Run N timed iterations with sync-bracketed measurements
    - Compute BenchmarkResult for each iteration
    - _Requirements: 1.2, 1.3, 1.4, 1.5, 1.9_

  - [x] 3.2 Implement device detection and environment info collection
    - Detect XPU devices, exit with clear error if none available
    - Collect PyTorch version, XPU device name, driver version
    - Report available devices before running
    - _Requirements: 5.3, 6.4_

  - [x] 3.3 Implement `--compile` flag support
    - Apply `torch.compile(backend="inductor", mode="reduce-overhead")` when enabled
    - Report success or fallback to eager mode
    - _Requirements: 4.1, 4.2_

  - [x] 3.4 Implement `--report-sdpa` flag support
    - Run diagnostic SDPA operation on XPU device
    - Report which backend is active (flash, math, efficient)
    - _Requirements: 4.4, 4.5_

  - [x] 3.5 Implement `main()` entry point wiring all components together
    - Parse args → detect devices → set seed → load model → warmup → bench → report
    - Handle errors per design error handling table
    - Set random seed before model loading
    - Use `torch.inference_mode()` for all forward passes
    - _Requirements: 1.9, 1.10, 5.1, 5.2, 6.1, 6.3_

- [x] 4. Checkpoint - Verify local tests pass
  - Ensure all property tests pass locally with: `LD_LIBRARY_PATH="/nix/store/cf1a53iqg6ncnygl698c4v0l8qam5a2q-gcc-14.3.0-lib/lib:$LD_LIBRARY_PATH" uv run --no-sync pytest src/exo/worker/engines/pytorch_xpu/tests/test_bench_xpu_properties.py -v --tb=short`
  - Ask the user if questions arise.

- [ ] 5. Deploy and run baseline benchmark on gremlin-1
  - [-] 5.1 Deploy benchmark script to gremlin-1
    - Push to xpu branch, run `bash force_update_gremlin1.sh`
    - Verify bench_xpu.py is accessible on gremlin-1
    - _Requirements: 1.10, 6.1, 6.2_

  - [~] 5.2 Run baseline benchmark and record results
    - SSH to gremlin-1 and run: `python -m exo.worker.engines.pytorch_xpu.bench_xpu --model_id Qwen/Qwen3.5-4B --device xpu:0 --gen_tokens 128 --iterations 3`
    - Record baseline TTFT and decode tok/s
    - Verify structured report output contains all required fields
    - _Requirements: 1.8, 3.1, 3.2, 5.3, 5.4_

- [ ] 6. Optimize exo generator decode loop
  - [~] 6.1 Refactor `src/exo/worker/engines/pytorch_xpu/generator.py` decode loop
    - Remove `torch.cat` for growing input_ids (use single token + KV cache)
    - Keep generated token IDs in CPU-side Python list
    - Defer `tokenizer.decode()` to after generation or streaming yield points
    - Reuse DynamicCache across decode steps
    - _Requirements: 2.1, 2.2, 2.3, 2.5_

  - [~] 6.2 Re-deploy and re-measure after decode loop optimization
    - Push to xpu branch, run `bash force_update_gremlin1.sh`
    - Run benchmark with same parameters as baseline
    - Compare decode tok/s before and after optimization
    - _Requirements: 3.1, 3.3_

- [~] 7. Checkpoint - Verify optimization results
  - Ensure all tests pass, ask the user if questions arise.
  - Compare baseline vs optimized decode TPS numbers.

- [ ] 8. Advanced optimizations (conditional on Phase 3 results)
  - [~] 8.1 Test torch.compile on XPU
    - Run benchmark with `--compile` flag on gremlin-1
    - Record whether compilation succeeds or falls back to eager
    - Compare performance with and without compile
    - _Requirements: 4.1, 4.2_

  - [~] 8.2 Test bf16 vs fp16 dtype comparison
    - Run benchmark with `--dtype bf16` and `--dtype fp16`
    - Compare decode TPS and TTFT between dtypes
    - _Requirements: 4.3_

  - [~] 8.3 Investigate SDPA backend
    - Run benchmark with `--report-sdpa` flag
    - Document which SDPA backend is active on Intel Arc
    - _Requirements: 4.4, 4.5_

- [~] 9. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
  - Document final performance numbers and optimization recommendations.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Property tests (tasks 1.2–1.8, 2.2–2.3) validate pure logic functions locally without XPU hardware
- Tasks 5–8 require deployment to gremlin-1 (10.1.1.12) with real XPU hardware
- The benchmark script operates independently of the exo service
- Run local tests with: `LD_LIBRARY_PATH="/nix/store/cf1a53iqg6ncnygl698c4v0l8qam5a2q-gcc-14.3.0-lib/lib:$LD_LIBRARY_PATH" uv run --no-sync pytest src/exo/worker/engines/pytorch_xpu/tests/test_bench_xpu_properties.py -v --tb=short`
- Deploy with: push to xpu branch → `bash force_update_gremlin1.sh`

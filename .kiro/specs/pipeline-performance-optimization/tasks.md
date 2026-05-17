# Tasks: Qwen3.5-27B Pipeline-Parallel XPU Inference Optimization

Tasks are ordered by dependency and expected implementation priority.

---

## Task 1: Performance Instrumentation

- [x] Add `src/exo/worker/engines/pytorch_xpu/instrumentation.py` with `PerformanceEvent`, `PerformanceSummary`, `PerformanceRecorder`, context-manager span API using `time.perf_counter()`, and JSON serialization
- [x] Add instrumentation configuration to `src/exo/worker/engines/pytorch_xpu/pipeline_config.py` with `PytorchXpuOptimizationConfiguration` using Pydantic `ConfigDict(frozen=True)`, including flags for detailed tracing and XPU synchronization timing
- [x] Instrument `pipeline_generator.py`: measure prefill wall time, decode wall time, scheduler wait time placeholder, and end-to-end token latency
- [x] Instrument `pipeline_parallel_shard.py`: measure per-stage compute time, per-layer compute time split by GatedDeltaNet vs full-attention, final normalization time, and `lm_head` time on rank 3
- [x] Instrument `distributed.py`: measure activation send time, activation receive time, metadata send time, metadata receive time, and token synchronization time
- [x] Instrument `distributed_generator.py`: measure logits processing time, filtering time, random selection time, and token return time
- [x] Instrument `gated_deltanet.py`: measure recurrent decode step time, prefill sequential time, and add placeholder metrics for chunked prefill
- [x] Update `bench_xpu.py`: add `--json-output-path`, add benchmark modes for single-request-decode, prefill, and continuous-batching, print tokens per second, TTFT, inter-token latency (mean/median/p95), per-rank utilization, communication time ratio, and pipeline bubble estimate
- [x] Add unit tests: `PerformanceRecorder` span duration is nonnegative, JSON serialization round trip, counters aggregate correctly, instrumentation disabled mode has minimal behavior
- [x] Run existing system before optimizations and store baseline metrics

### Dependencies
None. This task must be completed first.

---

## Task 2: Decode Communication Fast Path

- [x] Extend `distributed.py` with `DecodeActivationProtocol`, `TokenResultPacket`, `DecodeProtocolMismatchError`, and `PipelineCommunicationError` types
- [x] Add `src/exo/worker/engines/pytorch_xpu/buffer_pool.py` with `CommunicationBufferPool` implementing reusable CPU send/receive buffers keyed by name, dtype, shape, source rank, and destination rank
- [x] Implement protocol negotiation: `negotiate_decode_activation_protocol(...)` exchanging metadata once during decode initialization with protocol version and hidden size validation
- [x] Implement `send_decode_activation_fast(...)` and `receive_decode_activation_fast(...)` requiring contiguous tensors, using preallocated buffers, and avoiding per-token shape metadata
- [x] Keep existing generic path renamed for clarity, used for prefill, fallback, and debugging
- [x] Replace blocking token broadcast: add `send_token_results_to_rank_zero(...)` and `receive_token_results_from_final_rank(...)` using point-to-point Gloo send from rank 3 to rank 0 without requiring ranks 1 and 2 to participate
- [x] Update `pipeline_generator.py` to negotiate decode protocols after prefill and use fast path for decode activations when enabled
- [x] Update `pipeline_parallel_shard.py` to ensure decode activation shapes are stable and validated before fast send
- [x] Add instrumentation counters: fast-path activation messages, generic activation messages, shape metadata messages, protocol renegotiations, fast-path fallbacks
- [x] Add unit tests: protocol accepts matching tensors, rejects mismatched shapes, buffer pool reuses storage, token packet serialization
- [x] Add distributed integration test: verify fast path sends no per-token shape metadata after warm-up, verify rank 3 to rank 0 token result path works without ranks 1 and 2 waiting

### Dependencies
Depends on Task 1 instrumentation.

---

## Task 3: Fast Sampling

- [x] Refactor sampling in `distributed_generator.py`: keep public `sample_token(...)` compatibility, route to specialized implementations based on sampling configuration
- [x] Add `SamplingConfiguration` Pydantic model with `do_sample`, `top_k`, `top_p`, `temperature`, and `validate_logits` fields
- [x] Implement `sample_token_greedy(logits: torch.Tensor) -> torch.Tensor` using `torch.argmax` without calling `torch.sort`
- [x] Implement `sample_token_top_k(...)` using `torch.topk(logits, k=top_k)`, applying temperature, using `torch.multinomial` on top-k probability vector, and mapping back to vocabulary token identifier
- [x] Preserve fallback path for unsupported combinations (top-p with no top-k), make fallback explicit in instrumentation
- [x] Move full-logit NaN/infinity validation behind `validate_logits` flag, disabled by default in performance mode
- [x] Support batched per-request sampling: accept logits shaped `[batch, vocabulary]` with per-request `SamplingConfiguration`
- [x] Add instrumentation: count greedy/top-k/fallback samples, record sampling latency
- [x] Add tests: greedy returns argmax, top-k returns only tokens from `torch.topk`, top-k with `k=1` equals greedy, fallback matches old implementation under fixed seed, validation flag controls NaN checking

### Dependencies
Depends on Task 1 instrumentation. Can run in parallel with Task 2.

---

## Task 4: GatedDeltaNet State Optimization

- [x] Inspect current `gated_deltanet.py`: identify all per-token casts from bf16 to fp32, per-token recurrent state allocations, and output tensor allocations in decode path
- [x] Add `GatedDeltaNetPersistentState` container storing request identifier, layer index, fp32 recurrent tensors, device, shape metadata, and reset/recycle method
- [x] Integrate persistent state with `DynamicCache`: ensure state remains accessible through `past_key_values` (plural keyword), do not introduce singular `past_key_value`
- [x] Implement `initialize_gated_deltanet_state(...)`: initialize fp32 state once per request and layer on correct XPU device
- [x] Implement `gated_deltanet_decode_recurrent_step(...)`: accept persistent fp32 state, avoid per-token dtype promotion, update state in place where safe, use preallocated output tensor when provided
- [x] Add output buffer reuse: allocate decode output buffers per layer and microbatch shape, reuse across decode steps, fall back safely on shape change
- [x] Update `pipeline_parallel_shard.py`: ensure per-request local cache is initialized before decode, pass request identifiers to GatedDeltaNet state manager, reset state on request completion or cancellation
- [x] Add compatibility flag `enable_gated_deltanet_persistent_state` (if disabled, use existing behavior)
- [x] Add instrumentation: count per-token casts avoided, record recurrent step time by layer, record state allocation count
- [x] Add unit tests: compare optimized recurrent step to baseline, verify persistent state dtype is fp32, verify state storage reused across decode steps, verify reset removes request state
- [x] Add integration test: run short prefill plus decode through full shard path, compare logits before and after optimization within tolerance (`rtol <= 2e-2`, `atol <= 2e-2`)

### Dependencies
Depends on Task 1 instrumentation. Benefits from Task 2 buffer lifecycle.

---

## Task 5: Local-Shard Model Loading

- [x] Define pipeline configuration in `pipeline_config.py`: add `PipelineStageAssignment` and `PipelineLayerDistribution` with validation for total layer count (64) and four ranks
- [x] Refactor `model_loader.py`: add `load_qwen_local_shard_from_safetensors(...)` with safetensors index parsing, tensor-name ownership resolver, and local tensor manifest creation
- [x] Implement tensor ownership: rank 0 owns token embedding, ranks own configured contiguous layer ranges, final rank owns final normalization and `lm_head`, handle tied/shared weights explicitly
- [x] Avoid full model materialization: remove or bypass code paths that call full-model loading on every rank, load tensor slices by name from safetensors, do not build layers outside local stage assignment
- [x] Build local shard modules: instantiate only local Qwen3.5 layers using layer type metadata to choose full-attention or linear-attention implementation, preserve global layer indices for rotary position and cache keys
- [x] Update `pipeline_parallel_shard.py`: accept `PipelineStageAssignment`, expose loaded layer range, assert only assigned layers are present
- [x] Update startup path in `engine.py`: create rank-local stage assignment from configuration, call local-shard loader on each rank, log loaded tensor manifest summary
- [x] Add strict startup validation: missing tensor names fail startup, duplicate owned tensors fail startup, unassigned layer tensors fail startup, unsupported layer type fails startup
- [x] Add tests: synthetic safetensors checkpoint with embedding/layers/norm/head, test each rank loads only owned tensors, test invalid distribution fails, test toy model pipeline output equals monolithic output
- [x] Add memory benchmark: record peak resident memory during model load, add output to `bench_xpu.py`

### Dependencies
Depends on Task 1 instrumentation. Should be completed before large-scale 27B benchmarking. Independent of Tasks 2-4 but must be reconciled with Task 4 cache initialization.

---

## Task 6: Pipeline Stage Balancing

- [x] Add stage distribution support to configuration: use `PipelineLayerDistribution`, allow command-line override in `bench_xpu.py`, validate contiguous layer ranges
- [x] Update `model_loader.py` to use configured distribution when resolving local tensor ownership
- [x] Update `pipeline_parallel_shard.py` to use configured layer range for local forward, preserving global layer indices
- [x] Add per-layer timing export using Task 1 instrumentation data, exporting decode and prefill timing separately
- [x] Implement `recommend_pipeline_layer_distribution(...)`: include fixed rank costs (rank 0 embedding, rank 3 norm + lm_head + sampling), preserve layer order, minimize max stage time
- [x] Add benchmark support: `bench_xpu.py --pipeline-layer-distribution`, `--recommend-layer-distribution`, comparison output for multiple distributions
- [x] Test distribution validation: reject sums not equal to total layer count, reject zero-layer stages, reject negative counts
- [x] Test recommendation algorithm: use synthetic timing arrays, verify recommendation reduces max stage time when possible, verify no invalid distribution produced
- [x] Run empirical comparison: baseline `[16,16,16,16]`, try `[17,17,16,14]`, try timing-recommended distribution

### Dependencies
Depends on Task 1 instrumentation. Depends on Task 5 local-shard loading. Benefits from Task 3 (sampling overhead affects final-rank balancing).

---

## Task 7: Continuous Batching

- [~] Add `src/exo/worker/engines/pytorch_xpu/continuous_batching.py` with `ContinuousBatchScheduler`, request admission queue, prefill queue, decode-ready queue, and completed/cancelled request tracking
- [~] Define state types: `RequestRuntimeState`, immutable `RequestState` snapshot, `BatchSlotState`, `DecodeMicrobatch`, `TokenResultBatch`
- [~] Add request identifier mapping: map string identifiers to numeric identifiers for compact distributed packets, keep reverse mapping on rank 0, include slot generation to prevent stale message corruption
- [~] Implement per-request cache management: store `DynamicCache` by request identifier, store GatedDeltaNet persistent state by request identifier and layer index, ensure batch position is not the only cache key
- [~] Update `engine.py`: add continuous-batching generation mode, admit requests while decode is active, return token streams per request, support cancellation
- [~] Update `pipeline_generator.py`: implement `run_continuous_decode(...)`, build microbatches from scheduler, track in-flight microbatches, feed rank 0 with last generated tokens for each request, receive final token results from rank 3
- [~] Update `pipeline_parallel_shard.py`: accept `DecodeMicrobatch`, gather per-request cache entries, update cache entries by request identifier, handle finished requests safely
- [~] Extend decode fast path for variable active batch membership: reuse protocols for maximum microbatch size, include active size in lightweight control message if required
- [~] Add per-request sampling: integrate Task 3 sampler, support different sampling configurations in same microbatch
- [~] Add cancellation handling: mark request as cancelling, stop scheduling new decode steps, drain or discard in-flight messages tagged with request identifier and slot generation, reset cache after drain
- [~] Add scheduler metrics: active requests, average microbatch size, pipeline occupancy, admission rate, completion rate, cancellation count
- [~] Add property-based tests: request isolation under random interleavings, batch slot reuse safety, cancellation safety, deterministic greedy equivalence to independent generation
- [~] Add distributed integration tests: run four small model shards, submit at least eight requests, verify all complete, verify per-request token order, verify aggregate throughput improves over single request
- [~] Add benchmark modes: `bench_xpu.py --benchmark-mode continuous-batching --concurrent-requests 4` and `--concurrent-requests 8`, report aggregate and per-request tokens per second

### Dependencies
Depends on Task 1, Task 2, Task 3, Task 4, and Task 5. Benefits from Task 6.

---

## Task 8: Chunked GatedDeltaNet Prefill

- [~] Study current sequential prefill in `gated_deltanet.py`: identify recurrence equations, state transition form, and numerical stability requirements
- [~] Define chunk transform representation: add internal type for chunk-level transform, represent state update as affine or WY-style transform, include dtype and shape validation
- [~] Add `ChunkedGatedDeltaNetPrefillConfiguration` with `enabled`, `chunk_size`, and `fallback_on_unsupported_shape` fields, add to top-level optimization configuration
- [~] Implement chunk-local computation: compute local recurrence parameters for tokens inside each chunk, produce chunk output activations, produce chunk transform summary
- [~] Implement chunk transform composition: compose transforms associatively, support arbitrary number of chunks, support final partial chunk
- [~] Implement state prefix propagation: compute incoming state for each chunk, materialize per-token outputs in original token order, materialize final recurrent state for decode continuation
- [~] Add fallback logic: detect unsupported dtype/shape/parameter combination, emit structured warning, use sequential prefill when fallback enabled, raise structured error when fallback disabled
- [~] Integrate with `pipeline_parallel_shard.py`: use chunked path during prefill for GatedDeltaNet layers when enabled, keep full-attention layers unchanged, ensure resulting state stored in `DynamicCache`
- [~] Add instrumentation: measure chunk-local computation time, transform composition time, output materialization time, compare with sequential prefill time
- [~] Add numerical tests: toy GatedDeltaNet sequential vs chunked for prompt lengths smaller than chunk size, equal to chunk size, not divisible by chunk size, and multiple chunk sizes
- [~] Add property-based tests: chunk boundary property, associative composition property, final state equivalence property, decode continuation property, fallback safety property
- [~] Add benchmark coverage: `bench_xpu.py --benchmark-mode prefill --chunked-gated-deltanet-prefill` for prompt lengths 128, 512, 1024, 2048, report TTFT

### Dependencies
Depends on Task 1 instrumentation. Depends on Task 4 GatedDeltaNet persistent state design. Benefits from Task 5 for real-model testing.

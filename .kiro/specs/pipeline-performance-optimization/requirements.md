# Requirements: Qwen3.5-27B Pipeline-Parallel XPU Inference Optimization

## Overview

This specification defines performance, correctness, and implementation requirements for optimizing Qwen3.5-27B inference on four Intel Arc Meteor Lake-P iGPU nodes named `gremlin-1` through `gremlin-4`.

The current system uses:

- PyTorch 2.11+xpu
- Gloo distributed backend over 10 Gbps Ethernet
- CPU-staged tensor communication
- Four pipeline stages, approximately sixteen transformer layers per node
- Qwen3.5-27B hybrid architecture:
  - 64 total layers
  - 16 full-attention layers
  - 48 GatedDeltaNet linear-attention layers
- DynamicCache-based recurrent state
- `past_key_values` keyword argument for cache/state propagation

The optimization target is:

- Single request decode: `0.5–1.0 tokens/second`
- Continuous batched aggregate decode: `1–3 tokens/second`
- Correctness-equivalent outputs to the current implementation within explicitly defined numerical tolerances

## Non-Goals

The following are out of scope:

- CUDA support
- IPEX dependency
- XCCL support
- Model architecture changes
- Quantization as a required optimization
- Replacing Gloo with a non-Gloo backend
- Requiring discrete GPU memory behavior; Intel iGPU shared-memory assumptions remain valid

## Global Correctness Requirements

All requirements in this document must preserve the following correctness properties:

1. The model must support Qwen3.5 hybrid layer routing using both `full_attention` and `linear_attention` layer types.
2. The inference path must continue to use `past_key_values` as the cache/state keyword argument.
3. GatedDeltaNet recurrent state must remain compatible with `DynamicCache`.
4. Pipeline partitioning must not change logits compared with a non-pipeline reference except for accepted floating-point tolerance.
5. Greedy sampling with identical input and deterministic settings must produce identical token sequences before and after optimization.
6. Continuous batching must not allow request state, token history, random sampling state, or recurrent state to leak across requests.
7. Optimized communication protocols must be safe for variable sequence lengths, request termination, cancellation, and node failure.

---

## Requirement 1: Local-Shard Model Loading

### User Story

As an operator running Qwen3.5-27B on four memory-constrained iGPU nodes, I want each rank to load only the layers and auxiliary modules assigned to its pipeline stage, so that startup succeeds without temporarily materializing the full 54 GB bf16 model on every node.

### Acceptance Criteria

1. Each rank must load only tensors required by its assigned pipeline stage.
2. The implementation must not call a full-model `from_pretrained()` path that materializes all 64 layers on every rank.
3. Loading must read tensors directly from safetensors files by tensor name.
4. Rank ownership must be deterministic and configurable.
5. Default four-stage ownership must be:
   - Rank 0: token embedding and assigned early layers
   - Rank 1: assigned middle layers
   - Rank 2: assigned middle layers
   - Rank 3: assigned final layers, final normalization, and `lm_head`
6. Layer assignment must support non-uniform stage sizes for balancing.
7. The model loader must support Qwen3.5 layer-type metadata and instantiate correct layer classes for:
   - `full_attention`
   - `linear_attention`
8. Peak resident memory during load on each node must be less than:
   - `1.35 × assigned_weight_bytes + 6 GiB`
   - measured after Python interpreter startup and before first inference
9. The loader must expose a manifest of loaded tensor names for diagnostics.
10. Missing, duplicate, or unassigned layer tensors must produce a clear startup error before inference begins.
11. The implementation must preserve dtype as bf16 unless a module explicitly requires fp32 persistent state.
12. Rank-local loading must be covered by tests using a small synthetic safetensors checkpoint with the same naming conventions as Qwen3.5.

### Correctness Properties for Property-Based Testing

1. **Partition Coverage Property**: For any generated valid layer distribution over `N` layers and `R` ranks, every layer index from `0` to `N - 1` is assigned exactly once, no layer index is assigned to more than one rank, and stage ranges are ordered by rank.

2. **Tensor Ownership Property**: For any generated tensor name matching embedding, layer, final normalization, or language model head tensors, the ownership resolver returns exactly one rank or a clear "not part of model" result.

3. **Local Reference Equivalence Property**: For a toy hybrid model with randomly generated weights, a pipeline composed from locally loaded shards must produce the same final logits as a model loaded monolithically, within tolerance (bf16: `rtol <= 1e-2`, `atol <= 1e-2`).

4. **Deterministic Manifest Property**: Given the same checkpoint index, configuration, and rank, the loader must produce the same ordered tensor manifest across repeated runs.

---

## Requirement 2: Decode Communication Fast Path

### User Story

As a user generating tokens interactively, I want decode-step communication between pipeline stages to avoid avoidable metadata transfers, allocations, and synchronization broadcasts, so that each token step spends less time in Gloo overhead.

### Acceptance Criteria

1. Decode must use a steady-state fast path that does not send shape metadata before every activation transfer.
2. Shape and dtype metadata may be exchanged once during session initialization, when the active microbatch layout changes, when request membership changes, or when explicitly falling back to a generic path.
3. Decode activations must use preallocated CPU communication buffers suitable for Gloo send/receive.
4. XPU-to-CPU and CPU-to-XPU staging buffers must be reused across decode steps.
5. Decode-stage activation shape must be validated against the negotiated static protocol before communication.
6. Tensor allocation count inside the steady-state decode loop must be reduced relative to the baseline and reported by instrumentation.
7. Rank 3 must send sampled token results directly to rank 0 using point-to-point communication instead of blocking `dist.broadcast()` to all ranks.
8. Ranks that do not require the sampled token must not block on token synchronization.
9. Rank 0 must receive enough token metadata to update request state: request identifier, generated token identifier, finish reason if present, and optional sampling diagnostic fields in debug mode.
10. The generic metadata path must remain available for prefill, first token after prefill, nonstandard shapes, debugging, and protocol mismatch recovery.
11. Decode fast path must support batch size `1` and continuous batch sizes greater than `1`.
12. Communication errors must produce structured errors identifying source rank, destination rank, request identifier if available, expected shape, and received shape or byte count.
13. For single-request decode, a benchmark must show lower per-token communication time than the baseline.
14. For fixed-shape decode, instrumentation must show zero per-token shape metadata messages after warm-up.

### Correctness Properties for Property-Based Testing

1. **Protocol Round-Trip Property**: For generated valid activation metadata, protocol negotiation followed by packet validation must accept exactly tensors matching dtype, rank, shape, contiguous layout requirement, and microbatch slot count.

2. **Mismatched Shape Rejection Property**: For any tensor whose shape differs from the negotiated decode shape, the fast path must reject it and either use the generic path if allowed or raise a structured protocol error.

3. **Token Packet Integrity Property**: For generated token result packets, serialization and deserialization must preserve request identifier, token identifier, sequence position, finish reason, and rank source.

4. **No Unintended Broadcast Property**: In decode fast-path mode, only rank 0 and rank 3 participate in token-result synchronization. Ranks 1 and 2 must not wait on token-result messages.

---

## Requirement 3: Continuous Batching

### User Story

As a service user issuing multiple independent generation requests, I want requests to flow through the four pipeline stages concurrently, so that pipeline bubbles are reduced and aggregate token throughput increases.

### Acceptance Criteria

1. The engine must support multiple active requests at the same time.
2. Each request must maintain independent: input token history, generated token history, attention mask or equivalent position metadata, `DynamicCache`, GatedDeltaNet recurrent state, random sampling state (if stochastic sampling is enabled), and stopping criteria.
3. The scheduler must form decode microbatches from active requests.
4. The scheduler must support request admission while other requests are decoding.
5. The scheduler must support request completion without resetting other active requests.
6. The pipeline must keep up to four or more microbatches in flight when enough requests are available.
7. Continuous batching must support greedy sampling, top-k sampling, per-request maximum new tokens, end-of-sequence stopping, and cancellation.
8. Request identifiers must be stable and globally unique within an engine process.
9. Per-stage caches must be keyed by request identifier and must not be indexed only by batch position.
10. Batch slots may be reused only after the previous owner request has fully completed or been cancelled and all in-flight messages for that slot are drained.
11. Continuous batching must expose metrics: active request count, admitted request count, completed request count, cancelled request count, average batch size, pipeline occupancy, tokens per second aggregate, and tokens per second per request.
12. With at least four simultaneously active requests, measured pipeline utilization must be greater than single-request utilization.
13. With deterministic greedy sampling, continuous batching must produce the same output per request as running each request alone, within the same model and cache configuration.

### Correctness Properties for Property-Based Testing

1. **Request Isolation Property**: For generated interleavings of request admission, decode, completion, and cancellation, no request may observe another request's token history, cache entries, sampling state, or stopping state.

2. **Batch Slot Reuse Property**: A batch slot cannot be assigned to a new request until all messages tagged with the previous request identifier and slot generation have been consumed or discarded.

3. **Deterministic Greedy Equivalence Property**: For a toy model and generated prompts, batched greedy generation must equal independent greedy generation for each request.

4. **Cancellation Safety Property**: For any cancellation point, remaining active requests continue to produce the same output as if the cancelled request had never been admitted, excluding pipeline timing.

5. **Ordering Property**: Token results returned to clients must preserve per-request token order even when microbatches complete out of wall-clock order.

---

## Requirement 4: Fast Sampling

### User Story

As a user generating one token at a time, I want token sampling to avoid full-vocabulary sorts and unnecessary validation overhead, so that the final stage does not become the throughput bottleneck.

### Acceptance Criteria

1. The system must provide separate optimized paths for greedy sampling, top-k sampling, and fallback general sampling.
2. Greedy sampling must use `argmax` and must not sort logits.
3. Top-k sampling must use `torch.topk` with `k` values much smaller than vocabulary size.
4. Top-k-only sampling must not call `torch.sort` over the full vocabulary.
5. NaN and infinity validation over the full logits tensor must be controlled by a debug or validation flag and disabled by default in performance mode.
6. The sampler must support per-request sampling configuration in continuous batching.
7. The sampler must preserve existing behavior for unsupported combinations by using the fallback path.
8. Sampling must run on XPU when logits are already on XPU unless profiling shows CPU sampling is faster for the configured path.
9. The sampler must emit timing metrics split by logits preparation, filtering, random selection, and token return communication.
10. For greedy and top-k modes, sampling time must be lower than the baseline full-vocabulary sort path.

### Correctness Properties for Property-Based Testing

1. **Greedy Argmax Property**: For generated logits, greedy sampling must return an index whose value equals `torch.max(logits)`.

2. **Top-K Membership Property**: For generated logits and valid `k`, top-k sampling must return only token identifiers contained in `torch.topk(logits, k).indices`.

3. **Top-K One Equals Greedy Property**: For generated logits without ties, top-k sampling with `k = 1` must equal greedy sampling.

4. **Fallback Equivalence Property**: For sampling configurations that require the fallback path, the optimized sampler must match the previous implementation under the same random seed.

5. **Validation Flag Property**: NaN validation must run when enabled and must not run when disabled. When enabled, invalid logits must produce a structured sampling error.

---

## Requirement 5: GatedDeltaNet State Optimization

### User Story

As a user decoding long generations, I want GatedDeltaNet recurrent state to remain in persistent fp32 form without per-token casting and repeated allocation, so that linear-attention layers are efficient during decode.

### Acceptance Criteria

1. GatedDeltaNet recurrent state must be stored persistently in fp32 where required for numerical stability.
2. Decode must not cast recurrent state from bf16 to fp32 on every token.
3. Decode must not allocate new recurrent state tensors unnecessarily on every token.
4. Output tensors used by the GatedDeltaNet recurrent decode step must be preallocated where shape-stable.
5. The implementation must remain compatible with `DynamicCache`.
6. The model forward path must continue to pass cache state through the `past_key_values` keyword argument.
7. Cache initialization must define dtype, device, shape, and owning request identifier.
8. Cache reset must release or recycle state for completed and cancelled requests.
9. The optimized implementation must include a compatibility fallback for the current behavior.
10. Numerical difference versus the baseline implementation must be bounded: decode logits `rtol <= 2e-2`, `atol <= 2e-2` in bf16 mode; recurrent state `rtol <= 2e-2`, `atol <= 2e-2`.
11. Per-layer timing must show reduced GatedDeltaNet recurrent-step overhead relative to baseline.

### Correctness Properties for Property-Based Testing

1. **State Persistence Property**: Across generated decode steps, the state object identity or allocated storage identifier remains stable unless request reset, shape change, or explicit fallback occurs.

2. **Optimized-Baseline Equivalence Property**: For generated toy GatedDeltaNet inputs and states, optimized decode output must match baseline decode output within tolerance.

3. **No Cross-Request State Sharing Property**: Distinct request identifiers must map to distinct recurrent state storage unless explicit copy-on-write semantics are implemented and tested.

4. **Reset Property**: After request completion or cancellation, cache reset must remove or recycle all associated GatedDeltaNet state.

5. **Dtype Property**: Persistent recurrent state must remain fp32 during decode and must not silently degrade to bf16.

---

## Requirement 6: Pipeline Stage Balancing

### User Story

As an operator, I want layer distribution across the four nodes to be configurable and guided by profiling, so that slow stages do not limit pipeline throughput.

### Acceptance Criteria

1. Layer distribution must be configurable through a typed configuration object.
2. The default distribution may remain sixteen layers per stage, but alternate distributions must be supported.
3. The distribution must allow full-attention and GatedDeltaNet layers to be weighted differently.
4. Profiling data must identify per-layer and per-stage compute time separately for prefill and decode.
5. A balancing tool or function must recommend a new stage distribution based on measured timings.
6. The recommended distribution must preserve layer order.
7. Stage assignment must account for rank-specific auxiliary work: rank 0 embedding, rank 3 final normalization, `lm_head`, and sampling.
8. A proposed distribution must be validated before use: covers all layers once, uses all ranks, assigns final modules to the final rank, assigns embedding to the first rank.
9. The benchmark script must allow comparing multiple distributions.
10. Any distribution change must preserve output equivalence.

### Correctness Properties for Property-Based Testing

1. **Contiguous Partition Property**: Generated valid distributions must partition layers into contiguous rank-owned ranges.

2. **Load Balance Recommendation Property**: Given synthetic per-layer timings, the recommendation algorithm should reduce max stage time or leave the distribution unchanged when no improvement is possible.

3. **Output Equivalence Property**: For a toy model, any valid layer distribution must produce equivalent final logits.

4. **Auxiliary Cost Accounting Property**: Rank 0 and rank 3 fixed costs must be included in stage-time estimation.

---

## Requirement 7: Chunked GatedDeltaNet Prefill

### User Story

As a user submitting prompts with many tokens, I want GatedDeltaNet prefill to avoid strictly sequential token-by-token processing where mathematically possible, so that time to first token improves.

### Acceptance Criteria

1. GatedDeltaNet prefill must provide an optimized chunked algorithm using a WY-style decomposition or equivalent associative block composition.
2. The optimized prefill path must be optional and controlled by configuration.
3. The fallback sequential prefill path must remain available.
4. Chunk size must be configurable.
5. The chunked implementation must support prompt lengths not divisible by chunk size.
6. The implementation must preserve final recurrent state required for subsequent decode.
7. The implementation must preserve output activations for every prompt token.
8. Numerical difference versus sequential prefill must be bounded: bf16 activations `rtol <= 3e-2`, `atol <= 3e-2`; fp32 reference mode `rtol <= 1e-4`, `atol <= 1e-4`.
9. The implementation must support XPU execution with PyTorch operations only.
10. Chunked prefill must expose timing metrics: per-layer prefill time, chunk transform time, chunk composition time, and final state materialization time.
11. The implementation must be disabled automatically with a structured warning if unsupported shapes, dtypes, or layer parameters are encountered.
12. Benchmarks must show improved prefill time for long prompts compared with sequential GatedDeltaNet prefill.

### Correctness Properties for Property-Based Testing

1. **Chunk Boundary Property**: For generated sequence lengths and chunk sizes, concatenated chunk outputs must have the same sequence length and ordering as sequential outputs.

2. **Associative Composition Property**: For generated compatible chunk transforms, composing chunks as `(A ◦ B) ◦ C` must equal `A ◦ (B ◦ C)` within tolerance.

3. **Final State Equivalence Property**: The final recurrent state after chunked prefill must match the final recurrent state after sequential prefill within tolerance.

4. **Decode Continuation Property**: Running decode after chunked prefill must produce the same next-token logits as running decode after sequential prefill.

5. **Fallback Safety Property**: Unsupported configurations must use sequential prefill and must not silently produce approximate or incomplete state.

---

## Requirement 8: Performance Instrumentation

### User Story

As a developer optimizing the system, I want detailed stage, communication, sampling, and cache timing metrics, so that changes can be measured before and after implementation.

### Acceptance Criteria

1. Instrumentation must be implemented before optimization tasks that rely on performance claims.
2. Instrumentation must separately measure: prefill compute time, decode compute time, activation send time, activation receive time, token-result send time, token-result receive time, sampling time, `lm_head` time, GatedDeltaNet recurrent-step time, full-attention time, cache update time, scheduler wait time, and allocation count where measurable.
3. Metrics must be reported per rank and per pipeline stage.
4. Metrics must distinguish single-request mode, continuous-batching mode, prefill, and decode.
5. The benchmark script must output: tokens per second, time to first token, inter-token latency, stage utilization, pipeline bubble estimate, average microbatch size, and communication time ratio.
6. Instrumentation overhead must be low enough to remain enabled in normal benchmarking.
7. Detailed event tracing may be optionally enabled through configuration.
8. Metrics must be serializable as JSON for cross-run comparison.
9. The system must include regression thresholds for the target performance metrics.
10. Instrumentation must not change generated tokens.

### Correctness Properties for Property-Based Testing

1. **Timing Nesting Property**: Parent timing spans must contain child timing spans where nesting is declared.

2. **Nonnegative Duration Property**: All measured durations must be greater than or equal to zero.

3. **Token Count Conservation Property**: Reported generated token count must equal the number of token results emitted to clients.

4. **Mode Classification Property**: Events must be classified as prefill or decode, never both.

5. **Serialization Property**: Metrics serialized to JSON and deserialized back must preserve all numeric fields within normal floating-point representation.

# Requirements Document

## Introduction

This feature implements the distributed text generation pipeline for the exo inference system running across 4 NixOS gremlin nodes. The cluster has already achieved: cluster formation (libp2p gossipsub), Gloo process group initialization (all ranks connected), model loading with TransformerShard (layers split across nodes), and warmup (Ready state reached).

What is missing is the actual coordinated generation loop. Currently, `pytorch_xpu_generate()` calls `model.forward()` locally on each rank independently, which crashes because intermediate ranks only have a subset of layers — rank 0 produces hidden states (not logits) and cannot sample tokens, while the last rank never receives the hidden states it needs as input.

The distributed generation pipeline coordinates forward passes across ranks using the existing `send_activation`/`recv_activation` primitives (CPU tensor staging via Gloo), implements the autoregressive token generation loop where only rank 0 tokenizes and only the last rank produces logits, and streams tokens back to the user as they are generated.

**Cluster topology:**
- **Rank 0** (first shard, layers 0–N): Tokenizes prompt → embeds → forward through first layers → sends hidden states to rank 1
- **Rank 1..K-1** (middle shards): Receives hidden states → forward through assigned layers → sends hidden states to next rank
- **Last rank** (last shard, layers M–end): Receives hidden states → forward → norm → lm_head → produces logits → sends logits back to rank 0
- **Rank 0**: Receives logits → samples token → streams to user → repeats for next token

**Hardware:** All 4 gremlin nodes use Intel Arc Meteor Lake-P iGPUs with shared system memory. CPU↔GPU tensor staging is nearly free on these nodes due to shared memory architecture.

## Glossary

- **Generation_Pipeline**: The coordinated autoregressive loop that produces tokens by forwarding activations through pipeline-parallel shards across multiple ranks, with token sampling occurring only on rank 0
- **Rank**: A node's integer position (0 to world_size-1) within the distributed process group
- **First_Rank**: The rank (rank 0) responsible for tokenization, embedding, token sampling, and streaming results to the user
- **Last_Rank**: The rank (rank == world_size - 1) whose TransformerShard includes the final norm layer and lm_head, producing logits from hidden states
- **Middle_Rank**: Any rank between first and last that receives hidden states, forwards through its assigned layers, and sends hidden states to the next rank
- **Hidden_States**: The intermediate tensor output from a TransformerShard that does not include the final norm/lm_head layers; shape is (batch_size, seq_len, hidden_size)
- **Logits**: The output tensor from the last rank's lm_head layer; shape is (batch_size, seq_len, vocab_size); used by rank 0 for token sampling
- **Autoregressive_Loop**: The iterative process where each generated token is fed back as input for the next forward pass, continuing until an EOS token or max_tokens limit is reached
- **KV_Cache**: Per-layer key-value tensors cached from previous forward passes to avoid recomputation during autoregressive generation; each rank maintains KV cache only for its own layers
- **Prefill_Phase**: The first iteration of generation where the full tokenized prompt is processed through all ranks to populate the KV cache
- **Decode_Phase**: Subsequent iterations after prefill where only the single most-recently-generated token is forwarded through all ranks
- **TransformerShard**: The model wrapper that executes only a contiguous range of transformer layers, with optional embedding (first shard) and norm/lm_head (last shard)
- **CPU_Tensor_Staging**: The pattern of moving tensors GPU→CPU before `dist.send()` and CPU→GPU after `dist.recv()`, required by the Gloo backend
- **send_activation**: Function in distributed.py that stages a tensor to CPU and sends it to a destination rank via Gloo
- **recv_activation**: Function in distributed.py that receives a tensor on a CPU buffer and moves it to the target GPU device
- **Token_Streaming**: Yielding each generated token to the caller as soon as it is sampled, enabling real-time response delivery to the user
- **Sampling**: The process of selecting the next token from logits using temperature scaling, top-k filtering, and top-p (nucleus) filtering

## Requirements

### Requirement 1: Distributed Forward Pass Coordination

**User Story:** As a cluster operator, I want forward passes to be coordinated across all ranks in pipeline order, so that each rank processes its assigned layers and passes activations to the next rank until logits are produced.

#### Acceptance Criteria

1. WHEN the Generation_Pipeline begins a forward pass iteration, THE First_Rank SHALL compute its TransformerShard forward pass on the input tensor and send the resulting Hidden_States to rank 1 using send_activation
2. WHEN a Middle_Rank receives Hidden_States from the previous rank via recv_activation, THE Middle_Rank SHALL compute its TransformerShard forward pass and send the resulting Hidden_States to the next rank using send_activation
3. WHEN the Last_Rank receives Hidden_States from the previous rank via recv_activation, THE Last_Rank SHALL compute its TransformerShard forward pass (including norm and lm_head) to produce Logits
4. WHEN the Last_Rank produces Logits, THE Last_Rank SHALL send the Logits tensor back to the First_Rank using send_activation
5. THE Generation_Pipeline SHALL execute forward passes synchronously in pipeline order: First_Rank → Middle_Ranks (in rank order) → Last_Rank → Logits back to First_Rank
6. IF any rank's TransformerShard forward pass raises an exception, THEN THE Generation_Pipeline SHALL propagate the error and terminate generation on all participating ranks

### Requirement 2: Autoregressive Token Generation Loop

**User Story:** As a user, I want the system to generate tokens one at a time in an autoregressive loop, so that I receive a coherent text response built token by token.

#### Acceptance Criteria

1. WHEN the Generation_Pipeline starts, THE First_Rank SHALL tokenize the input prompt and begin the Prefill_Phase by forwarding the full token sequence through all ranks
2. WHEN the First_Rank receives Logits from the Last_Rank, THE First_Rank SHALL sample the next token from the logits of the last position
3. WHEN a token is sampled, THE First_Rank SHALL feed the sampled token as the sole input for the next Decode_Phase iteration
4. THE Generation_Pipeline SHALL repeat the Autoregressive_Loop (forward pass → receive logits → sample → feed back) until a termination condition is met
5. WHEN the sampled token matches an end-of-sequence token ID, THE Generation_Pipeline SHALL terminate with finish_reason "stop"
6. WHEN the number of generated tokens reaches the configured max_tokens limit, THE Generation_Pipeline SHALL terminate with finish_reason "length"
7. THE First_Rank SHALL broadcast each sampled token ID to all other ranks so they can use it as input for their next Decode_Phase forward pass

### Requirement 3: Token Sampling with Configurable Parameters

**User Story:** As a user, I want to control the randomness and diversity of generated text through temperature, top-k, and top-p parameters, so that I can tune output quality for my use case.

#### Acceptance Criteria

1. WHEN Logits are received by the First_Rank, THE Generation_Pipeline SHALL apply temperature scaling by dividing logits by the temperature value before sampling
2. WHERE top_k is configured, THE Generation_Pipeline SHALL zero out all logit values below the top-k highest values before sampling
3. WHERE top_p is configured, THE Generation_Pipeline SHALL zero out logit values whose cumulative probability exceeds the top_p threshold (nucleus sampling) before sampling
4. WHEN temperature is 1.0 and no top_k or top_p is configured, THE Generation_Pipeline SHALL sample from the unmodified softmax distribution
5. THE Generation_Pipeline SHALL apply sampling parameters in the order: temperature scaling → top-k filtering → top-p filtering → softmax → multinomial sampling

### Requirement 4: Streaming Token Output

**User Story:** As a user, I want to receive generated tokens as they are produced rather than waiting for the entire response, so that I experience low perceived latency.

#### Acceptance Criteria

1. WHEN the First_Rank samples a token, THE Generation_Pipeline SHALL yield a GenerationResponse containing the decoded token text, token ID, usage statistics, and generation stats
2. THE Generation_Pipeline SHALL yield each token immediately after sampling without buffering multiple tokens
3. WHEN generation terminates, THE Generation_Pipeline SHALL yield a final GenerationResponse with the appropriate finish_reason ("stop" or "length") and final usage statistics
4. IF generation encounters an error, THEN THE Generation_Pipeline SHALL yield a GenerationResponse with finish_reason "error" and a descriptive error message
5. THE Generation_Pipeline SHALL be implemented as a Python generator function that yields GenerationResponse objects, compatible with the existing runner event dispatch loop

### Requirement 5: KV Cache Management Across Iterations

**User Story:** As a cluster operator, I want each rank to maintain its own KV cache across autoregressive iterations, so that previously computed attention keys and values are reused without recomputation.

#### Acceptance Criteria

1. THE Generation_Pipeline SHALL maintain a separate KV_Cache for each rank, storing only the key-value pairs for that rank's assigned layers
2. WHEN the Prefill_Phase completes, THE Generation_Pipeline SHALL store the resulting KV_Cache on each rank for use in subsequent Decode_Phase iterations
3. WHEN a Decode_Phase iteration completes on a rank, THE Generation_Pipeline SHALL update that rank's KV_Cache with the new key-value pairs from the single-token forward pass
4. THE Generation_Pipeline SHALL pass the accumulated KV_Cache to TransformerShard.forward() on each iteration so that attention computation covers all previous positions
5. IF generation is terminated or an error occurs, THEN THE Generation_Pipeline SHALL release KV_Cache memory on all ranks

### Requirement 6: Rank Synchronization for Decode Phase

**User Story:** As a cluster operator, I want all ranks to stay synchronized during the decode phase, so that each rank processes the correct token at each iteration without deadlocks.

#### Acceptance Criteria

1. WHEN the First_Rank samples a token during the Decode_Phase, THE First_Rank SHALL send the token ID to all other ranks so they can prepare their input for the next iteration
2. WHEN a non-first rank receives a token ID from the First_Rank, THE non-first rank SHALL use that token ID as the input for its next forward pass (embedding lookup occurs only on the first shard; other ranks receive hidden states)
3. WHEN the First_Rank determines that generation should terminate (EOS or max_tokens), THE First_Rank SHALL send a termination signal to all other ranks
4. WHEN a non-first rank receives a termination signal, THE non-first rank SHALL exit its generation loop and release resources
5. THE Generation_Pipeline SHALL ensure that all ranks execute the same number of forward pass iterations to prevent Gloo send/recv deadlocks

### Requirement 7: Integration with Runner Task Dispatch

**User Story:** As a developer, I want the distributed generation pipeline to integrate with the existing runner's TextGeneration task handler, so that chat requests are handled by the distributed pipeline instead of the broken single-node generation path.

#### Acceptance Criteria

1. WHEN the Runner receives a TextGeneration task and the backend is "pytorch_xpu" with world_size greater than 1, THE Runner SHALL invoke the distributed Generation_Pipeline instead of the single-node pytorch_xpu_generate function
2. THE Generation_Pipeline SHALL accept the same parameters as pytorch_xpu_generate: model, tokenizer, prompt, device_type, device_id, max_tokens, temperature, top_k, top_p, and model_id
3. THE Generation_Pipeline SHALL additionally accept rank, world_size, and device string parameters to coordinate distributed communication
4. WHEN world_size equals 1, THE Runner SHALL continue to use the existing single-node pytorch_xpu_generate function without modification
5. THE Generation_Pipeline SHALL produce GenerationResponse objects compatible with the existing runner event dispatch loop that sends ChunkGenerated events

### Requirement 8: Prefill and Decode Phase Separation

**User Story:** As a cluster operator, I want the prefill phase (processing the full prompt) to be handled differently from the decode phase (generating one token at a time), so that the pipeline efficiently processes long prompts before switching to incremental generation.

#### Acceptance Criteria

1. WHEN generation begins, THE First_Rank SHALL forward the entire tokenized prompt (all tokens) through the pipeline in a single forward pass (Prefill_Phase)
2. WHEN the Prefill_Phase completes and the First_Rank receives Logits, THE Generation_Pipeline SHALL transition to the Decode_Phase where only single tokens are forwarded
3. DURING the Prefill_Phase, THE TransformerShard on each rank SHALL process the full sequence length and populate the KV_Cache for all prompt positions
4. DURING the Decode_Phase, THE TransformerShard on each rank SHALL process only a single token (seq_len=1) and append to the existing KV_Cache
5. THE Generation_Pipeline SHALL communicate the current phase (prefill vs decode) implicitly through the tensor shape: seq_len > 1 indicates prefill, seq_len == 1 indicates decode

### Requirement 9: Only Rank 0 Streams to User

**User Story:** As a developer, I want only rank 0 to produce user-visible output, so that the runner's event dispatch logic remains simple and tokens are not duplicated.

#### Acceptance Criteria

1. THE First_Rank SHALL be the only rank that yields GenerationResponse objects to the runner's event dispatch loop
2. THE non-first ranks SHALL execute their forward passes and communication silently without producing user-visible output
3. WHEN the runner dispatches ChunkGenerated events, THE Runner SHALL only do so from the rank 0 process (device_rank == 0)
4. THE non-first ranks SHALL run a loop that receives activations, computes forward passes, sends results, and waits for the next iteration signal without any token decoding or text generation logic

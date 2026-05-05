# Implementation Plan: Distributed Generation Pipeline

## Overview

Implement the coordinated autoregressive text generation loop for pipeline-parallel inference across 4 gremlin nodes. The pipeline coordinates forward passes across ranks using existing `send_activation`/`recv_activation` primitives, with rank 0 orchestrating token sampling and streaming, and non-rank-0 nodes running a blocking worker loop.

## Tasks

- [x] 1. Create the distributed_generator module with sample_token
  - [x] 1.1 Create `src/exo/worker/engines/pytorch_xpu/distributed_generator.py` with module docstring, imports, and TERMINATION_SENTINEL constant
    - Import torch, logging, time, Generator, Any, Optional
    - Import send_activation, recv_activation from distributed.py
    - Import GenerationResponse, Usage, GenerationStats, etc. from shared types
    - Define `TERMINATION_SENTINEL: int = -1`
    - _Requirements: 6.3, 6.4_

  - [x] 1.2 Implement `sample_token()` pure function
    - Accept logits tensor (shape `(1, seq_len, vocab_size)` or `(1, vocab_size)`), temperature, top_k, top_p
    - Extract last-position logits only: `logits[:, -1, :]` if 3D
    - Apply temperature scaling: `logits / temperature` when temperature != 1.0
    - Apply top-k filtering: zero out values below top-k threshold
    - Apply top-p (nucleus) filtering: zero out values above cumulative probability threshold
    - Apply softmax then multinomial sampling
    - Return sampled token ID as int
    - _Requirements: 2.2, 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 1.3 Write property test for sample_token — Property 1: Sampling uses last position only
    - **Property 1: Sampling uses last position only**
    - **Validates: Requirements 2.2**

  - [x] 1.4 Write property test for sample_token — Property 4: Temperature scaling is division
    - **Property 4: Temperature scaling is division**
    - **Validates: Requirements 3.1**

  - [x] 1.5 Write property test for sample_token — Property 5: Top-k retains exactly k values
    - **Property 5: Top-k retains exactly k values**
    - **Validates: Requirements 3.2**

  - [x] 1.6 Write property test for sample_token — Property 6: Top-p respects cumulative probability threshold
    - **Property 6: Top-p respects cumulative probability threshold**
    - **Validates: Requirements 3.3**

  - [x] 1.7 Write property test for sample_token — Property 12: Sampling pipeline order
    - **Property 12: Sampling pipeline order**
    - **Validates: Requirements 3.5**

- [x] 2. Implement distributed_generate (rank 0 orchestrator)
  - [x] 2.1 Implement `distributed_generate()` function signature and tokenization
    - Accept model, tokenizer, prompt, device_type, device_id, rank, world_size, max_tokens, temperature, top_k, top_p, model_id
    - Tokenize prompt using tokenizer.encode()
    - Determine EOS token IDs (including additional stop tokens)
    - Initialize KV cache as None, iteration counter, timing
    - _Requirements: 2.1, 7.2, 7.3_

  - [x] 2.2 Implement prefill phase in distributed_generate
    - Forward full input_ids through local TransformerShard (model.forward(input_ids, past_key_values=None))
    - Send hidden_states to rank 1 via send_activation
    - Receive logits from last rank via recv_activation (shape `(1, 1, vocab_size)`, dtype float32)
    - Sample first token using sample_token()
    - Broadcast sampled token to all other ranks (send int64 tensor of shape `(1,)` to each)
    - Store KV cache from prefill
    - Yield first GenerationResponse
    - _Requirements: 1.1, 1.4, 1.5, 2.1, 2.2, 2.7, 5.2, 8.1, 8.3_

  - [x] 2.3 Implement decode loop in distributed_generate
    - Loop up to max_tokens - 1 remaining iterations
    - Forward single token (shape `(1, 1)`) through local TransformerShard with KV cache
    - Send hidden_states to rank 1 via send_activation
    - Receive logits from last rank via recv_activation
    - Sample next token using sample_token()
    - Check termination: EOS → send SENTINEL to all ranks, yield final response with finish_reason="stop"
    - Check termination: max_tokens → send SENTINEL to all ranks, yield final response with finish_reason="length"
    - Otherwise: broadcast token to all ranks, yield GenerationResponse, continue
    - Update KV cache each iteration
    - _Requirements: 1.1, 1.4, 1.5, 2.3, 2.4, 2.5, 2.6, 2.7, 4.1, 4.2, 4.3, 5.3, 5.4, 6.1, 6.3, 8.2, 8.4_

  - [x] 2.4 Implement error handling in distributed_generate
    - Wrap forward/communication in try/except
    - On error: best-effort send TERMINATION_SENTINEL to all other ranks
    - Yield GenerationResponse with finish_reason="error" and error message
    - Set KV cache to None for cleanup
    - _Requirements: 1.6, 4.4, 5.5_

  - [x] 2.5 Write property test — Property 2: EOS token terminates with "stop"
    - **Property 2: EOS token terminates with "stop"**
    - **Validates: Requirements 2.5**

  - [x] 2.6 Write property test — Property 3: Max tokens terminates with "length"
    - **Property 3: Max tokens terminates with "length"**
    - **Validates: Requirements 2.6**

  - [x] 2.7 Write property test — Property 7: GenerationResponse contains all required fields
    - **Property 7: GenerationResponse contains all required fields**
    - **Validates: Requirements 4.1**

- [x] 3. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement distributed_worker_loop (non-rank-0 nodes)
  - [x] 4.1 Implement `distributed_worker_loop()` function
    - Accept model, device_type, device_id, rank, world_size, hidden_size, dtype
    - Determine if this rank is the last rank (rank == world_size - 1)
    - Initialize KV cache as None, iteration counter
    - _Requirements: 7.3, 9.2, 9.4_

  - [x] 4.2 Implement prefill iteration in worker loop
    - First iteration (iteration == 0): receive hidden_states from previous rank via recv_activation
    - Shape for prefill: unknown seq_len — receive shape metadata or use a fixed protocol
    - Forward through local TransformerShard with past_key_values=None
    - If last rank: send only last-position logits `(1, 1, vocab_size)` to rank 0
    - If middle rank: send hidden_states to next rank
    - Receive token from rank 0 (int64 tensor shape `(1,)`)
    - Check for TERMINATION_SENTINEL — if received, exit loop
    - Store KV cache
    - _Requirements: 1.2, 1.3, 1.4, 5.2, 6.2, 6.4, 8.3, 8.5_

  - [x] 4.3 Implement decode iterations in worker loop
    - Subsequent iterations: receive hidden_states from previous rank (shape `(1, 1, hidden_size)`)
    - Forward through local TransformerShard with accumulated KV cache
    - If last rank: send last-position logits to rank 0
    - If middle rank: send hidden_states to next rank
    - Receive next token from rank 0
    - Check for TERMINATION_SENTINEL — if received, exit loop
    - Update KV cache
    - _Requirements: 1.2, 1.3, 1.4, 5.3, 5.4, 6.2, 6.4, 8.4, 8.5_

  - [x] 4.4 Implement error handling and cleanup in worker loop
    - Wrap in try/except for RuntimeError from Gloo and model errors
    - On error: log, set KV cache to None, exit loop
    - On normal exit (sentinel): set KV cache to None
    - _Requirements: 1.6, 5.5, 6.4_

  - [x] 4.5 Write property test — Property 10: Only rank 0 produces output
    - **Property 10: Only rank 0 produces output**
    - **Validates: Requirements 9.1, 9.2, 9.4**

  - [x] 4.6 Write property test — Property 9: Input tensor shape reflects generation phase
    - **Property 9: Input tensor shape reflects generation phase**
    - **Validates: Requirements 8.1, 8.2, 8.5**

  - [x] 4.7 Write property test — Property 8: KV cache grows correctly across phases
    - **Property 8: KV cache grows correctly across phases**
    - **Validates: Requirements 5.2, 5.3, 8.3, 8.4**

  - [x] 4.8 Write property test — Property 11: All ranks execute equal iterations
    - **Property 11: All ranks execute equal iterations**
    - **Validates: Requirements 6.5**

- [x] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Integrate distributed pipeline into runner.py
  - [x] 6.1 Modify runner TextGeneration handler to dispatch to distributed pipeline
    - In the `if backend_type == "pytorch_xpu":` block (~line 670), add check for `world_size > 1`
    - If `world_size > 1` and `rank == 0`: call `distributed_generate()` with all parameters
    - If `world_size > 1` and `rank != 0`: call `distributed_worker_loop()` then set generator to empty iterator
    - If `world_size == 1`: keep existing `pytorch_xpu_generate()` path unchanged
    - Import distributed_generate and distributed_worker_loop at the top of the dispatch block
    - Determine `hidden_size` and `dtype` from the loaded model for worker_loop parameters
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 9.1, 9.3_

  - [x] 6.2 Write unit tests for runner dispatch logic
    - Test that world_size > 1 with rank 0 dispatches to distributed_generate
    - Test that world_size > 1 with rank != 0 dispatches to distributed_worker_loop
    - Test that world_size == 1 dispatches to pytorch_xpu_generate
    - _Requirements: 7.1, 7.4_

- [x] 7. Write unit tests for edge cases
  - [x] 7.1 Write unit tests for distributed_generator edge cases
    - Test EOS detection with multiple stop tokens (`<|im_end|>`, `<|endoftext|>`)
    - Test temperature edge cases: T approaching 0 (greedy-like), very high T
    - Test top-k edge cases: k=1 (greedy), k=vocab_size (no-op)
    - Test top-p edge cases: p close to 0 (top-1), p=1.0 (no-op)
    - Test sentinel value handling: worker loop exits cleanly on receiving -1
    - Test empty/short prompt handling
    - _Requirements: 2.5, 3.1, 3.2, 3.3, 6.3, 6.4_

- [x] 8. Write integration tests for multi-rank coordination
  - [x] 8.1 Write integration tests using torch.multiprocessing.spawn
    - Test 2-rank pipeline: hidden states flow rank 0 → rank 1, logits return
    - Test token broadcast: all ranks receive same token ID each iteration
    - Test termination: sentinel propagates and all ranks exit cleanly
    - Test error propagation: inject error on one rank, verify all terminate
    - Use mock TransformerShard that returns predictable tensors
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.7, 6.1, 6.3, 6.4, 6.5_

- [x] 9. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- Integration tests require `torch.multiprocessing.spawn()` for real multi-process Gloo groups
- Run tests with: `LD_LIBRARY_PATH="/nix/store/cf1a53iqg6ncnygl698c4v0l8qam5a2q-gcc-14.3.0-lib/lib:$LD_LIBRARY_PATH" uv run pytest`
- Type checking: `uv run basedpyright`
- Linting: `uv run ruff check`

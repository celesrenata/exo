# pyright: reportUnusedImport=false
"""
Distributed Generation Pipeline for Pipeline-Parallel Inference

Coordinates autoregressive text generation across multiple ranks using the
Gloo-based send_activation_generic/recv_activation_generic primitives from distributed.py.

This module provides three entry points:
- distributed_generate(): Python generator called by rank 0 that orchestrates
  the pipeline, samples tokens, and yields GenerationResponse objects
- distributed_worker_loop(): Blocking loop for non-rank-0 nodes that receives
  activations, computes forward passes, sends results, and waits for next token
- sample_token(): Pure function that samples a single token from logits with
  temperature scaling, top-k filtering, and top-p (nucleus) filtering

Requirements: 6.3, 6.4
"""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import torch

from exo.api.types import (
    CompletionTokensDetails,
    GenerationStats,
    PromptTokensDetails,
    Usage,
)
from exo.shared.types.memory import Memory
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.pytorch_xpu.distributed import (  # pyright: ignore[reportUnknownVariableType]
    recv_activation_generic as recv_activation,
    send_activation_generic as send_activation,
)

if TYPE_CHECKING:
    from exo.worker.engines.pytorch_xpu.instrumentation import PerformanceRecorder

logger = logging.getLogger(__name__)

TERMINATION_SENTINEL: int = -1
"""Token ID value that signals generation is complete to non-rank-0 nodes."""


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    model_id: str = "",
    enable_diagnostics: bool = False,
    performance_recorder: PerformanceRecorder | None = None,
) -> int:
    """
    Sample a single token from logits with temperature, top-k, and top-p.

    Routes to specialized implementations based on sampling configuration:
    - **Greedy**: ``torch.argmax`` when temperature is near zero or do_sample=False
    - **Top-K**: ``torch.topk`` when top_k is set without top_p
    - **Fallback**: Full pipeline (temperature + top-k + top-p + multinomial)

    The public signature is preserved for backward compatibility. Internally,
    this delegates to ``route_and_sample_token()`` in the ``sampling`` module.

    When enable_diagnostics is True, emits TP_LOGIT_DIAG with first-token top-k
    token IDs, raw decoded token strings, and logit stats.

    Args:
        logits: Shape (1, seq_len, vocab_size) or (1, vocab_size) — raw model output
        temperature: Divide logits by this value (1.0 = no change, near 0 = greedy)
        top_k: Keep only top-k highest logit values (None = disabled)
        top_p: Keep smallest set of tokens with cumulative prob ≥ top_p (None = disabled)
        model_id: Model identifier for diagnostic logging.
        enable_diagnostics: If True, emit TP_LOGIT_DIAG with top-k data.
        performance_recorder: Optional recorder for timing spans. When None,
            no instrumentation overhead is added.

    Returns:
        Sampled token ID as integer

    Requirements: 2.2, 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 5.2, 5.5, 6.2
    """
    from exo.worker.engines.pytorch_xpu.sampling import route_and_sample_token

    # --- Logits shape normalization ---
    # Extract last-position logits only (Requirement 2.2)
    if logits.dim() == 3:
        logits = logits[:, -1, :]  # Shape: (1, vocab_size)

    # Work with a 1D tensor for simplicity: shape (vocab_size,)
    logits_one_dimensional = logits[0].clone()

    # --- Route to specialized sampling implementation ---
    # Logit validation is enabled by default for backward compatibility.
    # The validate_logits flag controls NaN/inf checking (Requirement 4.5).
    token_identifier, route_used = route_and_sample_token(
        logits=logits_one_dimensional,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        validate_logits=True,
        performance_recorder=performance_recorder,
    )

    # --- Diagnostic logging (preserved from original implementation) ---
    if enable_diagnostics:
        top5_logits, top5_ids = torch.topk(logits_one_dimensional, min(5, logits_one_dimensional.size(0)))
        logits_range = logits_one_dimensional.max() - logits_one_dimensional.min()
        logger.info(
            f"TP_LOGIT_DIAG: "
            f"model_id={model_id} "
            f"sampled_token_id={token_identifier} "
            f"top5_token_ids={top5_ids.tolist()} "
            f"top5_logits={top5_logits.tolist()} "
            f"logits_mean={float(logits_one_dimensional.mean().item()):.4f} "
            f"logits_std={float(logits_one_dimensional.std().item()):.4f} "
            f"logits_range={float(logits_range):.6e} "
            f"temperature={temperature} "
            f"top_k={top_k} "
            f"top_p={top_p} "
            f"route={route_used}"
        )

    # --- Instrumentation: record sampling latency via parent span ---
    if performance_recorder is not None:
        performance_recorder.increment_counter("sample_token_calls")

    return token_identifier


def distributed_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device_type: str,
    device_id: int,
    rank: int,
    world_size: int,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    model_id: str = "",
) -> Generator[GenerationResponse, None, None]:
    """
    Orchestrate distributed generation from rank 0.

    Only called on rank 0. Drives the pipeline by:
    1. Tokenizing the prompt
    2. Running prefill (full sequence through pipeline)
    3. Running decode loop (one token at a time)
    4. Sampling tokens and broadcasting to other ranks
    5. Yielding GenerationResponse for each token

    Args:
        model: TransformerShard wrapping the local model layers
        tokenizer: HuggingFace tokenizer with encode/decode methods
        prompt: Input text prompt to generate from
        device_type: Device type ("xpu", "cuda", or "cpu")
        device_id: Device ID for tensor placement
        rank: This node's rank in the distributed group (must be 0)
        world_size: Total number of ranks in the distributed group
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter (None = disabled)
        top_p: Top-p (nucleus) sampling parameter (None = disabled)
        model_id: Model identifier for logging

    Yields:
        GenerationResponse objects containing generated tokens

    Requirements: 2.1, 7.2, 7.3
    """
    logger.info(
        f"Starting distributed generation: prompt_len={len(prompt)}, "
        f"max_tokens={max_tokens}, temperature={temperature}, "
        f"top_k={top_k}, top_p={top_p}, rank={rank}, world_size={world_size}, "
        f"model_id={model_id}"
    )

    # Tokenize prompt (Requirement 2.1)
    input_ids: list[int] = tokenizer.encode(prompt)  # pyright: ignore[reportAny]
    prompt_tokens: int = len(input_ids)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=f"{device_type}:{device_id}")

    logger.debug(f"Tokenized prompt: {prompt_tokens} tokens")

    # Determine EOS token IDs — check tokenizer for eos_token_id and additional stop tokens
    eos_token_ids: set[int] = set()

    # Primary EOS token
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:  # pyright: ignore[reportAny]
        eos_token_ids.add(int(tokenizer.eos_token_id))  # pyright: ignore[reportAny]

    # Additional special tokens that act as stop tokens
    if hasattr(tokenizer, "additional_special_tokens_ids"):  # pyright: ignore[reportAny]
        for special_id in tokenizer.additional_special_tokens_ids:  # pyright: ignore[reportAny]
            eos_token_ids.add(int(special_id))  # pyright: ignore[reportAny]

    # Check for common chat template stop tokens (<|im_end|>, <|endoftext|>)
    if hasattr(tokenizer, "all_special_ids"):  # pyright: ignore[reportAny]
        for special_id in tokenizer.all_special_ids:  # pyright: ignore[reportAny]
            special_tok: str = tokenizer.decode([special_id], skip_special_tokens=False)  # pyright: ignore[reportAny]
            if special_tok in ("<|im_end|>", "<|endoftext|>", "<|end|>"):
                eos_token_ids.add(int(special_id))  # pyright: ignore[reportAny]

    logger.debug(f"EOS token IDs: {eos_token_ids}")

    # Initialize generation state
    past_key_values: Any = None
    iteration: int = 0  # pyright: ignore[reportUnusedVariable]
    start_time: float = time.perf_counter()
    target_device: str = f"{device_type}:{device_id}"

    # Determine vocab_size from model config (lm_head is only on last rank)
    vocab_size: int = model.model.config.vocab_size  # pyright: ignore[reportAny]
    last_rank: int = world_size - 1

    # --- Prefill Phase + Decode Loop wrapped in error handling (task 2.4) ---
    # Requirements: 1.6, 4.4, 5.5
    try:
        # --- Prefill Phase (task 2.2) ---
        # Forward full input_ids through local TransformerShard (Requirement 8.1)
        logger.debug(f"Prefill: forwarding {prompt_tokens} tokens through local shard")
        hidden_states, past_key_values = model.forward(input_data=input_tensor, past_key_values=None)  # pyright: ignore[reportAny]

        # Send seq_len metadata to rank 1 so it knows the prefill shape (Requirement 8.5)
        seq_len_meta = torch.tensor([hidden_states.shape[1]], dtype=torch.int64)  # pyright: ignore[reportAny]
        send_activation(seq_len_meta, dst_rank=1)

        # Send hidden_states to rank 1 (Requirement 1.1)
        send_activation(hidden_states, dst_rank=1)  # pyright: ignore[reportAny]

        # Receive logits from last rank (Requirement 1.4, 1.5)
        # Last rank sends only last-position logits: shape (1, 1, vocab_size), dtype float32
        logits: torch.Tensor = recv_activation(  # pyright: ignore[reportUnknownVariableType]
            shape=(1, 1, vocab_size),
            dtype=torch.float32,
            src_rank=last_rank,
            target_device=target_device,
        )

        # Sample first token (Requirement 2.2)
        first_token_id: int = sample_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)  # pyright: ignore[reportUnknownArgumentType]
        logger.debug(f"Prefill: sampled first token {first_token_id}")

        # Broadcast sampled token to all other ranks (Requirement 2.7)
        token_tensor = torch.tensor([first_token_id], dtype=torch.int64)
        for dst in range(1, world_size):
            send_activation(token_tensor, dst_rank=dst)

        # Store KV cache from prefill (Requirement 5.2)
        # past_key_values is already set from model.forward above

        # Increment iteration to mark transition to decode phase
        iteration = 1  # pyright: ignore[reportUnusedVariable]

        # Decode first token text
        first_token_text: str = tokenizer.decode([first_token_id], skip_special_tokens=True)  # pyright: ignore[reportAny]

        # Calculate prefill timing
        prefill_elapsed: float = time.perf_counter() - start_time
        prompt_tps: float = prompt_tokens / prefill_elapsed if prefill_elapsed > 0 else 0.0

        # Check if first token is EOS (Requirement 2.5)
        if first_token_id in eos_token_ids:
            logger.debug(f"First token is EOS ({first_token_id}), terminating")
            # Send termination sentinel to all other ranks
            sentinel_tensor = torch.tensor([TERMINATION_SENTINEL], dtype=torch.int64)
            for dst in range(1, world_size):
                send_activation(sentinel_tensor, dst_rank=dst)

            yield GenerationResponse(
                text="",
                token=first_token_id,
                finish_reason="stop",
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=1,
                    total_tokens=prompt_tokens + 1,
                    prompt_tokens_details=PromptTokensDetails(cached_tokens=0, audio_tokens=0),
                    completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0, audio_tokens=0),
                ),
                stats=GenerationStats(
                    prompt_tps=prompt_tps,
                    generation_tps=1.0 / prefill_elapsed if prefill_elapsed > 0 else 0.0,
                    prompt_tokens=prompt_tokens,
                    generation_tokens=1,
                    peak_memory_usage=Memory(in_bytes=0),
                ),
            )
            return

        # Check if max_tokens == 1 (only one token requested, Requirement 2.6)
        if max_tokens <= 1:
            logger.debug("max_tokens reached after prefill (max_tokens=1)")
            # Send termination sentinel to all other ranks
            sentinel_tensor = torch.tensor([TERMINATION_SENTINEL], dtype=torch.int64)
            for dst in range(1, world_size):
                send_activation(sentinel_tensor, dst_rank=dst)

            yield GenerationResponse(
                text=first_token_text,
                token=first_token_id,
                finish_reason="length",
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=1,
                    total_tokens=prompt_tokens + 1,
                    prompt_tokens_details=PromptTokensDetails(cached_tokens=0, audio_tokens=0),
                    completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0, audio_tokens=0),
                ),
                stats=GenerationStats(
                    prompt_tps=prompt_tps,
                    generation_tps=1.0 / prefill_elapsed if prefill_elapsed > 0 else 0.0,
                    prompt_tokens=prompt_tokens,
                    generation_tokens=1,
                    peak_memory_usage=Memory(in_bytes=0),
                ),
            )
            return

        # Yield first GenerationResponse (Requirement 4.1, 4.2)
        yield GenerationResponse(
            text=first_token_text,
            token=first_token_id,
            finish_reason=None,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=1,
                total_tokens=prompt_tokens + 1,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=0, audio_tokens=0),
                completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0, audio_tokens=0),
            ),
            stats=GenerationStats(
                prompt_tps=prompt_tps,
                generation_tps=1.0 / prefill_elapsed if prefill_elapsed > 0 else 0.0,
                prompt_tokens=prompt_tokens,
                generation_tokens=1,
                peak_memory_usage=Memory(in_bytes=0),
            ),
        )

        # --- Decode Loop (task 2.3) ---
        # After prefill, we have first_token_id and past_key_values.
        # completion_tokens starts at 1 (the first token from prefill).
        completion_tokens: int = 1
        prev_token_id: int = first_token_id

        for _ in range(max_tokens - 1):
            # Create single-token input tensor: shape (1, 1) (Requirement 8.2, 8.4)
            token_input = torch.tensor([[prev_token_id]], dtype=torch.long, device=target_device)

            # Forward through local TransformerShard with KV cache (Requirement 5.3, 5.4)
            hidden_states, past_key_values = model.forward(input_data=token_input, past_key_values=past_key_values)  # pyright: ignore[reportAny]

            # Send hidden_states to rank 1 (Requirement 1.1)
            send_activation(hidden_states, dst_rank=1)  # pyright: ignore[reportAny]

            # Receive logits from last rank (Requirement 1.4, 1.5)
            logits = recv_activation(  # pyright: ignore[reportUnknownVariableType]
                shape=(1, 1, vocab_size),
                dtype=torch.float32,
                src_rank=last_rank,
                target_device=target_device,
            )

            # Sample next token (Requirement 2.2, 2.3)
            next_token_id: int = sample_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)  # pyright: ignore[reportUnknownArgumentType]
            completion_tokens += 1

            # Calculate generation stats
            elapsed: float = time.perf_counter() - start_time
            generation_tps: float = completion_tokens / elapsed if elapsed > 0 else 0.0

            # Check termination: EOS → send SENTINEL, yield final, return (Requirement 2.5)
            if next_token_id in eos_token_ids:
                logger.debug(f"EOS token {next_token_id} at iteration {completion_tokens}, terminating")
                sentinel_tensor = torch.tensor([TERMINATION_SENTINEL], dtype=torch.int64)
                for dst in range(1, world_size):
                    send_activation(sentinel_tensor, dst_rank=dst)

                yield GenerationResponse(
                    text="",
                    token=next_token_id,
                    finish_reason="stop",
                    usage=Usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                        prompt_tokens_details=PromptTokensDetails(cached_tokens=0, audio_tokens=0),
                        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0, audio_tokens=0),
                    ),
                    stats=GenerationStats(
                        prompt_tps=prompt_tps,
                        generation_tps=generation_tps,
                        prompt_tokens=prompt_tokens,
                        generation_tokens=completion_tokens,
                        peak_memory_usage=Memory(in_bytes=0),
                    ),
                )
                return

            # Check termination: max_tokens reached → send SENTINEL, yield final, return (Requirement 2.6)
            if completion_tokens >= max_tokens:
                logger.debug(f"max_tokens ({max_tokens}) reached, terminating")
                sentinel_tensor = torch.tensor([TERMINATION_SENTINEL], dtype=torch.int64)
                for dst in range(1, world_size):
                    send_activation(sentinel_tensor, dst_rank=dst)

                token_text: str = tokenizer.decode([next_token_id], skip_special_tokens=True)  # pyright: ignore[reportAny]
                yield GenerationResponse(
                    text=token_text,
                    token=next_token_id,
                    finish_reason="length",
                    usage=Usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                        prompt_tokens_details=PromptTokensDetails(cached_tokens=0, audio_tokens=0),
                        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0, audio_tokens=0),
                    ),
                    stats=GenerationStats(
                        prompt_tps=prompt_tps,
                        generation_tps=generation_tps,
                        prompt_tokens=prompt_tokens,
                        generation_tokens=completion_tokens,
                        peak_memory_usage=Memory(in_bytes=0),
                    ),
                )
                return

            # Not terminated: broadcast token to all ranks (Requirement 2.7, 6.1)
            token_broadcast = torch.tensor([next_token_id], dtype=torch.int64)
            for dst in range(1, world_size):
                send_activation(token_broadcast, dst_rank=dst)

            # Decode token text and yield GenerationResponse (Requirement 4.1, 4.2)
            token_text = tokenizer.decode([next_token_id], skip_special_tokens=True)  # pyright: ignore[reportAny]
            yield GenerationResponse(
                text=token_text,
                token=next_token_id,
                finish_reason=None,
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    prompt_tokens_details=PromptTokensDetails(cached_tokens=0, audio_tokens=0),
                    completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0, audio_tokens=0),
                ),
                stats=GenerationStats(
                    prompt_tps=prompt_tps,
                    generation_tps=generation_tps,
                    prompt_tokens=prompt_tokens,
                    generation_tokens=completion_tokens,
                    peak_memory_usage=Memory(in_bytes=0),
                ),
            )

            # Update prev_token_id for next iteration
            prev_token_id = next_token_id

    except Exception as exc:
        # --- Error Handling (task 2.4, Requirements: 1.6, 4.4, 5.5) ---
        error_msg: str = f"Distributed generation error on rank {rank}: {type(exc).__name__}: {exc}"
        logger.error(error_msg, exc_info=True)

        # Best-effort send TERMINATION_SENTINEL to all other ranks
        # Wrapped in its own try/except to avoid cascading failures
        try:
            sentinel_tensor = torch.tensor([TERMINATION_SENTINEL], dtype=torch.int64)
            for dst in range(1, world_size):
                send_activation(sentinel_tensor, dst_rank=dst)
        except Exception as sentinel_exc:
            logger.warning(
                f"Failed to send termination sentinel to other ranks: {sentinel_exc}"
            )

        # Release KV cache for cleanup (Requirement 5.5)
        past_key_values = None

        # Yield error response (Requirement 4.4)
        yield GenerationResponse(
            text=error_msg,
            token=0,
            finish_reason="error",
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                total_tokens=prompt_tokens,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=0, audio_tokens=0),
                completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0, audio_tokens=0),
            ),
            stats=None,
        )
        return


def distributed_worker_loop(
    model: Any,
    device_type: str,
    device_id: int,
    rank: int,
    world_size: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> None:
    """
    Blocking loop for non-rank-0 nodes.

    Runs until termination signal received:
    1. Receive hidden states from previous rank (or token from rank 0 for first shard)
    2. Forward through local TransformerShard
    3. Send output to next rank (or logits back to rank 0 if last rank)
    4. Receive next token ID from rank 0
    5. If token == SENTINEL, exit

    Args:
        model: TransformerShard wrapping the local model layers
        device_type: Device type ("xpu", "cuda", or "cpu")
        device_id: Device ID for tensor placement
        rank: This node's rank in the distributed group (must be > 0)
        world_size: Total number of ranks in the distributed group
        hidden_size: Hidden dimension size of the model
        dtype: Model tensor dtype (e.g. torch.float16, torch.bfloat16)

    Requirements: 7.3, 9.2, 9.4
    """
    logger.info(
        f"Starting distributed worker loop: rank={rank}, world_size={world_size}, "
        f"hidden_size={hidden_size}, dtype={dtype}, device={device_type}:{device_id}"
    )

    # Determine role based on rank position (Requirement 9.2, 9.4)
    is_last_rank: bool = rank == world_size - 1
    prev_rank: int = rank - 1

    # Initialize generation state (Requirement 7.3)
    past_key_values: Any = None
    iteration: int = 0
    target_device: str = f"{device_type}:{device_id}"

    # Determine vocab_size from model config (needed for logits shape on last rank)
    vocab_size: int = model.model.config.vocab_size if is_last_rank else 0  # pyright: ignore[reportAny, reportUnusedVariable]

    # --- Worker loop body (tasks 4.2, 4.3, 4.4) ---
    # Wrap in try/except for RuntimeError from Gloo and model errors (Requirements: 1.6, 5.5, 6.4)
    try:
        # --- Task 4.2: Prefill iteration (iteration == 0) ---
        # Receive seq_len metadata from previous rank (Requirement 8.5)
        seq_len_tensor: torch.Tensor = recv_activation(  # pyright: ignore[reportUnknownVariableType]
            shape=(1,),
            dtype=torch.int64,
            src_rank=prev_rank,
            target_device=target_device,
        )
        seq_len: int = int(seq_len_tensor[0].item())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        logger.debug(f"Worker rank {rank}: prefill seq_len={seq_len}")

        # Receive hidden_states from previous rank (Requirement 1.2)
        hidden_states: torch.Tensor = recv_activation(  # pyright: ignore[reportUnknownVariableType]
            shape=(1, seq_len, hidden_size),
            dtype=dtype,
            src_rank=prev_rank,
            target_device=target_device,
        )

        # Forward through local TransformerShard with past_key_values=None (Requirement 8.3)
        output, new_past_key_values = model.forward(input_data=hidden_states, past_key_values=None)  # pyright: ignore[reportAny]

        if is_last_rank:
            # Last rank: extract last-position logits and send to rank 0 (Requirement 1.4)
            logits: torch.Tensor = output[:, -1:, :]  # shape (1, 1, vocab_size)  # pyright: ignore[reportAny]
            send_activation(logits.float(), dst_rank=0)
        else:
            # Middle rank: send seq_len metadata + hidden_states to next rank (Requirement 1.3)
            seq_len_out = torch.tensor([output.shape[1]], dtype=torch.int64)  # pyright: ignore[reportAny]
            send_activation(seq_len_out, dst_rank=rank + 1)
            send_activation(output, dst_rank=rank + 1)  # pyright: ignore[reportAny]

        # Receive token from rank 0 (Requirement 6.2)
        token_tensor: torch.Tensor = recv_activation(  # pyright: ignore[reportUnknownVariableType]
            shape=(1,),
            dtype=torch.int64,
            src_rank=0,
            target_device=target_device,
        )

        # Check for TERMINATION_SENTINEL — if received, exit loop (Requirement 6.4)
        if int(token_tensor[0].item()) == TERMINATION_SENTINEL:  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            logger.debug(f"Worker rank {rank}: received termination sentinel during prefill")
            past_key_values = None
            return

        # Store KV cache from prefill (Requirement 5.2)
        past_key_values = new_past_key_values  # pyright: ignore[reportAny]

        # Transition to decode phase
        iteration = 1

        # --- Task 4.3: Decode iterations (iteration > 0) ---
        while True:
            # Receive hidden_states from previous rank (shape (1, 1, hidden_size) during decode)
            hidden_states = recv_activation(  # pyright: ignore[reportUnknownVariableType]
                shape=(1, 1, hidden_size),
                dtype=dtype,
                src_rank=prev_rank,
                target_device=target_device,
            )

            # Forward through local TransformerShard with KV cache (Requirement 5.3, 5.4)
            output, past_key_values = model.forward(input_data=hidden_states, past_key_values=past_key_values)  # pyright: ignore[reportAny]

            if is_last_rank:
                # Last rank: send last-position logits to rank 0 (Requirement 1.4)
                logits = output[:, -1:, :]  # shape (1, 1, vocab_size)  # pyright: ignore[reportAny]
                send_activation(logits.float(), dst_rank=0)
            else:
                # Middle rank: send hidden_states to next rank (Requirement 1.3)
                send_activation(output, dst_rank=rank + 1)  # pyright: ignore[reportAny]

            # Receive next token from rank 0 (Requirement 6.2)
            token_tensor = recv_activation(  # pyright: ignore[reportUnknownVariableType]
                shape=(1,),
                dtype=torch.int64,
                src_rank=0,
                target_device=target_device,
            )

            # Check for TERMINATION_SENTINEL — if received, exit loop (Requirement 6.4)
            if int(token_tensor[0].item()) == TERMINATION_SENTINEL:  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                logger.debug(f"Worker rank {rank}: received termination sentinel during decode")
                past_key_values = None
                return

            # Update iteration counter (Requirement 8.4)
            iteration += 1

    except Exception as exc:
        # --- Task 4.4: Error handling and cleanup (Requirements: 1.6, 5.5, 6.4) ---
        logger.error(
            f"Worker loop error on rank {rank}: {type(exc).__name__}: {exc}",
            exc_info=True,
        )
        # Release KV cache for cleanup (Requirement 5.5)
        past_key_values = None
        return

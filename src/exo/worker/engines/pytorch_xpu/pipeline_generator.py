# pyright: reportUnusedImport=false
"""
Pipeline-Parallel Generation Pipeline

Orchestrates autoregressive text generation with pipeline parallelism. Unlike
tensor parallelism where ALL ranks compute simultaneously through all layers
with all-reduce synchronization, pipeline parallelism assigns contiguous layer
ranges to each rank with point-to-point communication between adjacent stages.

Rank 0 drives generation: it embeds tokens, forwards through its local layers,
sends activations to rank 1, and waits for the last rank to broadcast the
sampled token back. Non-rank-0 nodes run a worker loop that receives activations,
forwards through local layers, and either sends to the next rank or (if last rank)
samples a token and broadcasts it.

This module provides:
- pipeline_parallel_generate(): Generator called by rank 0 that orchestrates
  generation, receives sampled tokens from last rank, and yields GenerationResponse
- pipeline_parallel_worker_loop(): Blocking loop for non-rank-0 nodes

Requirements: 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4, 9.4, 9.5, 11.1, 11.2, 11.3
"""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from typing import Any

import torch
import torch.distributed as dist

from exo.api.types import (
    CompletionTokensDetails,
    GenerationStats,
    PromptTokensDetails,
    Usage,
)
from exo.shared.types.memory import Memory
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.pytorch_xpu.distributed import recv_activation, send_activation
from exo.worker.engines.pytorch_xpu.distributed_generator import sample_token
from exo.worker.engines.pytorch_xpu.pipeline_parallel_shard import PipelineParallelShard

logger = logging.getLogger(__name__)

TERMINATION_SENTINEL: int = -1
"""Special token ID broadcast by last rank to signal all ranks to exit generation."""


def pipeline_parallel_generate(
    model: PipelineParallelShard,
    tokenizer: Any,
    prompt: str,
    device: str,
    rank: int,
    world_size: int,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> Generator[GenerationResponse, None, None]:
    """Drive autoregressive generation on rank 0 (pipeline parallelism).

    For multi-stage pipelines (world_size > 1):
    1. Embed token → forward through local layers (rank 0's layer range)
    2. send_activation to rank 1
    3. Wait for broadcast of next token from last rank (world_size - 1)
    4. If EOS or max_tokens: break
    5. Yield GenerationResponse with token text

    For single-stage (world_size == 1):
    Rank 0 is also the last stage, so it gets logits directly and samples
    locally without any communication.

    Args:
        model: PipelineParallelShard with local layers on this rank's device.
        tokenizer: HuggingFace tokenizer with encode/decode methods.
        prompt: Input text prompt to generate from.
        device: Device string for tensor placement (e.g., "xpu:0", "cpu").
        rank: This node's rank (must be 0 for this function).
        world_size: Total number of pipeline stages.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (higher = more random).
        top_k: Top-k sampling parameter (None = disabled).
        top_p: Top-p (nucleus) sampling parameter (None = disabled).

    Yields:
        GenerationResponse objects containing generated tokens.

    Requirements: 4.1, 4.2, 4.3, 4.4, 11.1, 11.2, 11.3
    """
    logger.info(
        f"Starting pipeline-parallel generation: prompt_len={len(prompt)}, "
        f"max_tokens={max_tokens}, temperature={temperature}, "
        f"top_k={top_k}, top_p={top_p}, rank={rank}, world_size={world_size}"
    )

    # Tokenize prompt
    input_ids: list[int] = tokenizer.encode(prompt)  # pyright: ignore[reportAny]
    prompt_tokens: int = len(input_ids)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    logger.debug(f"Tokenized prompt: {prompt_tokens} tokens")

    # Determine EOS token IDs
    eos_token_ids: set[int] = set()
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:  # pyright: ignore[reportAny]
        eos_token_ids.add(int(tokenizer.eos_token_id))  # pyright: ignore[reportAny]
    if hasattr(tokenizer, "all_special_ids"):  # pyright: ignore[reportAny]
        for special_id in tokenizer.all_special_ids:  # pyright: ignore[reportAny]
            special_tok: str = tokenizer.decode([special_id], skip_special_tokens=False)  # pyright: ignore[reportAny]
            if special_tok in ("<|im_end|>", "<|endoftext|>", "<|end|>"):
                eos_token_ids.add(int(special_id))  # pyright: ignore[reportAny]

    logger.debug(f"EOS token IDs: {eos_token_ids}")

    # Single-stage mode: rank 0 is both first and last stage
    is_single_stage: bool = world_size == 1

    start_time: float = time.perf_counter()

    try:
        # --- Prefill Phase ---
        prefill_start = time.perf_counter()
        output, _kv_cache = model.forward(input_data=input_tensor)
        prefill_elapsed = time.perf_counter() - prefill_start

        logger.debug(f"Prefill completed in {prefill_elapsed:.3f}s")

        if is_single_stage:
            # Single stage: output is logits, sample directly
            first_token_id: int = sample_token(
                output, temperature=temperature, top_k=top_k, top_p=top_p
            )
        else:
            # Multi-stage: output is hidden_state, send to rank 1
            _send_with_shape(output, dst_rank=1)

            # Wait for token broadcast from last rank
            token_tensor = torch.tensor([0], dtype=torch.long, device="cpu")
            dist.broadcast(token_tensor, src=world_size - 1)
            first_token_id = int(token_tensor.item())

        # Calculate prefill stats
        prompt_tps: float = prompt_tokens / prefill_elapsed if prefill_elapsed > 0 else 0.0

        # Check for termination sentinel from last rank (error/abort case)
        if first_token_id == TERMINATION_SENTINEL:
            logger.debug("Received termination sentinel after prefill")
            return

        # Check if first token is EOS
        if first_token_id in eos_token_ids:
            logger.debug(f"First token is EOS ({first_token_id}), terminating")
            if is_single_stage:
                # Broadcast termination to self (no-op in single stage, but consistent)
                pass
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

        # Check max_tokens == 1
        if max_tokens <= 1:
            logger.debug("max_tokens reached (max_tokens=1)")
            first_token_text: str = tokenizer.decode([first_token_id], skip_special_tokens=True)  # pyright: ignore[reportAny]
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

        # Yield first token response
        first_token_text = tokenizer.decode([first_token_id], skip_special_tokens=True)  # pyright: ignore[reportAny]
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

        # --- Decode Loop ---
        completion_tokens: int = 1
        prev_token_id: int = first_token_id
        decode_start: float = time.perf_counter()

        for _ in range(max_tokens - 1):
            # Embed the received token and forward through local layers
            token_input = torch.tensor([[prev_token_id]], dtype=torch.long, device=device)

            step_start = time.perf_counter()
            output, _kv_cache = model.forward(input_data=token_input)
            step_elapsed_ms = (time.perf_counter() - step_start) * 1000.0

            if is_single_stage:
                # Single stage: output is logits, sample directly
                next_token_id: int = sample_token(
                    output, temperature=temperature, top_k=top_k, top_p=top_p
                )
            else:
                # Multi-stage: send hidden_state to rank 1, wait for token broadcast
                _send_with_shape(output, dst_rank=1)

                token_tensor = torch.tensor([0], dtype=torch.long, device="cpu")
                dist.broadcast(token_tensor, src=world_size - 1)
                next_token_id = int(token_tensor.item())

            completion_tokens += 1

            # Check for termination sentinel (error/abort from last rank)
            if next_token_id == TERMINATION_SENTINEL:
                logger.debug("Received termination sentinel during decode")
                return

            # Calculate generation stats
            decode_elapsed = time.perf_counter() - decode_start
            generation_tps: float = (
                (completion_tokens - 1) / decode_elapsed if decode_elapsed > 0 else 0.0
            )

            # Check EOS termination
            if next_token_id in eos_token_ids:
                logger.debug(
                    f"EOS token {next_token_id} at step {completion_tokens}, terminating"
                )
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

            # Check max_tokens termination
            if completion_tokens >= max_tokens:
                logger.debug(f"max_tokens ({max_tokens}) reached, terminating")
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

            # Yield intermediate token response
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

            # Update for next iteration
            prev_token_id = next_token_id

    except Exception as exc:
        # --- Error Handling (Requirements: 11.1, 11.2, 11.3) ---
        error_msg: str = (
            f"Pipeline-parallel generation error on rank {rank}: "
            f"{type(exc).__name__}: {exc}"
        )
        logger.error(error_msg, exc_info=True)

        # Yield error response
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


def _send_with_shape(tensor: torch.Tensor, dst_rank: int) -> None:
    """Send a tensor preceded by its seq_len dimension as metadata.

    Since recv_activation() requires knowing the shape ahead of time, and
    seq_len varies between prefill (seq_len > 1) and decode (seq_len = 1),
    we send a 1-element int64 tensor with seq_len before the actual activation.

    Args:
        tensor: The activation tensor of shape [batch, seq_len, hidden_size].
        dst_rank: Destination rank for the send.
    """
    seq_len_tensor = torch.tensor([tensor.shape[1]], dtype=torch.long, device="cpu")
    dist.send(seq_len_tensor, dst=dst_rank)
    send_activation(tensor, dst_rank=dst_rank)


def _recv_with_shape(
    hidden_size: int,
    src_rank: int,
    target_device: str,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Receive a tensor preceded by its seq_len dimension as metadata.

    Reads the seq_len metadata first, then allocates the correct buffer
    and receives the actual activation tensor.

    Args:
        hidden_size: Model hidden dimension (fixed across all transfers).
        src_rank: Source rank to receive from.
        target_device: Device to place the received tensor on.
        dtype: Expected tensor dtype.

    Returns:
        The received activation tensor on target_device.
    """
    seq_len_tensor = torch.tensor([0], dtype=torch.long, device="cpu")
    dist.recv(seq_len_tensor, src=src_rank)
    seq_len: int = int(seq_len_tensor.item())

    shape = (1, seq_len, hidden_size)
    return recv_activation(shape=shape, dtype=dtype, src_rank=src_rank, target_device=target_device)


def pipeline_parallel_worker_loop(
    model: PipelineParallelShard,
    device: str,
    rank: int,
    world_size: int,
    tokenizer: Any = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> None:
    """Worker loop for non-rank-0 pipeline stages.

    Runs a blocking loop that:
    1. Receives activation from previous rank (with shape metadata)
    2. Forwards through local layers
    3. If not last rank: sends activation to next rank (with shape metadata)
    4. If last rank: samples token, broadcasts to all ranks
    5. All non-last ranks wait for token broadcast from last rank
    6. Checks for termination (EOS or sentinel -1)
    7. Loops until termination

    For the first iteration (prefill), seq_len > 1. For subsequent iterations
    (decode), seq_len == 1. The shape metadata exchange handles this transparently.

    Args:
        model: PipelineParallelShard with local layers on this rank's device.
        device: Device string for tensor placement (e.g., "xpu:0", "cpu").
        rank: This node's rank (must be > 0 for this function).
        world_size: Total number of pipeline stages.
        tokenizer: HuggingFace tokenizer (used to determine EOS token IDs).
        temperature: Sampling temperature (higher = more random).
        top_k: Top-k sampling parameter (None = disabled).
        top_p: Top-p (nucleus) sampling parameter (None = disabled).

    Requirements: 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 9.4, 9.5
    """
    logger.info(
        f"Starting pipeline-parallel worker loop: rank={rank}/{world_size}, "
        f"device={device}, temperature={temperature}, top_k={top_k}, top_p={top_p}"
    )

    prev_rank: int = rank - 1
    next_rank: int = rank + 1
    is_last_rank: bool = rank == world_size - 1
    hidden_size: int = model.config.hidden_size

    # Determine EOS token IDs
    eos_token_ids: set[int] = set()
    if tokenizer is not None:
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:  # pyright: ignore[reportAny]
            eos_token_ids.add(int(tokenizer.eos_token_id))  # pyright: ignore[reportAny]
        if hasattr(tokenizer, "all_special_ids"):  # pyright: ignore[reportAny]
            for special_id in tokenizer.all_special_ids:  # pyright: ignore[reportAny]
                special_tok: str = tokenizer.decode([special_id], skip_special_tokens=False)  # pyright: ignore[reportAny]
                if special_tok in ("<|im_end|>", "<|endoftext|>", "<|end|>"):
                    eos_token_ids.add(int(special_id))  # pyright: ignore[reportAny]

    logger.debug(f"Worker rank={rank}: EOS token IDs: {eos_token_ids}")

    try:
        while True:
            # Step 1: Receive activation from previous rank (with shape metadata)
            hidden_state = _recv_with_shape(
                hidden_size=hidden_size,
                src_rank=prev_rank,
                target_device=device,
            )

            # Step 2: Forward through local layers
            output, _kv_cache = model.forward(input_data=hidden_state)

            # Step 3/4: Send or sample depending on position in pipeline
            if not is_last_rank:
                # Middle stage: send activation to next rank
                _send_with_shape(output, dst_rank=next_rank)

                # Wait for token broadcast from last rank
                token_tensor = torch.tensor([0], dtype=torch.long, device="cpu")
                dist.broadcast(token_tensor, src=world_size - 1)
                token_id: int = int(token_tensor.item())
            else:
                # Last stage: output is logits, sample token
                token_id = sample_token(
                    output, temperature=temperature, top_k=top_k, top_p=top_p
                )

                # Broadcast sampled token to all ranks
                token_tensor = torch.tensor([token_id], dtype=torch.long, device="cpu")
                dist.broadcast(token_tensor, src=rank)

            # Step 5: Check for termination
            if token_id == TERMINATION_SENTINEL:
                logger.debug(f"Worker rank={rank}: received termination sentinel, exiting")
                break

            if token_id in eos_token_ids:
                logger.debug(f"Worker rank={rank}: EOS token {token_id}, exiting")
                break

    except Exception as exc:
        logger.error(
            f"Pipeline-parallel worker error on rank {rank}: "
            f"{type(exc).__name__}: {exc}",
            exc_info=True,
        )
        # If last rank, broadcast termination sentinel so other ranks can exit
        if is_last_rank:
            try:
                sentinel_tensor = torch.tensor(
                    [TERMINATION_SENTINEL], dtype=torch.long, device="cpu"
                )
                dist.broadcast(sentinel_tensor, src=rank)
            except Exception:
                logger.warning(
                    f"Worker rank={rank}: failed to broadcast termination sentinel "
                    f"after error",
                    exc_info=True,
                )
        raise

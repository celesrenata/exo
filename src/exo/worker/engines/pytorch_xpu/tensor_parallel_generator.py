# pyright: reportUnusedImport=false
"""
Tensor-Parallel Generation Pipeline

Orchestrates autoregressive text generation with tensor parallelism. Unlike
pipeline parallelism where only rank 0 drives generation and activations flow
sequentially through ranks, here ALL ranks compute simultaneously through all
layers, with all-reduce synchronization happening inside TensorParallelShard.forward().

Rank 0 handles token sampling and broadcasts the sampled token to all other ranks.
Non-rank-0 nodes run a simple worker loop: receive token → forward → wait.

This module provides:
- tensor_parallel_generate(): Generator called by rank 0 that orchestrates
  generation, samples tokens, and yields GenerationResponse objects
- tensor_parallel_worker_loop(): Blocking loop for non-rank-0 nodes
- TPPerformanceMetrics: Dataclass tracking all-reduce latencies and throughput

Requirements: 6.1, 6.2, 6.3, 6.4, 6.6, 10.1, 10.2, 10.3, 10.4, 10.5, 11.1, 11.2
"""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from exo.shared.types.api import (
    CompletionTokensDetails,
    GenerationStats,
    Memory,
    PromptTokensDetails,
    Usage,
)
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.pytorch_xpu.distributed_generator import sample_token
from exo.worker.engines.pytorch_xpu.tensor_parallel_shard import TensorParallelShard

logger = logging.getLogger(__name__)

TERMINATION_SENTINEL: int = -1
"""Special token ID broadcast by rank 0 to signal all ranks to exit generation."""

# Performance threshold: log warning if decode-phase all-reduce exceeds this (ms)
ALLREDUCE_LATENCY_WARNING_THRESHOLD_MS: float = 10.0
"""Default threshold for all-reduce latency warnings (milliseconds)."""

# TB4 theoretical bandwidth: 40 Gbps = 5 GB/s
TB4_BANDWIDTH_BYTES_PER_SECOND: float = 5_000_000_000.0
"""Theoretical TB4 bandwidth in bytes/second (40 Gbps)."""



@dataclass
class TPPerformanceMetrics:
    """Performance metrics collected during tensor-parallel generation.

    Tracks all-reduce latencies, throughput, and TB4 bandwidth utilization
    to enable comparison with pipeline-parallel performance and identify
    bottlenecks.

    Requirements: 10.1, 10.2, 10.3, 10.4
    """

    allreduce_latencies_ms: list[float] = field(default_factory=list)
    """Wall-clock time for each all-reduce operation during generation (ms)."""

    prefill_time_seconds: float = 0.0
    """Total wall-clock time for the prefill phase."""

    decode_tokens_per_second: float = 0.0
    """Tokens generated per second during the decode phase."""

    total_allreduce_bytes: int = 0
    """Total bytes transferred across all all-reduce operations."""

    total_allreduce_time_seconds: float = 0.0
    """Total wall-clock time spent in all-reduce operations."""

    @property
    def mean_allreduce_ms(self) -> float:
        """Mean all-reduce latency in milliseconds."""
        if not self.allreduce_latencies_ms:
            return 0.0
        return sum(self.allreduce_latencies_ms) / len(self.allreduce_latencies_ms)

    @property
    def p99_allreduce_ms(self) -> float:
        """99th percentile all-reduce latency in milliseconds."""
        if not self.allreduce_latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.allreduce_latencies_ms)
        idx = int(len(sorted_latencies) * 0.99)
        # Clamp to last element
        idx = min(idx, len(sorted_latencies) - 1)
        return sorted_latencies[idx]

    @property
    def tb4_bandwidth_utilization(self) -> float:
        """Effective TB4 bandwidth utilization as a fraction of theoretical 40 Gbps.

        Calculated as: total_bytes / total_time / theoretical_bandwidth.
        Returns 0.0 if no all-reduce operations were performed.

        Requirements: 10.3
        """
        if self.total_allreduce_time_seconds <= 0.0:
            return 0.0
        effective_bandwidth = self.total_allreduce_bytes / self.total_allreduce_time_seconds
        return effective_bandwidth / TB4_BANDWIDTH_BYTES_PER_SECOND



def tensor_parallel_generate(
    model: TensorParallelShard,
    tokenizer: Any,
    prompt: str,
    device: str,
    rank: int,
    world_size: int,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    model_id: str = "",
) -> Generator[GenerationResponse, None, None]:
    """
    Tensor-parallel generation orchestrator (rank 0 only).

    All ranks execute the forward pass simultaneously via all-reduce
    (which happens inside TensorParallelShard.forward()). Rank 0 samples
    tokens from the resulting logits and broadcasts the sampled token ID
    to all other ranks for the next decode step.

    Unlike pipeline parallelism, there is no sequential activation passing.
    All ranks have identical logits after each forward pass due to all-reduce
    synchronization within the model.

    Args:
        model: TensorParallelShard with sharded weights on this rank's device.
        tokenizer: HuggingFace tokenizer with encode/decode methods.
        prompt: Input text prompt to generate from.
        device: Device string for tensor placement (e.g., "xpu:0", "cpu").
        rank: This node's rank (must be 0 for this function).
        world_size: Total number of tensor-parallel ranks.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (higher = more random).
        top_k: Top-k sampling parameter (None = disabled).
        top_p: Top-p (nucleus) sampling parameter (None = disabled).
        model_id: Model identifier for logging.

    Yields:
        GenerationResponse objects containing generated tokens.

    Requirements: 6.1, 6.2, 6.3, 6.4, 6.6, 10.1, 10.2, 10.3, 10.4
    """
    logger.info(
        f"Starting tensor-parallel generation: prompt_len={len(prompt)}, "
        f"max_tokens={max_tokens}, temperature={temperature}, "
        f"top_k={top_k}, top_p={top_p}, rank={rank}, world_size={world_size}, "
        f"model_id={model_id}"
    )

    # Performance metrics collection
    metrics = TPPerformanceMetrics()

    # Tokenize prompt
    input_ids: list[int] = tokenizer.encode(prompt)  # pyright: ignore[reportAny]
    prompt_tokens: int = len(input_ids)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    logger.debug(f"Tokenized prompt: {prompt_tokens} tokens")

    # Determine EOS token IDs
    eos_token_ids: set[int] = set()
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:  # pyright: ignore[reportAny]
        eos_token_ids.add(int(tokenizer.eos_token_id))  # pyright: ignore[reportAny]
    if hasattr(tokenizer, "additional_special_tokens_ids"):  # pyright: ignore[reportAny]
        for special_id in tokenizer.additional_special_tokens_ids:  # pyright: ignore[reportAny]
            eos_token_ids.add(int(special_id))  # pyright: ignore[reportAny]
    if hasattr(tokenizer, "all_special_ids"):  # pyright: ignore[reportAny]
        for special_id in tokenizer.all_special_ids:  # pyright: ignore[reportAny]
            special_tok: str = tokenizer.decode([special_id], skip_special_tokens=False)  # pyright: ignore[reportAny]
            if special_tok in ("<|im_end|>", "<|endoftext|>", "<|end|>"):
                eos_token_ids.add(int(special_id))  # pyright: ignore[reportAny]

    logger.debug(f"EOS token IDs: {eos_token_ids}")

    # Initialize generation state
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    start_time: float = time.perf_counter()

    try:
        # --- Prefill Phase ---
        # Broadcast input token IDs to all ranks so they all process the same prompt
        # (Requirement 6.1: ALL ranks process the full tokenized prompt simultaneously)
        # NOTE: Gloo backend only supports CPU tensors for collective operations.
        # All tensors passed to dist.broadcast() must be on CPU.
        token_count_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device="cpu")
        dist.broadcast(token_count_tensor, src=0)
        input_tensor_cpu = input_tensor.cpu()
        dist.broadcast(input_tensor_cpu, src=0)

        # Forward full prompt through all layers (all-reduce happens internally)
        prefill_start = time.perf_counter()
        logits, past_key_values = model.forward(
            input_data=input_tensor,
            past_key_values=None,
        )
        prefill_elapsed = time.perf_counter() - prefill_start
        metrics.prefill_time_seconds = prefill_elapsed

        logger.debug(f"Prefill completed in {prefill_elapsed:.3f}s")

        # Sample first token from logits (Requirement 6.2: rank 0 samples)
        first_token_id: int = sample_token(
            logits, temperature=temperature, top_k=top_k, top_p=top_p
        )

        # Broadcast sampled token to all ranks (Requirement 6.3)
        token_tensor = torch.tensor([first_token_id], dtype=torch.long, device="cpu")
        dist.broadcast(token_tensor, src=0)

        # Calculate prefill stats
        prompt_tps: float = prompt_tokens / prefill_elapsed if prefill_elapsed > 0 else 0.0

        # Check if first token is EOS (Requirement 6.6)
        if first_token_id in eos_token_ids:
            logger.debug(f"First token is EOS ({first_token_id}), terminating")
            # Broadcast TERMINATION_SENTINEL to signal all ranks to exit
            sentinel = torch.tensor([TERMINATION_SENTINEL], dtype=torch.long, device="cpu")
            dist.broadcast(sentinel, src=0)

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
            sentinel = torch.tensor([TERMINATION_SENTINEL], dtype=torch.long, device="cpu")
            dist.broadcast(sentinel, src=0)

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
            # Create single-token input for decode step
            token_input = torch.tensor([[prev_token_id]], dtype=torch.long, device=device)

            # Forward through all layers with KV cache (all-reduce inside)
            step_start = time.perf_counter()
            logits, past_key_values = model.forward(
                input_data=token_input,
                past_key_values=past_key_values,
            )
            step_elapsed_ms = (time.perf_counter() - step_start) * 1000.0

            # Track all-reduce latency (approximate: forward includes compute + all-reduce)
            metrics.allreduce_latencies_ms.append(step_elapsed_ms)

            # Log warning if latency exceeds threshold (Requirement 10.5)
            if step_elapsed_ms > ALLREDUCE_LATENCY_WARNING_THRESHOLD_MS:
                logger.warning(
                    f"Decode step latency {step_elapsed_ms:.2f}ms exceeds threshold "
                    f"({ALLREDUCE_LATENCY_WARNING_THRESHOLD_MS}ms). "
                    f"Possible TB4 congestion or asymmetric topology."
                )

            # Sample next token (Requirement 6.2)
            next_token_id: int = sample_token(
                logits, temperature=temperature, top_k=top_k, top_p=top_p
            )
            completion_tokens += 1

            # Calculate generation stats
            decode_elapsed = time.perf_counter() - decode_start
            generation_tps: float = (completion_tokens - 1) / decode_elapsed if decode_elapsed > 0 else 0.0
            metrics.decode_tokens_per_second = generation_tps

            # Check EOS termination (Requirement 6.6)
            if next_token_id in eos_token_ids:
                logger.debug(f"EOS token {next_token_id} at step {completion_tokens}, terminating")
                sentinel = torch.tensor([TERMINATION_SENTINEL], dtype=torch.long, device="cpu")
                dist.broadcast(sentinel, src=0)

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

            # Check max_tokens termination (Requirement 6.6)
            if completion_tokens >= max_tokens:
                logger.debug(f"max_tokens ({max_tokens}) reached, terminating")
                sentinel = torch.tensor([TERMINATION_SENTINEL], dtype=torch.long, device="cpu")
                dist.broadcast(sentinel, src=0)

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

            # Not terminated: broadcast token to all ranks (Requirement 6.3)
            token_tensor = torch.tensor([next_token_id], dtype=torch.long, device="cpu")
            dist.broadcast(token_tensor, src=0)

            # Decode token text and yield response
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
        # --- Error Handling (Requirements: 11.1, 11.2) ---
        error_msg: str = (
            f"Tensor-parallel generation error on rank {rank}: "
            f"{type(exc).__name__}: {exc}"
        )
        logger.error(error_msg, exc_info=True)

        # Best-effort broadcast TERMINATION_SENTINEL to all ranks
        try:
            sentinel = torch.tensor([TERMINATION_SENTINEL], dtype=torch.long, device="cpu")
            dist.broadcast(sentinel, src=0)
        except Exception as sentinel_exc:
            logger.warning(
                f"Failed to broadcast termination sentinel: {sentinel_exc}"
            )

        # Release KV caches
        past_key_values = None

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



def tensor_parallel_worker_loop(
    model: TensorParallelShard,
    device: str,
    rank: int,
    world_size: int,
) -> None:
    """
    Worker loop for non-rank-0 nodes in tensor-parallel generation.

    Much simpler than the pipeline-parallel worker loop because all-reduce
    happens inside TensorParallelShard.forward() — no explicit activation
    sending is needed. The worker just:
    1. Receives token broadcasts from rank 0
    2. Executes forward pass (all-reduce synchronization is internal)
    3. Waits for next token broadcast
    4. Exits cleanly on TERMINATION_SENTINEL

    Args:
        model: TensorParallelShard with sharded weights on this rank's device.
        device: Device string for tensor placement (e.g., "xpu:0", "cpu").
        rank: This node's rank (must be > 0).
        world_size: Total number of tensor-parallel ranks.

    Requirements: 6.1, 6.2, 6.3, 6.4, 6.6
    """
    logger.info(
        f"Starting tensor-parallel worker loop: rank={rank}, "
        f"world_size={world_size}, device={device}"
    )

    past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None

    try:
        # --- Prefill Phase ---
        # Receive prompt token count and input_ids from rank 0
        # NOTE: Gloo backend only supports CPU tensors for collective operations.
        # All tensors passed to dist.broadcast() must be on CPU, then moved to device.
        token_count_tensor = torch.tensor([0], dtype=torch.long, device="cpu")
        dist.broadcast(token_count_tensor, src=0)
        prompt_tokens: int = int(token_count_tensor[0].item())

        # Receive the full input tensor (broadcast on CPU, then move to device for forward)
        input_tensor = torch.zeros(1, prompt_tokens, dtype=torch.long, device="cpu")
        dist.broadcast(input_tensor, src=0)
        input_tensor = input_tensor.to(device)

        # Forward full prompt (all-reduce happens internally in model.forward())
        _logits, past_key_values = model.forward(
            input_data=input_tensor,
            past_key_values=None,
        )

        # Receive first sampled token from rank 0
        token_tensor = torch.tensor([0], dtype=torch.long, device="cpu")
        dist.broadcast(token_tensor, src=0)
        token_id: int = int(token_tensor[0].item())

        # Check for immediate termination
        if token_id == TERMINATION_SENTINEL:
            logger.debug(f"Worker rank {rank}: received termination after prefill")
            past_key_values = None
            return

        # --- Decode Loop ---
        while True:
            # Forward the single token with KV cache
            token_input = torch.tensor([[token_id]], dtype=torch.long, device=device)
            _logits, past_key_values = model.forward(
                input_data=token_input,
                past_key_values=past_key_values,
            )

            # Receive next token from rank 0
            token_tensor = torch.tensor([0], dtype=torch.long, device="cpu")
            dist.broadcast(token_tensor, src=0)
            token_id = int(token_tensor[0].item())

            # Check for TERMINATION_SENTINEL
            if token_id == TERMINATION_SENTINEL:
                logger.debug(f"Worker rank {rank}: received termination sentinel")
                past_key_values = None
                return

    except Exception as exc:
        # Error handling: log and release KV caches
        logger.error(
            f"Tensor-parallel worker error on rank {rank}: "
            f"{type(exc).__name__}: {exc}",
            exc_info=True,
        )
        past_key_values = None
        return

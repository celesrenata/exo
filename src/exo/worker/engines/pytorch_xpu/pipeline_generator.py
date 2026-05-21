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

Requirements: 2.1, 2.2, 2.3, 2.5, 2.7, 2.8, 2.9, 2.11, 2.14,
             3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4, 9.4, 9.5, 11.1, 11.2, 11.3
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
from exo.worker.engines.pytorch_xpu.async_output_streamer import AsyncOutputStreamer
from exo.worker.engines.pytorch_xpu.buffer_pool import CommunicationBufferPool
from exo.worker.engines.pytorch_xpu.distributed import (
    DecodeActivationProtocol,
    TokenResultPacket,
    negotiate_decode_activation_protocol,
    receive_decode_activation_fast,
    receive_token_results_from_final_rank,
    recv_activation_generic as recv_activation,
    respond_to_decode_protocol_negotiation,
    send_activation_generic as send_activation,
    send_decode_activation_fast,
    send_token_results_to_rank_zero,
)
from exo.worker.engines.pytorch_xpu.distributed_generator import sample_token
from exo.worker.engines.pytorch_xpu.instrumentation import (
    EventMode,
    PerformanceRecorder,
)
from exo.worker.engines.pytorch_xpu.on_device_sampling import OnDeviceSampler
from exo.worker.engines.pytorch_xpu.pipeline_config import (
    PytorchXpuOptimizationConfiguration,
)
from exo.worker.engines.pytorch_xpu.pipeline_parallel_shard import PipelineParallelShard

logger = logging.getLogger(__name__)

TERMINATION_SENTINEL: int = -1
"""Special token ID broadcast by last rank to signal all ranks to exit generation."""

SHUTDOWN_SENTINEL: int = -2
"""Special token ID to signal worker processes to shut down entirely."""

_BACKPRESSURE_SLEEP_SECONDS: float = 0.001
"""Sleep interval when waiting for backpressure to clear (1ms)."""


def _create_recorder_from_configuration(
    configuration: PytorchXpuOptimizationConfiguration | None,
    rank: int,
    stage: int | None = None,
) -> PerformanceRecorder:
    """Create a PerformanceRecorder based on optimization configuration.

    If configuration is None or instrumentation is disabled, returns a
    disabled recorder that acts as a no-op for all span calls.

    Args:
        configuration: Optimization configuration with instrumentation flags.
        rank: Distributed rank for this recorder.
        stage: Pipeline stage index for this recorder.

    Returns:
        A PerformanceRecorder instance, enabled or disabled per configuration.
    """
    if configuration is None:
        return PerformanceRecorder(enabled=False, rank=rank, stage=stage)
    return PerformanceRecorder(
        enabled=configuration.enable_performance_instrumentation,
        rank=rank,
        stage=stage,
    )


def _is_fast_path_enabled(
    optimization_configuration: PytorchXpuOptimizationConfiguration | None,
) -> bool:
    """Check whether the decode fast path is enabled in configuration.

    Returns True when the configuration explicitly enables the fast path,
    or when no configuration is provided (defaults to enabled).
    """
    if optimization_configuration is None:
        return True
    return optimization_configuration.enable_decode_fast_path


def _create_on_device_sampler(
    optimization_configuration: PytorchXpuOptimizationConfiguration | None,
    device: str,
) -> OnDeviceSampler | None:
    """Create an OnDeviceSampler if on-device sampling is enabled.

    Returns None when the configuration disables on-device sampling or when
    the sampler cannot be created (logs at WARNING level per Req 11.3).

    Args:
        optimization_configuration: Optimization configuration with flags.
        device: Device string for the sampler (e.g., "xpu:0", "cpu").

    Returns:
        An OnDeviceSampler instance, or None if disabled or creation failed.
    """
    if optimization_configuration is None:
        return None
    if not optimization_configuration.enable_on_device_sampling:
        return None
    try:
        sampler = OnDeviceSampler(device=torch.device(device))
        logger.info("On-device sampling enabled on %s", device)
        return sampler
    except Exception as exc:
        logger.warning(
            "Failed to create OnDeviceSampler, falling back to CPU sampling: %s",
            exc,
        )
        return None


def _sample_with_on_device_sampler(
    logits: torch.Tensor,
    on_device_sampler: OnDeviceSampler | None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    """Sample a token using on-device sampling with fallback to CPU sampling.

    When on_device_sampler is provided, attempts to sample entirely on device.
    If on-device sampling fails at runtime, falls back to the existing
    sample_token() function and logs at WARNING level (Req 11.3).

    Args:
        logits: Raw logits tensor from the model.
        on_device_sampler: OnDeviceSampler instance, or None to use CPU path.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
        top_p: Top-p sampling parameter.

    Returns:
        Sampled token ID as integer.
    """
    if on_device_sampler is None:
        return sample_token(
            logits, temperature=temperature, top_k=top_k, top_p=top_p
        )

    try:
        # Normalize logits shape: extract last-position logits
        sampling_logits = logits
        if sampling_logits.dim() == 3:
            sampling_logits = sampling_logits[:, -1, :]  # [1, vocab_size]

        # Sample on device — returns [1] int64 tensor on device
        token_id_tensor = on_device_sampler.sample(
            sampling_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # If the token is already on CPU (e.g., CPU device or test mode),
        # read it directly. Otherwise, use async transfer for XPU→CPU.
        if token_id_tensor.device.type == "cpu":
            return int(token_id_tensor.item())

        # Transfer token ID to CPU asynchronously (XPU → pinned CPU memory)
        cpu_tensor = on_device_sampler.transfer_token_to_cpu_async(token_id_tensor)

        # Read the token ID (blocks until async copy completes)
        return int(cpu_tensor.item())

    except Exception as exc:
        logger.warning(
            "On-device sampling failed, falling back to CPU sampling: %s",
            exc,
        )
        return sample_token(
            logits, temperature=temperature, top_k=top_k, top_p=top_p
        )


def _get_default_process_group() -> dist.ProcessGroup:
    """Return the default process group for distributed communication.

    The default process group is initialized by ``init_process_group()`` and
    is used for all pipeline-parallel communication unless a specific group
    is provided.
    """
    group = dist.group.WORLD
    if group is None:
        raise RuntimeError(
            "Default process group is not initialized. "
            "Call dist.init_process_group() before using pipeline communication."
        )
    return group  # pyright: ignore[reportReturnType]


def _activation_matches_protocol(
    activation: torch.Tensor,
    protocol: DecodeActivationProtocol,
) -> bool:
    """Check whether an activation tensor matches the negotiated protocol.

    Returns True if the activation's shape and dtype match the protocol's
    expected shape and dtype. Used to decide whether to use the fast path
    or fall back to the generic path.
    """
    if tuple(activation.shape) != protocol.shape:
        return False
    if str(activation.dtype) != protocol.dtype_name:
        return False
    if protocol.requires_contiguous and not activation.is_contiguous():
        return False
    return True



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
    performance_recorder: PerformanceRecorder | None = None,
    optimization_configuration: PytorchXpuOptimizationConfiguration | None = None,
    async_output_streamer: AsyncOutputStreamer | None = None,
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

    After prefill completes, negotiates the decode fast-path protocol with
    rank 1 (if enabled). During decode, uses the fast path for activation
    sends when the protocol is available and the activation shape matches.
    Falls back to the generic path otherwise.

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
        performance_recorder: Optional pre-configured PerformanceRecorder instance.
            If provided, takes precedence over optimization_configuration.
        optimization_configuration: Optional configuration used to create a
            recorder when performance_recorder is not provided.
        async_output_streamer: Optional AsyncOutputStreamer instance for
            decoupling token output from the decode loop. When provided and
            ``enable_async_output`` is True in the optimization configuration,
            token IDs are pushed into the streamer immediately after sampling
            (before tokenizer decode), allowing an async consumer to read
            tokens without blocking the decode loop. Backpressure is applied
            when the streamer's queue exceeds its max_pending threshold.

    Yields:
        GenerationResponse objects containing generated tokens.

    Requirements: 2.1, 2.7, 2.8, 2.9, 2.11, 4.1, 4.2, 4.3, 4.4, 8.1, 8.2, 8.4, 9.1, 9.2, 9.3, 11.1, 11.2, 11.3
    """
    logger.info(
        f"Starting pipeline-parallel generation: prompt_len={len(prompt)}, "
        f"max_tokens={max_tokens}, temperature={temperature}, "
        f"top_k={top_k}, top_p={top_p}, rank={rank}, world_size={world_size}"
    )

    # Initialize performance recorder
    recorder: PerformanceRecorder
    if performance_recorder is not None:
        recorder = performance_recorder
    else:
        recorder = _create_recorder_from_configuration(
            optimization_configuration, rank=rank, stage=0
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

    # Determine whether async output streaming is active (Requirements 9.1, 9.2, 9.3)
    # When enabled, token IDs are pushed into the streamer immediately after
    # sampling, before tokenizer decode. This decouples the API consumer from
    # the decode loop — the consumer reads from the streamer asynchronously.
    async_output_active: bool = (
        async_output_streamer is not None
        and optimization_configuration is not None
        and optimization_configuration.enable_async_output
    )

    # Decode fast-path state (initialized after prefill)
    fast_path_enabled: bool = _is_fast_path_enabled(optimization_configuration) and not is_single_stage
    send_protocol: DecodeActivationProtocol | None = None
    buffer_pool: CommunicationBufferPool | None = None

    # --- On-device sampling (Requirements 8.1, 8.2, 8.3, 8.4, 8.5) ---
    # When enabled, sampling executes entirely on the XPU device. The only
    # CPU transfer is a single int64 token ID via non-blocking copy. Falls
    # back to CPU sampling (sample_token) on failure (Req 11.3).
    on_device_sampler: OnDeviceSampler | None = _create_on_device_sampler(
        optimization_configuration, device
    )

    _start_time: float = time.perf_counter()

    try:
        # --- Prefill Phase ---
        with recorder.span(
            "prefill",
            mode="prefill",
            metadata={"prompt_tokens": prompt_tokens},
        ):
            prefill_start = time.perf_counter()
            output, _kv_cache = model.forward(input_data=input_tensor)
            prefill_elapsed = time.perf_counter() - prefill_start

        logger.debug(f"Prefill completed in {prefill_elapsed:.3f}s")

        if is_single_stage:
            # Single stage: output is logits, sample directly
            # Use on-device sampling when enabled (Req 8.1, 8.4, 11.1)
            first_token_id: int = _sample_with_on_device_sampler(
                output,
                on_device_sampler,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        else:
            # Multi-stage: output is hidden_state, send to rank 1
            _send_with_shape(output, dst_rank=1, performance_recorder=recorder)

            # Wait for token broadcast from last rank
            token_tensor = torch.tensor([0], dtype=torch.long, device="cpu")
            dist.broadcast(token_tensor, src=world_size - 1)
            first_token_id = int(token_tensor.item())

        # Calculate prefill stats
        prompt_tps: float = prompt_tokens / prefill_elapsed if prefill_elapsed > 0 else 0.0

        # Check for termination sentinel from last rank (error/abort case)
        if first_token_id == TERMINATION_SENTINEL:
            logger.debug("Received termination sentinel after prefill")
            if async_output_active:
                assert async_output_streamer is not None
                async_output_streamer.signal_end()
            return

        # Check if first token is EOS
        if first_token_id in eos_token_ids:
            logger.debug(f"First token is EOS ({first_token_id}), terminating")
            if is_single_stage:
                # Broadcast termination to self (no-op in single stage, but consistent)
                pass
            if async_output_active:
                assert async_output_streamer is not None
                async_output_streamer.put_token(first_token_id)
                async_output_streamer.signal_end()
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
            if async_output_active:
                assert async_output_streamer is not None
                async_output_streamer.put_token(first_token_id)
                async_output_streamer.signal_end()
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
        # When async output is active, push the token into the streamer
        # immediately (Requirement 9.1, 9.2, 9.3). The streamer makes the
        # token available to the async consumer without waiting for tokenizer
        # decode or API consumption.
        if async_output_active:
            assert async_output_streamer is not None  # type narrowing
            _continue = async_output_streamer.put_token(first_token_id)
            if not _continue:
                # Backpressure active — wait for consumer to drain
                while not async_output_streamer.should_resume:
                    time.sleep(_BACKPRESSURE_SLEEP_SECONDS)
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

        # --- Negotiate Decode Fast Path (after prefill, before decode loop) ---
        if fast_path_enabled:
            try:
                process_group = _get_default_process_group()
                hidden_size: int = model.config.hidden_size
                maximum_microbatch_size: int = 1  # Single-request decode

                if optimization_configuration is not None:
                    maximum_microbatch_size = optimization_configuration.maximum_decode_microbatch_size

                send_protocol = negotiate_decode_activation_protocol(
                    process_group=process_group,
                    local_rank=rank,
                    world_size=world_size,
                    hidden_size=hidden_size,
                    dtype=torch.bfloat16,
                    maximum_microbatch_size=maximum_microbatch_size,
                )

                # Rank 0 also responds to upstream negotiation (returns None for rank 0)
                respond_to_decode_protocol_negotiation(
                    process_group=process_group,
                    local_rank=rank,
                    world_size=world_size,
                    hidden_size=hidden_size,
                    dtype=torch.bfloat16,
                    maximum_microbatch_size=maximum_microbatch_size,
                )

                # Create buffer pool for the decode session
                buffer_pool = CommunicationBufferPool()

                logger.info(
                    "Rank %d: decode fast path negotiated, send_protocol=%s",
                    rank,
                    f"shape={send_protocol.shape}" if send_protocol else "None (final rank)",
                )
            except Exception as exc:
                logger.warning(
                    "Rank %d: decode fast path negotiation failed, "
                    "falling back to generic path: %s",
                    rank,
                    exc,
                )
                send_protocol = None
                buffer_pool = None

        # --- Decode Loop ---
        completion_tokens: int = 1
        prev_token_id: int = first_token_id
        decode_start: float = time.perf_counter()

        # Determine whether to defer tokenizer decode (Requirement 1.4, 9.1)
        defer_tokenizer_decode: bool = (
            optimization_configuration is not None
            and optimization_configuration.enable_sync_removal
        )

        # State for deferred tokenizer decode: holds the token ID that has
        # been sampled but not yet decoded/yielded. The decode happens after
        # the NEXT forward pass is launched, overlapping CPU tokenizer work
        # with GPU compute.
        _pending_token_id: int | None = None
        _pending_completion_tokens: int = 0

        with recorder.span(
            "decode_total",
            mode="decode",
            metadata={"max_tokens": max_tokens},
        ):
            for step_index in range(max_tokens - 1):
                # Scheduler wait placeholder — records near-zero time until
                # continuous batching is implemented and real scheduling
                # latency is introduced.
                with recorder.span(
                    "scheduler_wait",
                    mode="decode",
                    metadata={"step": step_index},
                ):
                    pass

                # End-to-end token latency: from decode step start to token
                # availability (includes compute + communication).
                with recorder.span(
                    "end_to_end_token_latency",
                    mode="decode",
                    metadata={"step": step_index},
                ):
                    # Decode step: forward pass through local layers
                    with recorder.span(
                        "decode_step",
                        mode="decode",
                        metadata={"step": step_index},
                    ):
                        # Embed the received token and forward through local layers
                        token_input = torch.tensor(
                            [[prev_token_id]], dtype=torch.long, device=device
                        )

                        step_start = time.perf_counter()
                        output, _kv_cache = model.forward(input_data=token_input)
                        _step_elapsed_ms = (time.perf_counter() - step_start) * 1000.0

                    # --- Deferred tokenizer decode (Requirement 1.4, 9.1) ---
                    # After the forward pass has been launched (GPU work is
                    # queued), decode and yield the PREVIOUS token. This
                    # overlaps CPU-side tokenizer work with GPU compute.
                    if defer_tokenizer_decode and _pending_token_id is not None:
                        _deferred_decode_elapsed = time.perf_counter() - decode_start
                        _deferred_generation_tps: float = (
                            (_pending_completion_tokens - 1) / _deferred_decode_elapsed
                            if _deferred_decode_elapsed > 0
                            else 0.0
                        )
                        _deferred_token_text: str = tokenizer.decode(  # pyright: ignore[reportAny]
                            [_pending_token_id], skip_special_tokens=True
                        )
                        yield GenerationResponse(
                            text=_deferred_token_text,
                            token=_pending_token_id,
                            finish_reason=None,
                            usage=Usage(
                                prompt_tokens=prompt_tokens,
                                completion_tokens=_pending_completion_tokens,
                                total_tokens=prompt_tokens + _pending_completion_tokens,
                                prompt_tokens_details=PromptTokensDetails(
                                    cached_tokens=0, audio_tokens=0
                                ),
                                completion_tokens_details=CompletionTokensDetails(
                                    reasoning_tokens=0, audio_tokens=0
                                ),
                            ),
                            stats=GenerationStats(
                                prompt_tps=prompt_tps,
                                generation_tps=_deferred_generation_tps,
                                prompt_tokens=prompt_tokens,
                                generation_tokens=_pending_completion_tokens,
                                peak_memory_usage=Memory(in_bytes=0),
                            ),
                        )
                        _pending_token_id = None

                    if is_single_stage:
                        # Single stage: output is logits, sample directly
                        # Use on-device sampling when enabled (Req 8.1, 8.4, 11.1)
                        next_token_id: int = _sample_with_on_device_sampler(
                            output,
                            on_device_sampler,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                        )
                    else:
                        # Multi-stage: send hidden_state to rank 1, receive token from last rank
                        _send_decode_activation_rank0(
                            output=output,
                            send_protocol=send_protocol,
                            buffer_pool=buffer_pool,
                            recorder=recorder,
                        )

                        # Receive token result from final rank
                        next_token_id = _receive_token_rank0(
                            world_size=world_size,
                            send_protocol=send_protocol,
                            recorder=recorder,
                        )

                completion_tokens += 1
                recorder.increment_counter("tokens_generated")

                # --- Async output streaming (Requirements 9.1, 9.2, 9.3) ---
                # Push the token into the streamer immediately after sampling,
                # before tokenizer decode or EOS checking. This ensures the
                # async consumer receives tokens without waiting for CPU-side
                # processing. The next forward pass has already been launched
                # (via deferred decode pattern), so the GPU is not idle.
                if async_output_active:
                    assert async_output_streamer is not None  # type narrowing
                    _continue = async_output_streamer.put_token(next_token_id)
                    if not _continue:
                        # Backpressure active — wait for consumer to drain
                        while not async_output_streamer.should_resume:
                            time.sleep(_BACKPRESSURE_SLEEP_SECONDS)

                # Check for termination sentinel (error/abort from last rank)
                if next_token_id == TERMINATION_SENTINEL:
                    logger.debug("Received termination sentinel during decode")
                    # Signal end to streamer before returning
                    if async_output_active:
                        assert async_output_streamer is not None
                        async_output_streamer.signal_end()
                    return

                # Calculate generation stats
                decode_elapsed = time.perf_counter() - decode_start
                generation_tps: float = (
                    (completion_tokens - 1) / decode_elapsed
                    if decode_elapsed > 0
                    else 0.0
                )

                # Check EOS termination
                if next_token_id in eos_token_ids:
                    logger.debug(
                        f"EOS token {next_token_id} at step {completion_tokens}, terminating"
                    )
                    # Signal end to streamer before returning
                    if async_output_active:
                        assert async_output_streamer is not None
                        async_output_streamer.signal_end()
                    yield GenerationResponse(
                        text="",
                        token=next_token_id,
                        finish_reason="stop",
                        usage=Usage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                            prompt_tokens_details=PromptTokensDetails(
                                cached_tokens=0, audio_tokens=0
                            ),
                            completion_tokens_details=CompletionTokensDetails(
                                reasoning_tokens=0, audio_tokens=0
                            ),
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
                    # Signal end to streamer before returning
                    if async_output_active:
                        assert async_output_streamer is not None
                        async_output_streamer.signal_end()
                    token_text: str = tokenizer.decode([next_token_id], skip_special_tokens=True)  # pyright: ignore[reportAny]
                    yield GenerationResponse(
                        text=token_text,
                        token=next_token_id,
                        finish_reason="length",
                        usage=Usage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                            prompt_tokens_details=PromptTokensDetails(
                                cached_tokens=0, audio_tokens=0
                            ),
                            completion_tokens_details=CompletionTokensDetails(
                                reasoning_tokens=0, audio_tokens=0
                            ),
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

                if defer_tokenizer_decode:
                    # Deferred path: store token for decode after next forward
                    # pass launches (Requirement 1.4, 9.1). Token ordering is
                    # preserved because _pending_token_id is yielded at the
                    # start of the next iteration before any new token is
                    # produced.
                    _pending_token_id = next_token_id
                    _pending_completion_tokens = completion_tokens
                else:
                    # Standard path: decode and yield immediately
                    token_text = tokenizer.decode([next_token_id], skip_special_tokens=True)  # pyright: ignore[reportAny]
                    yield GenerationResponse(
                        text=token_text,
                        token=next_token_id,
                        finish_reason=None,
                        usage=Usage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                            prompt_tokens_details=PromptTokensDetails(
                                cached_tokens=0, audio_tokens=0
                            ),
                            completion_tokens_details=CompletionTokensDetails(
                                reasoning_tokens=0, audio_tokens=0
                            ),
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

            # --- End of decode loop: flush any pending deferred token ---
            # This handles the case where the loop completes all iterations
            # without hitting EOS or max_tokens (edge case: max_tokens - 1
            # iterations completed but completion_tokens < max_tokens due to
            # the first token being counted separately).
            if defer_tokenizer_decode and _pending_token_id is not None:
                _final_decode_elapsed = time.perf_counter() - decode_start
                _final_generation_tps: float = (
                    (_pending_completion_tokens - 1) / _final_decode_elapsed
                    if _final_decode_elapsed > 0
                    else 0.0
                )
                _final_token_text: str = tokenizer.decode(  # pyright: ignore[reportAny]
                    [_pending_token_id], skip_special_tokens=True
                )
                yield GenerationResponse(
                    text=_final_token_text,
                    token=_pending_token_id,
                    finish_reason=None,
                    usage=Usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=_pending_completion_tokens,
                        total_tokens=prompt_tokens + _pending_completion_tokens,
                        prompt_tokens_details=PromptTokensDetails(
                            cached_tokens=0, audio_tokens=0
                        ),
                        completion_tokens_details=CompletionTokensDetails(
                            reasoning_tokens=0, audio_tokens=0
                        ),
                    ),
                    stats=GenerationStats(
                        prompt_tps=prompt_tps,
                        generation_tps=_final_generation_tps,
                        prompt_tokens=prompt_tokens,
                        generation_tokens=_pending_completion_tokens,
                        peak_memory_usage=Memory(in_bytes=0),
                    ),
                )

            # --- Signal end to async output streamer ---
            # If the decode loop completed naturally (all iterations exhausted
            # without EOS or max_tokens), signal end to the streamer so the
            # async consumer knows no more tokens will arrive.
            if async_output_active:
                assert async_output_streamer is not None
                async_output_streamer.signal_end()

    except Exception as exc:
        # --- Error Handling (Requirements: 11.1, 11.2, 11.3) ---
        error_msg: str = (
            f"Pipeline-parallel generation error on rank {rank}: "
            f"{type(exc).__name__}: {exc}"
        )
        logger.error(error_msg, exc_info=True)

        # Signal end to streamer on error so the async consumer is not left waiting
        if async_output_active:
            assert async_output_streamer is not None
            async_output_streamer.signal_end()

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
    finally:
        # Clear the buffer pool at the end of the generation session
        if buffer_pool is not None:
            buffer_pool.clear()
            logger.debug("Rank %d: decode buffer pool cleared", rank)


def _send_decode_activation_rank0(
    output: torch.Tensor,
    send_protocol: DecodeActivationProtocol | None,
    buffer_pool: CommunicationBufferPool | None,
    recorder: PerformanceRecorder,
) -> None:
    """Send decode activation from rank 0 to rank 1, using fast path when available.

    Falls back to the generic path when:
    - The fast-path protocol is None (negotiation disabled or failed)
    - The buffer pool is None
    - The activation shape doesn't match the protocol
    - The fast-path send raises an exception

    Args:
        output: The activation tensor from rank 0's forward pass.
        send_protocol: Negotiated protocol for fast-path send, or None.
        buffer_pool: Buffer pool for preallocated communication buffers, or None.
        recorder: Performance recorder for instrumentation.
    """
    if (
        send_protocol is not None
        and buffer_pool is not None
        and _activation_matches_protocol(output, send_protocol)
    ):
        # Fast path: send without shape metadata
        try:
            process_group = _get_default_process_group()
            send_decode_activation_fast(
                activation=output,
                protocol=send_protocol,
                buffer_pool=buffer_pool,
                process_group=process_group,
            )
            recorder.increment_counter("fast_path_activation_sends")
            logger.debug(
                "Rank 0: fast-path send to rank %d, shape=%s",
                send_protocol.destination_rank,
                send_protocol.shape,
            )
            return
        except Exception as exc:
            logger.warning(
                "Rank 0: fast-path send failed, falling back to generic: %s",
                exc,
            )
            recorder.increment_counter("fast_path_fallbacks")

    # Generic path: send with shape metadata
    _send_with_shape(output, dst_rank=1, performance_recorder=recorder)
    recorder.increment_counter("generic_activation_sends")


def _receive_token_rank0(
    world_size: int,
    send_protocol: DecodeActivationProtocol | None,
    recorder: PerformanceRecorder,
) -> int:
    """Receive a token result on rank 0 from the final rank.

    When the fast path is active, uses point-to-point token result receive
    instead of blocking broadcast. Falls back to broadcast when the fast
    path is not available.

    Args:
        world_size: Total number of pipeline stages.
        send_protocol: Negotiated protocol (presence indicates fast path is active).
        recorder: Performance recorder for instrumentation.

    Returns:
        The token ID received from the final rank.
    """
    if send_protocol is not None:
        # Fast path: receive token result via point-to-point from final rank
        try:
            process_group = _get_default_process_group()
            packet = receive_token_results_from_final_rank(
                process_group=process_group,
                world_size=world_size,
                performance_recorder=recorder,
            )
            logger.debug(
                "Rank 0: received token result via fast path, token_id=%d",
                packet.token_identifier,
            )
            return packet.token_identifier
        except Exception as exc:
            logger.warning(
                "Rank 0: fast-path token receive failed, falling back to broadcast: %s",
                exc,
            )
            recorder.increment_counter("fast_path_fallbacks")

    # Generic path: wait for token broadcast from last rank
    token_tensor = torch.tensor([0], dtype=torch.long, device="cpu")
    dist.broadcast(token_tensor, src=world_size - 1)
    return int(token_tensor.item())



def _send_with_shape(
    tensor: torch.Tensor,
    dst_rank: int,
    performance_recorder: PerformanceRecorder | None = None,
) -> None:
    """Send a tensor preceded by its seq_len dimension as metadata.

    Since recv_activation() requires knowing the shape ahead of time, and
    seq_len varies between prefill (seq_len > 1) and decode (seq_len = 1),
    we send a 1-element int64 tensor with seq_len before the actual activation.

    Args:
        tensor: The activation tensor of shape [batch, seq_len, hidden_size].
        dst_rank: Destination rank for the send.
        performance_recorder: Optional recorder to increment shape_metadata_messages.
    """
    seq_len_tensor = torch.tensor([tensor.shape[1]], dtype=torch.long, device="cpu")
    dist.send(seq_len_tensor, dst=dst_rank)
    if performance_recorder is not None:
        performance_recorder.increment_counter("shape_metadata_messages")
    send_activation(tensor, dst_rank=dst_rank)


def _recv_with_shape(
    hidden_size: int,
    src_rank: int,
    target_device: str,
    dtype: torch.dtype = torch.bfloat16,
    performance_recorder: PerformanceRecorder | None = None,
) -> torch.Tensor:
    """Receive a tensor preceded by its seq_len dimension as metadata.

    Reads the seq_len metadata first, then allocates the correct buffer
    and receives the activation tensor.

    Args:
        hidden_size: Model hidden dimension (fixed across all transfers).
        src_rank: Source rank to receive from.
        target_device: Device to place the received tensor on.
        dtype: Expected tensor dtype.
        performance_recorder: Optional recorder to increment shape_metadata_messages.

    Returns:
        The received activation tensor on target_device.
    """
    seq_len_tensor = torch.tensor([0], dtype=torch.long, device="cpu")
    dist.recv(seq_len_tensor, src=src_rank)
    if performance_recorder is not None:
        performance_recorder.increment_counter("shape_metadata_messages")
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
    performance_recorder: PerformanceRecorder | None = None,
    optimization_configuration: PytorchXpuOptimizationConfiguration | None = None,
) -> None:
    """Worker loop for non-rank-0 pipeline stages.

    Runs a blocking loop that:
    1. Receives activation from previous rank (with shape metadata)
    2. Forwards through local layers
    3. If not last rank: sends activation to next rank (with shape metadata)
    4. If last rank: samples token, sends to rank 0 via point-to-point
    5. Middle ranks do NOT wait for token results (fast path)
    6. Checks for termination (EOS or sentinel -1)
    7. Loops until termination

    For the first iteration (prefill), seq_len > 1. For subsequent iterations
    (decode), seq_len == 1. The shape metadata exchange handles prefill
    transparently. After prefill, the decode fast path avoids per-token
    shape metadata when the protocol is negotiated.

    Args:
        model: PipelineParallelShard with local layers on this rank's device.
        device: Device string for tensor placement (e.g., "xpu:0", "cpu").
        rank: This node's rank (must be > 0 for this function).
        world_size: Total number of pipeline stages.
        tokenizer: HuggingFace tokenizer (used to determine EOS token IDs).
        temperature: Sampling temperature (higher = more random).
        top_k: Top-k sampling parameter (None = disabled).
        top_p: Top-p (nucleus) sampling parameter (None = disabled).
        performance_recorder: Optional pre-configured PerformanceRecorder instance.
            If provided, takes precedence over optimization_configuration.
        optimization_configuration: Optional configuration used to create a
            recorder when performance_recorder is not provided.

    Requirements: 2.1, 2.7, 2.8, 2.9, 2.11, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 8.1, 8.2, 8.4, 9.4, 9.5
    """
    logger.info(
        f"Starting pipeline-parallel worker loop: rank={rank}/{world_size}, "
        f"device={device}, temperature={temperature}, top_k={top_k}, top_p={top_p}"
    )

    # Initialize performance recorder
    recorder: PerformanceRecorder
    if performance_recorder is not None:
        recorder = performance_recorder
    else:
        recorder = _create_recorder_from_configuration(
            optimization_configuration, rank=rank, stage=rank
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

    # Decode fast-path state (initialized after prefill)
    fast_path_enabled: bool = _is_fast_path_enabled(optimization_configuration)
    send_protocol: DecodeActivationProtocol | None = None
    recv_protocol: DecodeActivationProtocol | None = None
    buffer_pool: CommunicationBufferPool | None = None

    # --- On-device sampling for last rank (Requirements 8.1, 8.2, 8.3, 8.4) ---
    # Only the last rank performs sampling. Create the sampler only if this
    # is the last rank and on-device sampling is enabled.
    on_device_sampler: OnDeviceSampler | None = None
    if is_last_rank:
        on_device_sampler = _create_on_device_sampler(
            optimization_configuration, device
        )

    is_first_iteration: bool = True
    step_index: int = 0

    try:
        while True:
            # Determine mode: first iteration is prefill, subsequent are decode
            current_mode: EventMode = "prefill" if is_first_iteration else "decode"

            # Scheduler wait placeholder — records near-zero time until
            # continuous batching is implemented.
            with recorder.span(
                "scheduler_wait",
                mode="decode",
                metadata={"step": step_index},
            ):
                pass

            # End-to-end token latency for this step
            with recorder.span(
                "end_to_end_token_latency",
                mode=current_mode,
                metadata={"step": step_index},
            ):
                if is_first_iteration or recv_protocol is None:
                    # Prefill or no fast path: receive with shape metadata (generic)
                    hidden_state = _recv_with_shape(
                        hidden_size=hidden_size,
                        src_rank=prev_rank,
                        target_device=device,
                        performance_recorder=recorder,
                    )
                    recorder.increment_counter("generic_activation_receives")
                else:
                    # Decode fast path: receive without shape metadata
                    hidden_state = _recv_decode_activation_worker(
                        recv_protocol=recv_protocol,
                        buffer_pool=buffer_pool,
                        hidden_size=hidden_size,
                        prev_rank=prev_rank,
                        device=device,
                        recorder=recorder,
                    )

                # Forward through local layers
                with recorder.span(
                    "prefill" if is_first_iteration else "decode_step",
                    mode=current_mode,
                    metadata={"step": step_index},
                ):
                    output, _kv_cache = model.forward(input_data=hidden_state)

                # Send or sample depending on position in pipeline
                if not is_last_rank:
                    if is_first_iteration or send_protocol is None:
                        # Prefill or no fast path: send with shape metadata (generic)
                        _send_with_shape(output, dst_rank=next_rank, performance_recorder=recorder)
                        recorder.increment_counter("generic_activation_sends")
                    else:
                        # Decode fast path: send without shape metadata
                        _send_decode_activation_worker(
                            output=output,
                            send_protocol=send_protocol,
                            buffer_pool=buffer_pool,
                            next_rank=next_rank,
                            recorder=recorder,
                        )

                    if is_first_iteration:
                        # During prefill, still use broadcast for token sync
                        token_tensor = torch.tensor([0], dtype=torch.long, device="cpu")
                        dist.broadcast(token_tensor, src=world_size - 1)
                        token_id: int = int(token_tensor.item())
                    else:
                        # During decode with fast path, middle ranks do NOT
                        # participate in token result communication. They
                        # continue to the next iteration without waiting.
                        # The token_id is not needed by middle ranks for
                        # termination detection in fast-path mode — they
                        # detect termination via communication failure or
                        # a sentinel in the activation stream.
                        #
                        # However, for graceful termination detection, middle
                        # ranks still need to know when to stop. In the current
                        # single-request mode, we use a lightweight broadcast
                        # for termination signaling when fast path is active.
                        if send_protocol is not None:
                            # Fast path active: middle ranks still need
                            # termination signal. Use broadcast for now.
                            token_tensor = torch.tensor([0], dtype=torch.long, device="cpu")
                            dist.broadcast(token_tensor, src=world_size - 1)
                            token_id = int(token_tensor.item())
                        else:
                            # Generic path: wait for token broadcast
                            token_tensor = torch.tensor([0], dtype=torch.long, device="cpu")
                            dist.broadcast(token_tensor, src=world_size - 1)
                            token_id = int(token_tensor.item())
                else:
                    # Last stage: output is logits, sample token
                    # Use on-device sampling when enabled (Req 8.1, 8.4, 11.1)
                    token_id = _sample_with_on_device_sampler(
                        output,
                        on_device_sampler,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )

                    if is_first_iteration or send_protocol is None:
                        # Prefill or no fast path: broadcast sampled token to all ranks
                        token_tensor = torch.tensor([token_id], dtype=torch.long, device="cpu")
                        dist.broadcast(token_tensor, src=rank)
                    else:
                        # Decode fast path: send token result to rank 0 via
                        # point-to-point, then broadcast for middle rank termination
                        _send_token_result_last_rank(
                            token_id=token_id,
                            step_index=step_index,
                            recorder=recorder,
                        )
                        # Broadcast for middle rank termination detection
                        token_tensor = torch.tensor([token_id], dtype=torch.long, device="cpu")
                        dist.broadcast(token_tensor, src=rank)

            recorder.increment_counter("tokens_generated")

            # Step 5: Check for termination or reset
            if token_id == SHUTDOWN_SENTINEL:
                logger.debug(f"Worker rank={rank}: received shutdown sentinel, exiting")
                break

            if token_id == TERMINATION_SENTINEL or token_id in eos_token_ids:
                logger.debug(f"Worker rank={rank}: generation ended (token={token_id}), resetting for next request")
                model.reset_state()
                send_protocol = None
                recv_protocol = None
                buffer_pool = None
                is_first_iteration = True
                step_index = 0
                continue

            # After first iteration (prefill), negotiate decode fast path
            if is_first_iteration and fast_path_enabled:
                try:
                    process_group = _get_default_process_group()
                    maximum_microbatch_size: int = 1
                    if optimization_configuration is not None:
                        maximum_microbatch_size = optimization_configuration.maximum_decode_microbatch_size

                    # Respond to upstream negotiation FIRST (receive protocol
                    # from upstream rank). This must happen before negotiate to
                    # avoid deadlock: upstream rank's negotiate blocks on send
                    # until we receive here.
                    recv_protocol = respond_to_decode_protocol_negotiation(
                        process_group=process_group,
                        local_rank=rank,
                        world_size=world_size,
                        hidden_size=hidden_size,
                        dtype=torch.bfloat16,
                        maximum_microbatch_size=maximum_microbatch_size,
                    )

                    # Then negotiate send protocol (to downstream rank)
                    send_protocol = negotiate_decode_activation_protocol(
                        process_group=process_group,
                        local_rank=rank,
                        world_size=world_size,
                        hidden_size=hidden_size,
                        dtype=torch.bfloat16,
                        maximum_microbatch_size=maximum_microbatch_size,
                    )

                    # Create buffer pool for the decode session
                    buffer_pool = CommunicationBufferPool()

                    logger.info(
                        "Rank %d: decode fast path negotiated, "
                        "send_protocol=%s, recv_protocol=%s",
                        rank,
                        f"shape={send_protocol.shape}" if send_protocol else "None",
                        f"shape={recv_protocol.shape}" if recv_protocol else "None",
                    )
                except Exception as exc:
                    logger.warning(
                        "Rank %d: decode fast path negotiation failed, "
                        "falling back to generic path: %s",
                        rank,
                        exc,
                    )
                    send_protocol = None
                    recv_protocol = None
                    buffer_pool = None

            is_first_iteration = False
            step_index += 1

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
    finally:
        # Clear the buffer pool at the end of the generation session
        if buffer_pool is not None:
            buffer_pool.clear()
            logger.debug("Rank %d: decode buffer pool cleared", rank)


def _recv_decode_activation_worker(
    recv_protocol: DecodeActivationProtocol | None,
    buffer_pool: CommunicationBufferPool | None,
    hidden_size: int,
    prev_rank: int,
    device: str,
    recorder: PerformanceRecorder,
) -> torch.Tensor:
    """Receive a decode activation on a worker rank, using fast path when available.

    Falls back to the generic path when:
    - The recv_protocol is None
    - The buffer pool is None
    - The fast-path receive raises an exception

    Args:
        recv_protocol: Negotiated protocol for fast-path receive, or None.
        buffer_pool: Buffer pool for preallocated communication buffers, or None.
        hidden_size: Model hidden dimension.
        prev_rank: Source rank to receive from.
        device: Target device for the received tensor.
        recorder: Performance recorder for instrumentation.

    Returns:
        The received activation tensor on the target device.
    """
    if recv_protocol is not None and buffer_pool is not None:
        try:
            process_group = _get_default_process_group()
            result = receive_decode_activation_fast(
                protocol=recv_protocol,
                buffer_pool=buffer_pool,
                process_group=process_group,
                target_device=device,
            )
            recorder.increment_counter("fast_path_activation_receives")
            logger.debug(
                "Rank %d: fast-path recv from rank %d, shape=%s",
                recv_protocol.destination_rank,
                recv_protocol.source_rank,
                recv_protocol.shape,
            )
            return result
        except Exception as exc:
            logger.warning(
                "Fast-path recv failed, falling back to generic: %s",
                exc,
            )
            recorder.increment_counter("fast_path_fallbacks")

    # Generic path: receive with shape metadata
    result = _recv_with_shape(
        hidden_size=hidden_size,
        src_rank=prev_rank,
        target_device=device,
        performance_recorder=recorder,
    )
    recorder.increment_counter("generic_activation_receives")
    return result


def _send_decode_activation_worker(
    output: torch.Tensor,
    send_protocol: DecodeActivationProtocol | None,
    buffer_pool: CommunicationBufferPool | None,
    next_rank: int,
    recorder: PerformanceRecorder,
) -> None:
    """Send a decode activation from a worker rank, using fast path when available.

    Falls back to the generic path when:
    - The send_protocol is None
    - The buffer pool is None
    - The activation shape doesn't match the protocol
    - The fast-path send raises an exception

    Args:
        output: The activation tensor from the worker's forward pass.
        send_protocol: Negotiated protocol for fast-path send, or None.
        buffer_pool: Buffer pool for preallocated communication buffers, or None.
        next_rank: Destination rank for the send.
        recorder: Performance recorder for instrumentation.
    """
    if (
        send_protocol is not None
        and buffer_pool is not None
        and _activation_matches_protocol(output, send_protocol)
    ):
        try:
            process_group = _get_default_process_group()
            send_decode_activation_fast(
                activation=output,
                protocol=send_protocol,
                buffer_pool=buffer_pool,
                process_group=process_group,
            )
            recorder.increment_counter("fast_path_activation_sends")
            logger.debug(
                "Rank %d: fast-path send to rank %d, shape=%s",
                send_protocol.source_rank,
                send_protocol.destination_rank,
                send_protocol.shape,
            )
            return
        except Exception as exc:
            logger.warning(
                "Fast-path send failed, falling back to generic: %s",
                exc,
            )
            recorder.increment_counter("fast_path_fallbacks")

    # Generic path: send with shape metadata
    _send_with_shape(output, dst_rank=next_rank, performance_recorder=recorder)
    recorder.increment_counter("generic_activation_sends")


def _send_token_result_last_rank(
    token_id: int,
    step_index: int,
    recorder: PerformanceRecorder,
) -> None:
    """Send a token result from the last rank to rank 0 via point-to-point.

    Uses the ``send_token_results_to_rank_zero`` function for direct
    communication without requiring middle ranks to participate.

    Args:
        token_id: The sampled token identifier.
        step_index: Current decode step index (used as position).
        recorder: Performance recorder for instrumentation.
    """
    process_group = _get_default_process_group()
    packet = TokenResultPacket(
        request_identifier="single_request",
        token_identifier=token_id,
        position=step_index + 1,  # +1 because step 0 is the first decode token
        finished=False,
        finish_reason=None,
    )
    send_token_results_to_rank_zero(
        packet=packet,
        process_group=process_group,
        performance_recorder=recorder,
    )
    logger.debug(
        "Rank 3: sent token result to rank 0 via point-to-point, token_id=%d",
        token_id,
    )

import builtins
import contextlib
import time
from typing import Callable, Generator

import torch

from exo.shared.types.api import ChatCompletionMessage, FinishReason
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.ipex import (
    IPEXDistributedError,
    IPEXEngineError,
    IPEXInferenceError,
    IPEXMemoryError,
    IPEXModel,
    IPEXTokenizerWrapper,
)
from exo.worker.engines.ipex.utils_ipex import (
    MAX_TOKENS,
    apply_chat_template,
)
from exo.worker.runner.bootstrap import logger


def warmup_inference(
    model: IPEXModel,
    tokenizer: IPEXTokenizerWrapper,
    sampler: Callable[[torch.Tensor], torch.Tensor],
) -> int:
    """Warm up the IPEX inference engine with Intel GPU optimizations."""
    from exo.worker.engines.ipex.utils_ipex import (
        log_ipex_error_context,
        log_ipex_performance_metrics,
    )

    content = "Prompt to warm up the inference engine. Repeat this."

    warmup_prompt = apply_chat_template(
        tokenizer=tokenizer,
        chat_task_data=ChatCompletionTaskParams(
            model="",
            messages=[
                ChatCompletionMessage(
                    role="user",
                    content=content,
                )
            ],
        ),
    )

    tokens_generated = 0
    warmup_start_time = time.perf_counter()

    logger.info("Generating warmup tokens with IPEX")

    # Tokenize the prompt and move to Intel GPU
    try:
        device = next(model.parameters()).device
        logger.info(f"Warmup using Intel GPU device: {device}")

        input_ids = torch.tensor(
            [tokenizer.encode(warmup_prompt)], dtype=torch.long, device=device
        )
        attention_mask = torch.ones_like(input_ids, device=device)

        # Generate a few tokens for warmup with Intel GPU optimizations
        with torch.no_grad():
            # Enable Intel GPU mixed precision for warmup
            with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16):
                for i in range(10):  # Generate 10 warmup tokens
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                    )
                    logits = outputs.logits[:, -1, :]
                    next_token = sampler(logits)

                    # Update sequences
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((1, 1), device=device)], dim=-1
                    )

                    # Decode the new token
                    token_text = tokenizer.decode(
                        [next_token.item()], skip_special_tokens=True
                    )
                    logger.debug(f"Generated warmup token {i + 1}: {token_text}")
                    tokens_generated += 1

                    # Stop if we hit EOS
                    if next_token.item() in tokenizer.eos_token_ids:
                        break

        # Clear Intel GPU cache after warmup
        torch.xpu.empty_cache()

    except torch.xpu.OutOfMemoryError as e:
        warmup_duration = time.perf_counter() - warmup_start_time
        logger.error(f"Intel GPU out of memory during warmup: {e}")
        log_ipex_error_context(
            e,
            "warmup",
            {"tokens_generated": tokens_generated, "duration": warmup_duration},
        )
        with contextlib.suppress(builtins.BaseException):
            torch.xpu.empty_cache()
        raise IPEXMemoryError(
            f"Intel GPU out of memory during warmup: {e}", device_id=0
        )
    except Exception as e:
        warmup_duration = time.perf_counter() - warmup_start_time
        logger.error(f"IPEX warmup failed: {e}")
        log_ipex_error_context(
            e,
            "warmup",
            {"tokens_generated": tokens_generated, "duration": warmup_duration},
        )
        # Clean up on error
        with contextlib.suppress(builtins.BaseException):
            torch.xpu.empty_cache()
        raise IPEXInferenceError(f"IPEX warmup failed: {e}", device_id=0, step="warmup")
    finally:
        # Log warmup performance metrics
        warmup_duration = time.perf_counter() - warmup_start_time
        log_ipex_performance_metrics(
            "warmup", warmup_duration, tokens_generated=tokens_generated, device_id=0
        )

    logger.info(
        f"Generated {tokens_generated} warmup tokens with IPEX in {warmup_duration:.2f}s"
    )
    return tokens_generated


def ipex_generate(
    model: IPEXModel,
    tokenizer: IPEXTokenizerWrapper,
    sampler: Callable[[torch.Tensor], torch.Tensor],
    task: ChatCompletionTaskParams,
) -> Generator[GenerationResponse, None, None]:
    """Generate text using IPEX-optimized model with streaming output."""
    from exo.worker.engines.ipex.utils_ipex import (
        log_ipex_error_context,
        log_ipex_inference_complete,
        log_ipex_inference_start,
        log_ipex_performance_metrics,
    )

    # Log inference start
    model_info = {
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "distributed": hasattr(model, "_ipex_dist_group")
        and model._ipex_dist_group is not None,
    }
    log_ipex_inference_start(task, model_info)

    prompt = apply_chat_template(
        tokenizer=tokenizer,
        chat_task_data=task,
    )

    logger.info(f"Generated prompt: {prompt[:200]}...")

    # Track performance metrics
    inference_start_time = time.perf_counter()
    tokens_generated = 0

    try:
        # Get model device (Intel GPU)
        device = next(model.parameters()).device
        logger.debug(f"Using Intel GPU device: {device}")

        # Check if this is distributed inference
        is_distributed = (
            hasattr(model, "_ipex_dist_group") and model._ipex_dist_group is not None
        )

        if is_distributed:
            logger.info("Running distributed IPEX inference")
            for response in _ipex_generate_distributed(
                model, tokenizer, sampler, task, prompt, device
            ):
                tokens_generated += 1
                yield response
                if response.finish_reason is not None:
                    break
        else:
            logger.info("Running single-device IPEX inference")
            for response in _ipex_generate_single_device(
                model, tokenizer, sampler, task, prompt, device
            ):
                tokens_generated += 1
                yield response
                if response.finish_reason is not None:
                    break

    except IPEXEngineError:
        # Log error context before re-raising
        inference_duration = time.perf_counter() - inference_start_time
        log_ipex_error_context(
            e,
            "inference_generation",
            {
                "tokens_generated": tokens_generated,
                "duration": inference_duration,
                "prompt_length": len(prompt),
            },
        )
        raise  # Re-raise IPEX-specific errors
    except torch.xpu.OutOfMemoryError as e:
        inference_duration = time.perf_counter() - inference_start_time
        logger.error(f"Intel GPU out of memory during generation: {e}")
        log_ipex_error_context(
            e,
            "inference_generation",
            {
                "tokens_generated": tokens_generated,
                "duration": inference_duration,
                "memory_error": True,
            },
        )
        with contextlib.suppress(builtins.BaseException):
            torch.xpu.empty_cache()
        raise IPEXMemoryError(
            f"Intel GPU out of memory during generation: {e}", device_id=0
        )
    except Exception as e:
        inference_duration = time.perf_counter() - inference_start_time
        logger.error(f"IPEX generation failed: {e}")
        log_ipex_error_context(
            e,
            "inference_generation",
            {"tokens_generated": tokens_generated, "duration": inference_duration},
        )
        # Clean up Intel GPU memory on error
        with contextlib.suppress(builtins.BaseException):
            torch.xpu.empty_cache()
        raise IPEXInferenceError(
            f"IPEX generation failed: {e}", device_id=0, step="generation"
        )
    finally:
        # Log completion metrics
        total_duration = time.perf_counter() - inference_start_time
        log_ipex_inference_complete(tokens_generated, total_duration)

        # Log performance metrics
        log_ipex_performance_metrics(
            "text_generation",
            total_duration,
            tokens_generated=tokens_generated,
            device_id=0,
        )


def _ipex_generate_single_device(
    model: IPEXModel,
    tokenizer: IPEXTokenizerWrapper,
    sampler: Callable[[torch.Tensor], torch.Tensor],
    task: ChatCompletionTaskParams,
    prompt: str,
    device: torch.device,
) -> Generator[GenerationResponse, None, None]:
    """Generate text on single Intel GPU device."""
    from exo.worker.engines.ipex.utils_ipex import (
        handle_intel_gpu_health_issues,
        monitor_intel_gpu_health,
    )

    # Check Intel GPU health before generation
    health_summary = monitor_intel_gpu_health(0)
    if health_summary.get("overall_health") != "healthy":
        logger.warning("Intel GPU health issues detected before generation")
        handle_intel_gpu_health_issues(health_summary)

    # Tokenize the prompt and move to Intel GPU with optimized tensor creation
    input_ids = torch.tensor(
        [tokenizer.encode(prompt)], dtype=torch.long, device=device
    )

    # Pre-allocate attention mask for Intel GPU optimization
    attention_mask = torch.ones_like(input_ids, device=device)

    max_tokens = task.max_tokens or MAX_TOKENS
    generated_tokens = 0

    logger.info(f"Starting IPEX generation with max_tokens={max_tokens}")

    # Intel GPU optimized inference with memory management
    with torch.no_grad():
        # Enable Intel GPU optimizations
        with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16):
            while generated_tokens < max_tokens:
                # Periodic health check during long generations
                if generated_tokens > 0 and generated_tokens % 50 == 0:
                    health_summary = monitor_intel_gpu_health(0)
                    if health_summary.get("overall_health") != "healthy":
                        logger.warning(
                            f"Intel GPU health issues detected during generation at token {generated_tokens}"
                        )
                        if not handle_intel_gpu_health_issues(health_summary):
                            logger.error(
                                "Could not resolve Intel GPU health issues, continuing generation"
                            )

                try:
                    # Forward pass on Intel GPU with attention mask
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,  # Enable KV cache for Intel GPU optimization
                    )
                    logits = outputs.logits[:, -1, :]

                    # Apply Intel GPU optimized sampling
                    next_token = sampler(logits)
                    next_token_id = next_token.item()

                    # Add token to sequence with Intel GPU memory optimization
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((1, 1), device=device)], dim=-1
                    )
                    generated_tokens += 1

                    # Decode the new token
                    try:
                        # Decode just the new token for streaming
                        token_text = tokenizer.decode(
                            [next_token_id], skip_special_tokens=True
                        )

                        # Check for finish conditions
                        finish_reason: FinishReason | None = None
                        if next_token_id in tokenizer.eos_token_ids:
                            finish_reason = "stop"
                        elif generated_tokens >= max_tokens:
                            finish_reason = "length"

                        logger.debug(
                            f"Generated token {generated_tokens}: '{token_text}' (id={next_token_id})"
                        )

                        yield GenerationResponse(
                            text=token_text,
                            token=next_token_id,
                            finish_reason=finish_reason,
                        )

                        if finish_reason is not None:
                            logger.info(f"IPEX generation finished: {finish_reason}")
                            break

                    except Exception as e:
                        logger.warning(f"Error decoding token {next_token_id}: {e}")
                        # Skip this token and continue
                        continue

                    # Intel GPU memory management - clear cache periodically
                    if generated_tokens % 50 == 0:
                        torch.xpu.empty_cache()

                except torch.xpu.OutOfMemoryError as e:
                    logger.error(
                        f"Intel GPU out of memory during generation at token {generated_tokens}: {e}"
                    )
                    # Try to recover
                    torch.xpu.empty_cache()
                    health_summary = monitor_intel_gpu_health(0)
                    if handle_intel_gpu_health_issues(health_summary):
                        logger.info(
                            "Attempting to continue generation after memory recovery"
                        )
                        continue
                    else:
                        raise IPEXMemoryError(
                            f"Intel GPU out of memory during generation: {e}",
                            device_id=0,
                        )

                except Exception as e:
                    logger.error(
                        f"Error during Intel GPU inference at token {generated_tokens}: {e}"
                    )
                    # Check if this is a recoverable error
                    health_summary = monitor_intel_gpu_health(0)
                    if health_summary.get("overall_health") == "healthy":
                        # Might be a transient error, try to continue
                        logger.info("Intel GPU appears healthy, attempting to continue")
                        continue
                    else:
                        # Health issues detected, try to recover
                        if handle_intel_gpu_health_issues(health_summary):
                            logger.info(
                                "Recovered from Intel GPU health issues, continuing"
                            )
                            continue
                        else:
                            raise IPEXInferenceError(
                                f"Intel GPU inference failed: {e}",
                                device_id=0,
                                step=f"token_{generated_tokens}",
                            )

    # Final health check
    final_health = monitor_intel_gpu_health(0)
    if final_health.get("overall_health") != "healthy":
        logger.warning("Intel GPU health issues detected after generation")
        handle_intel_gpu_health_issues(final_health)

    logger.info(f"IPEX generation complete. Generated {generated_tokens} tokens.")


def _ipex_generate_distributed(
    model: IPEXModel,
    tokenizer: IPEXTokenizerWrapper,
    sampler: Callable[[torch.Tensor], torch.Tensor],
    task: ChatCompletionTaskParams,
    prompt: str,
    device: torch.device,
) -> Generator[GenerationResponse, None, None]:
    """Generate text using distributed IPEX inference across multiple Intel GPUs."""
    try:
        import torch.distributed as dist

        from exo.worker.engines.ipex.utils_ipex import intel_gpu_distributed_barrier

        # Get distributed information
        dist_group = model._ipex_dist_group
        parallelism_type = model._ipex_parallelism_type
        rank = model._ipex_rank
        world_size = model._ipex_world_size

        logger.info(
            f"Distributed IPEX generation: rank={rank}, world_size={world_size}, type={parallelism_type}"
        )

        # Tokenize the prompt (only on rank 0 for pipeline parallelism)
        if parallelism_type == "pipeline" and rank == 0:
            input_ids = torch.tensor(
                [tokenizer.encode(prompt)], dtype=torch.long, device=device
            )
            attention_mask = torch.ones_like(input_ids, device=device)
        elif parallelism_type == "tensor":
            # All ranks need the input for tensor parallelism
            input_ids = torch.tensor(
                [tokenizer.encode(prompt)], dtype=torch.long, device=device
            )
            attention_mask = torch.ones_like(input_ids, device=device)
        else:
            input_ids = None
            attention_mask = None

        max_tokens = task.max_tokens or MAX_TOKENS
        generated_tokens = 0

        logger.info(
            f"Starting distributed IPEX generation with max_tokens={max_tokens}"
        )

        # Distributed Intel GPU inference
        with torch.no_grad():
            with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16):
                while generated_tokens < max_tokens:
                    # Handle different parallelism types
                    if parallelism_type == "pipeline":
                        # Pipeline parallelism: sequential processing across ranks
                        if rank == 0:
                            # First rank processes input
                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                use_cache=True,
                            )

                            # Send intermediate results to next rank
                            if world_size > 1:
                                hidden_states = outputs.last_hidden_state
                                dist.send(hidden_states, dst=1, group=dist_group)

                            logits = outputs.logits[:, -1, :]

                        elif rank == world_size - 1:
                            # Last rank receives from previous and generates output
                            hidden_states = torch.empty(
                                (
                                    1,
                                    input_ids.size(1) if input_ids is not None else 1,
                                    model.config.hidden_size,
                                ),
                                device=device,
                                dtype=torch.float16,
                            )
                            dist.recv(hidden_states, src=rank - 1, group=dist_group)

                            # Process through final layers
                            outputs = model(inputs_embeds=hidden_states, use_cache=True)
                            logits = outputs.logits[:, -1, :]

                        else:
                            # Middle ranks receive, process, and send
                            hidden_states = torch.empty(
                                (
                                    1,
                                    input_ids.size(1) if input_ids is not None else 1,
                                    model.config.hidden_size,
                                ),
                                device=device,
                                dtype=torch.float16,
                            )
                            dist.recv(hidden_states, src=rank - 1, group=dist_group)

                            outputs = model(inputs_embeds=hidden_states, use_cache=True)

                            if rank < world_size - 1:
                                dist.send(
                                    outputs.last_hidden_state,
                                    dst=rank + 1,
                                    group=dist_group,
                                )

                            logits = None  # Only last rank generates tokens

                    elif parallelism_type == "tensor":
                        # Tensor parallelism: parallel processing with communication
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=True,
                        )

                        # All-reduce logits across ranks
                        logits = outputs.logits[:, -1, :]
                        dist.all_reduce(logits, group=dist_group)
                        logits = logits / world_size

                    else:
                        raise IPEXDistributedError(
                            f"Unknown parallelism type: {parallelism_type}",
                            rank=rank,
                            world_size=world_size,
                        )

                    # Only generate tokens on appropriate ranks
                    if (
                        parallelism_type == "pipeline" and rank == world_size - 1
                    ) or parallelism_type == "tensor":
                        # Apply Intel GPU optimized sampling
                        next_token = sampler(logits)
                        next_token_id = next_token.item()

                        # Broadcast token to all ranks
                        token_tensor = torch.tensor([next_token_id], device=device)
                        dist.broadcast(
                            token_tensor,
                            src=world_size - 1 if parallelism_type == "pipeline" else 0,
                            group=dist_group,
                        )
                        next_token_id = token_tensor.item()
                    else:
                        # Receive token from generating rank
                        token_tensor = torch.tensor([0], device=device)
                        dist.broadcast(
                            token_tensor,
                            src=world_size - 1 if parallelism_type == "pipeline" else 0,
                            group=dist_group,
                        )
                        next_token_id = token_tensor.item()

                    # Update sequences on all ranks
                    if input_ids is not None:
                        new_token = torch.tensor([[next_token_id]], device=device)
                        input_ids = torch.cat([input_ids, new_token], dim=-1)
                        attention_mask = torch.cat(
                            [attention_mask, torch.ones((1, 1), device=device)], dim=-1
                        )

                    generated_tokens += 1

                    # Only output from rank 0 or last rank
                    if rank == 0 or (
                        parallelism_type == "pipeline" and rank == world_size - 1
                    ):
                        try:
                            # Decode the new token
                            token_text = tokenizer.decode(
                                [next_token_id], skip_special_tokens=True
                            )

                            # Check for finish conditions
                            finish_reason: FinishReason | None = None
                            if next_token_id in tokenizer.eos_token_ids:
                                finish_reason = "stop"
                            elif generated_tokens >= max_tokens:
                                finish_reason = "length"

                            logger.debug(
                                f"Generated token {generated_tokens}: '{token_text}' (id={next_token_id})"
                            )

                            yield GenerationResponse(
                                text=token_text,
                                token=next_token_id,
                                finish_reason=finish_reason,
                            )

                            if finish_reason is not None:
                                logger.info(
                                    f"Distributed IPEX generation finished: {finish_reason}"
                                )
                                break

                        except Exception as e:
                            logger.warning(f"Error decoding token {next_token_id}: {e}")
                            continue

                    # Synchronize all ranks
                    intel_gpu_distributed_barrier(dist_group)

                    # Intel GPU memory management
                    if generated_tokens % 50 == 0:
                        torch.xpu.empty_cache()

        logger.info(
            f"Distributed IPEX generation complete. Generated {generated_tokens} tokens."
        )

    except IPEXEngineError:
        raise  # Re-raise IPEX-specific errors
    except torch.xpu.OutOfMemoryError as e:
        logger.error(f"Intel GPU out of memory during distributed generation: {e}")
        with contextlib.suppress(builtins.BaseException):
            torch.xpu.empty_cache()
        raise IPEXMemoryError(
            f"Intel GPU out of memory during distributed generation: {e}", device_id=0
        )
    except Exception as e:
        logger.error(f"Distributed IPEX generation failed: {e}")
        # Fallback to single device generation on rank 0
        if hasattr(model, "_ipex_rank") and model._ipex_rank == 0:
            logger.warning("Falling back to single-device generation")
            try:
                yield from _ipex_generate_single_device(
                    model, tokenizer, sampler, task, prompt, device
                )
            except Exception as fallback_error:
                raise IPEXDistributedError(
                    f"Distributed generation failed and fallback also failed: {e}. Fallback error: {fallback_error}",
                    rank=getattr(model, "_ipex_rank", None),
                    world_size=getattr(model, "_ipex_world_size", None),
                )
        else:
            raise IPEXDistributedError(
                f"Distributed IPEX generation failed: {e}",
                rank=getattr(model, "_ipex_rank", None),
                world_size=getattr(model, "_ipex_world_size", None),
            )


def ipex_generate_simple(
    model: IPEXModel,
    tokenizer: IPEXTokenizerWrapper,
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.7,
) -> str:
    """Simple non-streaming generation for testing with IPEX Intel GPU optimizations."""
    try:
        # Get model device (Intel GPU)
        device = next(model.parameters()).device
        input_ids = torch.tensor(
            [tokenizer.encode(prompt)], dtype=torch.long, device=device
        )

        with torch.no_grad():
            # Use Intel GPU mixed precision for simple generation
            with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16):
                generated = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=tokenizer.tokenizer.pad_token_id,
                    eos_token_id=tokenizer.tokenizer.eos_token_id,
                    use_cache=True,  # Enable KV cache for Intel GPU
                )

        # Decode only the new tokens
        new_tokens = generated[0][input_ids.shape[1] :]
        result = tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)

        # Clean up Intel GPU memory
        torch.xpu.empty_cache()

        return result

    except torch.xpu.OutOfMemoryError as e:
        logger.error(f"Intel GPU out of memory during simple generation: {e}")
        with contextlib.suppress(builtins.BaseException):
            torch.xpu.empty_cache()
        raise IPEXMemoryError(
            f"Intel GPU out of memory during simple generation: {e}", device_id=0
        )
    except Exception as e:
        logger.error(f"IPEX simple generation failed: {e}")
        # Clean up on error
        with contextlib.suppress(builtins.BaseException):
            torch.xpu.empty_cache()
        raise IPEXInferenceError(
            f"IPEX simple generation failed: {e}", device_id=0, step="simple_generation"
        )

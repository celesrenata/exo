from typing import Any, Callable, Generator, cast, get_args
from datetime import datetime
import uuid

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.models.cache import KVCache
from mlx_lm.tokenizer_utils import TokenizerWrapper

# from exo.engines.mlx.cache import KVPrefixCache
from exo.shared.types.api import ChatCompletionMessage, FinishReason
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.validation import (
    EnhancedTokenChunk,
    ValidationStatus,
    CorruptionType,
)
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
)
from exo.shared.validation.token_validator import TokenValidator
from exo.shared.validation.corruption_detector import CorruptionDetector
from exo.shared.validation.diagnostics import (
    get_diagnostic_logger,
    DiagnosticLevel,
    DiagnosticCategory,
)
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.constants import KV_BITS, KV_GROUP_SIZE, MAX_TOKENS
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    make_kv_cache,
    mx_barrier,
)
from exo.worker.runner.bootstrap import logger

generation_stream = mx.new_stream(mx.default_device())


def maybe_quantize_kv_cache(
    prompt_cache: list[KVCache | Any],
    quantized_kv_start: int,
    kv_group_size: int,
    kv_bits: int | None,
) -> None:
    if kv_bits is None:
        return
    for e, c in enumerate(prompt_cache):
        if (
            hasattr(c, "to_quantized") and c.offset >= quantized_kv_start  # type: ignore
        ):
            prompt_cache[e] = c.to_quantized(group_size=kv_group_size, bits=kv_bits)


def warmup_inference(
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
) -> int:
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

    cache = make_kv_cache(
        model=model,
    )

    logger.info("Generating warmup tokens")
    for _r in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=warmup_prompt,
        max_tokens=50,
        sampler=sampler,
        prompt_cache=cache,
        prefill_step_size=65536,
        kv_group_size=KV_GROUP_SIZE,
        kv_bits=KV_BITS,
    ):
        logger.info("Generated warmup token: " + str(_r.text))
        tokens_generated += 1

    logger.info("Generated ALL warmup tokens")
    mx_barrier()

    return tokens_generated


def mlx_generate(
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
    task: ChatCompletionTaskParams,
) -> Generator[GenerationResponse]:
    # Currently we support chat-completion tasks only.
    logger.info(f"task_params: {task}")

    # Initialize diagnostic logging
    diagnostic_logger = get_diagnostic_logger()
    generation_id = str(uuid.uuid4())
    device_id = f"device_{getattr(mx.default_device(), 'index', 0)}"

    # Log generation start
    diagnostic_logger.log_generation_pipeline_event(
        event_type="generation_start",
        message=f"Starting MLX generation for model {task.model}",
        token_info={
            "generation_id": generation_id,
            "model": task.model,
            "max_tokens": task.max_tokens or MAX_TOKENS,
            "prompt_length": len(task.messages[0].content) if task.messages else 0,
        },
        device_id=device_id,
        correlation_id=generation_id,
    )

    # Start performance tracking
    perf_tracker = diagnostic_logger.start_performance_tracking(
        f"generation_{generation_id}"
    )

    prompt = apply_chat_template(
        tokenizer=tokenizer,
        chat_task_data=task,
    )

    caches = make_kv_cache(model=model)

    # Initialize validation components
    token_validator = TokenValidator()
    corruption_detector = CorruptionDetector()
    sequence_position = 0

    # Log validation initialization
    validation_id = f"validation_{generation_id}"
    diagnostic_logger.log_validation_start(
        validation_id=validation_id,
        token_count=0,  # Will be updated as we generate
        device_id=device_id,
        correlation_id=generation_id,
    )

    max_tokens = task.max_tokens or MAX_TOKENS
    tokens_generated = 0
    corruption_count = 0

    try:
        for out in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            prompt_cache=caches,
            prefill_step_size=65536,
            kv_group_size=KV_GROUP_SIZE,
            kv_bits=KV_BITS,
        ):
            logger.info(out.text)
            tokens_generated += 1

            # Log token generation event
            diagnostic_logger.log_generation_pipeline_event(
                event_type="token_generated",
                message=f"Generated token {sequence_position}: '{out.text[:50]}{'...' if len(out.text) > 50 else ''}'",
                token_info={
                    "sequence_position": sequence_position,
                    "token_id": out.token,
                    "text_length": len(out.text),
                    "finish_reason": out.finish_reason,
                },
                device_id=device_id,
                correlation_id=generation_id,
            )

            # Create enhanced token chunk with validation metadata
            enhanced_token = EnhancedTokenChunk(
                idx=out.token,
                model=task.model,
                text=out.text,
                token_id=out.token,
                finish_reason=out.finish_reason,
                generation_timestamp=datetime.now(),
                source_device_rank=getattr(mx.default_device(), "index", 0),
                sequence_position=sequence_position,
                validation_status=ValidationStatus.PENDING,
            )

            # Update checksum
            enhanced_token.update_checksum()

            # Real-time corruption detection with diagnostic logging
            corruption_report = corruption_detector.analyze_output(out.text)
            if corruption_report.corruption_type != CorruptionType.NONE:
                corruption_count += 1

                # Log corruption detection
                diagnostic_logger.log_corruption_detected(
                    corruption_type=corruption_report.corruption_type,
                    affected_tokens=[enhanced_token],
                    severity=corruption_report.severity.value,
                    device_id=device_id,
                    correlation_id=generation_id,
                )

                logger.warning(
                    f"Corruption detected in token {sequence_position}: {corruption_report.details}"
                )
                enhanced_token.mark_as_corrupted(
                    corruption_report.corruption_type,
                    corruption_report.details,
                    corruption_report.severity,
                )
            else:
                # Validate the token
                validation_result = token_validator.validate_token(enhanced_token)
                if validation_result.is_valid:
                    enhanced_token.mark_as_valid()
                else:
                    logger.warning(
                        f"Token validation failed for token {sequence_position}: {validation_result.error_details}"
                    )
                    enhanced_token.validation_status = ValidationStatus.INVALID
                    enhanced_token.validation_result = validation_result

                    # Log validation failure
                    diagnostic_logger.log_event(
                        level=DiagnosticLevel.WARNING,
                        category=DiagnosticCategory.VALIDATION,
                        event_type="token_validation_failed",
                        message=f"Token validation failed at position {sequence_position}",
                        details={
                            "validation_result": validation_result.dict()
                            if hasattr(validation_result, "dict")
                            else str(validation_result),
                            "token_text": out.text[:100],
                        },
                        device_id=device_id,
                        correlation_id=generation_id,
                    )

            # Validation checkpoint - log validation status and performance metrics
            if sequence_position % 10 == 0:  # Every 10 tokens
                logger.debug(
                    f"Validation checkpoint at token {sequence_position}: status={enhanced_token.validation_status}"
                )

                # Log performance metrics
                current_time = diagnostic_logger.end_performance_tracking(
                    f"checkpoint_{sequence_position}"
                )
                if current_time > 0:
                    diagnostic_logger.log_performance_metric(
                        name="tokens_per_second",
                        value=10.0 / (current_time / 1000.0)
                        if current_time > 0
                        else 0.0,
                        unit="tokens/sec",
                        device_id=device_id,
                    )

                # Restart tracking for next checkpoint
                diagnostic_logger.start_performance_tracking(
                    f"checkpoint_{sequence_position + 10}"
                )

            if out.finish_reason is not None and out.finish_reason not in get_args(
                FinishReason
            ):
                # We don't throw here as this failure case is really not all that bad
                # Just log the error and move on
                logger.warning(
                    f"Model generated unexpected finish_reason: {out.finish_reason}"
                )

                # Log unexpected finish reason
                diagnostic_logger.log_event(
                    level=DiagnosticLevel.WARNING,
                    category=DiagnosticCategory.GENERATION,
                    event_type="unexpected_finish_reason",
                    message=f"Unexpected finish reason: {out.finish_reason}",
                    details={"finish_reason": out.finish_reason},
                    device_id=device_id,
                    correlation_id=generation_id,
                )

            yield GenerationResponse(
                text=out.text,
                token=out.token,
                finish_reason=cast(FinishReason | None, out.finish_reason),
                enhanced_token=enhanced_token,  # Include enhanced token in response
            )

            if out.finish_reason is not None:
                break

            sequence_position += 1

    except Exception as e:
        # Log generation error
        diagnostic_logger.log_event(
            level=DiagnosticLevel.ERROR,
            category=DiagnosticCategory.GENERATION,
            event_type="generation_error",
            message=f"Error during generation: {str(e)}",
            details={
                "error_type": type(e).__name__,
                "tokens_generated": tokens_generated,
                "sequence_position": sequence_position,
            },
            device_id=device_id,
            correlation_id=generation_id,
        )
        raise

    finally:
        # Log generation completion and final metrics
        total_duration_ms = diagnostic_logger.end_performance_tracking(perf_tracker)

        # Log validation completion
        diagnostic_logger.log_validation_complete(
            validation_id=validation_id,
            token_count=tokens_generated,
            corruption_detected=corruption_count > 0,
            corruption_types=[CorruptionType.ENCODING_CORRUPTION]
            if corruption_count > 0
            else [],
            device_id=device_id,
            correlation_id=generation_id,
        )

        # Log final performance metrics
        if total_duration_ms > 0:
            diagnostic_logger.log_performance_metric(
                name="generation_duration",
                value=total_duration_ms,
                unit="ms",
                device_id=device_id,
            )

            diagnostic_logger.log_performance_metric(
                name="average_tokens_per_second",
                value=tokens_generated / (total_duration_ms / 1000.0)
                if total_duration_ms > 0
                else 0.0,
                unit="tokens/sec",
                device_id=device_id,
            )

        # Log generation completion
        diagnostic_logger.log_generation_pipeline_event(
            event_type="generation_complete",
            message=f"MLX generation completed: {tokens_generated} tokens, {corruption_count} corruptions",
            token_info={
                "generation_id": generation_id,
                "tokens_generated": tokens_generated,
                "corruption_count": corruption_count,
                "duration_ms": total_duration_ms,
                "success": True,
            },
            device_id=device_id,
            correlation_id=generation_id,
        )

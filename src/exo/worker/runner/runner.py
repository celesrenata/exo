import time
import asyncio
from typing import List, Any, Optional

from exo.shared.types.api import ChatCompletionMessageText
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import (
    ChatCompletion,
    ConnectToGroup,
    DownloadModel,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskStatus,
)
from exo.shared.types.validation import (
    EnhancedTokenChunk, CorruptionType, 
    FailureMode, ValidationStatus
)
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
)
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerFailed,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.shared.validation.corruption_detector import CorruptionDetector
from exo.shared.validation.recovery_manager import RecoveryManager
from exo.shared.validation.fallback_coordinator import FallbackCoordinator
from exo.utils.channels import ClosedResourceError, MpReceiver, MpSender
from exo.worker.engines.engine_init import (
    generate_with_engine,
    initialize_engine,
    warmup_engine,
)
from exo.worker.runner.bootstrap import logger

# Conditional import for MLX utilities (only needed for testing)
try:
    from exo.worker.engines.mlx.utils_mlx import mlx_force_oom
except ImportError:

    def mlx_force_oom():
        """Fallback function when MLX is not available"""
        raise RuntimeError("MLX not available - cannot force OOM")


def main(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
):
    instance, runner_id, shard_metadata = (
        bound_instance.instance,
        bound_instance.bound_runner_id,
        bound_instance.bound_shard,
    )
    
    # Initialize validation and recovery components
    corruption_detector = CorruptionDetector()
    recovery_manager = RecoveryManager()
    fallback_coordinator = FallbackCoordinator()
    
    # Track corruption events for fallback decisions
    corruption_count = 0
    consecutive_corruptions = 0
    max_consecutive_corruptions = 5
    
    try:
        logger.info("hello from the runner")
        if getattr(shard_metadata, "immediate_exception", False):
            raise Exception("Fake exception - runner failed to spin up.")
        if timeout := getattr(shard_metadata, "should_timeout", 0):
            time.sleep(timeout)

        setup_start_time = time.time()

        model = None
        tokenizer = None
        sampler = None
        group = None

        current_status: RunnerStatus = RunnerIdle()
        logger.info("runner created")
        event_sender.send(
            RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status)
        )
        with task_receiver as tasks:
            for task in tasks:
                event_sender.send(
                    TaskStatusUpdated(
                        task_id=task.task_id, task_status=TaskStatus.Running
                    )
                )
                event_sender.send(TaskAcknowledged(task_id=task.task_id))
                match task:
                    case ConnectToGroup() if isinstance(
                        current_status, (RunnerIdle, RunnerFailed)
                    ):
                        logger.info("runner connecting")
                        current_status = RunnerConnecting()
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )
                        # Initialize engine (will handle MLX group setup if needed)
                        group = initialize_engine(bound_instance, connect_only=True)
                        
                        # For engines that don't need separate connection (like torch/cpu),
                        # set group to a placeholder to indicate connection is established
                        if group is None:
                            group = "connected"  # Placeholder for non-MLX engines

                        logger.info("runner connected")
                        current_status = RunnerConnected()

                    case DownloadModel(shard_metadata=shard_metadata) if isinstance(
                        current_status, (RunnerIdle, RunnerConnected, RunnerFailed)
                    ):
                        logger.info(f"Downloading model: {shard_metadata.model_meta.model_id}")
                        
                        try:
                            import asyncio
                            from exo.worker.download.download_utils import download_shard
                            
                            def on_progress(shard, progress):
                                logger.debug(f"Download progress: {progress.completed_files}/{progress.total_files} files, {progress.downloaded_bytes}/{progress.total_bytes}")
                            
                            # Download the model using asyncio.run since we're in a sync context
                            model_path, download_progress = asyncio.run(download_shard(
                                shard_metadata,
                                on_progress=on_progress
                            ))
                            
                            logger.info(f"Model downloaded successfully to: {model_path}")
                            
                        except Exception as e:
                            logger.error(f"Model download failed: {e}")
                            logger.opt(exception=e).error("Full model download error traceback")
                            
                            # Send failed status with detailed error message
                            current_status = RunnerFailed(error_message=f"Model download failed: {str(e)}")
                            event_sender.send(
                                RunnerStatusUpdated(
                                    runner_id=runner_id, runner_status=current_status
                                )
                            )
                            
                            # Send task failure
                            event_sender.send(
                                TaskStatusUpdated(
                                    task_id=task.task_id, task_status=TaskStatus.Failed
                                )
                            )
                            continue

                    # we load the model if it's connected with a group, or idle without a group. we should never tell a model to connect if it doesn't need to
                    case LoadModel() if (
                        isinstance(current_status, RunnerConnected)
                        and group is not None
                    ) or (isinstance(current_status, RunnerIdle) and group is None):
                        logger.debug(f"LoadModel accepted: status={current_status}, group={'present' if group is not None else 'None'}")
                        current_status = RunnerLoading()
                        logger.info("runner loading")
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )

                        # For actual model loading, we need the full initialization
                        # Reset group to None for non-MLX engines before full initialization
                        if group == "connected":
                            group = None
                        
                        try:
                            logger.info(f"Initializing engine for model: {bound_instance.bound_shard.model_meta.model_id}")
                            model, tokenizer, sampler = initialize_engine(bound_instance)
                            logger.info("Engine initialization completed successfully")
                        except Exception as e:
                            logger.error(f"Model loading failed: {e}")
                            logger.opt(exception=e).error("Full model loading error traceback")
                            
                            # Send failed status with detailed error message
                            current_status = RunnerFailed(error_message=f"Model loading failed: {str(e)}")
                            event_sender.send(
                                RunnerStatusUpdated(
                                    runner_id=runner_id, runner_status=current_status
                                )
                            )
                            
                            # Send task failure
                            event_sender.send(
                                TaskStatusUpdated(
                                    task_id=task.task_id, task_status=TaskStatus.Failed
                                )
                            )
                            continue

                        current_status = RunnerLoaded()
                        logger.info("runner loaded")
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )
                    case StartWarmup() if isinstance(current_status, RunnerLoaded):
                        assert model
                        assert tokenizer
                        assert sampler
                        current_status = RunnerWarmingUp()
                        logger.info("runner warming up")
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )

                        logger.info(f"warming up inference for instance: {instance}")
                        toks = warmup_engine(
                            model=model,
                            tokenizer=tokenizer,
                            sampler=sampler,
                        )
                        logger.info(f"warmed up by generating {toks} tokens")
                        logger.info(
                            f"runner initialized in {time.time() - setup_start_time} seconds"
                        )
                        current_status = RunnerReady()
                        logger.info("runner ready")
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )
                    case ChatCompletion(
                        task_params=task_params, command_id=command_id
                    ) if isinstance(current_status, RunnerReady):
                        assert model
                        assert tokenizer
                        assert sampler
                        logger.info(f"received chat request: {str(task)[:500]}")
                        current_status = RunnerRunning()
                        logger.info("runner running")
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )
                        assert task_params.messages[0].content is not None
                        _check_for_debug_prompts(task_params.messages[0].content)

                        # Generate responses using the selected engine
                        for response in generate_with_engine(
                            model=model,
                            tokenizer=tokenizer,
                            sampler=sampler,
                            task=task_params,
                        ):
                            match response:
                                case GenerationResponse():
                                    if shard_metadata.device_rank == 0:
                                        event_sender.send(
                                            ChunkGenerated(
                                                command_id=command_id,
                                                chunk=TokenChunk(
                                                    corruption_type=enhanced_token.validation_result.error_type or CorruptionType.ENCODING_CORRUPTION,
                                                    affected_range=(enhanced_token.sequence_position, enhanced_token.sequence_position),
                                                    severity=enhanced_token.validation_result.error_details.severity,
                                                    recovery_possible=enhanced_token.validation_result.error_details.recoverable,
                                                    details=enhanced_token.validation_result.error_details.message
                                                )
                                                fallback_coordinator.report_corruption_event(corruption_report)
                                            
                                            # Attempt recovery
                                            failure_mode = _determine_failure_mode(enhanced_token)
                                            try:
                                                # Since we're not in an async context, we need to run the async recovery
                                                import asyncio
                                                recovery_result = asyncio.run(recovery_manager.recover_from_corruption(failure_mode))
                                                
                                                # Report recovery result to fallback coordinator
                                                fallback_coordinator.report_recovery_failure(recovery_result)
                                                
                                                if recovery_result.success:
                                                    logger.info(f"Recovery successful: {recovery_result.details}")
                                                    consecutive_corruptions = 0  # Reset consecutive count on successful recovery
                                                else:
                                                    logger.error(f"Recovery failed: {recovery_result.error_message}")
                                                    
                                                    # Check if we should trigger fallback
                                                    if consecutive_corruptions >= max_consecutive_corruptions:
                                                        logger.error(f"Too many consecutive corruptions ({consecutive_corruptions}), triggering fallback")
                                                        generation_successful = False
                                                        break
                                            except Exception as recovery_error:
                                                logger.error(f"Recovery attempt failed with exception: {recovery_error}")
                                                if consecutive_corruptions >= max_consecutive_corruptions:
                                                    generation_successful = False
                                                    break
                                        else:
                                            # Reset consecutive corruption count on valid token
                                            consecutive_corruptions = 0
                                        
                                        # Add to sequence for potential batch validation
                                        if enhanced_token:
                                            tokens_in_sequence.append(enhanced_token)
                                        
                                        # Send event only for device rank 0 (as before)
                                        if shard_metadata.device_rank == 0:
                                            # Create enhanced event with validation metadata
                                            if enhanced_token:
                                                enhanced_event = _create_enhanced_chunk_event(
                                                    command_id, response, enhanced_token, shard_metadata
                                                )
                                            else:
                                                # Fallback to basic event if no enhanced token
                                                chunk = TokenChunk(
                                                    idx=response.token,
                                                    model=shard_metadata.model_meta.model_id,
                                                    text=response.text,
                                                    token_id=response.token,
                                                    finish_reason=response.finish_reason,
                                                )
                                                enhanced_event = ChunkGenerated(
                                                    command_id=command_id,
                                                    chunk=chunk
                                                )
                                            
                                            event_sender.send(enhanced_event)
                                        # case TokenizedResponse():
                                        # TODO: something here ig
                            
                            # Perform batch validation on the complete sequence
                            if tokens_in_sequence and generation_successful:
                                sequence_reports = corruption_detector.detect_token_corruption(tokens_in_sequence)
                                if sequence_reports:
                                    logger.warning(f"Sequence-level corruption detected: {len(sequence_reports)} issues found")
                                    for report in sequence_reports:
                                        logger.warning(f"Sequence corruption: {report.details}")
                        
                        except Exception as e:
                            logger.error(f"Generation failed with exception: {e}")
                            generation_successful = False
                            
                            # Attempt recovery from generation failure
                            recovery_result = asyncio.run(recovery_manager.recover_from_corruption(FailureMode.PIPELINE_FAILURE))
                            if not recovery_result.success:
                                logger.error("Failed to recover from generation failure")

                        current_status = RunnerReady()
                        logger.info("runner ready")
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )
                    case Shutdown():
                        logger.info("runner shutting down")
                        event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id, task_status=TaskStatus.Complete
                            )
                        )
                        break
                    case _:
                        # Add detailed logging for debugging state machine issues
                        logger.error(
                            f"Task {task.__class__.__name__} rejected: "
                            f"current_status={current_status}, "
                            f"group={'present' if group is not None else 'None'}, "
                            f"group_value={group}, "
                            f"task_details={task}"
                        )
                        
                        # For LoadModel tasks, provide specific guidance
                        if isinstance(task, LoadModel):
                            if isinstance(current_status, RunnerConnected) and group is None:
                                logger.error("LoadModel rejected: RunnerConnected but no group available - this indicates a ConnectToGroup failure")
                            elif isinstance(current_status, RunnerIdle) and group is not None:
                                logger.error("LoadModel rejected: RunnerIdle but group is present - unexpected state")
                            elif isinstance(current_status, RunnerConnected) and group is not None:
                                logger.error("LoadModel rejected: RunnerConnected with group present - condition should have matched")
                            else:
                                logger.error(f"LoadModel rejected: Unexpected state combination - status={current_status}, group={group}")
                        
                        # Instead of crashing with ValueError, gracefully reject the task
                        error_message = f"Task {task.__class__.__name__} rejected due to invalid state transition: current_status={current_status}, group={group}"
                        logger.warning(f"Gracefully rejecting task: {error_message}")
                        
                        # Brief delay to allow for potential state synchronization
                        time.sleep(0.1)
                        
                        # Send task failure status instead of crashing
                        event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id, task_status=TaskStatus.Failed
                            )
                        )
                        
                        # Continue processing instead of crashing - skip to next task
                        continue
                event_sender.send(
                    TaskStatusUpdated(
                        task_id=task.task_id, task_status=TaskStatus.Complete
                    )
                )
                event_sender.send(
                    RunnerStatusUpdated(
                        runner_id=runner_id, runner_status=current_status
                    )
                )
        event_sender.send(
            RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerShutdown())
        )
    except ClosedResourceError:
        logger.warning("runner communication closed unexpectedly")
    except Exception as e:
        logger.opt(exception=e).warning(
            f"Runner {runner_id} crashed with critical exception {e}"
        )
        event_sender.send(
            RunnerStatusUpdated(
                runner_id=runner_id,
                runner_status=RunnerFailed(error_message=str(e)),
            )
        )
    finally:
        event_sender.close()
        task_receiver.close()
        event_sender.join()
        task_receiver.join()
        logger.info("bye from the runner")


EXO_RUNNER_MUST_FAIL = "EXO RUNNER MUST FAIL"
EXO_RUNNER_MUST_OOM = "EXO RUNNER MUST OOM"
EXO_RUNNER_MUST_TIMEOUT = "EXO RUNNER MUST TIMEOUT"


def _determine_failure_mode(enhanced_token: EnhancedTokenChunk) -> FailureMode:
    """Determine the failure mode based on the enhanced token's validation result."""
    if not enhanced_token.validation_result or not enhanced_token.validation_result.error_details:
        return FailureMode.TOKEN_CORRUPTION
    
    error_type = enhanced_token.validation_result.error_details.error_type
    
    if error_type == CorruptionType.ENCODING_CORRUPTION:
        return FailureMode.ENCODING_ERROR
    elif error_type == CorruptionType.CHECKSUM_MISMATCH:
        return FailureMode.COMMUNICATION_FAILURE
    elif error_type == CorruptionType.SEQUENCE_CORRUPTION:
        return FailureMode.SYNCHRONIZATION_FAILURE
    else:
        return FailureMode.TOKEN_CORRUPTION


def _create_enhanced_chunk_event(
    command_id: str, 
    response: GenerationResponse, 
    enhanced_token: Optional[EnhancedTokenChunk],
    shard_metadata: Any
) -> ChunkGenerated:
    """Create an enhanced ChunkGenerated event with validation metadata."""
    # Create the basic chunk
    chunk = TokenChunk(
        idx=response.token,
        model=shard_metadata.model_meta.model_id,
        text=response.text,
        token_id=response.token,
        finish_reason=response.finish_reason,
    )
    
    # Create the enhanced event with validation metadata
    return ChunkGenerated(
        command_id=command_id,
        chunk=chunk,
        validation_result=enhanced_token.validation_result if enhanced_token else None,
        corruption_report=None,  # Could be populated from corruption detector
        device_rank=shard_metadata.device_rank,
        generation_timestamp=enhanced_token.generation_timestamp if enhanced_token else None,
        sequence_position=enhanced_token.sequence_position if enhanced_token else 0
    )


def _create_enhanced_chunk_from_response(response: GenerationResponse, enhanced_token: EnhancedTokenChunk) -> TokenChunk:
    """Create a TokenChunk for events, optionally enhanced with validation metadata."""
    if enhanced_token:
        # Create an enhanced TokenChunk that includes validation information
        # For now, we'll use the standard TokenChunk but could extend this
        return TokenChunk(
            idx=response.token,
            model=enhanced_token.model,
            text=response.text,
            token_id=response.token,
            finish_reason=response.finish_reason,
        )
    else:
        # Fallback to standard TokenChunk
        return TokenChunk(
            idx=response.token,
            model="unknown",  # This should be filled from shard_metadata
            text=response.text,
            token_id=response.token,
            finish_reason=response.finish_reason,
        )


def _check_for_debug_prompts(
    prompt: str | ChatCompletionMessageText | list[ChatCompletionMessageText],
):
    if isinstance(prompt, list):
        if len(prompt) == 0:
            logger.debug("Empty message prompt received in debug prompt")
            return
        prompt = prompt[0]

    if isinstance(prompt, ChatCompletionMessageText):
        prompt = prompt.text

    if EXO_RUNNER_MUST_FAIL in prompt:
        logger.info("raising exception")
        raise Exception("Artificial runner exception - for testing purposes only.")
    if EXO_RUNNER_MUST_OOM in prompt:
        mlx_force_oom()
    if EXO_RUNNER_MUST_TIMEOUT in prompt:
        time.sleep(100)

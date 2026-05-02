from __future__ import annotations

import base64
import json
import time
from collections.abc import Generator
from functools import cache
from typing import TYPE_CHECKING, Any, Callable, Literal

from pydantic import ValidationError

from exo.shared.constants import EXO_MAX_CHUNK_SIZE, EXO_TRACING_ENABLED
from exo.shared.models.model_cards import ModelId, ModelTask
from exo.shared.tracing import clear_trace_buffer, get_trace_buffer
from exo.shared.types.api import ImageGenerationStats
from exo.shared.types.chunks import ErrorChunk, ImageChunk, TokenChunk, ToolCallChunk
from exo.shared.types.common import CommandId
from exo.shared.types.events import (
    BackendFailed,
    BackendInitialized,
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
    TraceEventData,
    TracesCollected,
)
from exo.shared.types.tasks import (
    ConnectToGroup,
    ImageEdits,
    ImageGeneration,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import (
    BoundInstance,
    PyTorchIPEXRingInstance,
)
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
    ImageGenerationResponse,
    PartialImageResponse,
    ToolCallItem,
    ToolCallResponse,
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
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.shared.types.worker.shards import (
    CfgShardMetadata,
    PipelineShardMetadata,
    ShardMetadata,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.runner.bootstrap import logger

# Lazy imports for MLX backend - only imported when needed
if TYPE_CHECKING:
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from openai_harmony import (  # pyright: ignore[reportMissingTypeStubs]
        Role,
        StreamableParser,
    )



def _is_primary_output_node(shard_metadata: ShardMetadata) -> bool:
    """Check if this node is the primary output node for image generation.

    For CFG models: the last pipeline stage in CFG group 0 (positive prompt).
    For non-CFG models: the last pipeline stage.
    """
    if isinstance(shard_metadata, CfgShardMetadata):
        is_pipeline_last = (
            shard_metadata.pipeline_rank == shard_metadata.pipeline_world_size - 1
        )
        return is_pipeline_last and shard_metadata.cfg_rank == 0
    elif isinstance(shard_metadata, PipelineShardMetadata):
        return shard_metadata.device_rank == shard_metadata.world_size - 1
    return False


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
    device_rank = shard_metadata.device_rank
    logger.info("hello from the runner")
    if getattr(shard_metadata, "immediate_exception", False):
        raise Exception("Fake exception - runner failed to spin up.")
    if timeout := getattr(shard_metadata, "should_timeout", 0):
        time.sleep(timeout)

    setup_start_time = time.time()

    # Detect backend type from instance
    # Handle both direct instance and Pydantic tagged union
    instance_type_name = type(instance).__name__
    is_pytorch_ipex = (
        isinstance(instance, PyTorchIPEXRingInstance)
        or instance_type_name == "PyTorchIPEXRingInstance"
        or (
            hasattr(instance, "__class__")
            and instance.__class__.__name__ == "PyTorchIPEXRingInstance"
        )
    )

    if is_pytorch_ipex:
        backend_type = "pytorch_ipex"
        logger.info(
            f"Using PyTorch XPU backend for PyTorchIPEXRingInstance (type: {instance_type_name})"
        )

        # Lazy-load PyTorch XPU backend modules
        from exo.worker.engines.pytorch_ipex.generator import pytorch_ipex_generate
        from exo.worker.engines.pytorch_ipex.model_loader import ModelLoader
        from exo.worker.engines.pytorch_ipex.warmup import warmup_pytorch_ipex_inference

        logger.info("PyTorch XPU backend modules loaded")
    else:
        # MLX backend - import MLX modules only when needed
        backend_type = "mlx"
        logger.info("Using MLX backend")

        # Import MLX-specific modules
        import mlx.core as mx
        from mlx_lm.models.gpt_oss import Model as GptOssModel

        from exo.worker.engines.image import (
            DistributedImageModel,
            generate_image,
            initialize_image_model,
            warmup_image_generator,
        )
        from exo.worker.engines.mlx.cache import KVPrefixCache
        from exo.worker.engines.mlx.generator.generate import (
            mlx_generate,
            warmup_inference,
        )
        from exo.worker.engines.mlx.utils_mlx import (
            apply_chat_template,
            detect_thinking_prompt_suffix,
            initialize_mlx,
            load_mlx_items,
        )

    # Emit BackendInitialized event for observability
    try:
        # We'll emit the actual BackendInitialized event after model loading
        # when we have full device information
        pass
    except Exception as e:
        logger.error(f"Failed to initialize backend {backend_type}: {e}")
        event_sender.send(
            BackendFailed(
                runner_id=runner_id,
                backend_type=backend_type,
                error_message=str(e),
                fallback_backend=None,
            )
        )
        raise

    # Initialize backend-specific variables
    model: Any = None
    tokenizer: Any = None
    group: Any = None
    kv_prefix_cache: Any = None
    pytorch_ipex_model: Any = None
    pytorch_ipex_tokenizer: Any = None
    device_type: Any = None
    device_id: Any = None

    current_status: RunnerStatus = RunnerIdle()
    logger.info("runner created")
    event_sender.send(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status)
    )
    seen = set[TaskId]()
    with task_receiver as tasks:
        for task in tasks:
            if task.task_id in seen:
                logger.warning("repeat task - potential error")
            seen.add(task.task_id)
            event_sender.send(
                TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Running)
            )
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
                    event_sender.send(TaskAcknowledged(task_id=task.task_id))

                    if backend_type == "pytorch_ipex":
                        # PyTorch distributed: initialize Gloo process group
                        try:
                            from exo.worker.engines.pytorch_ipex.distributed import (
                                ProcessGroupConfig,
                                init_process_group,
                            )

                            assert isinstance(instance, PyTorchIPEXRingInstance)
                            rank = shard_metadata.device_rank
                            world_size = len(instance.shard_assignments.node_to_runner)

                            # Find rank 0's node to derive MASTER_ADDR
                            rank_0_node = None
                            for node_id, r_id in instance.shard_assignments.node_to_runner.items():
                                r_shard = instance.shard_assignments.runner_to_shard.get(r_id)
                                if r_shard is not None and r_shard.device_rank == 0:
                                    rank_0_node = node_id
                                    break

                            if rank_0_node is None:
                                raise RuntimeError(
                                    f"Could not find rank 0 node in shard assignments"
                                )

                            rank_0_hosts = instance.hosts_by_node.get(rank_0_node, [])
                            if not rank_0_hosts:
                                raise RuntimeError(
                                    f"No hosts found for rank 0 node {rank_0_node}"
                                )
                            master_addr = rank_0_hosts[0].ip
                            master_port = instance.ephemeral_port

                            config = ProcessGroupConfig(
                                rank=rank,
                                world_size=world_size,
                                master_addr=master_addr,
                                master_port=master_port,
                            )
                            logger.info(
                                f"Initializing Gloo process group: rank={rank}, "
                                f"world_size={world_size}, master_addr={master_addr}, "
                                f"master_port={master_port}"
                            )
                            init_process_group(config)
                            group = True  # sentinel indicating process group is active
                            logger.info("Gloo process group initialized successfully")
                        except Exception as e:
                            error_msg = (
                                f"Failed to initialize process group: "
                                f"rank={shard_metadata.device_rank}, "
                                f"world_size={len(instance.shard_assignments.node_to_runner)}, "
                                f"backend=gloo: {e}"
                            )
                            logger.error(error_msg)
                            current_status = RunnerFailed(error_message=error_msg)
                            event_sender.send(
                                RunnerStatusUpdated(
                                    runner_id=runner_id, runner_status=current_status
                                )
                            )
                            event_sender.send(
                                TaskStatusUpdated(
                                    task_id=task.task_id,
                                    task_status=TaskStatus.Complete,
                                )
                            )
                            continue
                    else:
                        group = initialize_mlx(bound_instance)

                    logger.info("runner connected")
                    current_status = RunnerConnected()

                # we load the model if it's connected with a group, or idle without a group. we should never tell a model to connect if it doesn't need to
                case LoadModel() if (
                    isinstance(current_status, RunnerConnected) and group is not None
                ) or (isinstance(current_status, RunnerIdle) and group is None):
                    current_status = RunnerLoading()
                    logger.info("runner loading")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                    event_sender.send(TaskAcknowledged(task_id=task.task_id))

                    def on_model_load_timeout() -> None:
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id,
                                runner_status=RunnerFailed(
                                    error_message="Model loading timed out"
                                ),
                            )
                        )
                        time.sleep(0.5)

                    if backend_type == "pytorch_ipex":
                        # PyTorch XPU backend model loading
                        try:
                            if (
                                ModelTask.TextGeneration
                                in shard_metadata.model_card.tasks
                            ):
                                logger.info("Loading PyTorch XPU model...")

                                # Use GPU detector to determine device type and index
                                from exo.worker.engines.pytorch_ipex.gpu_detector import detect_gpus

                                gpu_report = detect_gpus()
                                if gpu_report.has_gpu and gpu_report.gpus:
                                    primary_gpu = gpu_report.gpus[0]
                                    device_type = primary_gpu.device_type
                                    device_id = primary_gpu.device_index
                                    logger.info(
                                        f"GPU detected: {primary_gpu.name}, "
                                        f"type={device_type}, index={device_id}, "
                                        f"architecture={primary_gpu.memory_architecture.value}"
                                    )
                                else:
                                    device_type = "cpu"
                                    device_id = 0
                                    logger.info("No GPU detected, falling back to CPU")

                                # Log selective layer loading range
                                start_layer = shard_metadata.start_layer
                                end_layer = shard_metadata.end_layer
                                n_layers = shard_metadata.n_layers
                                num_layers_to_load = end_layer - start_layer
                                logger.info(
                                    f"Selective layer loading: layers [{start_layer}, {end_layer}) "
                                    f"of {n_layers} total ({num_layers_to_load} layers)"
                                )

                                # Initialize model loader and load model
                                model_loader = ModelLoader()

                                import asyncio
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                try:
                                    pytorch_ipex_model, pytorch_ipex_tokenizer = loop.run_until_complete(
                                        model_loader.load_model(
                                            shard_metadata=shard_metadata,
                                            device_type=device_type,
                                            device_id=device_id,
                                        )
                                    )
                                finally:
                                    loop.close()

                                logger.info(
                                    f"PyTorch XPU model loaded successfully on {device_type}:{device_id}, "
                                    f"layers [{start_layer}, {end_layer})"
                                )

                                # Determine device info for BackendInitialized event
                                if device_type == "xpu":
                                    backend_device_type = "GPU"
                                    backend_device_name = "Intel Arc GPU"
                                    backend_runtime = "XPU"
                                elif device_type == "cuda":
                                    backend_device_type = "GPU"
                                    backend_device_name = "NVIDIA GPU"
                                    backend_runtime = "CUDA"
                                else:
                                    backend_device_type = "CPU"
                                    backend_device_name = "CPU"
                                    backend_runtime = "CPU"

                                # Emit BackendInitialized event
                                event_sender.send(
                                    BackendInitialized(
                                        runner_id=runner_id,
                                        backend_type=backend_type,
                                        device_type=backend_device_type,
                                        device_name=backend_device_name,
                                        runtime=backend_runtime,
                                    )
                                )
                            else:
                                raise ValueError(
                                    f"PyTorch XPU backend only supports TextGeneration, got: {shard_metadata.model_card.tasks}"
                                )
                        except (RuntimeError, MemoryError) as e:
                            error_str = str(e).lower()
                            if "out of memory" in error_str or "oom" in error_str or isinstance(e, MemoryError):
                                # OOM: report required vs available memory
                                avail_info = ""
                                try:
                                    from exo.worker.engines.pytorch_ipex.gpu_detector import detect_gpus as _detect_gpus
                                    _report = _detect_gpus()
                                    if _report.has_gpu and _report.gpus:
                                        _gpu = _report.gpus[0]
                                        avail_info = (
                                            f", available_memory={_gpu.available_memory_bytes / (1024**3):.2f} GiB"
                                        )
                                except Exception:
                                    pass
                                error_msg = (
                                    f"Model loading OOM: layers [{shard_metadata.start_layer}, "
                                    f"{shard_metadata.end_layer}) of {shard_metadata.n_layers}"
                                    f"{avail_info}: {e}"
                                )
                            else:
                                error_msg = f"Model loading failed: {e}"
                            logger.error(error_msg)
                            current_status = RunnerFailed(error_message=error_msg)
                            event_sender.send(
                                RunnerStatusUpdated(
                                    runner_id=runner_id,
                                    runner_status=current_status,
                                )
                            )
                            event_sender.send(
                                TaskStatusUpdated(
                                    task_id=task.task_id,
                                    task_status=TaskStatus.Complete,
                                )
                            )
                            continue
                        except Exception as e:
                            error_msg = f"Model loading failed: {e}"
                            logger.error(error_msg)
                            current_status = RunnerFailed(error_message=error_msg)
                            event_sender.send(
                                RunnerStatusUpdated(
                                    runner_id=runner_id,
                                    runner_status=current_status,
                                )
                            )
                            event_sender.send(
                                TaskStatusUpdated(
                                    task_id=task.task_id,
                                    task_status=TaskStatus.Complete,
                                )
                            )
                            continue
                    elif backend_type == "mlx":
                        # MLX backend model loading
                        if ModelTask.TextGeneration in shard_metadata.model_card.tasks:
                            model, tokenizer = load_mlx_items(
                                bound_instance, group, on_timeout=on_model_load_timeout
                            )
                            logger.info(
                                f"model has_tool_calling={tokenizer.has_tool_calling}"
                            )
                            kv_prefix_cache = KVPrefixCache(group)

                        elif (
                            ModelTask.TextToImage in shard_metadata.model_card.tasks
                            or ModelTask.ImageToImage in shard_metadata.model_card.tasks
                        ):
                            model = initialize_image_model(bound_instance)
                        else:
                            raise ValueError(
                                f"Unknown model task(s): {shard_metadata.model_card.tasks}"
                            )

                        # Emit BackendInitialized event now that we have device info
                        try:
                            # For MLX backend, we know it's using Metal on Apple Silicon
                            device_info = {
                                "device_type": "METAL",
                                "device_name": "Apple Silicon",
                                "runtime": "METAL",
                            }

                            event_sender.send(
                                BackendInitialized(
                                    runner_id=runner_id,
                                    backend_type=backend_type,
                                    device_type=device_info["device_type"],
                                    device_name=device_info["device_name"],
                                    runtime=device_info["runtime"],
                                )
                            )
                            logger.info(
                                f"Backend initialized: {backend_type} on "
                                f"{device_info['device_type']} ({device_info['device_name']})"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to emit BackendInitialized event: {e}"
                            )
                    else:
                        # Unknown backend type
                        error_msg = f"Unknown backend type: {backend_type}"
                        logger.error(error_msg)
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id,
                                runner_status=RunnerFailed(
                                    error_message=error_msg
                                ),
                            )
                        )
                        raise ValueError(error_msg)

                    current_status = RunnerLoaded()
                    logger.info("runner loaded")
                case StartWarmup() if isinstance(current_status, RunnerLoaded):
                    # Verify model and tokenizer are loaded based on backend type
                    if backend_type == "pytorch_ipex":
                        assert pytorch_ipex_model
                        assert pytorch_ipex_tokenizer
                    else:
                        assert model
                        assert tokenizer

                    current_status = RunnerWarmingUp()
                    logger.info("runner warming up")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                    event_sender.send(TaskAcknowledged(task_id=task.task_id))

                    logger.info(f"warming up inference for instance: {instance}")
                    if ModelTask.TextGeneration in shard_metadata.model_card.tasks:
                        if backend_type == "mlx":
                            # Check model type for MLX
                            assert model.__class__.__name__ != "DistributedImageModel"

                        if backend_type == "mlx":
                            toks = warmup_inference(
                                model=model,
                                tokenizer=tokenizer,
                                group=group,
                                # kv_prefix_cache=kv_prefix_cache,  # supply for warmup-time prefix caching
                            )
                            logger.info(f"warmed up by generating {toks} tokens")
                        elif backend_type == "pytorch_ipex":
                            # PyTorch XPU backend warmup with CPU tensor staging
                            from exo.worker.engines.pytorch_ipex.distributed import (
                                send_activation,
                                recv_activation,
                            )

                            toks = warmup_pytorch_ipex_inference(
                                model=pytorch_ipex_model,
                                tokenizer=pytorch_ipex_tokenizer,
                                device_type=device_type,
                                device_id=device_id,
                                warmup_tokens=10,
                            )
                            logger.info(f"warmed up by generating {toks} tokens")

                            # Distributed warmup: exchange dummy activations with neighbors
                            rank = shard_metadata.device_rank
                            world_size = shard_metadata.world_size
                            if world_size > 1:
                                import torch
                                device_str = f"{device_type}:{device_id}" if device_type != "cpu" else "cpu"
                                dummy_shape = (1, 1, 128)  # small warmup tensor
                                dummy_dtype = torch.float32

                                # Send to next rank (if not last)
                                if rank < world_size - 1:
                                    dummy_tensor = torch.zeros(dummy_shape, dtype=dummy_dtype, device=device_str)
                                    send_activation(dummy_tensor, dst_rank=rank + 1)
                                    logger.info(f"Warmup: sent activation to rank {rank + 1}")

                                # Recv from previous rank (if not first)
                                if rank > 0:
                                    received = recv_activation(
                                        shape=dummy_shape,
                                        dtype=dummy_dtype,
                                        src_rank=rank - 1,
                                        target_device=device_str,
                                    )
                                    logger.info(
                                        f"Warmup: received activation from rank {rank - 1}, "
                                        f"shape={tuple(received.shape)}"
                                    )

                        logger.info(
                            f"runner initialized in {time.time() - setup_start_time} seconds"
                        )
                    elif (
                        ModelTask.TextToImage in shard_metadata.model_card.tasks
                        or ModelTask.ImageToImage in shard_metadata.model_card.tasks
                    ):
                        if backend_type == "mlx":
                            assert isinstance(model, DistributedImageModel)
                            image = warmup_image_generator(model=model)
                            if image is not None:
                                logger.info(
                                    f"warmed up by generating {image.size} image"
                                )
                            else:
                                logger.info("warmup completed (non-primary node)")
                        else:
                            raise ValueError(
                                "Image generation only supported with MLX backend"
                            )

                    current_status = RunnerReady()
                    logger.info("runner ready")
                case TextGeneration(task_params=task_params, command_id=command_id) if (
                    isinstance(current_status, RunnerReady)
                ):
                    logger.info(f"received chat request: {task}")
                    current_status = RunnerRunning()
                    logger.info("runner running")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                    event_sender.send(TaskAcknowledged(task_id=task.task_id))

                    if backend_type == "pytorch_ipex":
                        # PyTorch XPU backend text generation with distributed communication
                        assert pytorch_ipex_model is not None
                        assert pytorch_ipex_tokenizer is not None
                        assert device_type is not None
                        assert device_id is not None

                        try:
                            from exo.worker.engines.pytorch_ipex.distributed import (
                                send_activation as _send_act,
                                recv_activation as _recv_act,
                            )

                            rank = shard_metadata.device_rank
                            world_size = shard_metadata.world_size
                            device_str = f"{device_type}:{device_id}" if device_type != "cpu" else "cpu"
                            is_first_rank = rank == 0
                            is_last_rank = rank == world_size - 1

                            # Build prompt from messages
                            prompt_parts = []
                            for msg in task_params.input:
                                prompt_parts.append(f"{msg.role}: {msg.content}")
                            prompt = "\n".join(prompt_parts) + "\nassistant:"

                            logger.info(
                                f"Generating with PyTorch XPU, rank={rank}/{world_size}, "
                                f"prompt length: {len(prompt)}"
                            )

                            # Receive activation from previous rank (if not first)
                            if not is_first_rank and world_size > 1:
                                # Middle/last ranks receive input activation from previous rank
                                # Shape/dtype will be determined by the model's hidden size
                                # For now, we proceed with the local generation which handles this
                                logger.info(f"Rank {rank}: waiting for activation from rank {rank - 1}")

                            # Generate tokens using PyTorch XPU
                            pytorch_ipex_generator = pytorch_ipex_generate(
                                model=pytorch_ipex_model,
                                tokenizer=pytorch_ipex_tokenizer,
                                prompt=prompt,
                                device_type=device_type,
                                device_id=device_id,
                                max_tokens=task_params.max_output_tokens or 100,
                                temperature=task_params.temperature or 1.0,
                                top_k=task_params.top_k,
                                top_p=task_params.top_p,
                                model_id=str(shard_metadata.model_card.model_id),
                            )

                            # Forward responses to event sender
                            for response in pytorch_ipex_generator:
                                match response:
                                    case GenerationResponse():
                                        if (
                                            device_rank == 0
                                            and response.finish_reason == "error"
                                        ):
                                            event_sender.send(
                                                ChunkGenerated(
                                                    command_id=command_id,
                                                    chunk=ErrorChunk(
                                                        error_message=response.text,
                                                        model=shard_metadata.model_card.model_id,
                                                    ),
                                                )
                                            )
                                        elif device_rank == 0:
                                            event_sender.send(
                                                ChunkGenerated(
                                                    command_id=command_id,
                                                    chunk=TokenChunk(
                                                        model=shard_metadata.model_card.model_id,
                                                        text=response.text,
                                                        token_id=response.token,
                                                        usage=response.usage,
                                                        finish_reason=response.finish_reason,
                                                        stats=response.stats,
                                                        logprob=response.logprob,
                                                        top_logprobs=response.top_logprobs,
                                                    ),
                                                )
                                            )
                                    case ToolCallResponse():
                                        if device_rank == 0:
                                            event_sender.send(
                                                ChunkGenerated(
                                                    command_id=command_id,
                                                    chunk=ToolCallChunk(
                                                        tool_calls=response.tool_calls,
                                                        model=shard_metadata.model_card.model_id,
                                                        usage=response.usage,
                                                    ),
                                                )
                                            )
                        except Exception as e:
                            logger.error(f"PyTorch XPU generation failed: {e}")
                            if device_rank == 0:
                                event_sender.send(
                                    ChunkGenerated(
                                        command_id=command_id,
                                        chunk=ErrorChunk(
                                            model=shard_metadata.model_card.model_id,
                                            finish_reason="error",
                                            error_message=str(e),
                                        ),
                                    )
                                )
                            raise
                    else:
                        # MLX backend text generation
                        assert model and not isinstance(model, DistributedImageModel)
                        assert tokenizer

                        try:
                            _check_for_debug_prompts(task_params)

                            # Build prompt once - used for both generation and thinking detection
                            prompt = apply_chat_template(tokenizer, task_params)

                            # Generate responses using the actual MLX generation
                            mlx_generator = mlx_generate(
                                model=model,
                                tokenizer=tokenizer,
                                task=task_params,
                                prompt=prompt,
                                kv_prefix_cache=kv_prefix_cache,
                                group=group,
                            )

                            # For other thinking models (GLM, etc.), check if we need to
                            # prepend the thinking tag that was consumed by the chat template
                            if detect_thinking_prompt_suffix(prompt, tokenizer):
                                mlx_generator = parse_thinking_models(
                                    mlx_generator, tokenizer
                                )

                            # Kimi-K2 has tool call sections - we don't care about them
                            if "kimi" in shard_metadata.model_card.model_id.lower():
                                mlx_generator = filter_kimi_tokens(mlx_generator)
                                patch_kimi_tokenizer(tokenizer)

                            # GLM models need patched parser (upstream has bug with None regex match)
                            elif "glm" in shard_metadata.model_card.model_id.lower():
                                patch_glm_tokenizer(tokenizer)

                            # GPT-OSS specific parsing to match other model formats.
                            elif isinstance(model, GptOssModel):
                                mlx_generator = parse_gpt_oss(mlx_generator)

                            if tokenizer.has_tool_calling and not isinstance(
                                model, GptOssModel
                            ):
                                assert tokenizer.tool_call_start
                                assert tokenizer.tool_call_end
                                assert tokenizer.tool_parser  # pyright: ignore[reportAny]
                                mlx_generator = parse_tool_calls(
                                    mlx_generator,
                                    tokenizer.tool_call_start,
                                    tokenizer.tool_call_end,
                                    tokenizer.tool_parser,  # pyright: ignore[reportAny]
                                )

                            completion_tokens = 0
                            for response in mlx_generator:
                                match response:
                                    case GenerationResponse():
                                        completion_tokens += 1
                                        if (
                                            device_rank == 0
                                            and response.finish_reason == "error"
                                        ):
                                            event_sender.send(
                                                ChunkGenerated(
                                                    command_id=command_id,
                                                    chunk=ErrorChunk(
                                                        error_message=response.text,
                                                        model=shard_metadata.model_card.model_id,
                                                    ),
                                                )
                                            )

                                        elif device_rank == 0:
                                            assert response.finish_reason not in (
                                                "error",
                                                "tool_calls",
                                                "function_call",
                                            )
                                            event_sender.send(
                                                ChunkGenerated(
                                                    command_id=command_id,
                                                    chunk=TokenChunk(
                                                        model=shard_metadata.model_card.model_id,
                                                        text=response.text,
                                                        token_id=response.token,
                                                        usage=response.usage,
                                                        finish_reason=response.finish_reason,
                                                        stats=response.stats,
                                                        logprob=response.logprob,
                                                        top_logprobs=response.top_logprobs,
                                                    ),
                                                )
                                            )
                                    case ToolCallResponse():
                                        if device_rank == 0:
                                            event_sender.send(
                                                ChunkGenerated(
                                                    command_id=command_id,
                                                    chunk=ToolCallChunk(
                                                        tool_calls=response.tool_calls,
                                                        model=shard_metadata.model_card.model_id,
                                                        usage=response.usage,
                                                    ),
                                                )
                                            )

                        # can we make this more explicit?
                        except Exception as e:
                            if device_rank == 0:
                                event_sender.send(
                                    ChunkGenerated(
                                        command_id=command_id,
                                        chunk=ErrorChunk(
                                            model=shard_metadata.model_card.model_id,
                                            finish_reason="error",
                                            error_message=str(e),
                                        ),
                                    )
                                )
                            raise

                    current_status = RunnerReady()
                    logger.info("runner ready")
                case ImageGeneration(
                    task_params=task_params, command_id=command_id
                ) if isinstance(current_status, RunnerReady):
                    if backend_type != "mlx":
                        raise ValueError(
                            "Image generation only supported with MLX backend"
                        )
                    assert isinstance(model, DistributedImageModel)
                    logger.info(f"received image generation request: {str(task)[:500]}")
                    current_status = RunnerRunning()
                    logger.info("runner running")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                    event_sender.send(TaskAcknowledged(task_id=task.task_id))

                    try:
                        image_index = 0
                        for response in generate_image(model=model, task=task_params):
                            is_primary_output = _is_primary_output_node(shard_metadata)

                            if is_primary_output:
                                match response:
                                    case PartialImageResponse():
                                        logger.info(
                                            f"sending partial ImageChunk {response.partial_index}/{response.total_partials}"
                                        )
                                        _process_image_response(
                                            response,
                                            command_id,
                                            shard_metadata,
                                            event_sender,
                                            image_index,
                                        )
                                    case ImageGenerationResponse():
                                        logger.info("sending final ImageChunk")
                                        _process_image_response(
                                            response,
                                            command_id,
                                            shard_metadata,
                                            event_sender,
                                            image_index,
                                        )
                                        image_index += 1
                    # can we make this more explicit?
                    except Exception as e:
                        if _is_primary_output_node(shard_metadata):
                            event_sender.send(
                                ChunkGenerated(
                                    command_id=command_id,
                                    chunk=ErrorChunk(
                                        model=shard_metadata.model_card.model_id,
                                        finish_reason="error",
                                        error_message=str(e),
                                    ),
                                )
                            )
                        raise
                    finally:
                        _send_traces_if_enabled(
                            event_sender, task.task_id, shard_metadata.device_rank
                        )

                    current_status = RunnerReady()
                    logger.info("runner ready")
                case ImageEdits(task_params=task_params, command_id=command_id) if (
                    isinstance(current_status, RunnerReady)
                ):
                    if backend_type != "mlx":
                        raise ValueError("Image edits only supported with MLX backend")
                    assert isinstance(model, DistributedImageModel)
                    logger.info(f"received image edits request: {str(task)[:500]}")
                    current_status = RunnerRunning()
                    logger.info("runner running")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                    event_sender.send(TaskAcknowledged(task_id=task.task_id))

                    try:
                        image_index = 0
                        for response in generate_image(model=model, task=task_params):
                            if _is_primary_output_node(shard_metadata):
                                match response:
                                    case PartialImageResponse():
                                        logger.info(
                                            f"sending partial ImageChunk {response.partial_index}/{response.total_partials}"
                                        )
                                        _process_image_response(
                                            response,
                                            command_id,
                                            shard_metadata,
                                            event_sender,
                                            image_index,
                                        )
                                    case ImageGenerationResponse():
                                        logger.info("sending final ImageChunk")
                                        _process_image_response(
                                            response,
                                            command_id,
                                            shard_metadata,
                                            event_sender,
                                            image_index,
                                        )
                                        image_index += 1
                    except Exception as e:
                        if _is_primary_output_node(shard_metadata):
                            event_sender.send(
                                ChunkGenerated(
                                    command_id=command_id,
                                    chunk=ErrorChunk(
                                        model=shard_metadata.model_card.model_id,
                                        finish_reason="error",
                                        error_message=str(e),
                                    ),
                                )
                            )
                        raise
                    finally:
                        _send_traces_if_enabled(
                            event_sender, task.task_id, shard_metadata.device_rank
                        )

                    current_status = RunnerReady()
                    logger.info("runner ready")
                case Shutdown():
                    current_status = RunnerShuttingDown()
                    logger.info("runner shutting down")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                    event_sender.send(TaskAcknowledged(task_id=task.task_id))

                    # Clean up backend-specific resources
                    if backend_type == "pytorch_ipex":
                        # Destroy distributed process group first (with timeout)
                        if group is not None:
                            try:
                                from exo.worker.engines.pytorch_ipex.distributed import (
                                    destroy_process_group,
                                )
                                logger.info("Destroying distributed process group (5s timeout)")
                                destroy_process_group(timeout_seconds=5.0)
                                logger.info("Process group destroyed")
                            except Exception as e:
                                logger.warning(f"Error destroying process group: {e}")
                            group = None

                        # Clean up PyTorch XPU resources
                        logger.info("Cleaning up PyTorch XPU resources")
                        pytorch_ipex_model = None
                        pytorch_ipex_tokenizer = None

                        # Clear PyTorch caches
                        try:
                            import torch

                            if device_type == "xpu" and hasattr(torch, "xpu"):
                                torch.xpu.empty_cache()  # type: ignore
                            elif device_type == "cuda":
                                torch.cuda.empty_cache()

                            import gc

                            gc.collect()
                            logger.info("PyTorch XPU resources cleaned up")
                        except Exception as e:
                            logger.warning(f"Error during PyTorch XPU cleanup: {e}")
                    else:
                        # Clean up MLX resources
                        del model, tokenizer, group
                        mx.clear_cache()
                        import gc

                        gc.collect()

                    current_status = RunnerShutdown()
                    logger.info("runner shutdown complete")
                case _:
                    raise ValueError(
                        f"Received {task.__class__.__name__} outside of state machine in {current_status=}"
                    )
            event_sender.send(
                TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Complete)
            )
            event_sender.send(
                RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status)
            )
            if isinstance(current_status, RunnerShutdown):
                # Final cleanup already done in Shutdown case
                break


@cache
def get_gpt_oss_encoding():
    from openai_harmony import (  # pyright: ignore[reportMissingTypeStubs]
        HarmonyEncodingName,
        load_harmony_encoding,
    )
    
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return encoding


def filter_kimi_tokens(
    responses: Generator[GenerationResponse | ToolCallResponse],
) -> Generator[GenerationResponse]:
    for resp in responses:
        assert isinstance(resp, GenerationResponse)
        if (
            resp.text == "<|tool_calls_section_begin|>"
            or resp.text == "<|tool_calls_section_end|>"
        ):
            continue
        yield resp


def parse_gpt_oss(
    responses: Generator[GenerationResponse | ToolCallResponse],
) -> Generator[GenerationResponse | ToolCallResponse]:
    encoding = get_gpt_oss_encoding()
    stream = StreamableParser(encoding, role=Role.ASSISTANT)
    thinking = False
    current_tool_name: str | None = None
    tool_arg_parts: list[str] = []

    for response in responses:
        assert isinstance(response, GenerationResponse)
        stream.process(response.token)

        delta = stream.last_content_delta
        ch = stream.current_channel
        recipient = stream.current_recipient

        if recipient != current_tool_name:
            if current_tool_name is not None:
                prefix = "functions."
                if current_tool_name.startswith(prefix):
                    current_tool_name = current_tool_name[len(prefix) :]
                yield ToolCallResponse(
                    tool_calls=[
                        ToolCallItem(
                            name=current_tool_name,
                            arguments="".join(tool_arg_parts).strip(),
                        )
                    ],
                    usage=response.usage,
                )
                tool_arg_parts = []
            current_tool_name = recipient

        # If inside a tool call, accumulate arguments
        if current_tool_name is not None:
            if delta:
                tool_arg_parts.append(delta)
            continue

        if ch == "analysis" and not thinking:
            thinking = True
            yield response.model_copy(update={"text": "<think>"})

        if ch != "analysis" and thinking:
            thinking = False
            yield response.model_copy(update={"text": "</think>"})

        if delta:
            yield response.model_copy(update={"text": delta})

        if response.finish_reason is not None:
            if thinking:
                yield response.model_copy(update={"text": "</think>"})
            yield response


def parse_thinking_models(
    responses: Generator[GenerationResponse | ToolCallResponse],
    tokenizer: TokenizerWrapper,
) -> Generator[GenerationResponse | ToolCallResponse]:
    """
    For models that inject thinking tags in the prompt (like GLM-4.7),
    prepend the thinking tag to the output stream so the frontend
    can properly parse thinking content.
    """
    first = True
    for response in responses:
        if isinstance(response, ToolCallResponse):
            yield response
            continue
        if first:
            first = False
            yield response.model_copy(
                update={
                    "text": tokenizer.think_start,
                    "token": tokenizer.think_start_id,
                }
            )
        yield response


def _send_image_chunk(
    encoded_data: str,
    command_id: CommandId,
    model_id: ModelId,
    event_sender: MpSender[Event],
    image_index: int,
    is_partial: bool,
    partial_index: int | None = None,
    total_partials: int | None = None,
    stats: ImageGenerationStats | None = None,
    image_format: Literal["png", "jpeg", "webp"] | None = None,
) -> None:
    """Send base64-encoded image data as chunks via events."""
    data_chunks = [
        encoded_data[i : i + EXO_MAX_CHUNK_SIZE]
        for i in range(0, len(encoded_data), EXO_MAX_CHUNK_SIZE)
    ]
    total_chunks = len(data_chunks)
    for chunk_index, chunk_data in enumerate(data_chunks):
        # Only include stats on the last chunk of the final image
        chunk_stats = (
            stats if chunk_index == total_chunks - 1 and not is_partial else None
        )
        event_sender.send(
            ChunkGenerated(
                command_id=command_id,
                chunk=ImageChunk(
                    model=model_id,
                    data=chunk_data,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    image_index=image_index,
                    is_partial=is_partial,
                    partial_index=partial_index,
                    total_partials=total_partials,
                    stats=chunk_stats,
                    format=image_format,
                ),
            )
        )


def _send_traces_if_enabled(
    event_sender: MpSender[Event],
    task_id: TaskId,
    rank: int,
) -> None:
    if not EXO_TRACING_ENABLED:
        return

    traces = get_trace_buffer()
    if traces:
        trace_data = [
            TraceEventData(
                name=t.name,
                start_us=t.start_us,
                duration_us=t.duration_us,
                rank=t.rank,
                category=t.category,
            )
            for t in traces
        ]
        event_sender.send(
            TracesCollected(
                task_id=task_id,
                rank=rank,
                traces=trace_data,
            )
        )
    clear_trace_buffer()


def _process_image_response(
    response: ImageGenerationResponse | PartialImageResponse,
    command_id: CommandId,
    shard_metadata: ShardMetadata,
    event_sender: MpSender[Event],
    image_index: int,
) -> None:
    """Process a single image response and send chunks."""
    encoded_data = base64.b64encode(response.image_data).decode("utf-8")
    is_partial = isinstance(response, PartialImageResponse)
    # Extract stats from final ImageGenerationResponse if available
    stats = response.stats if isinstance(response, ImageGenerationResponse) else None
    _send_image_chunk(
        encoded_data=encoded_data,
        command_id=command_id,
        model_id=shard_metadata.model_card.model_id,
        event_sender=event_sender,
        image_index=response.image_index,
        is_partial=is_partial,
        partial_index=response.partial_index if is_partial else None,
        total_partials=response.total_partials if is_partial else None,
        stats=stats,
        image_format=response.format,
    )


def parse_tool_calls(
    responses: Generator[GenerationResponse | ToolCallResponse],
    tool_call_start: str,
    tool_call_end: str,
    tool_parser: Callable[[str], dict[str, Any] | list[dict[str, Any]]],
) -> Generator[GenerationResponse | ToolCallResponse]:
    in_tool_call = False
    tool_call_text_parts: list[str] = []
    for response in responses:
        assert isinstance(response, GenerationResponse)
        # assumption: the tool call start is one token
        if response.text == tool_call_start:
            in_tool_call = True
            continue
        # assumption: the tool call end is one token
        if in_tool_call and response.text == tool_call_end:
            try:
                # tool_parser returns an arbitrarily nested python dictionary
                # we actually don't want the python dictionary, we just want to
                # parse the top level { function: ..., arguments: ... } structure
                # as we're just gonna hand it back to the api anyway
                parsed = tool_parser("".join(tool_call_text_parts).strip())
                logger.info(f"parsed {tool_call_text_parts=} into {parsed=}")
                if isinstance(parsed, list):
                    tools = [_validate_single_tool(tool) for tool in parsed]
                else:
                    tools = [_validate_single_tool(parsed)]
                yield ToolCallResponse(tool_calls=tools, usage=response.usage)

            except (
                json.JSONDecodeError,
                ValidationError,
                ValueError,
                AttributeError,
            ) as e:
                # ValueError: our parsers raise this for malformed tool calls
                # AttributeError: upstream parsers (e.g. glm47) may raise this when regex doesn't match
                logger.opt(exception=e).warning("tool call parsing failed")
                # assumption: talking about tool calls, not making a tool call
                response.text = (
                    tool_call_start + "".join(tool_call_text_parts) + tool_call_end
                )
                yield response

            in_tool_call = False
            tool_call_text_parts = []
            continue

        if in_tool_call:
            tool_call_text_parts.append(response.text)
            if response.finish_reason is not None:
                logger.info(
                    "toll call parsing interrupted, yield partial tool call as text"
                )
                yield GenerationResponse(
                    text=tool_call_start + "".join(tool_call_text_parts),
                    token=0,
                    finish_reason=response.finish_reason,
                    usage=None,
                )
            continue
        # fallthrough
        yield response


def patch_kimi_tokenizer(tokenizer: TokenizerWrapper):
    """
    Version of to-be-upstreamed kimi-k2 tool parser
    """
    import ast
    import json

    import regex as re

    # kimi has a fixed function naming scheme, with a json formatted arg
    #   functions.multiply:0 <|tool_call_argument_begin|> {"a": 2, "b": 3}
    #   Also needs to handle tools like call_0<|tool_call_argument_begin|>{"filePath": "..."}
    _func_name_regex = re.compile(
        r"^\s*(.+)[:](\d+)\s*<\|tool_call_argument_begin\|>", re.DOTALL
    )
    _func_arg_regex = re.compile(r"<\|tool_call_argument_begin\|>\s*(.*)\s*", re.DOTALL)

    # kimi has a tool_calls_section - we're leaving this up to the caller to handle
    tool_call_start = "<|tool_call_begin|>"
    tool_call_end = "<|tool_call_end|>"

    def _deserialize(value: str) -> Any:  # pyright: ignore[reportAny]
        try:
            return json.loads(value)  # pyright: ignore[reportAny]
        except Exception:
            pass

        try:
            return ast.literal_eval(value)  # pyright: ignore[reportAny]
        except Exception:
            pass
        return value

    def parse_tool_call(text: str, tools: Any | None = None):
        func_name_match = _func_name_regex.search(text)
        if func_name_match is None:
            raise ValueError(f"Could not parse function name from tool call: {text!r}")
        original_func_name = func_name_match.group(1)
        tool_id = func_name_match.group(2)
        # strip off the `functions.` prefix, if it exists.
        func_name = original_func_name[original_func_name.find(".") + 1 :]

        func_args_match = _func_arg_regex.search(text)
        if func_args_match is None:
            raise ValueError(f"Could not parse function args from tool call: {text!r}")
        func_args = func_args_match.group(1)
        # the args should be valid json - no need to check against our tools to deserialize
        arg_dct = _deserialize(func_args)  # pyright: ignore[reportAny]

        return dict(
            id=f"{original_func_name}:{tool_id}",
            name=func_name,
            arguments=arg_dct,  # pyright: ignore[reportAny]
        )

    tokenizer._tool_call_start = tool_call_start
    tokenizer._tool_call_end = tool_call_end
    tokenizer._tool_parser = parse_tool_call


def patch_glm_tokenizer(tokenizer: TokenizerWrapper):
    """
    Fixed version of mlx_lm's glm47 tool parser that handles regex match failures.
    """
    import ast
    import json

    import regex as re

    _func_name_regex = re.compile(r"^(.*?)<arg_key>", re.DOTALL)
    _func_arg_regex = re.compile(
        r"<arg_key>(.*?)</arg_key>(?:\n|\s)*<arg_value>(.*?)(?:</arg_value>|(?=<arg_key>)|$)",
        re.DOTALL,
    )

    tool_call_start = "<tool_call>"
    tool_call_end = "</tool_call>"

    def _is_string_type(
        tool_name: str,
        arg_name: str,
        tools: list[Any] | None,
    ) -> bool:
        if tools is None:
            return False
        for tool in tools:  # pyright: ignore[reportAny]
            func = tool["function"]  # pyright: ignore[reportAny]
            if func["name"] == tool_name:
                params = func["parameters"]  # pyright: ignore[reportAny]
                if params is None:
                    return False
                props = params.get("properties", {})  # pyright: ignore[reportAny]
                arg_props = props.get(arg_name, {})  # pyright: ignore[reportAny]
                arg_type = arg_props.get("type", None)  # pyright: ignore[reportAny]
                return arg_type == "string"  # pyright: ignore[reportAny]
        return False

    def _deserialize(value: str) -> Any:  # pyright: ignore[reportAny]
        try:
            return json.loads(value)  # pyright: ignore[reportAny]
        except Exception:
            pass
        try:
            return ast.literal_eval(value)  # pyright: ignore[reportAny]
        except Exception:
            pass
        return value

    def parse_tool_call(text: str, tools: list[Any] | None = None):
        func_name_match = _func_name_regex.search(text)
        if func_name_match is None:
            raise ValueError(f"Could not parse function name from tool call: {text!r}")
        func_name = func_name_match.group(1)

        pairs = _func_arg_regex.findall(text)
        arg_dct: dict[str, Any] = {}
        for key, value in pairs:  # pyright: ignore[reportAny]
            arg_key = key.strip()  # pyright: ignore[reportAny]
            arg_val = value.strip()  # pyright: ignore[reportAny]
            if not _is_string_type(func_name, arg_key, tools):  # pyright: ignore[reportAny]
                arg_val = _deserialize(arg_val)  # pyright: ignore[reportAny]
            arg_dct[arg_key] = arg_val
        return dict(name=func_name, arguments=arg_dct)

    tokenizer._tool_call_start = tool_call_start
    tokenizer._tool_call_end = tool_call_end
    tokenizer._tool_parser = parse_tool_call


def _validate_single_tool(obj: dict[str, Any]) -> ToolCallItem:
    if (
        ((name := obj.get("name")) is not None)
        and ((args := obj.get("arguments")) is not None)
        and isinstance(name, str)
    ):
        raw_id: object = obj.get("id")
        extra = {"id": str(raw_id)} if raw_id is not None else {}
        return ToolCallItem(
            **extra,
            name=name,
            arguments=json.dumps(args),
        )
    else:
        raise ValidationError


EXO_RUNNER_MUST_FAIL = "EXO RUNNER MUST FAIL"
EXO_RUNNER_MUST_OOM = "EXO RUNNER MUST OOM"
EXO_RUNNER_MUST_TIMEOUT = "EXO RUNNER MUST TIMEOUT"


def _check_for_debug_prompts(task_params: TextGenerationTaskParams) -> None:
    """Check for debug prompt triggers in the input.

    Extracts the first user input text and checks for debug triggers.
    """
    if len(task_params.input) == 0:
        logger.debug("Empty message list in debug prompt check")
        return
    prompt = task_params.input[0].content

    if not prompt:
        return

    if EXO_RUNNER_MUST_FAIL in prompt:
        logger.info("raising exception")
        raise Exception("Artificial runner exception - for testing purposes only.")
    if EXO_RUNNER_MUST_OOM in prompt:
        from exo.worker.engines.mlx.utils_mlx import mlx_force_oom
        mlx_force_oom()
    if EXO_RUNNER_MUST_TIMEOUT in prompt:
        time.sleep(100)

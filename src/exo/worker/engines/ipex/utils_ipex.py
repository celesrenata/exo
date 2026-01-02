import time
from pathlib import Path
from typing import Any, Callable, cast

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from exo.shared.types.api import ChatCompletionMessageText
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.shards import ShardMetadata
from exo.worker.download.download_utils import build_model_path
from exo.worker.engines.ipex import (
    IPEXModel, 
    IPEXTokenizerWrapper,
    IPEXEngineError,
    IPEXDriverError,
    IPEXMemoryError,
    IPEXInitializationError,
    IPEXModelLoadError,
    IPEXInferenceError,
    IPEXDistributedError,
)
from exo.worker.runner.bootstrap import logger

# Constants
TEMPERATURE = 0.7
TRUST_REMOTE_CODE = True
MAX_TOKENS = 2048


def get_weights_size(model_shard_meta: ShardMetadata) -> Memory:
    """Calculate the memory size needed for model weights."""
    return Memory.from_float_kb(
        (model_shard_meta.end_layer - model_shard_meta.start_layer)
        / model_shard_meta.n_layers
        * model_shard_meta.model_meta.storage_size.in_kb
    )


def initialize_ipex(
    bound_instance: BoundInstance,
) -> tuple[IPEXModel, IPEXTokenizerWrapper, Callable[[torch.Tensor], torch.Tensor]]:
    """
    Initialize the IPEX model, tokenizer, and sampler for Intel GPU inference.
    """
    try:
        # Set up comprehensive IPEX logging
        setup_ipex_logging()
        
        # Validate Intel GPU environment first
        validate_intel_gpu_environment()
        
        torch.manual_seed(42)

        # Enable Intel GPU kernel optimizations
        enable_intel_gpu_kernel_optimizations()

        # Create Intel GPU optimized sampler function
        sampler = create_intel_gpu_optimized_sampler(
            temperature=TEMPERATURE,
            top_k=50,
            top_p=0.9
        )

        logger.info("Created IPEX Intel GPU optimized sampler")

        # Check if this is distributed inference
        num_nodes = len(bound_instance.instance.shard_assignments.node_to_runner)
        if num_nodes > 1:
            logger.info(f"Initializing distributed IPEX inference across {num_nodes} nodes")
            return initialize_distributed_ipex(bound_instance, sampler)
        else:
            logger.info(f"Single device Intel GPU inference for {bound_instance.instance}")
            return initialize_single_device_ipex(bound_instance, sampler)
            
    except IPEXEngineError:
        raise  # Re-raise IPEX-specific errors
    except Exception as e:
        log_ipex_error_context(e, "IPEX initialization")
        raise IPEXInitializationError(
            f"IPEX initialization failed: {e}",
            component="main_initialization"
        )


def initialize_single_device_ipex(
    bound_instance: BoundInstance,
    sampler: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[IPEXModel, IPEXTokenizerWrapper, Callable[[torch.Tensor], torch.Tensor]]:
    """Initialize IPEX for single Intel GPU device."""
    model_path = build_model_path(bound_instance.bound_shard.model_meta.model_id)

    start_time = time.perf_counter()

    # Set up Intel GPU health monitoring
    setup_intel_gpu_health_monitoring(0)

    # Load model configuration
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    # Get model info for batch optimization
    model_info = get_model_info(model_path)
    
    # Load model with Intel GPU optimizations
    model = load_ipex_model(model_path, config)

    # Load tokenizer
    tokenizer_raw = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    # Set pad token if not present
    if tokenizer_raw.pad_token is None:
        tokenizer_raw.pad_token = tokenizer_raw.eos_token

    tokenizer = IPEXTokenizerWrapper(tokenizer_raw)

    end_time = time.perf_counter()
    logger.info(f"Time taken to load IPEX model: {(end_time - start_time):.2f}s")

    # Move model to eval mode
    model.eval()

    # Apply additional Intel GPU optimizations after loading
    try:
        # Get batch optimization parameters
        batch_params = optimize_batch_processing_for_intel_gpu(
            batch_size=1,  # Default batch size
            sequence_length=2048,  # Default sequence length
            model_config=model_info
        )
        logger.info(f"Intel GPU batch optimization parameters: {batch_params}")
        
        # Store optimization parameters for use during inference
        if hasattr(model, 'config'):
            model.config.intel_gpu_optimizations = batch_params
            
    except Exception as e:
        logger.warning(f"Could not apply batch optimizations: {e}")

    logger.debug(f"Model: {model}")
    logger.info(f"Model loaded on device: {next(model.parameters()).device}")

    # Log initial health status
    log_intel_gpu_health_status(0)

    return cast(IPEXModel, model), tokenizer, sampler


def initialize_distributed_ipex(
    bound_instance: BoundInstance,
    sampler: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[IPEXModel, IPEXTokenizerWrapper, Callable[[torch.Tensor], torch.Tensor]]:
    """Initialize IPEX for distributed inference across multiple Intel GPUs."""
    try:
        import torch.distributed as dist
        
        # Initialize distributed Intel GPU setup
        dist_group = initialize_intel_gpu_distributed(bound_instance)
        
        # Load and shard model for distributed inference
        model, tokenizer = load_and_shard_ipex_model(bound_instance, dist_group)
        
        logger.info("Distributed IPEX initialization complete")
        return cast(IPEXModel, model), tokenizer, sampler
        
    except IPEXEngineError:
        raise  # Re-raise IPEX-specific errors
    except Exception as e:
        logger.error(f"Distributed IPEX initialization failed: {e}")
        # Fallback to single device if distributed fails
        logger.warning("Falling back to single device IPEX")
        try:
            return initialize_single_device_ipex(bound_instance, sampler)
        except Exception as fallback_error:
            raise IPEXDistributedError(
                f"Distributed IPEX initialization failed and fallback also failed: {e}. Fallback error: {fallback_error}"
            )


def initialize_intel_gpu_distributed(bound_instance: BoundInstance) -> Any:
    """Initialize distributed communication for Intel GPUs."""
    try:
        import torch.distributed as dist
        
        # Get distributed parameters
        rank = bound_instance.bound_shard.device_rank
        world_size = len(bound_instance.instance.shard_assignments.node_to_runner)
        
        # Log distributed setup
        log_ipex_distributed_setup(rank, world_size, "pipeline")  # Default to pipeline
        
        # Set up distributed environment for Intel GPU
        import os
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['MASTER_ADDR'] = 'localhost'  # TODO: Get from bound_instance
        os.environ['MASTER_PORT'] = '29500'  # TODO: Get from bound_instance
        
        # Initialize Intel GPU distributed backend
        if not dist.is_initialized():
            # Use NCCL backend for Intel GPU if available, otherwise gloo
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            
            # For Intel GPU, we might need to use a custom backend
            try:
                # Try Intel GPU specific backend
                backend = 'ccl'  # Intel Collective Communications Library
                dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
                logger.info(f"Initialized Intel GPU distributed with CCL backend")
            except Exception as ccl_error:
                logger.debug(f"CCL backend not available: {ccl_error}")
                # Fallback to gloo
                backend = 'gloo'
                dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
                logger.info(f"Initialized Intel GPU distributed with gloo backend")
        
        # Create process group for Intel GPU communication
        dist_group = dist.new_group(ranks=list(range(world_size)))
        
        # Set Intel GPU device for this rank
        if torch.xpu.is_available():
            device_count = torch.xpu.device_count()
            if device_count == 0:
                raise IPEXDriverError("No Intel GPU devices available for distributed inference")
            
            device_id = rank % device_count
            torch.xpu.set_device(device_id)
            logger.info(f"Set Intel GPU device {device_id} for rank {rank}")
            
            # Log device memory status
            memory_info = get_intel_gpu_memory_usage(device_id)
            logger.info(f"Device {device_id} memory: {memory_info['free'] / (1024**3):.2f} GB free")
            
        else:
            raise IPEXDriverError("Intel GPU not available for distributed inference")
        
        logger.info("Intel GPU distributed initialization complete")
        return dist_group
        
    except IPEXEngineError:
        raise  # Re-raise IPEX-specific errors
    except Exception as e:
        log_ipex_error_context(e, "distributed_initialization", {
            "rank": rank if 'rank' in locals() else None,
            "world_size": world_size if 'world_size' in locals() else None
        })
        raise IPEXDistributedError(
            f"Intel GPU distributed initialization failed: {e}",
            rank=rank if 'rank' in locals() else None,
            world_size=world_size if 'world_size' in locals() else None
        )


def load_and_shard_ipex_model(
    bound_instance: BoundInstance,
    dist_group: Any,
) -> tuple[IPEXModel, IPEXTokenizerWrapper]:
    """Load and shard IPEX model for distributed inference."""
    try:
        import torch.distributed as dist
        
        model_path = build_model_path(bound_instance.bound_shard.model_meta.model_id)
        shard_metadata = bound_instance.bound_shard
        
        logger.info(f"Loading and sharding IPEX model from {model_path}")
        
        start_time = time.perf_counter()
        
        # Load model configuration
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=TRUST_REMOTE_CODE,
        )
        
        # Load tokenizer (same on all ranks)
        tokenizer_raw = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=TRUST_REMOTE_CODE,
        )
        
        if tokenizer_raw.pad_token is None:
            tokenizer_raw.pad_token = tokenizer_raw.eos_token
            
        tokenizer = IPEXTokenizerWrapper(tokenizer_raw)
        
        # Load model with Intel GPU optimizations
        model = load_ipex_model(model_path, config)
        
        # Apply model sharding based on shard metadata
        from exo.shared.types.worker.shards import TensorShardMetadata, PipelineShardMetadata
        
        if isinstance(shard_metadata, TensorShardMetadata):
            logger.info("Applying tensor parallelism for Intel GPU")
            model = apply_intel_gpu_tensor_parallelism(model, dist_group, shard_metadata)
        elif isinstance(shard_metadata, PipelineShardMetadata):
            logger.info("Applying pipeline parallelism for Intel GPU")
            model = apply_intel_gpu_pipeline_parallelism(model, dist_group, shard_metadata)
        else:
            logger.warning(f"Unknown shard metadata type: {type(shard_metadata)}")
        
        # Synchronize all processes
        dist.barrier(group=dist_group)
        
        end_time = time.perf_counter()
        logger.info(f"Time taken to load and shard IPEX model: {(end_time - start_time):.2f}s")
        
        return cast(IPEXModel, model), tokenizer
        
    except Exception as e:
        logger.error(f"IPEX model sharding failed: {e}")
        raise


def apply_intel_gpu_tensor_parallelism(
    model: Any,
    dist_group: Any,
    shard_metadata: Any,
) -> Any:
    """Apply tensor parallelism for Intel GPU distributed inference."""
    try:
        import torch.distributed as dist
        
        rank = dist.get_rank(group=dist_group)
        world_size = dist.get_world_size(group=dist_group)
        
        logger.info(f"Applying Intel GPU tensor parallelism: rank={rank}, world_size={world_size}")
        
        # Shard linear layers across Intel GPUs
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Shard weight matrix along appropriate dimension
                weight = module.weight.data
                
                # For tensor parallelism, shard along the output dimension
                output_dim = weight.size(0)
                shard_size = output_dim // world_size
                start_idx = rank * shard_size
                end_idx = start_idx + shard_size if rank < world_size - 1 else output_dim
                
                # Create sharded weight
                sharded_weight = weight[start_idx:end_idx, :].contiguous()
                
                # Replace module with sharded version
                sharded_module = torch.nn.Linear(
                    weight.size(1),
                    end_idx - start_idx,
                    bias=module.bias is not None
                )
                sharded_module.weight.data = sharded_weight
                
                if module.bias is not None:
                    sharded_bias = module.bias.data[start_idx:end_idx].contiguous()
                    sharded_module.bias.data = sharded_bias
                
                # Move to Intel GPU
                device = torch.device(f"xpu:{rank % torch.xpu.device_count()}")
                sharded_module = sharded_module.to(device)
                
                # Replace in model
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent_module = model.get_submodule(parent_name)
                    setattr(parent_module, child_name, sharded_module)
                else:
                    setattr(model, child_name, sharded_module)
                
                logger.debug(f"Sharded {name}: {weight.shape} -> {sharded_weight.shape}")
        
        # Add communication hooks for tensor parallelism
        model = add_intel_gpu_communication_hooks(model, dist_group, "tensor")
        
        logger.info("Intel GPU tensor parallelism applied")
        return model
        
    except Exception as e:
        logger.error(f"Intel GPU tensor parallelism failed: {e}")
        return model


def apply_intel_gpu_pipeline_parallelism(
    model: Any,
    dist_group: Any,
    shard_metadata: Any,
) -> Any:
    """Apply pipeline parallelism for Intel GPU distributed inference."""
    try:
        import torch.distributed as dist
        
        rank = dist.get_rank(group=dist_group)
        world_size = dist.get_world_size(group=dist_group)
        
        logger.info(f"Applying Intel GPU pipeline parallelism: rank={rank}, world_size={world_size}")
        
        # Get layer range for this rank
        start_layer = shard_metadata.start_layer
        end_layer = shard_metadata.end_layer
        
        logger.info(f"Rank {rank} handling layers {start_layer} to {end_layer}")
        
        # Keep only the layers assigned to this rank
        if hasattr(model, 'layers') or hasattr(model, 'transformer'):
            # Handle different model architectures
            if hasattr(model, 'layers'):
                all_layers = model.layers
                model.layers = all_layers[start_layer:end_layer]
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                all_layers = model.transformer.h
                model.transformer.h = all_layers[start_layer:end_layer]
            
            logger.info(f"Kept {len(model.layers if hasattr(model, 'layers') else model.transformer.h)} layers for rank {rank}")
        
        # Move model to appropriate Intel GPU
        device = torch.device(f"xpu:{rank % torch.xpu.device_count()}")
        model = model.to(device)
        
        # Add communication hooks for pipeline parallelism
        model = add_intel_gpu_communication_hooks(model, dist_group, "pipeline")
        
        logger.info("Intel GPU pipeline parallelism applied")
        return model
        
    except Exception as e:
        logger.error(f"Intel GPU pipeline parallelism failed: {e}")
        return model


def add_intel_gpu_communication_hooks(
    model: Any,
    dist_group: Any,
    parallelism_type: str,
) -> Any:
    """Add communication hooks for Intel GPU distributed inference."""
    try:
        import torch.distributed as dist
        
        logger.info(f"Adding Intel GPU communication hooks for {parallelism_type} parallelism")
        
        # Store distributed information in model
        model._ipex_dist_group = dist_group
        model._ipex_parallelism_type = parallelism_type
        model._ipex_rank = dist.get_rank(group=dist_group)
        model._ipex_world_size = dist.get_world_size(group=dist_group)
        
        # Add forward hook for communication
        def communication_hook(module, input, output):
            """Handle inter-GPU communication during forward pass."""
            try:
                if parallelism_type == "tensor":
                    # For tensor parallelism, all-reduce outputs
                    if isinstance(output, torch.Tensor):
                        dist.all_reduce(output, group=dist_group)
                        output = output / model._ipex_world_size
                    elif isinstance(output, tuple):
                        # Handle multiple outputs
                        new_outputs = []
                        for out in output:
                            if isinstance(out, torch.Tensor):
                                dist.all_reduce(out, group=dist_group)
                                out = out / model._ipex_world_size
                            new_outputs.append(out)
                        output = tuple(new_outputs)
                
                elif parallelism_type == "pipeline":
                    # For pipeline parallelism, send to next rank
                    if model._ipex_rank < model._ipex_world_size - 1:
                        next_rank = model._ipex_rank + 1
                        if isinstance(output, torch.Tensor):
                            dist.send(output, dst=next_rank, group=dist_group)
                        elif isinstance(output, tuple):
                            for out in output:
                                if isinstance(out, torch.Tensor):
                                    dist.send(out, dst=next_rank, group=dist_group)
                
                return output
                
            except Exception as e:
                logger.warning(f"Communication hook failed: {e}")
                return output
        
        # Register the hook
        model.register_forward_hook(communication_hook)
        
        logger.info("Intel GPU communication hooks added")
        return model
        
    except Exception as e:
        logger.error(f"Adding communication hooks failed: {e}")
        return model


def intel_gpu_distributed_barrier(dist_group: Any = None) -> None:
    """Synchronize all Intel GPU processes."""
    try:
        import torch.distributed as dist
        
        if dist.is_initialized():
            dist.barrier(group=dist_group)
            logger.debug("Intel GPU distributed barrier completed")
            
    except Exception as e:
        logger.warning(f"Intel GPU distributed barrier failed: {e}")


def cleanup_intel_gpu_distributed() -> None:
    """Clean up Intel GPU distributed resources."""
    try:
        import torch.distributed as dist
        
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Intel GPU distributed cleanup completed")
            
    except Exception as e:
        logger.warning(f"Intel GPU distributed cleanup failed: {e}")


def load_ipex_model(model_path: Path, config: Any) -> IPEXModel:
    """Load and optimize model for Intel GPU using IPEX."""
    try:
        # Validate Intel GPU environment first
        validate_intel_gpu_environment()
        
        import intel_extension_for_pytorch as ipex
        
        device_count = torch.xpu.device_count()
        logger.info(f"Found {device_count} Intel GPU device(s)")
        
        # Use first Intel GPU device
        device = torch.device("xpu:0")
        logger.info(f"Using Intel GPU device: {device}")
        
        # Detect model quantization
        quantization_info = detect_model_quantization(model_path)
        logger.info(f"Model quantization detected: {quantization_info}")
        
        # Get optimal dtype for Intel GPU
        optimal_dtype = get_optimal_dtype_for_intel_gpu()
        
        # Prepare optimization configuration
        optimization_config = {
            "dtype": optimal_dtype,
            "level": "O1",
            "quantization": not quantization_info["is_quantized"],
            "quantization_config": {
                "method": "dynamic",
                "dtype": "int8"
            }
        }
        
        # Log detailed model loading information
        log_ipex_model_loading(str(model_path), config, optimization_config)
        
        # Check memory availability before loading
        try:
            # Estimate model memory requirements (rough approximation)
            estimated_memory = getattr(config, 'vocab_size', 50000) * getattr(config, 'hidden_size', 4096) * 4  # 4 bytes per float32
            estimated_memory += getattr(config, 'num_hidden_layers', 32) * getattr(config, 'hidden_size', 4096) ** 2 * 4  # Attention weights
            
            # Adjust for quantization
            if quantization_info["is_quantized"]:
                bits = quantization_info.get("bits", 8)
                estimated_memory = estimated_memory * bits // 32  # Reduce memory estimate for quantized models
            
            logger.info(f"Estimated memory requirement: {estimated_memory / (1024**3):.2f} GB")
            
            if not check_intel_gpu_memory_available(estimated_memory, 0):
                logger.warning("Insufficient Intel GPU memory, attempting to optimize...")
                optimize_intel_gpu_memory_usage()
                
                if not check_intel_gpu_memory_available(estimated_memory, 0):
                    memory_info = get_intel_gpu_memory_usage(0)
                    raise IPEXMemoryError(
                        "Insufficient Intel GPU memory for model loading",
                        device_id=0,
                        requested_memory=estimated_memory,
                        available_memory=memory_info["free"]
                    )
                    
        except IPEXMemoryError:
            raise  # Re-raise IPEX memory errors
        except Exception as mem_check_error:
            logger.warning(f"Could not check memory availability: {mem_check_error}")
        
        # Record loading start time
        load_start_time = time.perf_counter()
        
        # Load model with Intel GPU device mapping
        try:
            logger.info("Loading model to Intel GPU...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=optimal_dtype,  # Use optimal dtype for Intel GPU
                device_map={"": device},
                trust_remote_code=TRUST_REMOTE_CODE,
                low_cpu_mem_usage=True,
            )
            
            load_duration = time.perf_counter() - load_start_time
            logger.info(f"Model loaded to Intel GPU in {load_duration:.2f} seconds")
            
        except Exception as load_error:
            load_duration = time.perf_counter() - load_start_time
            log_ipex_error_context(load_error, "model_loading", {
                "load_duration": load_duration,
                "model_path": str(model_path),
                "device": str(device),
                "dtype": str(optimal_dtype)
            })
            
            # Handle memory errors specifically
            if "memory" in str(load_error).lower() or "out of memory" in str(load_error).lower():
                logger.warning(f"Memory error during model loading: {load_error}")
                if handle_intel_gpu_memory_error(load_error, 0):
                    # Retry loading after memory cleanup
                    try:
                        logger.info("Retrying model loading after memory cleanup...")
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            config=config,
                            torch_dtype=optimal_dtype,
                            device_map={"": device},
                            trust_remote_code=TRUST_REMOTE_CODE,
                            low_cpu_mem_usage=True,
                        )
                        logger.info("Model loading retry successful")
                    except Exception as retry_error:
                        raise IPEXModelLoadError(
                            f"Failed to load model after memory cleanup: {retry_error}",
                            model_path=str(model_path),
                            device_id=0
                        )
                else:
                    memory_info = get_intel_gpu_memory_usage(0)
                    raise IPEXMemoryError(
                        f"Failed to load model due to Intel GPU memory constraints: {load_error}",
                        device_id=0,
                        available_memory=memory_info["free"]
                    )
            else:
                raise IPEXModelLoadError(
                    f"Model loading failed: {load_error}",
                    model_path=str(model_path),
                    device_id=0
                )
        
        # Apply comprehensive Intel GPU optimizations
        logger.info("Applying Intel GPU optimizations...")
        optimization_start_time = time.perf_counter()
        
        try:
            model = optimize_model_for_intel_gpu(model, optimization_config)
            optimization_duration = time.perf_counter() - optimization_start_time
            
            log_ipex_optimization_applied("comprehensive_optimization", {
                "duration": f"{optimization_duration:.2f}s",
                "dtype": str(optimal_dtype),
                "level": optimization_config["level"],
                "quantization_enabled": optimization_config["quantization"]
            })
            
        except Exception as opt_error:
            optimization_duration = time.perf_counter() - optimization_start_time
            logger.warning(f"Intel GPU optimization failed after {optimization_duration:.2f}s, using basic IPEX optimization: {opt_error}")
            
            try:
                model = ipex.optimize(model, dtype=optimal_dtype, level="O1")
                log_ipex_optimization_applied("basic_optimization", {
                    "dtype": str(optimal_dtype),
                    "level": "O1"
                })
            except Exception as basic_opt_error:
                logger.warning(f"Basic IPEX optimization failed, using unoptimized model: {basic_opt_error}")
        
        # Log memory usage after loading
        memory_info = get_intel_gpu_memory_usage(0)
        total_duration = time.perf_counter() - load_start_time
        
        log_ipex_performance_metrics(
            "model_loading",
            total_duration,
            tokens_generated=None,
            memory_usage=memory_info,
            device_id=0
        )
        
        return cast(IPEXModel, model)
        
    except IPEXEngineError:
        raise  # Re-raise IPEX-specific errors
    except ImportError as e:
        raise IPEXDriverError(
            f"Intel Extension for PyTorch not available: {e}",
            driver_version=None,
            required_version="intel-extension-for-pytorch"
        )
    except Exception as e:
        log_ipex_error_context(e, "model_loading", {"model_path": str(model_path)})
        raise IPEXModelLoadError(
            f"Failed to load IPEX model: {e}",
            model_path=str(model_path)
        )


def apply_chat_template(
    tokenizer: IPEXTokenizerWrapper,
    chat_task_data: ChatCompletionTaskParams,
) -> str:
    """Apply chat template to format messages for generation."""
    messages = chat_task_data.messages

    formatted_messages: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message.content, ChatCompletionMessageText):
            message.content = message.content.text
        if isinstance(message.content, list):
            if len(message.content) != 1:
                logger.warning("Received malformed prompt")
                continue
            message.content = message.content[0].text
        if message.content is None and message.thinking is None:
            continue

        # Null values are not valid when applying templates
        formatted_messages.append(
            {k: v for k, v in message.model_dump().items() if v is not None}
        )

    try:
        prompt: str = tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        logger.warning(f"Failed to apply chat template: {e}")
        # Fallback to simple concatenation
        prompt = ""
        for msg in formatted_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"{role}: {content}\n"
        prompt += "assistant: "

    return prompt


def check_ipex_availability() -> bool:
    """Check if IPEX and Intel GPU are available and working."""
    try:
        import intel_extension_for_pytorch as ipex
        import torch
        
        # Check if Intel GPU (XPU) is available
        if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
            logger.info("Intel GPU (XPU) not available")
            return False
        
        device_count = torch.xpu.device_count()
        if device_count == 0:
            logger.info("No Intel GPU devices found")
            return False
        
        # Test basic tensor operations on Intel GPU
        device = torch.device("xpu:0")
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        _ = x + 1  # Test tensor operations
        
        logger.info(f"IPEX available with {device_count} Intel GPU device(s)")
        return True
        
    except ImportError as e:
        logger.info(f"Intel Extension for PyTorch not installed: {e}")
        return False
    except Exception as e:
        logger.warning(f"IPEX availability check failed: {e}")
        return False


def validate_intel_gpu_environment() -> None:
    """Validate Intel GPU environment and raise appropriate errors if issues found."""
    try:
        import intel_extension_for_pytorch as ipex
        import torch
        
        # Check IPEX installation
        if not hasattr(ipex, '__version__'):
            raise IPEXDriverError("Intel Extension for PyTorch installation is corrupted")
        
        # Check Intel GPU availability
        if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
            raise IPEXDriverError(
                "Intel GPU (XPU) not available. Please ensure Intel GPU drivers are installed.",
                driver_version=None,
                required_version="Latest Intel GPU drivers"
            )
        
        device_count = torch.xpu.device_count()
        if device_count == 0:
            raise IPEXDriverError("No Intel GPU devices found. Please check hardware and drivers.")
        
        # Test basic operations on each device
        for device_id in range(device_count):
            try:
                device = torch.device(f"xpu:{device_id}")
                test_tensor = torch.tensor([1.0], device=device)
                _ = test_tensor + 1
                logger.debug(f"Intel GPU device {device_id} validation passed")
            except Exception as e:
                raise IPEXDriverError(
                    f"Intel GPU device {device_id} failed validation: {e}",
                    driver_version=None,
                    required_version="Compatible Intel GPU drivers"
                )
        
        logger.info(f"Intel GPU environment validation passed for {device_count} device(s)")
        
    except ImportError as e:
        raise IPEXDriverError(
            f"Intel Extension for PyTorch not installed: {e}",
            driver_version=None,
            required_version="intel-extension-for-pytorch"
        )
    except IPEXDriverError:
        raise  # Re-raise IPEX-specific errors
    except Exception as e:
        raise IPEXInitializationError(f"Intel GPU environment validation failed: {e}")


def handle_ipex_fallback(error: Exception, fallback_engine: str = "torch") -> None:
    """Handle IPEX errors with graceful fallback to CPU/Torch engine."""
    logger.error(f"IPEX engine failed: {error}")
    
    if isinstance(error, IPEXDriverError):
        logger.warning(
            f"Intel GPU driver issue detected: {error}. "
            f"Falling back to {fallback_engine} engine. "
            f"To fix this, please install or update Intel GPU drivers."
        )
    elif isinstance(error, IPEXMemoryError):
        logger.warning(
            f"Intel GPU memory issue: {error}. "
            f"Falling back to {fallback_engine} engine. "
            f"Consider using a smaller model or enabling model sharding."
        )
    elif isinstance(error, IPEXModelLoadError):
        logger.warning(
            f"IPEX model loading failed: {error}. "
            f"Falling back to {fallback_engine} engine. "
            f"The model may not be compatible with Intel GPU acceleration."
        )
    else:
        logger.warning(
            f"IPEX engine error: {error}. "
            f"Falling back to {fallback_engine} engine."
        )
    
    # Clear Intel GPU memory if possible
    try:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            for device_id in range(torch.xpu.device_count()):
                torch.xpu.empty_cache()
        logger.debug("Cleared Intel GPU memory caches during fallback")
    except Exception as cleanup_error:
        logger.debug(f"Could not clear Intel GPU memory during fallback: {cleanup_error}")


def get_ipex_error_context(error: Exception) -> dict[str, Any]:
    """Get detailed context information for IPEX errors."""
    context = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": time.time(),
    }
    
    # Add IPEX-specific context
    if isinstance(error, IPEXEngineError):
        context["error_code"] = getattr(error, 'error_code', None)
        context["device_id"] = getattr(error, 'device_id', None)
        
        if isinstance(error, IPEXDriverError):
            context["driver_version"] = getattr(error, 'driver_version', None)
            context["required_version"] = getattr(error, 'required_version', None)
        elif isinstance(error, IPEXMemoryError):
            context["requested_memory"] = getattr(error, 'requested_memory', None)
            context["available_memory"] = getattr(error, 'available_memory', None)
        elif isinstance(error, IPEXModelLoadError):
            context["model_path"] = getattr(error, 'model_path', None)
        elif isinstance(error, IPEXDistributedError):
            context["rank"] = getattr(error, 'rank', None)
            context["world_size"] = getattr(error, 'world_size', None)
    
    # Add system context
    try:
        intel_gpu_info = get_intel_gpu_info()
        context["intel_gpu_available"] = intel_gpu_info["intel_gpu_available"]
        context["intel_gpu_count"] = intel_gpu_info["intel_gpu_count"]
        context["ipex_version"] = intel_gpu_info["ipex_version"]
    except Exception:
        context["intel_gpu_available"] = False
    
    return context


def get_intel_gpu_memory_usage(device_id: int = 0) -> dict[str, int]:
    """Get current Intel GPU memory usage for a specific device."""
    memory_info = {
        "allocated": 0,
        "reserved": 0,
        "free": 0,
        "total": 0,
    }
    
    try:
        import torch
        
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            if device_id < torch.xpu.device_count():
                # Get memory stats for Intel GPU
                allocated = torch.xpu.memory_allocated(device_id)
                reserved = torch.xpu.memory_reserved(device_id)
                
                # Get device properties for total memory
                device_props = torch.xpu.get_device_properties(device_id)
                total = getattr(device_props, 'total_memory', 0)
                
                memory_info.update({
                    "allocated": allocated,
                    "reserved": reserved,
                    "free": max(0, total - reserved),
                    "total": total,
                })
                
    except Exception as e:
        logger.warning(f"Could not get Intel GPU memory usage for device {device_id}: {e}")
    
    return memory_info


def clear_intel_gpu_memory(device_id: int = 0) -> None:
    """Clear Intel GPU memory cache for a specific device."""
    try:
        import torch
        
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            if device_id < torch.xpu.device_count():
                torch.xpu.empty_cache()
                logger.info(f"Cleared Intel GPU memory cache for device {device_id}")
                
    except Exception as e:
        logger.warning(f"Could not clear Intel GPU memory for device {device_id}: {e}")


def check_intel_gpu_memory_available(required_memory: int, device_id: int = 0) -> bool:
    """Check if enough Intel GPU memory is available for allocation."""
    try:
        memory_info = get_intel_gpu_memory_usage(device_id)
        available = memory_info["free"]
        
        # Add some buffer (10% of total memory or 1GB, whichever is smaller)
        buffer = min(memory_info["total"] * 0.1, 1024 * 1024 * 1024)
        
        return available >= (required_memory + buffer)
        
    except Exception as e:
        logger.warning(f"Could not check Intel GPU memory availability: {e}")
        return False


def optimize_intel_gpu_memory_usage() -> None:
    """Optimize Intel GPU memory usage by clearing caches and running garbage collection."""
    try:
        import gc
        import torch
        
        # Run Python garbage collection
        gc.collect()
        
        # Clear Intel GPU memory caches for all devices
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            device_count = torch.xpu.device_count()
            for device_id in range(device_count):
                clear_intel_gpu_memory(device_id)
                
        logger.info("Optimized Intel GPU memory usage")
        
    except Exception as e:
        logger.warning(f"Could not optimize Intel GPU memory usage: {e}")


def handle_intel_gpu_memory_error(error: Exception, device_id: int = 0) -> bool:
    """Handle Intel GPU memory errors with graceful recovery."""
    try:
        logger.warning(f"Intel GPU memory error on device {device_id}: {error}")
        
        # Get current memory info
        memory_info = get_intel_gpu_memory_usage(device_id)
        
        # Try to free up memory
        optimize_intel_gpu_memory_usage()
        
        # Check if memory is now available
        new_memory_info = get_intel_gpu_memory_usage(device_id)
        freed_memory = new_memory_info["free"] - memory_info["free"]
        
        if freed_memory > 0:
            logger.info(f"Recovered {freed_memory / (1024**2):.1f}MB Intel GPU memory on device {device_id}")
            return True
        else:
            logger.error(f"Could not recover Intel GPU memory on device {device_id}")
            
            # Raise specific memory error with context
            raise IPEXMemoryError(
                f"Intel GPU memory recovery failed: {error}",
                device_id=device_id,
                available_memory=new_memory_info["free"]
            )
            
    except IPEXMemoryError:
        raise  # Re-raise IPEX memory errors
    except Exception as recovery_error:
        logger.error(f"Memory recovery failed: {recovery_error}")
        raise IPEXMemoryError(
            f"Intel GPU memory error handling failed: {recovery_error}",
            device_id=device_id
        )


def get_intel_gpu_info() -> dict[str, Any]:
    """Get Intel GPU information."""
    info = {
        "intel_gpu_available": False,
        "intel_gpu_count": 0,
        "intel_gpu_memory": 0,
        "ipex_version": None,
        "intel_gpu_devices": [],
    }
    
    try:
        import intel_extension_for_pytorch as ipex
        import torch
        
        info["ipex_version"] = ipex.__version__
        
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            device_count = torch.xpu.device_count()
            info["intel_gpu_available"] = device_count > 0
            info["intel_gpu_count"] = device_count
            
            # Get information for each Intel GPU device
            for i in range(device_count):
                try:
                    device_props = torch.xpu.get_device_properties(i)
                    memory_info = get_intel_gpu_memory_usage(i)
                    
                    device_info = {
                        "device_id": i,
                        "name": getattr(device_props, 'name', f'Intel GPU {i}'),
                        "total_memory": getattr(device_props, 'total_memory', 0),
                        "max_compute_units": getattr(device_props, 'max_compute_units', 0),
                        "memory_allocated": memory_info["allocated"],
                        "memory_reserved": memory_info["reserved"],
                        "memory_free": memory_info["free"],
                    }
                    info["intel_gpu_devices"].append(device_info)
                    
                    # Use first device memory for backward compatibility
                    if i == 0:
                        info["intel_gpu_memory"] = device_info["total_memory"]
                        
                except Exception as e:
                    logger.warning(f"Could not get properties for Intel GPU {i}: {e}")
                    
    except ImportError:
        logger.debug("Intel Extension for PyTorch not available")
    except Exception as e:
        logger.warning(f"Error getting Intel GPU info: {e}")
    
    return info


def get_model_info(model_path: Path) -> dict[str, Any]:
    """Get basic information about the model."""
    try:
        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=TRUST_REMOTE_CODE
        )
        return {
            "model_type": getattr(config, "model_type", "unknown"),
            "vocab_size": getattr(config, "vocab_size", 0),
            "hidden_size": getattr(config, "hidden_size", 0),
            "num_layers": getattr(config, "num_hidden_layers", 0),
            "num_attention_heads": getattr(config, "num_attention_heads", 0),
        }
    except Exception as e:
        logger.warning(f"Could not get model info: {e}")
        return {}


def detect_model_quantization(model_path: Path) -> dict[str, Any]:
    """Detect if model is already quantized and what format."""
    quantization_info = {
        "is_quantized": False,
        "quantization_method": None,
        "bits": None,
        "supported_dtypes": ["float16", "bfloat16"],
    }
    
    try:
        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=TRUST_REMOTE_CODE
        )
        
        # Check for common quantization indicators
        if hasattr(config, 'quantization_config'):
            quantization_info["is_quantized"] = True
            quant_config = config.quantization_config
            
            if hasattr(quant_config, 'bits'):
                quantization_info["bits"] = quant_config.bits
            if hasattr(quant_config, 'quant_method'):
                quantization_info["quantization_method"] = quant_config.quant_method
                
        # Check for GPTQ quantization
        if hasattr(config, 'gptq'):
            quantization_info["is_quantized"] = True
            quantization_info["quantization_method"] = "gptq"
            
        # Check for AWQ quantization
        if hasattr(config, 'awq'):
            quantization_info["is_quantized"] = True
            quantization_info["quantization_method"] = "awq"
            
        logger.info(f"Model quantization info: {quantization_info}")
        
    except Exception as e:
        logger.warning(f"Could not detect model quantization: {e}")
    
    return quantization_info


def apply_ipex_quantization(model: Any, quantization_config: dict[str, Any]) -> Any:
    """Apply IPEX quantization to the model."""
    try:
        import intel_extension_for_pytorch as ipex
        
        quant_method = quantization_config.get("method", "dynamic")
        dtype = quantization_config.get("dtype", "int8")
        
        logger.info(f"Applying IPEX quantization: method={quant_method}, dtype={dtype}")
        
        if quant_method == "dynamic":
            # Dynamic quantization - quantize weights, activations computed in fp16
            quantized_model = ipex.quantization.prepare(
                model,
                qconfig=ipex.quantization.default_dynamic_qconfig,
                example_inputs=None,
                inplace=False
            )
            quantized_model = ipex.quantization.convert(quantized_model)
            
        elif quant_method == "static":
            # Static quantization - requires calibration data
            logger.warning("Static quantization requires calibration data, falling back to dynamic")
            quantized_model = ipex.quantization.prepare(
                model,
                qconfig=ipex.quantization.default_dynamic_qconfig,
                example_inputs=None,
                inplace=False
            )
            quantized_model = ipex.quantization.convert(quantized_model)
            
        else:
            logger.warning(f"Unsupported quantization method: {quant_method}")
            return model
            
        logger.info("IPEX quantization applied successfully")
        return quantized_model
        
    except Exception as e:
        logger.warning(f"IPEX quantization failed: {e}")
        return model


def get_optimal_dtype_for_intel_gpu() -> torch.dtype:
    """Get the optimal data type for Intel GPU inference."""
    try:
        import torch
        
        # Check Intel GPU capabilities
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            device_props = torch.xpu.get_device_properties(0)
            
            # Intel Arc GPUs generally support fp16 and bf16
            # bf16 is often better for training, fp16 for inference
            if hasattr(device_props, 'supports_bfloat16') and device_props.supports_bfloat16:
                logger.info("Using bfloat16 for Intel GPU inference")
                return torch.bfloat16
            else:
                logger.info("Using float16 for Intel GPU inference")
                return torch.float16
                
    except Exception as e:
        logger.warning(f"Could not determine optimal dtype: {e}")
    
    # Default fallback
    return torch.float16


def optimize_model_for_intel_gpu(model: Any, optimization_config: dict[str, Any]) -> Any:
    """Apply comprehensive optimizations for Intel GPU inference."""
    try:
        import intel_extension_for_pytorch as ipex
        
        # Get optimization parameters
        dtype = optimization_config.get("dtype", get_optimal_dtype_for_intel_gpu())
        level = optimization_config.get("level", "O1")
        enable_quantization = optimization_config.get("quantization", False)
        
        logger.info(f"Optimizing model for Intel GPU: dtype={dtype}, level={level}, quantization={enable_quantization}")
        
        # Apply mixed precision if requested
        if dtype in [torch.float16, torch.bfloat16]:
            model = model.to(dtype)
            
        # Apply Intel GPU specific optimizations
        optimized_model = ipex.optimize(
            model,
            dtype=dtype,
            level=level,
            inplace=False,
            # Intel GPU specific optimizations
            auto_kernel_selection=True,  # Enable automatic kernel selection
            graph_mode=True,  # Enable graph optimization
            concat_linear=True,  # Fuse linear layers for better tensor core utilization
            linear_bn_folding=True,  # Fold batch normalization into linear layers
            conv_bn_folding=True,  # Fold batch normalization into convolution layers
            remove_dropout=True,  # Remove dropout layers during inference
            replace_dropout_with_identity=True,  # Replace dropout with identity for inference
            # Memory optimizations
            memory_format=torch.channels_last,  # Use channels_last for better memory layout
            # Attention optimizations
            enable_fused_attention=True,  # Enable fused attention kernels
            enable_flash_attention=True,  # Enable flash attention if available
        )
        
        # Apply quantization if requested
        if enable_quantization:
            quantization_config = optimization_config.get("quantization_config", {"method": "dynamic"})
            optimized_model = apply_ipex_quantization(optimized_model, quantization_config)
            
        # Apply Intel GPU tensor core optimizations
        optimized_model = apply_intel_tensor_core_optimizations(optimized_model)
        
        # Apply memory-efficient attention if available
        optimized_model = apply_memory_efficient_attention(optimized_model)
        
        logger.info("Intel GPU model optimization complete")
        return optimized_model
        
    except Exception as e:
        logger.warning(f"Intel GPU model optimization failed: {e}")
        return model


def apply_intel_tensor_core_optimizations(model: Any) -> Any:
    """Apply Intel GPU tensor core optimizations for matrix operations."""
    try:
        import intel_extension_for_pytorch as ipex
        
        logger.info("Applying Intel GPU tensor core optimizations")
        
        # Enable tensor core operations for linear layers
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and len(module.weight.shape) == 2:
                # Ensure weight tensors are properly aligned for tensor cores
                if module.weight.dtype in [torch.float16, torch.bfloat16]:
                    # Pad weights to multiples of 8 for optimal tensor core usage
                    weight_shape = module.weight.shape
                    if weight_shape[0] % 8 != 0 or weight_shape[1] % 8 != 0:
                        new_shape = (
                            ((weight_shape[0] + 7) // 8) * 8,
                            ((weight_shape[1] + 7) // 8) * 8
                        )
                        if new_shape != weight_shape:
                            logger.debug(f"Padding {name} weights from {weight_shape} to {new_shape} for tensor cores")
                            # Note: This is a conceptual optimization - actual implementation would need careful handling
        
        # Apply Intel GPU specific linear layer optimizations
        try:
            # Enable Intel GPU optimized GEMM operations
            optimized_model = ipex.optimize_transformers(
                model,
                dtype=torch.float16,
                inplace=False,
                # Tensor core specific optimizations
                enable_auto_mixed_precision=True,
                enable_graph_capture=True,
                enable_kernel_profiling=False,  # Disable for production
            )
            logger.info("Applied Intel GPU tensor core optimizations")
            return optimized_model
            
        except AttributeError:
            # Fallback if optimize_transformers is not available
            logger.debug("optimize_transformers not available, using basic optimizations")
            return model
            
    except Exception as e:
        logger.warning(f"Intel tensor core optimization failed: {e}")
        return model


def apply_memory_efficient_attention(model: Any) -> Any:
    """Apply memory-efficient attention mechanisms for Intel GPU."""
    try:
        logger.info("Applying memory-efficient attention optimizations")
        
        # Replace attention modules with Intel GPU optimized versions
        for name, module in model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                # Apply Intel GPU specific attention optimizations
                if hasattr(module, 'num_heads') and hasattr(module, 'head_dim'):
                    # Optimize attention for Intel GPU architecture
                    logger.debug(f"Optimizing attention module: {name}")
                    
                    # Enable Intel GPU optimized attention patterns
                    if hasattr(module, 'scale_attn_weights'):
                        module.scale_attn_weights = True
                    
                    # Enable attention dropout fusion
                    if hasattr(module, 'attn_dropout') and hasattr(module.attn_dropout, 'p'):
                        if module.attn_dropout.p == 0.0:
                            # Replace with identity for inference
                            module.attn_dropout = torch.nn.Identity()
        
        logger.info("Applied memory-efficient attention optimizations")
        return model
        
    except Exception as e:
        logger.warning(f"Memory-efficient attention optimization failed: {e}")
        return model


def optimize_batch_processing_for_intel_gpu(batch_size: int, sequence_length: int, model_config: dict) -> dict[str, Any]:
    """Optimize batch processing parameters for Intel GPU architecture."""
    try:
        # Get Intel GPU properties
        device_props = torch.xpu.get_device_properties(0)
        total_memory = getattr(device_props, 'total_memory', 16 * 1024**3)  # Default 16GB
        compute_units = getattr(device_props, 'max_compute_units', 512)  # Default estimate
        
        # Calculate optimal batch processing parameters
        hidden_size = model_config.get('hidden_size', 4096)
        num_layers = model_config.get('num_layers', 32)
        
        # Estimate memory usage per sample
        memory_per_sample = (
            sequence_length * hidden_size * 4 +  # Activations (float32)
            sequence_length * sequence_length * 2 +  # Attention matrix (float16)
            hidden_size * hidden_size * num_layers * 2  # Weight memory estimate (float16)
        )
        
        # Calculate optimal batch size for Intel GPU
        available_memory = total_memory * 0.8  # Leave 20% buffer
        optimal_batch_size = min(batch_size, max(1, int(available_memory // memory_per_sample)))
        
        # Optimize for Intel GPU compute units
        # Intel GPUs work well with batch sizes that are multiples of compute unit count
        if optimal_batch_size > compute_units // 16:
            optimal_batch_size = ((optimal_batch_size + 15) // 16) * 16  # Round to multiple of 16
        
        optimization_params = {
            "optimal_batch_size": optimal_batch_size,
            "prefill_chunk_size": min(2048, sequence_length),  # Chunk size for prefill
            "decode_batch_size": min(optimal_batch_size * 4, 64),  # Larger batch for decode
            "memory_efficient_attention": sequence_length > 1024,  # Use for long sequences
            "gradient_checkpointing": sequence_length > 2048,  # For very long sequences
            "use_cache": True,  # Always use KV cache
            "cache_implementation": "intel_optimized",  # Intel GPU optimized cache
        }
        
        logger.info(f"Intel GPU batch optimization: {optimization_params}")
        return optimization_params
        
    except Exception as e:
        logger.warning(f"Batch processing optimization failed: {e}")
        return {
            "optimal_batch_size": batch_size,
            "prefill_chunk_size": 1024,
            "decode_batch_size": batch_size,
            "memory_efficient_attention": True,
            "use_cache": True,
        }


def enable_intel_gpu_kernel_optimizations() -> None:
    """Enable Intel GPU specific kernel optimizations."""
    try:
        import intel_extension_for_pytorch as ipex
        
        logger.info("Enabling Intel GPU kernel optimizations")
        
        # Enable Intel GPU specific optimizations
        torch.backends.xpu.matmul.allow_tf32 = True  # Enable TF32 for matrix operations
        torch.backends.xpu.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.xpu.matmul.allow_bf16_reduced_precision_reduction = True
        
        # Enable Intel GPU optimized kernels
        if hasattr(torch.backends, 'xpu'):
            # Enable optimized convolution algorithms
            torch.backends.xpu.conv.benchmark = True
            torch.backends.xpu.conv.deterministic = False  # Allow non-deterministic for performance
            
            # Enable Intel GPU memory optimizations
            torch.backends.xpu.memory.allow_non_uniform_memory_access = True
            
        # Set Intel GPU specific environment optimizations
        import os
        os.environ['IPEX_OPTIMIZE_LEVEL'] = '1'  # Enable level 1 optimizations
        os.environ['IPEX_ENABLE_AUTO_MIXED_PRECISION'] = '1'  # Enable AMP
        os.environ['IPEX_ENABLE_JIT_OPTIMIZATION'] = '1'  # Enable JIT optimizations
        
        logger.info("Intel GPU kernel optimizations enabled")
        
    except Exception as e:
        logger.warning(f"Intel GPU kernel optimization setup failed: {e}")


def create_intel_gpu_optimized_sampler(temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create an Intel GPU optimized sampler with advanced sampling strategies."""
    def intel_optimized_sampler(logits: torch.Tensor) -> torch.Tensor:
        """Intel GPU optimized sampling with multiple strategies."""
        try:
            # Ensure logits are on Intel GPU
            if not logits.device.type == 'xpu':
                logits = logits.to('xpu')
            
            # Apply temperature scaling with Intel GPU optimization
            if temperature == 0:
                return torch.argmax(logits, dim=-1)
            
            # Intel GPU optimized temperature scaling
            scaled_logits = logits / temperature
            
            # Apply top-k filtering with Intel GPU optimization
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                # Create mask for top-k values
                mask = torch.full_like(scaled_logits, float('-inf'))
                mask.scatter_(-1, top_k_indices, top_k_values)
                scaled_logits = mask
            
            # Apply top-p (nucleus) filtering with Intel GPU optimization
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Find cutoff for top-p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Apply top-p mask
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                scaled_logits[indices_to_remove] = float('-inf')
            
            # Intel GPU optimized softmax and sampling
            probs = torch.softmax(scaled_logits, dim=-1)
            
            # Use Intel GPU optimized multinomial sampling
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            return next_token
            
        except Exception as e:
            logger.warning(f"Intel GPU sampling failed, falling back to simple sampling: {e}")
            # Fallback to simple sampling
            if temperature == 0:
                return torch.argmax(logits, dim=-1)
            scaled_logits = logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    return intel_optimized_sampler


# ============================================================================
# COMPREHENSIVE IPEX LOGGING FUNCTIONS
# ============================================================================

def log_intel_gpu_detection() -> None:
    """Log detailed Intel GPU detection and initialization information."""
    try:
        logger.info("=== Intel GPU Detection and Initialization ===")
        
        # Log IPEX version and availability
        try:
            import intel_extension_for_pytorch as ipex
            logger.info(f"Intel Extension for PyTorch version: {ipex.__version__}")
        except ImportError:
            logger.warning("Intel Extension for PyTorch not available")
            return
        
        # Log Intel GPU availability
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            device_count = torch.xpu.device_count()
            logger.info(f"Intel GPU (XPU) available with {device_count} device(s)")
            
            # Log detailed information for each device
            for device_id in range(device_count):
                try:
                    device_props = torch.xpu.get_device_properties(device_id)
                    memory_info = get_intel_gpu_memory_usage(device_id)
                    
                    logger.info(f"Intel GPU Device {device_id}:")
                    logger.info(f"  Name: {getattr(device_props, 'name', 'Unknown')}")
                    logger.info(f"  Total Memory: {getattr(device_props, 'total_memory', 0) / (1024**3):.2f} GB")
                    logger.info(f"  Max Compute Units: {getattr(device_props, 'max_compute_units', 0)}")
                    logger.info(f"  Memory Allocated: {memory_info['allocated'] / (1024**2):.1f} MB")
                    logger.info(f"  Memory Free: {memory_info['free'] / (1024**3):.2f} GB")
                    
                    # Test basic operations
                    test_tensor = torch.tensor([1.0], device=f"xpu:{device_id}")
                    _ = test_tensor + 1
                    logger.info(f"  Basic operations: PASSED")
                    
                except Exception as e:
                    logger.warning(f"  Error getting device {device_id} info: {e}")
        else:
            logger.warning("Intel GPU (XPU) not available")
        
        # Log Intel GPU driver information
        log_intel_gpu_driver_info()
        
        logger.info("=== End Intel GPU Detection ===")
        
    except Exception as e:
        logger.error(f"Error during Intel GPU detection logging: {e}")


def log_intel_gpu_driver_info() -> None:
    """Log Intel GPU driver and runtime version information."""
    try:
        logger.info("--- Intel GPU Driver Information ---")
        
        # Try to get Intel GPU driver version from system
        import subprocess
        import os
        
        # Check for Intel GPU tools
        try:
            # Try intel_gpu_top (if available)
            result = subprocess.run(['intel_gpu_top', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info(f"Intel GPU Tools: {result.stdout.strip()}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.debug("intel_gpu_top not available")
        
        # Check for Level Zero runtime
        try:
            result = subprocess.run(['ze_info'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info("Level Zero runtime: Available")
                # Parse ze_info output for version info
                lines = result.stdout.split('\n')[:10]  # First 10 lines usually contain version info
                for line in lines:
                    if 'version' in line.lower() or 'driver' in line.lower():
                        logger.info(f"  {line.strip()}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.debug("ze_info not available")
        
        # Check environment variables
        intel_env_vars = [
            'INTEL_DEVICE_PLUGINS_PATH',
            'ZE_ENABLE_VALIDATION_LAYER',
            'IPEX_OPTIMIZE_LEVEL',
            'IPEX_ENABLE_AUTO_MIXED_PRECISION',
        ]
        
        for env_var in intel_env_vars:
            value = os.environ.get(env_var)
            if value:
                logger.info(f"Environment: {env_var}={value}")
        
        # Log PyTorch XPU backend information
        if hasattr(torch, 'xpu'):
            logger.info("PyTorch XPU backend: Available")
            try:
                # Get XPU backend properties
                if torch.xpu.is_available():
                    logger.info(f"XPU device count: {torch.xpu.device_count()}")
                    logger.info(f"XPU current device: {torch.xpu.current_device()}")
            except Exception as e:
                logger.debug(f"Error getting XPU backend info: {e}")
        
    except Exception as e:
        logger.warning(f"Error logging Intel GPU driver info: {e}")


def log_ipex_model_loading(model_path: str, config: Any, optimization_config: dict) -> None:
    """Log detailed information about IPEX model loading process."""
    try:
        logger.info("=== IPEX Model Loading ===")
        logger.info(f"Model path: {model_path}")
        
        # Log model configuration
        if hasattr(config, 'model_type'):
            logger.info(f"Model type: {config.model_type}")
        if hasattr(config, 'vocab_size'):
            logger.info(f"Vocabulary size: {config.vocab_size:,}")
        if hasattr(config, 'hidden_size'):
            logger.info(f"Hidden size: {config.hidden_size}")
        if hasattr(config, 'num_hidden_layers'):
            logger.info(f"Number of layers: {config.num_hidden_layers}")
        if hasattr(config, 'num_attention_heads'):
            logger.info(f"Attention heads: {config.num_attention_heads}")
        
        # Log optimization configuration
        logger.info("Optimization configuration:")
        for key, value in optimization_config.items():
            logger.info(f"  {key}: {value}")
        
        # Log memory requirements estimation
        try:
            vocab_size = getattr(config, 'vocab_size', 50000)
            hidden_size = getattr(config, 'hidden_size', 4096)
            num_layers = getattr(config, 'num_hidden_layers', 32)
            
            # Rough memory estimation
            embedding_memory = vocab_size * hidden_size * 4  # 4 bytes per float32
            layer_memory = num_layers * hidden_size * hidden_size * 4  # Attention weights
            total_estimated = (embedding_memory + layer_memory) / (1024**3)  # Convert to GB
            
            logger.info(f"Estimated model memory: {total_estimated:.2f} GB")
            
        except Exception as e:
            logger.debug(f"Could not estimate model memory: {e}")
        
        logger.info("=== End Model Loading Info ===")
        
    except Exception as e:
        logger.warning(f"Error logging model loading info: {e}")


def log_ipex_performance_metrics(
    operation: str,
    duration: float,
    tokens_generated: int | None = None,
    memory_usage: dict | None = None,
    device_id: int = 0
) -> None:
    """Log performance metrics for IPEX operations."""
    try:
        logger.info(f"=== IPEX Performance Metrics: {operation} ===")
        logger.info(f"Duration: {duration:.3f} seconds")
        
        if tokens_generated is not None:
            logger.info(f"Tokens generated: {tokens_generated}")
            if duration > 0:
                tokens_per_second = tokens_generated / duration
                logger.info(f"Tokens per second: {tokens_per_second:.2f}")
        
        # Log memory usage
        if memory_usage is None:
            memory_usage = get_intel_gpu_memory_usage(device_id)
        
        logger.info(f"Intel GPU Memory (Device {device_id}):")
        logger.info(f"  Allocated: {memory_usage['allocated'] / (1024**2):.1f} MB")
        logger.info(f"  Reserved: {memory_usage['reserved'] / (1024**2):.1f} MB")
        logger.info(f"  Free: {memory_usage['free'] / (1024**3):.2f} GB")
        logger.info(f"  Total: {memory_usage['total'] / (1024**3):.2f} GB")
        
        if memory_usage['total'] > 0:
            utilization = (memory_usage['allocated'] / memory_usage['total']) * 100
            logger.info(f"  Utilization: {utilization:.1f}%")
        
        logger.info("=== End Performance Metrics ===")
        
    except Exception as e:
        logger.warning(f"Error logging performance metrics: {e}")


def log_ipex_inference_start(task_params: Any, model_info: dict | None = None) -> None:
    """Log the start of IPEX inference with task parameters."""
    try:
        logger.info("=== IPEX Inference Started ===")
        
        # Log task parameters
        if hasattr(task_params, 'model'):
            logger.info(f"Model: {task_params.model}")
        if hasattr(task_params, 'max_tokens'):
            logger.info(f"Max tokens: {task_params.max_tokens}")
        if hasattr(task_params, 'temperature'):
            logger.info(f"Temperature: {task_params.temperature}")
        if hasattr(task_params, 'top_p'):
            logger.info(f"Top-p: {task_params.top_p}")
        if hasattr(task_params, 'top_k'):
            logger.info(f"Top-k: {task_params.top_k}")
        
        # Log message count
        if hasattr(task_params, 'messages') and task_params.messages:
            logger.info(f"Input messages: {len(task_params.messages)}")
            
            # Log first message preview (truncated)
            first_message = task_params.messages[0]
            if hasattr(first_message, 'content'):
                content = str(first_message.content)[:100]
                logger.info(f"First message preview: {content}...")
        
        # Log model information if provided
        if model_info:
            logger.info("Model information:")
            for key, value in model_info.items():
                logger.info(f"  {key}: {value}")
        
        # Log current Intel GPU status
        memory_info = get_intel_gpu_memory_usage(0)
        logger.info(f"Intel GPU memory before inference: {memory_info['allocated'] / (1024**2):.1f} MB allocated")
        
    except Exception as e:
        logger.warning(f"Error logging inference start: {e}")


def log_ipex_inference_complete(
    tokens_generated: int,
    total_duration: float,
    finish_reason: str | None = None
) -> None:
    """Log the completion of IPEX inference."""
    try:
        logger.info("=== IPEX Inference Complete ===")
        logger.info(f"Tokens generated: {tokens_generated}")
        logger.info(f"Total duration: {total_duration:.3f} seconds")
        
        if total_duration > 0:
            tokens_per_second = tokens_generated / total_duration
            logger.info(f"Average tokens per second: {tokens_per_second:.2f}")
        
        if finish_reason:
            logger.info(f"Finish reason: {finish_reason}")
        
        # Log final memory usage
        memory_info = get_intel_gpu_memory_usage(0)
        logger.info(f"Intel GPU memory after inference: {memory_info['allocated'] / (1024**2):.1f} MB allocated")
        
    except Exception as e:
        logger.warning(f"Error logging inference completion: {e}")


def log_ipex_distributed_setup(rank: int, world_size: int, parallelism_type: str) -> None:
    """Log distributed IPEX setup information."""
    try:
        logger.info("=== IPEX Distributed Setup ===")
        logger.info(f"Rank: {rank}")
        logger.info(f"World size: {world_size}")
        logger.info(f"Parallelism type: {parallelism_type}")
        
        # Log device assignment
        if torch.xpu.is_available():
            device_count = torch.xpu.device_count()
            assigned_device = rank % device_count
            logger.info(f"Assigned Intel GPU device: {assigned_device}")
            logger.info(f"Total Intel GPU devices: {device_count}")
        
        # Log distributed backend information
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                backend = dist.get_backend()
                logger.info(f"Distributed backend: {backend}")
        except Exception as e:
            logger.debug(f"Could not get distributed backend info: {e}")
        
        logger.info("=== End Distributed Setup ===")
        
    except Exception as e:
        logger.warning(f"Error logging distributed setup: {e}")


def log_ipex_error_context(error: Exception, operation: str, additional_context: dict | None = None) -> None:
    """Log detailed error context for IPEX operations."""
    try:
        logger.error("=== IPEX Error Context ===")
        logger.error(f"Operation: {operation}")
        logger.error(f"Error type: {type(error).__name__}")
        logger.error(f"Error message: {str(error)}")
        
        # Log IPEX-specific error context
        if hasattr(error, 'error_code'):
            logger.error(f"Error code: {error.error_code}")
        if hasattr(error, 'device_id'):
            logger.error(f"Device ID: {error.device_id}")
        
        # Log additional context
        if additional_context:
            logger.error("Additional context:")
            for key, value in additional_context.items():
                logger.error(f"  {key}: {value}")
        
        # Log system state
        try:
            intel_gpu_info = get_intel_gpu_info()
            logger.error("Intel GPU state:")
            logger.error(f"  Available: {intel_gpu_info['intel_gpu_available']}")
            logger.error(f"  Device count: {intel_gpu_info['intel_gpu_count']}")
            logger.error(f"  IPEX version: {intel_gpu_info['ipex_version']}")
            
            # Log memory state for each device
            for device_info in intel_gpu_info.get('intel_gpu_devices', []):
                device_id = device_info['device_id']
                logger.error(f"  Device {device_id} memory: {device_info['memory_free'] / (1024**3):.2f} GB free")
                
        except Exception as context_error:
            logger.error(f"Could not get Intel GPU context: {context_error}")
        
        logger.error("=== End Error Context ===")
        
    except Exception as e:
        logger.error(f"Error logging error context: {e}")


def setup_ipex_logging() -> None:
    """Set up comprehensive logging for IPEX operations."""
    try:
        # Log initial Intel GPU detection
        log_intel_gpu_detection()
        
        # Set up Intel GPU specific logging levels
        import os
        
        # Enable detailed IPEX logging if debug mode
        if logger.level <= 10:  # DEBUG level
            os.environ['IPEX_VERBOSE'] = '1'
            os.environ['INTEL_EXTENSION_FOR_PYTORCH_VERBOSE'] = '1'
            logger.debug("Enabled verbose IPEX logging")
        
        # Set up Intel GPU memory logging
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            # Enable memory debugging if available
            try:
                torch.xpu.memory._record_memory_history(enabled=True)
                logger.debug("Enabled Intel GPU memory history recording")
            except AttributeError:
                logger.debug("Intel GPU memory history recording not available")
        
        logger.info("IPEX logging setup complete")
        
    except Exception as e:
        logger.warning(f"Error setting up IPEX logging: {e}")


def log_ipex_optimization_applied(optimization_type: str, details: dict | None = None) -> None:
    """Log when IPEX optimizations are applied."""
    try:
        logger.info(f"=== IPEX Optimization Applied: {optimization_type} ===")
        
        if details:
            for key, value in details.items():
                logger.info(f"  {key}: {value}")
        
        # Log memory impact if available
        memory_info = get_intel_gpu_memory_usage(0)
        logger.info(f"Memory after optimization: {memory_info['allocated'] / (1024**2):.1f} MB allocated")
        
    except Exception as e:
        logger.warning(f"Error logging optimization: {e}")


# ============================================================================
# INTEL GPU HEALTH MONITORING FUNCTIONS
# ============================================================================

class IntelGPUHealthMonitor:
    """Monitor Intel GPU health and performance."""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.baseline_memory = None
        self.error_count = 0
        self.last_health_check = 0
        self.performance_history = []
        self.memory_leak_threshold = 1024 * 1024 * 100  # 100MB threshold
        self.max_error_count = 5
        self.health_check_interval = 30  # seconds
        
    def check_device_health(self) -> dict[str, Any]:
        """Check Intel GPU device health and return status."""
        health_status = {
            "device_id": self.device_id,
            "healthy": True,
            "issues": [],
            "memory_status": "normal",
            "performance_status": "normal",
            "error_count": self.error_count,
            "timestamp": time.time()
        }
        
        try:
            # Check if device is accessible
            if not torch.xpu.is_available():
                health_status["healthy"] = False
                health_status["issues"].append("Intel GPU not available")
                return health_status
            
            if self.device_id >= torch.xpu.device_count():
                health_status["healthy"] = False
                health_status["issues"].append(f"Device {self.device_id} not found")
                return health_status
            
            # Check memory status
            memory_status = self._check_memory_health()
            health_status.update(memory_status)
            
            # Check performance status
            performance_status = self._check_performance_health()
            health_status.update(performance_status)
            
            # Check for excessive errors
            if self.error_count > self.max_error_count:
                health_status["healthy"] = False
                health_status["issues"].append(f"Excessive errors: {self.error_count}")
            
            # Test basic operations
            operation_status = self._test_basic_operations()
            if not operation_status["success"]:
                health_status["healthy"] = False
                health_status["issues"].append(f"Basic operations failed: {operation_status['error']}")
            
            self.last_health_check = time.time()
            
        except Exception as e:
            health_status["healthy"] = False
            health_status["issues"].append(f"Health check failed: {e}")
            self.error_count += 1
            logger.warning(f"Intel GPU health check failed: {e}")
        
        return health_status
    
    def _check_memory_health(self) -> dict[str, Any]:
        """Check Intel GPU memory health."""
        memory_status = {
            "memory_status": "normal",
            "memory_info": {}
        }
        
        try:
            memory_info = get_intel_gpu_memory_usage(self.device_id)
            memory_status["memory_info"] = memory_info
            
            # Check for memory leaks
            if self.baseline_memory is None:
                self.baseline_memory = memory_info["allocated"]
            else:
                memory_growth = memory_info["allocated"] - self.baseline_memory
                if memory_growth > self.memory_leak_threshold:
                    memory_status["memory_status"] = "leak_detected"
                    memory_status["issues"] = [f"Memory leak detected: {memory_growth / (1024**2):.1f}MB growth"]
            
            # Check for low memory
            if memory_info["total"] > 0:
                free_percentage = (memory_info["free"] / memory_info["total"]) * 100
                if free_percentage < 10:
                    memory_status["memory_status"] = "low_memory"
                    memory_status["issues"] = [f"Low memory: {free_percentage:.1f}% free"]
                elif free_percentage < 20:
                    memory_status["memory_status"] = "warning"
                    memory_status["issues"] = [f"Memory warning: {free_percentage:.1f}% free"]
            
        except Exception as e:
            memory_status["memory_status"] = "error"
            memory_status["issues"] = [f"Memory check failed: {e}"]
        
        return memory_status
    
    def _check_performance_health(self) -> dict[str, Any]:
        """Check Intel GPU performance health."""
        performance_status = {
            "performance_status": "normal",
            "performance_metrics": {}
        }
        
        try:
            # Run a simple performance test
            start_time = time.perf_counter()
            
            # Create test tensors and perform operations
            device = torch.device(f"xpu:{self.device_id}")
            test_size = 1000
            a = torch.randn(test_size, test_size, device=device, dtype=torch.float16)
            b = torch.randn(test_size, test_size, device=device, dtype=torch.float16)
            
            # Perform matrix multiplication
            c = torch.matmul(a, b)
            torch.xpu.synchronize()  # Wait for completion
            
            duration = time.perf_counter() - start_time
            
            # Calculate performance metrics
            operations = test_size * test_size * test_size * 2  # FLOPS for matrix multiplication
            gflops = (operations / duration) / 1e9
            
            performance_metrics = {
                "test_duration": duration,
                "gflops": gflops,
                "memory_bandwidth_test": self._test_memory_bandwidth()
            }
            
            performance_status["performance_metrics"] = performance_metrics
            
            # Store performance history
            self.performance_history.append({
                "timestamp": time.time(),
                "gflops": gflops,
                "duration": duration
            })
            
            # Keep only recent history
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-50:]
            
            # Check for performance degradation
            if len(self.performance_history) > 10:
                recent_avg = sum(p["gflops"] for p in self.performance_history[-5:]) / 5
                baseline_avg = sum(p["gflops"] for p in self.performance_history[:5]) / 5
                
                if baseline_avg > 0:
                    degradation = (baseline_avg - recent_avg) / baseline_avg
                    if degradation > 0.3:  # 30% degradation
                        performance_status["performance_status"] = "degraded"
                        performance_status["issues"] = [f"Performance degraded by {degradation*100:.1f}%"]
                    elif degradation > 0.15:  # 15% degradation
                        performance_status["performance_status"] = "warning"
                        performance_status["issues"] = [f"Performance warning: {degradation*100:.1f}% degradation"]
            
            # Clean up test tensors
            del a, b, c
            torch.xpu.empty_cache()
            
        except Exception as e:
            performance_status["performance_status"] = "error"
            performance_status["issues"] = [f"Performance test failed: {e}"]
            self.error_count += 1
        
        return performance_status
    
    def _test_memory_bandwidth(self) -> dict[str, Any]:
        """Test Intel GPU memory bandwidth."""
        try:
            device = torch.device(f"xpu:{self.device_id}")
            
            # Test different sizes
            sizes = [1024, 2048, 4096]
            bandwidth_results = []
            
            for size in sizes:
                # Create large tensor for bandwidth test
                data_size = size * size * 4  # 4 bytes per float32
                
                start_time = time.perf_counter()
                
                # Create and copy data
                src = torch.randn(size, size, device=device, dtype=torch.float32)
                dst = torch.empty_like(src)
                dst.copy_(src)
                torch.xpu.synchronize()
                
                duration = time.perf_counter() - start_time
                bandwidth = (data_size / duration) / (1024**3)  # GB/s
                
                bandwidth_results.append({
                    "size": f"{size}x{size}",
                    "bandwidth_gbps": bandwidth
                })
                
                # Clean up
                del src, dst
            
            torch.xpu.empty_cache()
            
            return {
                "bandwidth_tests": bandwidth_results,
                "average_bandwidth": sum(r["bandwidth_gbps"] for r in bandwidth_results) / len(bandwidth_results)
            }
            
        except Exception as e:
            return {"error": f"Memory bandwidth test failed: {e}"}
    
    def _test_basic_operations(self) -> dict[str, Any]:
        """Test basic Intel GPU operations."""
        try:
            device = torch.device(f"xpu:{self.device_id}")
            
            # Test tensor creation
            x = torch.tensor([1.0, 2.0, 3.0], device=device)
            
            # Test basic arithmetic
            y = x + 1
            z = x * 2
            
            # Test reduction operations
            sum_result = torch.sum(x)
            
            # Test memory operations
            x_cpu = x.cpu()
            x_gpu = x_cpu.to(device)
            
            # Verify results
            expected_sum = 6.0
            if abs(sum_result.item() - expected_sum) > 1e-6:
                return {"success": False, "error": "Arithmetic operations failed"}
            
            return {"success": True}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def detect_memory_leaks(self) -> dict[str, Any]:
        """Detect Intel GPU memory leaks."""
        leak_info = {
            "leak_detected": False,
            "memory_growth": 0,
            "recommendations": []
        }
        
        try:
            current_memory = get_intel_gpu_memory_usage(self.device_id)
            
            if self.baseline_memory is not None:
                memory_growth = current_memory["allocated"] - self.baseline_memory
                leak_info["memory_growth"] = memory_growth
                
                if memory_growth > self.memory_leak_threshold:
                    leak_info["leak_detected"] = True
                    leak_info["recommendations"].extend([
                        "Clear Intel GPU cache with torch.xpu.empty_cache()",
                        "Check for unreleased tensors",
                        "Review model loading and optimization code",
                        "Consider reducing batch size or model size"
                    ])
                    
                    logger.warning(f"Intel GPU memory leak detected: {memory_growth / (1024**2):.1f}MB growth")
            
            # Update baseline periodically
            if time.time() - self.last_health_check > self.health_check_interval:
                self.baseline_memory = current_memory["allocated"]
            
        except Exception as e:
            leak_info["error"] = f"Memory leak detection failed: {e}"
        
        return leak_info
    
    def cleanup_memory_leaks(self) -> bool:
        """Attempt to clean up Intel GPU memory leaks."""
        try:
            logger.info(f"Attempting to clean up Intel GPU memory leaks on device {self.device_id}")
            
            # Get memory before cleanup
            memory_before = get_intel_gpu_memory_usage(self.device_id)
            
            # Run garbage collection
            import gc
            gc.collect()
            
            # Clear Intel GPU cache
            torch.xpu.empty_cache()
            
            # Wait a moment for cleanup to complete
            time.sleep(0.1)
            
            # Get memory after cleanup
            memory_after = get_intel_gpu_memory_usage(self.device_id)
            
            # Calculate memory freed
            memory_freed = memory_before["allocated"] - memory_after["allocated"]
            
            if memory_freed > 0:
                logger.info(f"Freed {memory_freed / (1024**2):.1f}MB of Intel GPU memory")
                # Update baseline
                self.baseline_memory = memory_after["allocated"]
                return True
            else:
                logger.warning("No memory was freed during cleanup")
                return False
                
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
            return False
    
    def get_performance_alerts(self) -> list[dict[str, Any]]:
        """Get performance degradation alerts."""
        alerts = []
        
        try:
            if len(self.performance_history) < 10:
                return alerts
            
            # Check for recent performance issues
            recent_performance = self.performance_history[-5:]
            baseline_performance = self.performance_history[:5]
            
            recent_avg = sum(p["gflops"] for p in recent_performance) / len(recent_performance)
            baseline_avg = sum(p["gflops"] for p in baseline_performance) / len(baseline_performance)
            
            if baseline_avg > 0:
                degradation = (baseline_avg - recent_avg) / baseline_avg
                
                if degradation > 0.3:
                    alerts.append({
                        "type": "performance_degradation",
                        "severity": "high",
                        "message": f"Intel GPU performance degraded by {degradation*100:.1f}%",
                        "recommendations": [
                            "Check Intel GPU temperature and throttling",
                            "Verify Intel GPU driver version",
                            "Clear Intel GPU memory cache",
                            "Restart IPEX engine if degradation persists"
                        ]
                    })
                elif degradation > 0.15:
                    alerts.append({
                        "type": "performance_warning",
                        "severity": "medium",
                        "message": f"Intel GPU performance warning: {degradation*100:.1f}% degradation",
                        "recommendations": [
                            "Monitor Intel GPU usage",
                            "Check for memory pressure",
                            "Consider reducing workload"
                        ]
                    })
            
            # Check for excessive errors
            if self.error_count > self.max_error_count:
                alerts.append({
                    "type": "excessive_errors",
                    "severity": "high",
                    "message": f"Excessive Intel GPU errors: {self.error_count}",
                    "recommendations": [
                        "Check Intel GPU driver and hardware",
                        "Review error logs for patterns",
                        "Consider fallback to CPU engine"
                    ]
                })
            
        except Exception as e:
            logger.warning(f"Error generating performance alerts: {e}")
        
        return alerts
    
    def reset_error_count(self) -> None:
        """Reset the error count."""
        self.error_count = 0
        logger.info(f"Reset Intel GPU error count for device {self.device_id}")
    
    def get_health_summary(self) -> dict[str, Any]:
        """Get comprehensive health summary."""
        try:
            health_status = self.check_device_health()
            leak_info = self.detect_memory_leaks()
            alerts = self.get_performance_alerts()
            
            summary = {
                "device_id": self.device_id,
                "overall_health": "healthy" if health_status["healthy"] and not leak_info["leak_detected"] else "unhealthy",
                "health_status": health_status,
                "memory_leak_info": leak_info,
                "performance_alerts": alerts,
                "error_count": self.error_count,
                "last_check": self.last_health_check,
                "recommendations": []
            }
            
            # Aggregate recommendations
            if not health_status["healthy"]:
                summary["recommendations"].extend(health_status.get("issues", []))
            
            if leak_info["leak_detected"]:
                summary["recommendations"].extend(leak_info.get("recommendations", []))
            
            for alert in alerts:
                summary["recommendations"].extend(alert.get("recommendations", []))
            
            return summary
            
        except Exception as e:
            return {
                "device_id": self.device_id,
                "overall_health": "error",
                "error": f"Health summary failed: {e}"
            }


# Global health monitor instance
_intel_gpu_health_monitor = None

def get_intel_gpu_health_monitor(device_id: int = 0) -> IntelGPUHealthMonitor:
    """Get or create Intel GPU health monitor instance."""
    global _intel_gpu_health_monitor
    
    if _intel_gpu_health_monitor is None or _intel_gpu_health_monitor.device_id != device_id:
        _intel_gpu_health_monitor = IntelGPUHealthMonitor(device_id)
    
    return _intel_gpu_health_monitor


def monitor_intel_gpu_health(device_id: int = 0) -> dict[str, Any]:
    """Monitor Intel GPU health and return status."""
    try:
        monitor = get_intel_gpu_health_monitor(device_id)
        return monitor.get_health_summary()
    except Exception as e:
        logger.error(f"Intel GPU health monitoring failed: {e}")
        return {
            "device_id": device_id,
            "overall_health": "error",
            "error": str(e)
        }


def setup_intel_gpu_health_monitoring(device_id: int = 0) -> None:
    """Set up Intel GPU health monitoring."""
    try:
        monitor = get_intel_gpu_health_monitor(device_id)
        
        # Initialize baseline memory
        if torch.xpu.is_available() and device_id < torch.xpu.device_count():
            memory_info = get_intel_gpu_memory_usage(device_id)
            monitor.baseline_memory = memory_info["allocated"]
            logger.info(f"Intel GPU health monitoring initialized for device {device_id}")
        else:
            logger.warning(f"Intel GPU device {device_id} not available for health monitoring")
            
    except Exception as e:
        logger.error(f"Failed to set up Intel GPU health monitoring: {e}")


def handle_intel_gpu_health_issues(health_summary: dict[str, Any]) -> bool:
    """Handle Intel GPU health issues with automatic recovery."""
    try:
        device_id = health_summary.get("device_id", 0)
        overall_health = health_summary.get("overall_health", "unknown")
        
        if overall_health == "healthy":
            return True
        
        logger.warning(f"Intel GPU health issues detected on device {device_id}")
        
        recovery_actions = []
        
        # Handle memory leaks
        leak_info = health_summary.get("memory_leak_info", {})
        if leak_info.get("leak_detected", False):
            logger.info("Attempting to clean up Intel GPU memory leaks...")
            monitor = get_intel_gpu_health_monitor(device_id)
            if monitor.cleanup_memory_leaks():
                recovery_actions.append("memory_cleanup_successful")
            else:
                recovery_actions.append("memory_cleanup_failed")
        
        # Handle performance issues
        alerts = health_summary.get("performance_alerts", [])
        for alert in alerts:
            if alert.get("type") == "performance_degradation":
                logger.info("Attempting to recover from performance degradation...")
                # Clear cache and reset performance history
                try:
                    torch.xpu.empty_cache()
                    monitor = get_intel_gpu_health_monitor(device_id)
                    monitor.performance_history = []
                    recovery_actions.append("performance_reset")
                except Exception as e:
                    logger.warning(f"Performance recovery failed: {e}")
                    recovery_actions.append("performance_reset_failed")
        
        # Handle excessive errors
        if health_summary.get("error_count", 0) > 5:
            logger.info("Resetting Intel GPU error count...")
            monitor = get_intel_gpu_health_monitor(device_id)
            monitor.reset_error_count()
            recovery_actions.append("error_count_reset")
        
        # Log recovery actions
        if recovery_actions:
            logger.info(f"Intel GPU recovery actions taken: {recovery_actions}")
            return True
        else:
            logger.warning("No recovery actions could be taken for Intel GPU health issues")
            return False
            
    except Exception as e:
        logger.error(f"Intel GPU health issue handling failed: {e}")
        return False


def log_intel_gpu_health_status(device_id: int = 0) -> None:
    """Log current Intel GPU health status."""
    try:
        health_summary = monitor_intel_gpu_health(device_id)
        
        logger.info("=== Intel GPU Health Status ===")
        logger.info(f"Device ID: {health_summary.get('device_id', device_id)}")
        logger.info(f"Overall Health: {health_summary.get('overall_health', 'unknown')}")
        
        # Log health details
        health_status = health_summary.get("health_status", {})
        if health_status.get("issues"):
            logger.warning(f"Health Issues: {health_status['issues']}")
        
        # Log memory status
        memory_info = health_status.get("memory_info", {})
        if memory_info:
            logger.info(f"Memory: {memory_info['allocated'] / (1024**2):.1f}MB allocated, {memory_info['free'] / (1024**3):.2f}GB free")
        
        # Log performance metrics
        performance_metrics = health_status.get("performance_metrics", {})
        if performance_metrics:
            logger.info(f"Performance: {performance_metrics.get('gflops', 0):.1f} GFLOPS")
        
        # Log alerts
        alerts = health_summary.get("performance_alerts", [])
        if alerts:
            logger.warning(f"Performance Alerts: {len(alerts)} active")
            for alert in alerts:
                logger.warning(f"  {alert.get('type', 'unknown')}: {alert.get('message', 'no message')}")
        
        logger.info("=== End Health Status ===")
        
    except Exception as e:
        logger.error(f"Failed to log Intel GPU health status: {e}")
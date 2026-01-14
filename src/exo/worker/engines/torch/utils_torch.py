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
from exo.worker.engines.torch import Model, TokenizerWrapper
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


def initialize_torch(
    bound_instance: BoundInstance,
) -> tuple[Model, TokenizerWrapper, Callable[[torch.Tensor], torch.Tensor]]:
    """
    Initialize the PyTorch model, tokenizer, and sampler for CPU inference.
    Supports both single-node and distributed inference.
    """
    torch.manual_seed(42)

    # Create sampler function
    def sampler(logits: torch.Tensor) -> torch.Tensor:
        """Simple temperature-based sampling."""
        if TEMPERATURE == 0:
            return torch.argmax(logits, dim=-1)

        # Apply temperature
        logits = logits / TEMPERATURE
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    logger.info("Created PyTorch CPU sampler")

    # Check if this is distributed inference
    shard_assignments = bound_instance.instance.shard_assignments
    is_distributed = len(shard_assignments.node_to_runner) > 1

    if is_distributed:
        logger.info(
            f"Initializing distributed PyTorch inference with {len(shard_assignments.node_to_runner)} nodes"
        )
        return _initialize_distributed_torch(bound_instance, sampler)
    else:
        logger.info(f"Single device CPU inference for {bound_instance.instance}")
        return _initialize_single_torch(bound_instance, sampler)


def _initialize_single_torch(
    bound_instance: BoundInstance, sampler: Callable[[torch.Tensor], torch.Tensor]
) -> tuple[Model, TokenizerWrapper, Callable[[torch.Tensor], torch.Tensor]]:
    """Initialize PyTorch for single-node inference."""
    model_path = build_model_path(bound_instance.bound_shard.model_meta.model_id)

    start_time = time.perf_counter()

    # Load model configuration
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    # Load model with CPU device mapping
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map="cpu",
        trust_remote_code=TRUST_REMOTE_CODE,
        low_cpu_mem_usage=True,
    )

    # Load tokenizer
    tokenizer_raw = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    # Set pad token if not present
    if tokenizer_raw.pad_token is None:
        tokenizer_raw.pad_token = tokenizer_raw.eos_token

    tokenizer = TokenizerWrapper(tokenizer_raw)

    end_time = time.perf_counter()
    logger.info(f"Time taken to load PyTorch model: {(end_time - start_time):.2f}s")

    # Move model to eval mode
    model.eval()

    logger.debug(f"Model: {model}")
    logger.info(f"Model loaded on device: {next(model.parameters()).device}")

    return cast(Model, model), tokenizer, sampler


def _initialize_distributed_torch(
    bound_instance: BoundInstance, sampler: Callable[[torch.Tensor], torch.Tensor]
) -> tuple[Model, TokenizerWrapper, Callable[[torch.Tensor], torch.Tensor]]:
    """Initialize PyTorch for distributed inference using pipeline parallelism."""
    from exo.shared.types.worker.shards import PipelineShardMetadata

    shard_metadata = bound_instance.bound_shard
    if not isinstance(shard_metadata, PipelineShardMetadata):
        raise NotImplementedError(
            "Only pipeline parallelism is supported for PyTorch distributed inference"
        )

    model_path = build_model_path(shard_metadata.model_meta.model_id)
    device_rank = shard_metadata.device_rank
    world_size = shard_metadata.world_size
    start_layer = shard_metadata.start_layer
    end_layer = shard_metadata.end_layer

    logger.info(
        f"Loading PyTorch model shard: rank {device_rank}/{world_size}, layers {start_layer}-{end_layer}"
    )

    start_time = time.perf_counter()

    # Load model configuration
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    # For distributed inference, we need to load only the layers for this shard
    # This is a simplified implementation - in practice, you'd want more sophisticated layer extraction
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=TRUST_REMOTE_CODE,
        low_cpu_mem_usage=True,
    )

    # Extract only the layers needed for this shard
    # This is a basic implementation - real distributed inference would need more sophisticated handling
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        # GPT-style models
        total_layers = len(model.transformer.h)
        if end_layer > total_layers:
            end_layer = total_layers
        model.transformer.h = model.transformer.h[start_layer:end_layer]
        logger.info(
            f"Extracted layers {start_layer}:{end_layer} from {total_layers} total layers"
        )
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        # Llama-style models
        total_layers = len(model.model.layers)
        if end_layer > total_layers:
            end_layer = total_layers
        model.model.layers = model.model.layers[start_layer:end_layer]
        logger.info(
            f"Extracted layers {start_layer}:{end_layer} from {total_layers} total layers"
        )
    else:
        logger.warning(
            "Could not identify model layer structure for sharding - using full model"
        )

    # Load tokenizer (needed on all ranks for warmup and generation)
    tokenizer_raw = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    if tokenizer_raw.pad_token is None:
        tokenizer_raw.pad_token = tokenizer_raw.eos_token

    tokenizer = TokenizerWrapper(tokenizer_raw)

    end_time = time.perf_counter()
    logger.info(
        f"Time taken to load PyTorch model shard: {(end_time - start_time):.2f}s"
    )

    # Move model to eval mode
    model.eval()

    logger.info(
        f"Distributed PyTorch model shard loaded: rank {device_rank}, device: {next(model.parameters()).device}"
    )

    return cast(Model, model), tokenizer, sampler


def apply_chat_template(
    tokenizer: TokenizerWrapper,
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


def check_torch_availability() -> bool:
    """Check if PyTorch is available and working."""
    try:
        import torch

        # Test basic tensor operations
        x = torch.tensor([1.0, 2.0, 3.0])
        _ = x + 1  # Test tensor operations
        return True
    except Exception as e:
        logger.error(f"PyTorch not available: {e}")
        return False


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

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, cast

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from exo.shared.types.api import ChatCompletionMessageText
from exo.shared.types.common import Host
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
    """
    torch.manual_seed(42)

    # Set device to CPU
    device = torch.device("cpu")

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

    # For now, only support single device (no distributed inference)
    if len(bound_instance.instance.shard_assignments.node_to_runner) > 1:
        raise NotImplementedError(
            "Distributed inference not yet supported for PyTorch engine"
        )

    logger.info(f"Single device CPU inference for {bound_instance.instance}")
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
        y = x + 1
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

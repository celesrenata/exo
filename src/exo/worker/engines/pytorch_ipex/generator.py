"""
Text Generation for PyTorch + IPEX Backend

This module provides text generation functionality for the PyTorch+IPEX backend,
including streaming token generation with temperature, top-k, and top-p sampling.

Requirements addressed:
- 3.1: Async text generation
- 3.2: Streaming token generation
- 3.3: Sampling with temperature, top-k, top-p
"""

import logging
import time
from collections.abc import Generator
from typing import Any, Optional

import numpy as np
import torch

from exo.shared.types.api import GenerationStats, Memory, Usage
from exo.shared.types.worker.runner_response import GenerationResponse

logger = logging.getLogger(__name__)


def pytorch_ipex_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device_type: str,
    device_id: int,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    model_id: str = "",
) -> Generator[GenerationResponse, None, None]:
    """
    Generate text using PyTorch+IPEX backend.

    This function:
    1. Tokenizes the input prompt
    2. Generates tokens one at a time
    3. Applies temperature, top-k, and top-p sampling
    4. Yields GenerationResponse for each token
    5. Handles EOS token and max_tokens limit

    Args:
        model: Loaded PyTorch model
        tokenizer: HuggingFace tokenizer
        prompt: Input text prompt
        device_type: Device type ("xpu", "cuda", or "cpu")
        device_id: Device ID
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter (None = disabled)
        top_p: Top-p (nucleus) sampling parameter (None = disabled)
        model_id: Model identifier for logging

    Yields:
        GenerationResponse objects containing generated tokens

    Requirements: 3.1, 3.2, 3.3
    """
    try:
        logger.info(
            f"Starting PyTorch+IPEX generation: max_tokens={max_tokens}, "
            f"temperature={temperature}, top_k={top_k}, top_p={top_p}"
        )

        # Create device
        device = torch.device(f"{device_type}:{device_id}")

        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        logger.debug(f"Tokenized prompt: {input_ids.shape[1]} tokens")

        # Get EOS token ID
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            logger.warning("No EOS token ID found, using 0")
            eos_token_id = 0

        # Initialize generation state
        past_key_values = None
        generated_tokens = 0
        prompt_tokens = input_ids.shape[1]
        start_time = time.time()

        # Set model to eval mode
        model.eval()

        # Generation loop
        with torch.no_grad():
            for step in range(max_tokens):
                # Forward pass
                if past_key_values is None:
                    # First forward pass - use full prompt
                    outputs = model(
                        input_ids=input_ids,
                        past_key_values=None,
                        use_cache=True,
                    )
                else:
                    # Subsequent passes - use only last token
                    outputs = model(
                        input_ids=input_ids[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                # Extract logits and KV cache
                logits = outputs.logits
                past_key_values = outputs.past_key_values

                # Get logits for last position
                next_token_logits = logits[:, -1, :]

                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Apply top-k filtering
                if top_k is not None and top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(
                        next_token_logits, top_k
                    )[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float("-inf")

                # Apply top-p (nucleus) filtering
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
                    )

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    # Scatter to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float("-inf")

                # Sample next token
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to input_ids
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                generated_tokens += 1

                # Decode token
                token_id = next_token.item()
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)

                # Calculate stats
                elapsed_time = time.time() - start_time
                generation_tps = generated_tokens / elapsed_time if elapsed_time > 0 else 0
                prompt_tps = prompt_tokens / elapsed_time if elapsed_time > 0 and step == 0 else 0

                # Check for EOS token
                finish_reason = None
                if token_id == eos_token_id:
                    finish_reason = "stop"
                    logger.debug(f"EOS token reached at step {step}")
                elif step == max_tokens - 1:
                    finish_reason = "length"
                    logger.debug(f"Max tokens reached at step {step}")

                # Create proper Usage and GenerationStats objects
                usage = Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=generated_tokens,
                    total_tokens=prompt_tokens + generated_tokens,
                )

                stats = GenerationStats(
                    prompt_tps=prompt_tps if step == 0 else generation_tps,
                    generation_tps=generation_tps,
                    prompt_tokens=prompt_tokens,
                    generation_tokens=generated_tokens,
                    peak_memory_usage=Memory(bytes=0),  # TODO: Track actual memory usage
                )

                # Yield response
                yield GenerationResponse(
                    text=token_text,
                    token=token_id,
                    finish_reason=finish_reason,
                    usage=usage,
                    stats=stats,
                    logprob=None,
                    top_logprobs=None,
                )

                # Stop if EOS reached
                if finish_reason is not None:
                    break

        logger.info(
            f"Generation complete: {generated_tokens} tokens in {elapsed_time:.2f}s "
            f"({generation_tps:.2f} tokens/s)"
        )

    except Exception as e:
        logger.error(f"PyTorch+IPEX generation failed: {e}")
        # Yield error response
        yield GenerationResponse(
            text=f"Error: {str(e)}",
            token=0,
            finish_reason="error",
            usage=None,
            stats=None,
            logprob=None,
            top_logprobs=None,
        )
        raise

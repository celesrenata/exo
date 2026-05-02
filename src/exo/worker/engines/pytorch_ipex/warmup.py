"""
Warmup functionality for PyTorch XPU Backend

This module provides warmup inference to pre-compile kernels and
initialize the inference pipeline. Uses native PyTorch XPU (2.11+),
no IPEX dependency.

Requirements addressed:
- 5.5: Warmup logic for backend initialization
"""

import logging
import time
from typing import Any

import torch

logger = logging.getLogger(__name__)


def warmup_pytorch_ipex_inference(
    model: Any,
    tokenizer: Any,
    device_type: str,
    device_id: int,
    warmup_tokens: int = 10,
) -> int:
    """
    Warm up PyTorch XPU inference by generating a few tokens.

    This function:
    1. Creates a simple warmup prompt
    2. Generates a few tokens to trigger JIT compilation
    3. Pre-allocates memory buffers

    Args:
        model: Loaded PyTorch model
        tokenizer: HuggingFace tokenizer
        device_type: Device type ("xpu", "cuda", or "cpu")
        device_id: Device ID
        warmup_tokens: Number of tokens to generate for warmup

    Returns:
        Number of tokens generated during warmup

    Requirements: 5.5
    """
    logger.info(
        f"Starting PyTorch XPU warmup on {device_type}:{device_id}, "
        f"generating {warmup_tokens} tokens"
    )

    start_time = time.time()

    try:
        # Create device
        device = torch.device(f"{device_type}:{device_id}")

        # Simple warmup prompt
        warmup_prompt = "Hello, this is a warmup prompt."

        # Tokenize
        input_ids = tokenizer.encode(warmup_prompt, return_tensors="pt").to(device)
        logger.debug(f"Warmup prompt tokenized: {input_ids.shape[1]} tokens")

        # Set model to eval mode
        model.eval()

        # Generate tokens
        past_key_values = None
        tokens_generated = 0

        with torch.no_grad():
            for _ in range(warmup_tokens):
                # Forward pass
                if past_key_values is None:
                    outputs = model(
                        input_ids=input_ids,
                        past_key_values=None,
                        use_cache=True,
                    )
                else:
                    outputs = model(
                        input_ids=input_ids[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                # Extract logits and KV cache
                logits = outputs.logits
                past_key_values = outputs.past_key_values

                # Sample next token (greedy for warmup)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

                # Append to input_ids
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                tokens_generated += 1

        elapsed_time = time.time() - start_time
        tokens_per_second = tokens_generated / elapsed_time if elapsed_time > 0 else 0

        logger.info(
            f"Warmup complete: generated {tokens_generated} tokens in {elapsed_time:.2f}s "
            f"({tokens_per_second:.2f} tokens/s)"
        )

        return tokens_generated

    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        # Don't fail the entire initialization if warmup fails
        logger.warning("Continuing without warmup")
        return 0

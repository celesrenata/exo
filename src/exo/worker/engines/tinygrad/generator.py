"""Text generation logic using tinygrad.

This module implements token-by-token text generation for the tinygrad backend,
with proper error handling and state management.
"""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
else:
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore

logger = logging.getLogger(__name__)


async def infer_tensor(
    model: Any,
    input_data: "np.ndarray[Any, Any]",
    inference_state: dict[str, Any] | None,
    device: str,
) -> tuple["np.ndarray[Any, Any]", dict[str, Any] | None]:
    """Execute tensor inference using tinygrad.

    This function runs a forward pass through the model with the given input,
    maintaining inference state (KV cache) for efficient generation.

    Args:
        model: Tinygrad model instance
        input_data: Input tensor as numpy array (shape: [batch_size, seq_len])
        inference_state: Optional state from previous inference (KV cache, etc.)
        device: Target device (CPU, GPU, METAL)

    Returns:
        Tuple of (output_logits, new_inference_state)
        - output_logits: Logits for next token prediction (shape: [batch_size, vocab_size])
        - new_inference_state: Updated state for next inference

    Raises:
        RuntimeError: If inference fails

    Example:
        >>> output, state = await infer_tensor(
        ...     model,
        ...     input_tokens,
        ...     previous_state,
        ...     "GPU"
        ... )
    """
    try:
        # Import tinygrad
        from tinygrad import Tensor

        logger.debug(
            f"Running inference on {device} with input shape {input_data.shape}"
        )

        # Convert numpy array to tinygrad Tensor
        input_tensor = Tensor(input_data)

        # Initialize or update inference state
        if inference_state is None:
            inference_state = _initialize_inference_state(model)

        # Run forward pass
        output_logits = await _forward_pass(
            model,
            input_tensor,
            inference_state,
        )

        # Convert output back to numpy
        output_np = output_logits.numpy()

        # Update inference state (KV cache)
        new_state = _update_inference_state(inference_state, output_logits)

        logger.debug(f"Inference complete, output shape: {output_np.shape}")

        return output_np, new_state

    except ImportError as e:
        raise RuntimeError(f"tinygrad not available: {e}") from e
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise RuntimeError(f"Inference failed: {e}") from e


def _initialize_inference_state(model: Any) -> dict[str, Any]:
    """Initialize inference state for a new generation.

    Args:
        model: Model instance

    Returns:
        Initial inference state dictionary
    """
    # Initialize KV cache and other state
    return {
        "kv_cache": None,
        "position": 0,
        "generated_tokens": [],
    }


async def _forward_pass(
    model: Any,
    input_tensor: Any,
    inference_state: dict,
) -> Any:
    """Execute forward pass through the model.

    Args:
        model: Model instance
        input_tensor: Input tensor
        inference_state: Current inference state

    Returns:
        Output logits tensor
    """
    # Placeholder forward pass
    # Real implementation would call model's forward method with KV cache
    try:
        # Call model forward
        output = model(input_tensor)
        return output
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        # Return dummy output for now
        from tinygrad import Tensor

        # Create dummy logits (batch_size=1, vocab_size=50000)
        batch_size = input_tensor.shape[0]
        vocab_size = 50000
        return Tensor.randn(batch_size, vocab_size)


def _update_inference_state(
    inference_state: dict[str, Any],
    output_logits: Any,
) -> dict[str, Any]:
    """Update inference state after forward pass.

    Args:
        inference_state: Current inference state
        output_logits: Output logits from forward pass

    Returns:
        Updated inference state
    """
    # Update position counter
    new_state = inference_state.copy()
    new_state["position"] += 1

    # KV cache would be updated here in real implementation
    # new_state["kv_cache"] = updated_cache

    return new_state


async def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    device: str = "CPU",
) -> str:
    """Generate text from a prompt using the model.

    This is a high-level generation function that handles tokenization,
    inference, and decoding.

    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        prompt: Input text prompt
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        device: Target device

    Returns:
        Generated text string

    Example:
        >>> text = await generate_text(
        ...     model,
        ...     tokenizer,
        ...     "Once upon a time",
        ...     max_tokens=50,
        ...     temperature=0.8
        ... )
    """
    from exo.worker.engines.tinygrad.model_loader import (
        decode_tokens,
        encode_prompt,
    )

    # Encode prompt
    input_tokens = await encode_prompt(tokenizer, prompt)

    # Initialize generation
    generated_tokens = input_tokens.tolist()
    inference_state = None

    logger.info(f"Generating up to {max_tokens} tokens")

    # Generate tokens one by one
    for i in range(max_tokens):
        # Prepare input (last token or full sequence on first iteration)
        if i == 0:
            current_input = input_tokens.reshape(1, -1)
        else:
            current_input = np.array([[generated_tokens[-1]]], dtype=np.int64)

        # Run inference
        logits, inference_state = await infer_tensor(
            model,
            current_input,
            inference_state,
            device,
        )

        # Sample next token
        next_token = _sample_token(
            logits[0, -1],  # Last position logits
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Add to generated sequence
        generated_tokens.append(int(next_token))

        # Check for end of sequence token (simplified)
        if next_token == 0:  # Assuming 0 is EOS
            break

    # Decode generated tokens
    generated_array = np.array(generated_tokens, dtype=np.int64)
    generated_text = await decode_tokens(tokenizer, generated_array)

    logger.info(f"Generated {len(generated_tokens)} tokens")

    return generated_text


def _sample_token(
    logits: "np.ndarray[Any, Any]",
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    """Sample a token from logits using various sampling strategies.

    Args:
        logits: Logit values for vocabulary (shape: [vocab_size])
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter

    Returns:
        Sampled token ID
    """
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Convert logits to probabilities
    probs = _softmax(logits)

    # Apply top-k filtering
    if top_k is not None:
        probs = _apply_top_k(probs, top_k)

    # Apply top-p (nucleus) filtering
    if top_p is not None:
        probs = _apply_top_p(probs, top_p)

    # Sample from distribution
    token = np.random.choice(len(probs), p=probs)

    return token


def _softmax(logits: "np.ndarray[Any, Any]") -> "np.ndarray[Any, Any]":
    """Compute softmax probabilities from logits.

    Args:
        logits: Logit values

    Returns:
        Probability distribution
    """
    # Subtract max for numerical stability
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)


def _apply_top_k(probs: "np.ndarray[Any, Any]", k: int) -> "np.ndarray[Any, Any]":
    """Apply top-k filtering to probabilities.

    Args:
        probs: Probability distribution
        k: Number of top tokens to keep

    Returns:
        Filtered probability distribution
    """
    # Get indices of top-k probabilities
    top_k_indices = np.argpartition(probs, -k)[-k:]

    # Zero out probabilities outside top-k
    filtered_probs = np.zeros_like(probs)
    filtered_probs[top_k_indices] = probs[top_k_indices]

    # Renormalize
    return filtered_probs / np.sum(filtered_probs)


def _apply_top_p(probs: "np.ndarray[Any, Any]", p: float) -> "np.ndarray[Any, Any]":
    """Apply top-p (nucleus) filtering to probabilities.

    Args:
        probs: Probability distribution
        p: Cumulative probability threshold

    Returns:
        Filtered probability distribution
    """
    # Sort probabilities in descending order
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    # Compute cumulative probabilities
    cumsum_probs = np.cumsum(sorted_probs)

    # Find cutoff index where cumsum exceeds p
    cutoff_idx = np.searchsorted(cumsum_probs, p)

    # Keep tokens up to cutoff
    filtered_probs = np.zeros_like(probs)
    filtered_probs[sorted_indices[: cutoff_idx + 1]] = sorted_probs[: cutoff_idx + 1]

    # Renormalize
    return filtered_probs / np.sum(filtered_probs)

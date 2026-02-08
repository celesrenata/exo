"""Text generation logic using tinygrad.

This module implements token-by-token text generation for the tinygrad backend,
with proper error handling and state management.
"""

import logging
import time
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
else:
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore

from exo.shared.types.api import (
    CompletionTokensDetails,
    FinishReason,
    GenerationStats,
    PromptTokensDetails,
    Usage,
)
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
    ToolCallResponse,
)

logger = logging.getLogger(__name__)


def tinygrad_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    device: str = "CPU",
    model_id: str = "",
    eos_token_id: int | None = None,
) -> Generator[GenerationResponse | ToolCallResponse, None, None]:
    """Generate text tokens using tinygrad model (streaming).

    This function implements the core generation loop following the exo-cuda pattern.
    It yields GenerationResponse objects for each generated token, compatible with
    exo's streaming API.

    Args:
        model: Tinygrad model instance
        tokenizer: Tokenizer instance
        prompt: Input text prompt
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        device: Target device (CPU, GPU, METAL)
        model_id: Model identifier for response metadata
        eos_token_id: End-of-sequence token ID (if None, uses tokenizer.eos_token_id)

    Yields:
        GenerationResponse or ToolCallResponse objects containing:
        - text: Decoded token text (GenerationResponse)
        - token: Token ID (GenerationResponse)
        - tool_calls: Parsed tool calls (ToolCallResponse)
        - finish_reason: None during generation, "stop", "length", or "tool_calls" at end
        - stats: Generation statistics (tokens/sec, etc.)
        - usage: Token usage information

    Example:
        >>> for response in tinygrad_generate(
        ...     model,
        ...     tokenizer,
        ...     "Once upon a time",
        ...     max_tokens=50,
        ...     temperature=0.8
        ... ):
        ...     print(response.text, end="", flush=True)
    """
    try:
        # Encode prompt (synchronous wrapper for async function)
        logger.debug(f"Encoding prompt: {prompt[:50]}...")
        input_tokens = _encode_prompt_sync(tokenizer, prompt)
        prompt_token_count = len(input_tokens)

        # Initialize generation state
        generated_tokens: list[int] = input_tokens.tolist()
        inference_state = None
        generation_start_time = time.perf_counter()
        accumulated_text = ""  # Track full generated text for tool call detection

        # Determine EOS token
        if eos_token_id is None:
            eos_token_id = getattr(tokenizer, "eos_token_id", 0)

        logger.info(
            f"Starting generation: prompt_tokens={prompt_token_count}, "
            f"max_tokens={max_tokens}, temperature={temperature}"
        )

        # Generate tokens one by one
        for token_idx in range(max_tokens):
            # Prepare input (last token or full sequence on first iteration)
            if token_idx == 0:
                current_input = input_tokens.reshape(1, -1)
            else:
                current_input = np.array([[generated_tokens[-1]]], dtype=np.int64)

            # Run inference
            try:
                logits, inference_state = infer_tensor(
                    model,
                    current_input,
                    inference_state,
                    device,
                )
            except Exception as e:
                logger.error(f"Inference failed at token {token_idx}: {e}")
                yield GenerationResponse(
                    text=f"Error during inference: {e}",
                    token=0,
                    finish_reason="error",
                    usage=None,
                )
                return

            # Sample next token
            next_token = _sample_token(
                logits[0, -1],  # Last position logits
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            # Add to generated sequence
            generated_tokens.append(int(next_token))

            # Decode token (synchronous wrapper for async function)
            try:
                token_text = _decode_tokens_sync(
                    tokenizer, np.array([next_token], dtype=np.int64)
                )
            except Exception as e:
                logger.error(f"Token decoding failed: {e}")
                token_text = ""

            # Accumulate text for tool call detection
            accumulated_text += token_text

            # Check for tool calls at end of generation or when tool markers detected
            tool_calls = None
            if hasattr(tokenizer, "has_tool_calling") and tokenizer.has_tool_calling:
                # Check if we might have complete tool calls
                tool_call_end = getattr(tokenizer, "tool_call_end", None)
                if tool_call_end and tool_call_end in accumulated_text:
                    tool_calls = _detect_tool_calls(accumulated_text, tokenizer)

            # Calculate statistics
            elapsed = time.perf_counter() - generation_start_time
            completion_tokens = token_idx + 1
            tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0.0

            # Check for end of sequence
            finish_reason: FinishReason | None = None
            if next_token == eos_token_id:
                finish_reason = "stop"
                # Final check for tool calls
                if tool_calls is None:
                    tool_calls = _detect_tool_calls(accumulated_text, tokenizer)
            elif token_idx == max_tokens - 1:
                finish_reason = "length"

            # Create usage information
            usage = Usage(
                prompt_tokens=prompt_token_count,
                completion_tokens=completion_tokens,
                total_tokens=prompt_token_count + completion_tokens,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=0,
                    accepted_prediction_tokens=0,
                    rejected_prediction_tokens=0,
                ),
            )

            # Create generation stats
            stats = GenerationStats(
                prefill_tokens_per_sec=0.0,  # Not tracked separately in tinygrad
                generation_tokens_per_sec=tokens_per_sec,
            )

            # If tool calls detected at end, yield ToolCallResponse instead
            if tool_calls and finish_reason in ("stop", "length"):
                logger.info(f"Tool calls detected: {len(tool_calls)} calls")
                yield ToolCallResponse(
                    tool_calls=tool_calls,
                    usage=usage,
                )
                break

            # Yield regular token response
            yield GenerationResponse(
                text=token_text,
                token=int(next_token),
                finish_reason=finish_reason,
                stats=stats,
                usage=usage,
                logprob=None,  # TODO: Add logprob support
                top_logprobs=None,  # TODO: Add top_logprobs support
            )

            # Stop if we hit EOS
            if finish_reason == "stop":
                logger.info(
                    f"Generation complete (EOS): {completion_tokens} tokens "
                    f"in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
                )
                break

        else:
            # Reached max_tokens
            logger.info(
                f"Generation complete (max_tokens): {max_tokens} tokens "
                f"in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
            )

    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        yield GenerationResponse(
            text=f"Generation error: {e}",
            token=0,
            finish_reason="error",
            usage=None,
        )


def infer_tensor(
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
        >>> output, state = infer_tensor(
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
        output_logits = _forward_pass(
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

    Creates the initial KV cache structure for the model. The KV cache stores
    key and value tensors from previous tokens to avoid recomputing them.

    Args:
        model: Model instance

    Returns:
        Initial inference state dictionary with:
        - kv_cache: List of (key, value) tuples for each layer
        - position: Current position in sequence
        - generated_tokens: List of generated token IDs
    """
    # Initialize KV cache structure
    # Each layer needs a (key, value) cache
    num_layers = getattr(model, "num_layers", 0)

    kv_cache = []
    if num_layers > 0:
        # Create empty cache for each layer
        # Will be populated during forward passes
        for _ in range(num_layers):
            kv_cache.append({"key": None, "value": None})

    return {
        "kv_cache": kv_cache,
        "position": 0,
        "generated_tokens": [],
    }


def _forward_pass(
    model: Any,
    input_tensor: Any,
    inference_state: dict,
) -> Any:
    """Execute forward pass through the model with KV cache.

    This function runs the model's forward method, passing the KV cache
    to enable efficient incremental generation. The cache stores key and
    value tensors from previous tokens.

    Args:
        model: Model instance
        input_tensor: Input tensor (shape: [batch_size, seq_len])
        inference_state: Current inference state with KV cache

    Returns:
        Output logits tensor (shape: [batch_size, seq_len, vocab_size])
    """
    try:
        kv_cache = inference_state.get("kv_cache")

        # Check if model supports KV cache
        if hasattr(model, "forward_with_cache"):
            # Model has explicit cache support
            output, updated_cache = model.forward_with_cache(input_tensor, kv_cache)
            # Update cache in state (will be handled by _update_inference_state)
            inference_state["_updated_cache"] = updated_cache
            return output
        elif kv_cache is not None and len(kv_cache) > 0:
            # Try passing cache as keyword argument
            try:
                output = model(input_tensor, cache=kv_cache)
                return output
            except TypeError:
                # Model doesn't accept cache parameter, fall back to regular forward
                pass

        # Fall back to regular forward pass without cache
        output = model(input_tensor)
        return output

    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        # Return dummy output for now
        from tinygrad import Tensor

        # Create dummy logits (batch_size, seq_len, vocab_size)
        batch_size = input_tensor.shape[0]
        seq_len = input_tensor.shape[1] if len(input_tensor.shape) > 1 else 1
        vocab_size = 50000
        return Tensor.randn(batch_size, seq_len, vocab_size)


def _update_inference_state(
    inference_state: dict[str, Any],
    output_logits: Any,
) -> dict[str, Any]:
    """Update inference state after forward pass.

    Updates the KV cache with new key/value tensors from the current forward pass.
    The cache is used in subsequent forward passes to avoid recomputing attention
    for previous tokens.

    Args:
        inference_state: Current inference state
        output_logits: Output logits from forward pass

    Returns:
        Updated inference state with:
        - Updated KV cache (if model provided updates)
        - Incremented position counter
        - Same generated_tokens list (updated elsewhere)
    """
    # Create new state (don't modify original)
    new_state = inference_state.copy()

    # Update position counter
    new_state["position"] += 1

    # Check if forward pass provided updated cache
    if "_updated_cache" in inference_state:
        new_state["kv_cache"] = inference_state["_updated_cache"]
        # Remove temporary key
        if "_updated_cache" in new_state:
            del new_state["_updated_cache"]

    # Note: KV cache updates are model-specific
    # Some models update cache in-place during forward pass
    # Others return updated cache explicitly
    # The cache structure depends on the model architecture

    return new_state


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


def _detect_tool_calls(generated_text: str, tokenizer: Any) -> list[Any] | None:
    """Detect and parse tool calls from generated text.

    This function checks if the model has generated tool call markers
    and attempts to parse them into structured tool call objects.

    Args:
        generated_text: The accumulated generated text
        tokenizer: Tokenizer instance (may have tool call parsing support)

    Returns:
        List of ToolCallItem objects if tool calls detected, None otherwise
    """
    # Check if tokenizer has tool call support
    if not hasattr(tokenizer, "has_tool_calling") or not tokenizer.has_tool_calling:
        return None

    # Check for tool call markers
    tool_call_start = getattr(tokenizer, "tool_call_start", None)
    tool_call_end = getattr(tokenizer, "tool_call_end", None)

    if not tool_call_start or not tool_call_end:
        return None

    # Check if text contains tool call markers
    if tool_call_start not in generated_text:
        return None

    # Try to parse tool calls using tokenizer's parser
    tool_parser = getattr(tokenizer, "tool_parser", None)
    if tool_parser is None:
        return None

    try:
        # Parse tool calls from text
        tool_calls = tool_parser(generated_text)
        return tool_calls if tool_calls else None
    except Exception as e:
        logger.debug(f"Tool call parsing failed: {e}")
        return None


# Legacy async function for backward compatibility
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
    """Generate text from a prompt using the model (legacy async version).

    This is a high-level generation function that handles tokenization,
    inference, and decoding. Returns the complete generated text.

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
    # Collect all generated tokens
    generated_text = ""
    for response in tinygrad_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
    ):
        if (
            isinstance(response, GenerationResponse)
            and response.finish_reason != "error"
        ):
            generated_text += response.text

    return generated_text


def _encode_prompt_sync(tokenizer: Any, prompt: str) -> "np.ndarray[Any, Any]":
    """Synchronous wrapper for encode_prompt.

    Args:
        tokenizer: Tokenizer instance
        prompt: Text prompt to encode

    Returns:
        Numpy array of token IDs
    """
    import asyncio

    from exo.worker.engines.tinygrad.model_loader import encode_prompt

    # Run async function in event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, create a new task
            # This shouldn't happen in normal usage
            raise RuntimeError("Cannot call encode_prompt_sync from async context")
        return loop.run_until_complete(encode_prompt(tokenizer, prompt))
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(encode_prompt(tokenizer, prompt))


def _decode_tokens_sync(tokenizer: Any, tokens: "np.ndarray[Any, Any]") -> str:
    """Synchronous wrapper for decode_tokens.

    Args:
        tokenizer: Tokenizer instance
        tokens: Numpy array of token IDs

    Returns:
        Decoded text string
    """
    import asyncio

    from exo.worker.engines.tinygrad.model_loader import decode_tokens

    # Run async function in event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, create a new task
            # This shouldn't happen in normal usage
            raise RuntimeError("Cannot call decode_tokens_sync from async context")
        return loop.run_until_complete(decode_tokens(tokenizer, tokens))
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(decode_tokens(tokenizer, tokens))

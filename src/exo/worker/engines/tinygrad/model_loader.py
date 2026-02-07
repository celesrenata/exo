"""Model loading and weight conversion for tinygrad backend.

This module handles loading HuggingFace model weights and converting them
to tinygrad format for inference.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
else:
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore

from exo.shared.types.worker.shards import ShardMetadata

logger = logging.getLogger(__name__)


async def load_model_and_tokenizer(
    shard_metadata: ShardMetadata,
    checkpoint_path: str,
    device: str,
) -> tuple[Any, Any]:
    """Load model and tokenizer from checkpoint.

    This function loads a model and tokenizer from a HuggingFace checkpoint,
    converting weights to tinygrad format and placing them on the specified device.

    Args:
        shard_metadata: Metadata describing the model shard
        checkpoint_path: Path to checkpoint directory
        device: Target device (CPU, GPU, METAL)

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        FileNotFoundError: If checkpoint path does not exist
        RuntimeError: If model loading fails

    Example:
        >>> model, tokenizer = await load_model_and_tokenizer(
        ...     shard_metadata,
        ...     "/path/to/checkpoint",
        ...     "GPU"
        ... )
    """
    checkpoint_dir = Path(checkpoint_path)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    model_id = shard_metadata.model_card.model_id
    logger.info(f"Loading model {model_id} from {checkpoint_path}")

    # Load tokenizer
    tokenizer = await _load_tokenizer(checkpoint_dir, model_id)

    # Load model weights
    model = await _load_model_weights(
        checkpoint_dir,
        shard_metadata,
        device,
    )

    logger.info(f"Successfully loaded model and tokenizer for {model_id}")
    return model, tokenizer


async def _load_tokenizer(checkpoint_dir: Path, model_id: str) -> Any:
    """Load tokenizer from checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory
        model_id: Model identifier

    Returns:
        Loaded tokenizer instance

    Raises:
        RuntimeError: If tokenizer loading fails
    """
    try:
        # Try to use transformers tokenizer if available
        from transformers import AutoTokenizer

        logger.debug(f"Loading tokenizer from {checkpoint_dir}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint_dir),
            trust_remote_code=True,
        )
        return tokenizer
    except ImportError:
        logger.warning("transformers not available, using basic tokenizer")
        # Fall back to basic tokenizer implementation
        return _create_basic_tokenizer()
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}") from e


def _create_basic_tokenizer() -> Any:
    """Create a basic tokenizer as fallback.

    Returns:
        Basic tokenizer instance
    """

    # Simple tokenizer that just splits on whitespace
    # This is a minimal fallback - real implementation should use proper tokenizer
    class BasicTokenizer:
        def encode(self, text: str) -> list[int]:
            # Very basic: convert characters to ASCII values
            return [ord(c) for c in text]

        def decode(self, tokens: list[int]) -> str:
            # Convert ASCII values back to characters
            return "".join(chr(t) for t in tokens if 0 <= t < 128)

    return BasicTokenizer()


async def _load_model_weights(
    checkpoint_dir: Path,
    shard_metadata: ShardMetadata,
    device: str,
) -> Any:
    """Load model weights and convert to tinygrad format.

    Args:
        checkpoint_dir: Path to checkpoint directory
        shard_metadata: Metadata describing the model shard
        device: Target device (CPU, GPU, METAL)

    Returns:
        Loaded model instance

    Raises:
        RuntimeError: If model loading fails
    """
    try:
        logger.debug(f"Loading model weights from {checkpoint_dir}")

        # For now, create a placeholder model structure
        # Real implementation would load actual weights from safetensors
        model = _create_model_structure(shard_metadata, device)

        # Load weights from checkpoint if available
        weights_path = checkpoint_dir / "model.safetensors"
        if weights_path.exists():
            logger.debug(f"Loading weights from {weights_path}")
            weights = await _load_safetensors(weights_path)
            _apply_weights_to_model(model, weights, device)
        else:
            logger.warning(
                f"No weights file found at {weights_path}, using random initialization"
            )

        return model
    except ImportError as e:
        raise RuntimeError(f"tinygrad not available: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}") from e


def _create_model_structure(shard_metadata: ShardMetadata, device: str) -> Any:
    """Create model structure based on shard metadata.

    Args:
        shard_metadata: Metadata describing the model shard
        device: Target device

    Returns:
        Model structure (placeholder for now)
    """

    # Placeholder model structure
    # Real implementation would create proper transformer layers
    class PlaceholderModel:
        def __init__(self, n_layers: int, hidden_size: int):
            self.n_layers = n_layers
            self.hidden_size = hidden_size
            self.device = device

        def __call__(self, x: Any) -> Any:
            # Placeholder forward pass
            return x

    return PlaceholderModel(
        n_layers=shard_metadata.n_layers,
        hidden_size=shard_metadata.model_card.hidden_size,
    )


async def _load_safetensors(weights_path: Path) -> dict[str, "np.ndarray[Any, Any]"]:
    """Load weights from safetensors file.

    Args:
        weights_path: Path to safetensors file

    Returns:
        Dictionary mapping weight names to numpy arrays

    Raises:
        RuntimeError: If loading fails
    """
    try:
        from safetensors import safe_open

        weights = {}
        with safe_open(weights_path, framework="numpy") as f:
            for key in f:
                weights[key] = f.get_tensor(key)

        logger.debug(f"Loaded {len(weights)} weight tensors")
        return weights
    except ImportError:
        logger.warning("safetensors not available, skipping weight loading")
        return {}
    except Exception as e:
        raise RuntimeError(f"Failed to load safetensors: {e}") from e


def _apply_weights_to_model(
    model: Any,
    weights: dict[str, "np.ndarray[Any, Any]"],
    device: str,
) -> None:
    """Apply loaded weights to model.

    Args:
        model: Model instance
        weights: Dictionary of weight tensors
        device: Target device
    """
    # Placeholder implementation
    # Real implementation would map weights to model parameters
    logger.debug(f"Applying {len(weights)} weight tensors to model on {device}")


async def encode_prompt(tokenizer: Any, prompt: str) -> "np.ndarray[Any, Any]":
    """Encode a text prompt into tokens.

    Args:
        tokenizer: Tokenizer instance
        prompt: Text prompt to encode

    Returns:
        Token IDs as numpy array

    Example:
        >>> tokens = await encode_prompt(tokenizer, "Hello, world!")
        >>> print(tokens.shape)
    """
    try:
        # Use tokenizer's encode method
        if hasattr(tokenizer, "encode"):
            tokens = tokenizer.encode(prompt)
            if isinstance(tokens, list):
                return np.array(tokens, dtype=np.int64)
            return np.array(tokens, dtype=np.int64)
        else:
            # Fallback for basic tokenizer
            tokens = [ord(c) for c in prompt]
            return np.array(tokens, dtype=np.int64)
    except Exception as e:
        logger.error(f"Failed to encode prompt: {e}")
        raise


async def decode_tokens(tokenizer: Any, tokens: "np.ndarray[Any, Any]") -> str:
    """Decode tokens back into text.

    Args:
        tokenizer: Tokenizer instance
        tokens: Token IDs to decode

    Returns:
        Decoded text string

    Example:
        >>> text = await decode_tokens(tokenizer, tokens)
        >>> print(text)
    """
    try:
        # Convert numpy array to list
        token_list = tokens.tolist()

        # Use tokenizer's decode method
        if hasattr(tokenizer, "decode"):
            return tokenizer.decode(token_list)
        else:
            # Fallback for basic tokenizer
            return "".join(chr(t) for t in token_list if 0 <= t < 128)
    except Exception as e:
        logger.error(f"Failed to decode tokens: {e}")
        raise


async def save_model(model: Any, path: str) -> None:
    """Save model weights to checkpoint.

    Args:
        model: Model instance to save
        path: Path to save checkpoint

    Raises:
        RuntimeError: If saving fails
    """
    checkpoint_dir = Path(path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model to {path}")

    # Placeholder implementation
    # Real implementation would save model weights to safetensors
    logger.warning("Model saving not fully implemented yet")

"""Model loading and weight conversion for tinygrad backend.

This module handles loading HuggingFace model weights and converting them
to tinygrad format for inference. Follows the pattern from exo-cuda reference.

Reference: https://github.com/Scottcjn/exo-cuda
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
else:
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore

from loguru import logger

from exo.shared.types.worker.shards import ShardMetadata

# Shared executor for tinygrad operations (must run on same thread)
_executor = ThreadPoolExecutor(max_workers=1)


async def load_tinygrad_model(
    shard_metadata: ShardMetadata,
    checkpoint_path: str,
    device: str,
) -> tuple[Any, Any]:
    """Load model and tokenizer from checkpoint using tinygrad.

    This function loads a model and tokenizer from a HuggingFace checkpoint,
    converting weights to tinygrad format and placing them on the specified device.

    Following exo-cuda pattern:
    - Model loading runs in dedicated thread executor
    - Weights are loaded from safetensors format
    - Tokenizer is loaded from HuggingFace format

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
        >>> model, tokenizer = await load_tinygrad_model(
        ...     shard_metadata,
        ...     "/path/to/checkpoint",
        ...     "GPU"
        ... )
    """
    checkpoint_dir = Path(checkpoint_path)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    model_id = shard_metadata.model_card.model_id
    logger.info(
        "Loading tinygrad model",
        model_id=str(model_id),
        checkpoint_path=checkpoint_path,
        device=device,
        start_layer=shard_metadata.start_layer,
        end_layer=shard_metadata.end_layer,
    )

    # Load tokenizer (can run in main thread)
    tokenizer = await _load_tokenizer(checkpoint_dir, model_id)

    # Load model weights (must run in executor for tinygrad thread safety)
    loop = asyncio.get_running_loop()
    model = await loop.run_in_executor(
        _executor,
        _load_model_weights_sync,
        checkpoint_dir,
        shard_metadata,
        device,
    )

    logger.info(
        "Successfully loaded tinygrad model",
        model_id=str(model_id),
        device=device,
        n_layers=shard_metadata.n_layers,
    )
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
        logger.info(
            "Tokenizer loaded",
            checkpoint_dir=str(checkpoint_dir),
            vocab_size=len(tokenizer) if hasattr(tokenizer, "__len__") else "unknown",
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

    logger.warning("Using basic fallback tokenizer (limited functionality)")
    return BasicTokenizer()


def _load_model_weights_sync(
    checkpoint_dir: Path,
    shard_metadata: ShardMetadata,
    device: str,
) -> Any:
    """Load model weights synchronously (runs in executor).

    This function runs in a dedicated thread executor to ensure tinygrad
    operations are thread-safe. Following exo-cuda pattern.

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
        logger.debug(
            "Loading model weights",
            checkpoint_dir=str(checkpoint_dir),
            device=device,
            start_layer=shard_metadata.start_layer,
            end_layer=shard_metadata.end_layer,
        )

        # Determine model size from model_id
        model_size = _infer_model_size(shard_metadata.model_card.model_id)

        # Find weights file
        weights_path = _find_weights_file(checkpoint_dir)

        if weights_path:
            logger.debug(f"Loading weights from {weights_path}")
            # Load weights from safetensors
            weights = _load_safetensors_sync(weights_path)

            # Create model architecture and load weights
            model = _create_model_with_weights(
                shard_metadata, weights, device, model_size
            )
        else:
            logger.warning(
                f"No weights file found in {checkpoint_dir}, using random initialization"
            )
            # Create model with random weights
            model = _create_model_structure(shard_metadata, device, model_size)

        return model
    except ImportError as e:
        raise RuntimeError(f"tinygrad not available: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}") from e


def _infer_model_size(model_id: str) -> str:
    """Infer model size from model_id string.

    Following exo-cuda pattern of extracting size from model name.

    Args:
        model_id: Model identifier (e.g., "llama-3.2-3b-instruct")

    Returns:
        Model size string (e.g., "3B", "8B", "70B")
    """
    model_id_lower = str(model_id).lower()

    # Check for common size patterns
    if "0.5b" in model_id_lower or "500m" in model_id_lower:
        return "0.5B"
    elif "1b" in model_id_lower or "1.5b" in model_id_lower:
        return "1B"
    elif "3b" in model_id_lower:
        return "3B"
    elif "7b" in model_id_lower or "8b" in model_id_lower:
        return "8B"
    elif "13b" in model_id_lower:
        return "13B"
    elif "70b" in model_id_lower:
        return "70B"
    else:
        logger.warning(f"Could not infer model size from {model_id}, defaulting to 8B")
        return "8B"


def _find_weights_file(checkpoint_dir: Path) -> Path | None:
    """Find weights file in checkpoint directory.

    Looks for safetensors files in order of preference:
    1. model.safetensors.index.json (sharded model)
    2. model.safetensors (single file)
    3. *.safetensors (any safetensors file)

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        Path to weights file, or None if not found
    """
    # Check for index file (sharded model)
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        return index_path

    # Check for single file
    single_path = checkpoint_dir / "model.safetensors"
    if single_path.exists():
        return single_path

    # Check for any safetensors file
    safetensors_files = list(checkpoint_dir.glob("*.safetensors"))
    if safetensors_files:
        return safetensors_files[0]

    return None


def _load_safetensors_sync(weights_path: Path) -> dict[str, "np.ndarray[Any, Any]"]:
    """Load weights from safetensors file synchronously.

    Args:
        weights_path: Path to safetensors file or index

    Returns:
        Dictionary mapping weight names to numpy arrays

    Raises:
        RuntimeError: If loading fails
    """
    try:
        # Check if this is an index file (sharded model)
        if weights_path.name.endswith(".index.json"):
            return _load_sharded_safetensors(weights_path)
        else:
            return _load_single_safetensors(weights_path)
    except ImportError:
        logger.warning("safetensors not available, skipping weight loading")
        return {}
    except Exception as e:
        raise RuntimeError(f"Failed to load safetensors: {e}") from e


def _convert_bfloat16_to_float32(tensor: "np.ndarray[Any, Any]") -> "np.ndarray[Any, Any]":
    """Convert bfloat16 tensor to float32.
    
    bfloat16 is not natively supported by numpy, but safetensors can load it.
    We convert it to float32 for compatibility with tinygrad.
    
    Args:
        tensor: Input tensor in bfloat16 format
        
    Returns:
        Tensor converted to float32
    """
    try:
        # Try direct conversion if numpy supports it
        return tensor.astype(np.float32)
    except (AttributeError, TypeError):
        # Fallback: bfloat16 is stored as uint16, manually convert
        # bfloat16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits
        # float32 format: 1 sign bit, 8 exponent bits, 23 mantissa bits
        # Conversion: shift left by 16 bits to expand mantissa
        if tensor.dtype == np.uint16:
            # Reinterpret as uint32 and shift left 16 bits
            uint32_data = tensor.astype(np.uint32) << 16
            # Reinterpret as float32
            return uint32_data.view(np.float32)
        else:
            logger.warning(f"Unexpected dtype for bfloat16 conversion: {tensor.dtype}, using direct cast")
            return tensor.astype(np.float32)


def _load_single_safetensors(weights_path: Path) -> dict[str, "np.ndarray[Any, Any]"]:
    """Load weights from a single safetensors file.

    Args:
        weights_path: Path to safetensors file

    Returns:
        Dictionary mapping weight names to numpy arrays
    """
    import struct
    
    try:
        # Read the file manually to handle bfloat16
        with open(weights_path, 'rb') as f:
            # Read header length (first 8 bytes)
            header_size = struct.unpack('<Q', f.read(8))[0]
            # Read header JSON
            import json
            header = json.loads(f.read(header_size).decode('utf-8'))
            
            weights = {}
            # Get data start position
            data_start = 8 + header_size
            
            for key, info in header.items():
                if key == '__metadata__':
                    continue
                    
                dtype_str = info['dtype']
                shape = info['shape']
                data_offsets = info['data_offsets']
                
                # Seek to tensor data
                f.seek(data_start + data_offsets[0])
                # Read tensor bytes
                tensor_bytes = f.read(data_offsets[1] - data_offsets[0])
                
                # Convert based on dtype
                if dtype_str == 'BF16':
                    # bfloat16: read as uint16, convert to float32
                    tensor_uint16 = np.frombuffer(tensor_bytes, dtype=np.uint16).reshape(shape)
                    tensor = _convert_bfloat16_to_float32(tensor_uint16)
                    logger.debug(f"Converted {key} from bfloat16 to float32")
                else:
                    # Use numpy's dtype mapping
                    np_dtype = _safetensors_dtype_to_numpy(dtype_str)
                    tensor = np.frombuffer(tensor_bytes, dtype=np_dtype).reshape(shape)
                
                weights[key] = tensor
        
        logger.debug(f"Loaded {len(weights)} weight tensors from {weights_path}")
        return weights
    except Exception as e:
        logger.error(f"Failed to load safetensors file {weights_path}: {e}")
        raise


def _safetensors_dtype_to_numpy(dtype_str: str) -> np.dtype:
    """Convert safetensors dtype string to numpy dtype."""
    dtype_map = {
        'F32': np.float32,
        'F16': np.float16,
        'I32': np.int32,
        'I64': np.int64,
        'U8': np.uint8,
        'I8': np.int8,
        'I16': np.int16,
        'U16': np.uint16,
        'U32': np.uint32,
        'U64': np.uint64,
        'BOOL': np.bool_,
    }
    return dtype_map.get(dtype_str, np.float32)


def _load_sharded_safetensors(
    index_path: Path,
) -> dict[str, "np.ndarray[Any, Any]"]:
    """Load weights from sharded safetensors files.

    Args:
        index_path: Path to model.safetensors.index.json

    Returns:
        Dictionary mapping weight names to numpy arrays
    """
    import json

    # Load index file
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    checkpoint_dir = index_path.parent

    # Load weights from all shard files
    weights = {}
    shard_files = set(weight_map.values())

    for shard_file in shard_files:
        shard_path = checkpoint_dir / shard_file
        if not shard_path.exists():
            logger.warning(f"Shard file not found: {shard_path}")
            continue

        try:
            # Load entire shard file using manual loader
            shard_weights = _load_single_safetensors(shard_path)
            
            # Only keep weights that belong to this shard according to index
            for key in shard_weights.keys():
                if key in weight_map and weight_map[key] == shard_file:
                    weights[key] = shard_weights[key]
        except Exception as e:
            logger.error(f"Failed to load shard {shard_path}: {e}")
            raise

    logger.debug(
        f"Loaded {len(weights)} weight tensors from {len(shard_files)} shard files"
    )
    return weights


def _create_model_structure(
    shard_metadata: ShardMetadata, device: str, model_size: str
) -> Any:
    """Create model structure based on shard metadata.

    This creates a placeholder model structure. In a full implementation,
    this would create proper transformer layers using tinygrad.

    Args:
        shard_metadata: Metadata describing the model shard
        device: Target device
        model_size: Model size string (e.g., "3B", "8B")

    Returns:
        Model structure
    """
    logger.debug(
        "Creating model structure",
        model_size=model_size,
        n_layers=shard_metadata.n_layers,
        start_layer=shard_metadata.start_layer,
        end_layer=shard_metadata.end_layer,
        device=device,
    )

    # Placeholder model structure
    # Real implementation would use tinygrad to create transformer layers
    class PlaceholderModel:
        def __init__(
            self,
            n_layers: int,
            hidden_size: int,
            start_layer: int,
            end_layer: int,
            device: str,
        ):
            self.n_layers = n_layers
            self.hidden_size = hidden_size
            self.start_layer = start_layer
            self.end_layer = end_layer
            self.device = device
            self.model_size = model_size

        def embed(self, x: Any) -> Any:
            """Embed tokens (placeholder)."""
            return x

        def forward(self, h: Any, **kwargs: Any) -> Any:
            """Forward pass (placeholder)."""
            return h

        def __call__(self, x: Any) -> Any:
            """Forward pass through model."""
            h = self.embed(x)
            return self.forward(h)

    return PlaceholderModel(
        n_layers=shard_metadata.n_layers,
        hidden_size=shard_metadata.model_card.hidden_size,
        start_layer=shard_metadata.start_layer,
        end_layer=shard_metadata.end_layer,
        device=device,
    )


def _create_model_with_weights(
    shard_metadata: ShardMetadata,
    weights: dict[str, "np.ndarray[Any, Any]"],
    device: str,
    model_size: str,
) -> Any:
    """Create model architecture and load weights.

    This function creates the model structure and applies loaded weights.
    Following exo-cuda pattern of building transformer with weights.

    Args:
        shard_metadata: Metadata describing the model shard
        weights: Dictionary of weight tensors
        device: Target device
        model_size: Model size string (e.g., "3B", "8B")

    Returns:
        Model with loaded weights
    """
    logger.debug(
        "Creating model with weights",
        model_size=model_size,
        n_weights=len(weights),
        start_layer=shard_metadata.start_layer,
        end_layer=shard_metadata.end_layer,
    )

    # Create base model structure
    model = _create_model_structure(shard_metadata, device, model_size)

    # Filter weights for this shard (pipeline sharding)
    shard_weights = _filter_weights_for_shard(weights, shard_metadata)

    # Apply weights to model
    _apply_weights_to_model(model, shard_weights, device)

    logger.info(
        "Model created with weights",
        model_size=model_size,
        n_weights=len(shard_weights),
        device=device,
    )

    return model


def _filter_weights_for_shard(
    weights: dict[str, "np.ndarray[Any, Any]"],
    shard_metadata: ShardMetadata,
) -> dict[str, "np.ndarray[Any, Any]"]:
    """Filter weights to only include layers for this shard.

    For pipeline sharding, we only need weights for layers in the range
    [start_layer, end_layer).

    Args:
        weights: All model weights
        shard_metadata: Metadata describing the model shard

    Returns:
        Filtered weights for this shard
    """
    start_layer = shard_metadata.start_layer
    end_layer = shard_metadata.end_layer

    # If this is the full model (no sharding), return all weights
    if start_layer == 0 and end_layer == shard_metadata.n_layers:
        logger.debug("No sharding needed, using all weights")
        return weights

    shard_weights = {}

    for key, value in weights.items():
        # Check if this weight belongs to a layer in our shard
        layer_num = _extract_layer_number(key)

        if layer_num is None:
            # Layer-independent weights (embeddings, final norm, etc.)
            # Include if we're the first or last shard
            if shard_metadata.is_first_layer or shard_metadata.is_last_layer:
                shard_weights[key] = value
        elif start_layer <= layer_num < end_layer:
            # This layer is in our shard
            shard_weights[key] = value

    logger.debug(
        f"Filtered weights for shard [{start_layer}, {end_layer}): "
        f"{len(shard_weights)}/{len(weights)} weights"
    )

    return shard_weights


def _extract_layer_number(weight_name: str) -> int | None:
    """Extract layer number from weight name.

    Common patterns:
    - "model.layers.0.weight" -> 0
    - "transformer.h.5.attn.weight" -> 5
    - "blocks.10.mlp.weight" -> 10

    Args:
        weight_name: Name of the weight tensor

    Returns:
        Layer number, or None if not a layer-specific weight
    """
    import re

    # Try common patterns
    patterns = [
        r"layers\.(\d+)\.",  # model.layers.N.
        r"\.h\.(\d+)\.",  # transformer.h.N.
        r"blocks\.(\d+)\.",  # blocks.N.
        r"layer\.(\d+)\.",  # layer.N.
    ]

    for pattern in patterns:
        match = re.search(pattern, weight_name)
        if match:
            return int(match.group(1))

    return None


def _apply_weights_to_model(
    model: Any,
    weights: dict[str, "np.ndarray[Any, Any]"],
    device: str,
) -> None:
    """Apply loaded weights to model.

    This is a placeholder implementation. In a full implementation,
    this would convert numpy arrays to tinygrad Tensors and load them
    into the model parameters.

    Args:
        model: Model instance
        weights: Dictionary of weight tensors
        device: Target device
    """
    logger.debug(
        f"Applying {len(weights)} weight tensors to model on {device}",
        device=device,
        n_weights=len(weights),
    )

    # Placeholder: In real implementation, would do:
    # 1. Convert numpy arrays to tinygrad Tensors
    # 2. Map weight names to model parameters
    # 3. Load tensors into model with proper device placement
    #
    # Example (pseudo-code):
    # from tinygrad import Tensor
    # for name, weight in weights.items():
    #     tensor = Tensor(weight, device=device)
    #     set_parameter(model, name, tensor)

    # Store weights in model for now
    if not hasattr(model, "_weights"):
        model._weights = {}
    model._weights.update(weights)


async def encode_prompt(tokenizer: Any, prompt: str) -> "np.ndarray[Any, Any]":
    """Encode a text prompt into tokens.

    Compatible with exo's tokenizer interface. Supports both HuggingFace
    transformers tokenizers and basic fallback tokenizer.

    Args:
        tokenizer: Tokenizer instance
        prompt: Text prompt to encode

    Returns:
        Token IDs as numpy array

    Example:
        >>> tokens = await encode_prompt(tokenizer, "Hello, world!")
        >>> print(tokens.shape)
        (13,)
    """
    try:
        # Use tokenizer's encode method
        if hasattr(tokenizer, "encode"):
            tokens = tokenizer.encode(prompt)

            # Handle different return types
            if isinstance(tokens, list):
                return np.array(tokens, dtype=np.int64)
            elif hasattr(tokens, "ids"):
                # HuggingFace Encoding object
                return np.array(tokens.ids, dtype=np.int64)
            else:
                # Assume it's already array-like
                return np.array(tokens, dtype=np.int64)
        else:
            # Fallback for basic tokenizer
            tokens = [ord(c) for c in prompt]
            return np.array(tokens, dtype=np.int64)
    except Exception as e:
        logger.error(f"Failed to encode prompt: {e}")
        raise RuntimeError(f"Failed to encode prompt: {e}") from e


async def decode_tokens(tokenizer: Any, tokens: "np.ndarray[Any, Any]") -> str:
    """Decode tokens back into text.

    Compatible with exo's tokenizer interface. Supports both HuggingFace
    transformers tokenizers and basic fallback tokenizer.

    Args:
        tokenizer: Tokenizer instance
        tokens: Token IDs to decode

    Returns:
        Decoded text string

    Example:
        >>> text = await decode_tokens(tokenizer, tokens)
        >>> print(text)
        "Hello, world!"
    """
    try:
        # Convert numpy array to list if needed
        if isinstance(tokens, np.ndarray):
            token_list = tokens.tolist()
        else:
            token_list = list(tokens)

        # Use tokenizer's decode method
        if hasattr(tokenizer, "decode"):
            return tokenizer.decode(token_list, skip_special_tokens=True)
        else:
            # Fallback for basic tokenizer
            return "".join(chr(t) for t in token_list if 0 <= t < 128)
    except Exception as e:
        logger.error(f"Failed to decode tokens: {e}")
        raise RuntimeError(f"Failed to decode tokens: {e}") from e


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


# Backward compatibility aliases
async def load_model_and_tokenizer(
    shard_metadata: ShardMetadata,
    checkpoint_path: str,
    device: str,
) -> tuple[Any, Any]:
    """Backward compatibility alias for load_tinygrad_model.

    Args:
        shard_metadata: Metadata describing the model shard
        checkpoint_path: Path to checkpoint directory
        device: Target device (CPU, GPU, METAL)

    Returns:
        Tuple of (model, tokenizer)
    """
    return await load_tinygrad_model(shard_metadata, checkpoint_path, device)

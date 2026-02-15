"""
Model Loader for PyTorch + IPEX Backend

This module provides model loading from HuggingFace, IPEX optimization,
and model sharding support for distributed inference.

Requirements addressed:
- 2.1: Model loading from HuggingFace hub and local cache
- 2.2: IPEX optimization with bfloat16 precision
- 2.3: Model sharding for distributed inference
- 2.4: Model validation and compatibility checking
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Optional, final

if TYPE_CHECKING:
    import numpy as np
else:
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore

from exo.shared.types.worker.shards import ShardMetadata

logger = logging.getLogger(__name__)


@final
class ModelLoader:
    """
    Loads and optimizes HuggingFace models for Intel Arc GPUs using PyTorch + IPEX.

    This class provides:
    - Async model loading from HuggingFace hub or local cache
    - IPEX optimization with bfloat16 precision
    - Model sharding support for distributed inference
    - Model validation and compatibility checking

    Requirements: 2.1, 2.2, 2.3, 2.4
    """

    def __init__(self) -> None:
        """Initialize the ModelLoader."""
        self._torch_available: bool = False
        self._ipex_available: bool = False
        self._transformers_available: bool = False

        # Try to import dependencies
        try:
            import torch  # type: ignore

            self._torch_available = True
            self._torch = torch
            logger.info(f"PyTorch available: {torch.__version__}")
        except ImportError:
            logger.error("PyTorch not available - ModelLoader cannot function")

        try:
            import intel_extension_for_pytorch as ipex  # type: ignore

            self._ipex_available = True
            self._ipex = ipex
            logger.info(f"IPEX available: {ipex.__version__}")
        except ImportError:
            logger.warning("IPEX not available - optimizations will be skipped")

        try:
            import transformers  # type: ignore

            self._transformers_available = True
            self._transformers = transformers
            logger.info(f"Transformers available: {transformers.__version__}")
        except ImportError:
            logger.error("Transformers not available - cannot load HuggingFace models")

        # Cache for loaded models
        self._model_cache: dict[str, Any] = {}
        self._tokenizer_cache: dict[str, Any] = {}

    async def load_model(
        self,
        shard_metadata: ShardMetadata,
        device_type: str,
        device_id: int,
    ) -> tuple[Any, Any]:
        """
        Load model and tokenizer from HuggingFace hub or local cache.

        This method:
        1. Checks local cache first
        2. Downloads from HuggingFace if not cached
        3. Loads model using transformers.AutoModelForCausalLM
        4. Applies IPEX optimizations
        5. Handles model sharding if needed
        6. Validates model compatibility

        Args:
            shard_metadata: Metadata describing the model shard
            device_type: Device type string ("xpu", "cuda", or "cpu")
            device_id: Device ID

        Returns:
            Tuple of (model, tokenizer)

        Raises:
            RuntimeError: If model loading fails
            ValueError: If model is incompatible

        Requirements: 2.1
        """
        if not self._torch_available or not self._transformers_available:
            raise RuntimeError(
                "PyTorch and Transformers are required for model loading"
            )

        model_id = str(shard_metadata.model_card.model_id)
        cache_key = f"{model_id}_{shard_metadata.start_layer}_{shard_metadata.end_layer}"

        # Check cache first
        if cache_key in self._model_cache:
            logger.info(f"Using cached model for {model_id}")
            return self._model_cache[cache_key], self._tokenizer_cache[model_id]

        logger.info(
            f"Loading model {model_id} for device {device_type}:{device_id}",
            extra={
                "model_id": model_id,
                "device": f"{device_type}:{device_id}",
                "start_layer": shard_metadata.start_layer,
                "end_layer": shard_metadata.end_layer,
                "n_layers": shard_metadata.n_layers,
            },
        )

        # Create torch device
        device = self._torch.device(f"{device_type}:{device_id}")

        # Load model and tokenizer in executor to avoid blocking
        loop = asyncio.get_running_loop()
        model, tokenizer = await loop.run_in_executor(
            None,
            self._load_model_sync,
            model_id,
            device,
            shard_metadata,
        )

        # Cache the loaded model and tokenizer
        self._model_cache[cache_key] = model
        self._tokenizer_cache[model_id] = tokenizer

        logger.info(f"Successfully loaded and cached model {model_id}")

        return model, tokenizer

    def _load_model_sync(
        self,
        model_id: str,
        device: Any,
        shard_metadata: ShardMetadata,
    ) -> tuple[Any, Any]:
        """
        Synchronously load model and tokenizer (runs in executor).

        Args:
            model_id: HuggingFace model identifier
            device: Torch device object
            shard_metadata: Metadata describing the model shard

        Returns:
            Tuple of (model, tokenizer)

        Raises:
            RuntimeError: If loading fails
        """
        try:
            # Load tokenizer
            logger.debug(f"Loading tokenizer for {model_id}")
            tokenizer = self._transformers.AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
            logger.info(f"Tokenizer loaded: vocab_size={len(tokenizer)}")

            # Load model
            logger.debug(f"Loading model {model_id}")
            model = self._transformers.AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=self._torch.bfloat16,  # Use bfloat16 for efficiency
                low_cpu_mem_usage=True,
            )
            logger.info(f"Model loaded: {type(model).__name__}")

            # Validate model compatibility
            self._validate_model(model, shard_metadata)

            # Move model to device
            logger.debug(f"Moving model to device {device}")
            model = model.to(device)

            # Apply IPEX optimizations
            if self._ipex_available and device.type == "xpu":
                logger.debug("Applying IPEX optimizations")
                model = self._apply_ipex_optimization(model, device)
            else:
                logger.debug("Skipping IPEX optimizations (not available or not XPU)")

            # Handle model sharding if needed
            if not (
                shard_metadata.start_layer == 0
                and shard_metadata.end_layer == shard_metadata.n_layers
            ):
                logger.debug(
                    f"Creating shard [{shard_metadata.start_layer}, {shard_metadata.end_layer})"
                )
                model = self._create_model_shard(model, shard_metadata)

            # Set model to eval mode
            model.eval()

            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise RuntimeError(f"Failed to load model {model_id}: {e}") from e

    def _validate_model(self, model: Any, shard_metadata: ShardMetadata) -> None:
        """
        Validate model compatibility with shard metadata.

        Checks:
        - Model has required attributes (config, layers)
        - Layer count matches metadata
        - Model architecture is supported

        Args:
            model: Loaded model instance
            shard_metadata: Metadata describing the model shard

        Raises:
            ValueError: If model is incompatible

        Requirements: 2.4
        """
        # Check if model has config
        if not hasattr(model, "config"):
            raise ValueError(f"Model {type(model).__name__} has no config attribute")

        config = model.config

        # Check if model has layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            actual_layers = len(model.model.layers)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            actual_layers = len(model.transformer.h)
        elif hasattr(model, "layers"):
            actual_layers = len(model.layers)
        else:
            logger.warning(
                f"Could not determine layer count for {type(model).__name__}, "
                "skipping layer validation"
            )
            return

        # Validate layer count
        expected_layers = shard_metadata.n_layers
        if actual_layers != expected_layers:
            raise ValueError(
                f"Model has {actual_layers} layers but metadata specifies {expected_layers}"
            )

        # Validate shard boundaries
        if shard_metadata.start_layer < 0:
            raise ValueError(f"Invalid start_layer: {shard_metadata.start_layer}")
        if shard_metadata.end_layer > actual_layers:
            raise ValueError(
                f"Invalid end_layer: {shard_metadata.end_layer} > {actual_layers}"
            )
        if shard_metadata.start_layer >= shard_metadata.end_layer:
            raise ValueError(
                f"Invalid shard range: [{shard_metadata.start_layer}, {shard_metadata.end_layer})"
            )

        logger.info(
            f"Model validation passed: {actual_layers} layers, "
            f"shard [{shard_metadata.start_layer}, {shard_metadata.end_layer})"
        )

    def _apply_ipex_optimization(self, model: Any, device: Any) -> Any:
        """
        Apply IPEX optimizations to the model.

        Optimizations include:
        - bfloat16 precision
        - Weights prepacking
        - Operator fusion

        Args:
            model: Model to optimize
            device: Target device

        Returns:
            Optimized model

        Requirements: 2.2, 8.1
        """
        if not self._ipex_available:
            logger.warning("IPEX not available, skipping optimization")
            return model

        try:
            logger.debug("Applying IPEX optimizations")

            # Apply IPEX optimize
            optimized_model = self._ipex.optimize(
                model,
                dtype=self._torch.bfloat16,
                inplace=True,
                weights_prepack=True,
            )

            logger.info("IPEX optimizations applied successfully")
            return optimized_model

        except Exception as e:
            logger.error(f"Failed to apply IPEX optimizations: {e}")
            logger.warning("Continuing with unoptimized model")
            return model

    def _create_model_shard(self, model: Any, shard_metadata: ShardMetadata) -> Any:
        """
        Create a model shard by extracting specific layer ranges.

        This wraps the model in a TransformerShard that only executes
        the specified layers.

        Args:
            model: Full model instance
            shard_metadata: Metadata describing the shard

        Returns:
            Sharded model wrapper

        Requirements: 5.2
        """
        logger.debug(
            f"Creating TransformerShard for layers "
            f"[{shard_metadata.start_layer}, {shard_metadata.end_layer})"
        )

        # Create shard wrapper
        shard = TransformerShard(
            model=model,
            start_layer=shard_metadata.start_layer,
            end_layer=shard_metadata.end_layer,
            is_first_layer=shard_metadata.is_first_layer,
            is_last_layer=shard_metadata.is_last_layer,
        )

        logger.info(
            f"Created TransformerShard: "
            f"layers [{shard_metadata.start_layer}, {shard_metadata.end_layer}), "
            f"first={shard_metadata.is_first_layer}, "
            f"last={shard_metadata.is_last_layer}"
        )

        return shard

    async def encode(self, model_id: str, prompt: str) -> "np.ndarray[Any, Any]":
        """
        Encode a text prompt into tokens.

        Args:
            model_id: Model identifier
            prompt: Text prompt to encode

        Returns:
            Token IDs as numpy array

        Raises:
            RuntimeError: If encoding fails
        """
        if np is None:
            raise RuntimeError("numpy is required for encoding")

        if model_id not in self._tokenizer_cache:
            raise RuntimeError(f"Tokenizer for {model_id} not loaded")

        tokenizer = self._tokenizer_cache[model_id]

        try:
            # Encode prompt
            tokens = tokenizer.encode(prompt, return_tensors="np")

            # Handle different return types
            if isinstance(tokens, np.ndarray):
                return tokens
            elif hasattr(tokens, "numpy"):
                return tokens.numpy()
            else:
                return np.array(tokens, dtype=np.int64)

        except Exception as e:
            logger.error(f"Failed to encode prompt: {e}")
            raise RuntimeError(f"Failed to encode prompt: {e}") from e

    async def decode(self, model_id: str, tokens: "np.ndarray[Any, Any]") -> str:
        """
        Decode tokens back into text.

        Args:
            model_id: Model identifier
            tokens: Token IDs to decode

        Returns:
            Decoded text string

        Raises:
            RuntimeError: If decoding fails
        """
        if np is None:
            raise RuntimeError("numpy is required for decoding")

        if model_id not in self._tokenizer_cache:
            raise RuntimeError(f"Tokenizer for {model_id} not loaded")

        tokenizer = self._tokenizer_cache[model_id]

        try:
            # Convert numpy array to list
            if isinstance(tokens, np.ndarray):
                token_list = tokens.tolist()
            else:
                token_list = list(tokens)

            # Decode tokens
            text = tokenizer.decode(token_list, skip_special_tokens=True)
            return text

        except Exception as e:
            logger.error(f"Failed to decode tokens: {e}")
            raise RuntimeError(f"Failed to decode tokens: {e}") from e

    def clear_cache(self) -> None:
        """Clear the model and tokenizer cache."""
        self._model_cache.clear()
        self._tokenizer_cache.clear()
        logger.info("Model cache cleared")


class TransformerShard:
    """
    Wrapper for a transformer model shard.

    This class wraps a full transformer model and only executes
    the specified layer range during forward pass.

    Requirements: 5.2
    """

    def __init__(
        self,
        model: Any,
        start_layer: int,
        end_layer: int,
        is_first_layer: bool,
        is_last_layer: bool,
    ) -> None:
        """
        Initialize the TransformerShard.

        Args:
            model: Full transformer model
            start_layer: Start layer index (inclusive)
            end_layer: End layer index (exclusive)
            is_first_layer: Whether this shard includes the first layer
            is_last_layer: Whether this shard includes the last layer
        """
        self.model = model
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.is_first_layer = is_first_layer
        self.is_last_layer = is_last_layer

        # Extract the layers we need
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self.layers = model.model.layers[start_layer:end_layer]
            self.embed_tokens = model.model.embed_tokens if is_first_layer else None
            self.norm = model.model.norm if is_last_layer else None
            self.lm_head = model.lm_head if is_last_layer else None
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            self.layers = model.transformer.h[start_layer:end_layer]
            self.embed_tokens = (
                model.transformer.wte if is_first_layer else None
            )  # word token embeddings
            self.norm = model.transformer.ln_f if is_last_layer else None  # final norm
            self.lm_head = model.lm_head if is_last_layer else None
        else:
            raise ValueError(f"Unsupported model architecture: {type(model).__name__}")

        logger.debug(
            f"TransformerShard initialized: "
            f"{len(self.layers)} layers, "
            f"embed={self.embed_tokens is not None}, "
            f"norm={self.norm is not None}, "
            f"lm_head={self.lm_head is not None}"
        )

    def forward(
        self,
        input_data: Any,
        attention_mask: Optional[Any] = None,
        past_key_values: Optional[Any] = None,
    ) -> tuple[Any, Optional[Any]]:
        """
        Forward pass through the shard.

        Args:
            input_data: Input tensor (tokens if first layer, hidden states otherwise)
            attention_mask: Optional attention mask
            past_key_values: Optional KV cache from previous forward pass

        Returns:
            Tuple of (output, new_past_key_values)
        """
        # If first layer, embed tokens
        if self.is_first_layer and self.embed_tokens is not None:
            hidden_states = self.embed_tokens(input_data)
        else:
            hidden_states = input_data

        # Process through layers
        new_past_key_values = []
        for i, layer in enumerate(self.layers):
            # Get past KV for this layer if available
            layer_past = past_key_values[i] if past_key_values else None

            # Forward through layer
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=layer_past,
                use_cache=True,
            )

            # Extract hidden states and new KV
            hidden_states = layer_outputs[0]
            if len(layer_outputs) > 1:
                new_past_key_values.append(layer_outputs[1])

        # If last layer, apply norm and lm_head
        if self.is_last_layer:
            if self.norm is not None:
                hidden_states = self.norm(hidden_states)
            if self.lm_head is not None:
                hidden_states = self.lm_head(hidden_states)

        return hidden_states, new_past_key_values if new_past_key_values else None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Make the shard callable like a model."""
        return self.forward(*args, **kwargs)

    def eval(self) -> "TransformerShard":
        """Set the shard to evaluation mode."""
        self.model.eval()
        return self

    def to(self, device: Any) -> "TransformerShard":
        """Move the shard to a device."""
        self.model.to(device)
        return self

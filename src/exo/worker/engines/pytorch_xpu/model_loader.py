"""
Model Loader for PyTorch Backend

This module provides model loading from HuggingFace and model sharding
support for distributed inference. Uses native PyTorch XPU (2.11+) for
Intel Arc GPU support — no IPEX dependency.

Requirements addressed:
- 2.1: Model loading from HuggingFace hub and local cache
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
from exo.worker.engines.pytorch_xpu.tensor_parallel_shard import (
    TensorParallelShard,
    TPShardConfig,
)

logger = logging.getLogger(__name__)


@final
class ModelLoader:
    """
    Loads and optimizes HuggingFace models for Intel Arc GPUs using native PyTorch XPU.

    This class provides:
    - Async model loading from HuggingFace hub or local cache
    - Native PyTorch XPU device placement (no IPEX dependency)
    - Model sharding support for distributed inference
    - Model validation and compatibility checking

    Requirements: 2.1, 2.2, 2.3, 2.4
    """

    def __init__(self) -> None:
        """Initialize the ModelLoader."""
        self._torch_available: bool = False
        self._transformers_available: bool = False

        # Try to import dependencies
        try:
            import torch  # type: ignore

            self._torch_available = True
            self._torch = torch
            logger.info(f"PyTorch available: {torch.__version__}")
        except ImportError as e:
            logger.error(f"PyTorch not available - ModelLoader cannot function: {e}")

        try:
            import transformers  # type: ignore

            self._transformers_available = True
            self._transformers = transformers
            logger.info(f"Transformers available: {transformers.__version__}")
        except ImportError as e:
            logger.error(f"Transformers not available - cannot load HuggingFace models: {e}")

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
        4. Places model on target device (XPU, CUDA, or CPU)
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
            # Check if this is a large model that should use streaming loading
            is_tensor_parallel = (
                shard_metadata.start_layer == 0
                and shard_metadata.end_layer == shard_metadata.n_layers
                and shard_metadata.world_size > 1
            )
            model_size_bytes = shard_metadata.model_card.storage_size.in_bytes if hasattr(shard_metadata, 'model_card') else 0
            use_streaming = is_tensor_parallel and model_size_bytes > 20_000_000_000

            if use_streaming:
                # STREAMING PATH: load directly from safetensors without from_pretrained()
                logger.info(
                    f"Using streaming safetensors loader for {model_id} "
                    f"({model_size_bytes / 1e9:.1f}GB)"
                )
                return self._load_model_streaming(model_id, device, shard_metadata)

            # STANDARD PATH: load via from_pretrained()
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
                torch_dtype=self._torch.bfloat16,  # bf16 works on Intel XPU (Meteor Lake+)
                low_cpu_mem_usage=True,
            )
            logger.info(f"Model loaded: {type(model).__name__}")

            # Validate model compatibility
            self._validate_model(model, shard_metadata)

            # Move model to device
            logger.debug(f"Moving model to device {device}")
            model = model.to(device)

            # Note: Native PyTorch 2.11+ handles XPU optimization internally.
            # No IPEX optimization needed — just use model.to(device).
            logger.debug("Model placed on device (native PyTorch XPU, no IPEX optimization)")

            # Handle model sharding if needed
            # Detect tensor-parallel configuration BEFORE pipeline-parallel:
            # Tensor-parallel: all layers on every node (start_layer=0, end_layer=n_layers, world_size>1)
            # Pipeline-parallel: different layer ranges per node
            is_tensor_parallel = (
                shard_metadata.start_layer == 0
                and shard_metadata.end_layer == shard_metadata.n_layers
                and shard_metadata.world_size > 1
            )

            if is_tensor_parallel:
                logger.debug(
                    f"Creating TensorParallelShard (all layers, world_size={shard_metadata.world_size},"
                    f" rank={shard_metadata.device_rank})"
                )
                model = self._create_tensor_parallel_shard(model, shard_metadata, device)
            elif not (
                shard_metadata.start_layer == 0
                and shard_metadata.end_layer == shard_metadata.n_layers
            ) or shard_metadata.world_size > 1:
                # Pipeline-parallel or partial layer range: create TransformerShard
                # The distributed generator expects the TransformerShard.forward(input_data=...)
                # interface, not the raw HuggingFace model.forward(input_ids=...) interface.
                logger.debug(
                    f"Creating TransformerShard [{shard_metadata.start_layer}, {shard_metadata.end_layer})"
                    f" (world_size={shard_metadata.world_size})"
                )
                model = self._create_model_shard(model, shard_metadata)

            # Set model to eval mode (TensorParallelShard doesn't need this —
            # it's not an nn.Module and has no training mode)
            if hasattr(model, "eval"):
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

    def _resolve_model_path(self, model_id: str) -> str:
        """Resolve the local filesystem path for a downloaded model."""
        import os
        dir_name = model_id.replace("/", "--")
        base_dir = os.path.expanduser("~/.local/share/exo/models")
        model_path = os.path.join(base_dir, dir_name)
        if os.path.exists(model_path):
            return model_path
        from huggingface_hub import snapshot_download
        return snapshot_download(model_id, local_files_only=True)

    def _load_model_streaming(
        self, model_id: str, device: Any, shard_metadata: ShardMetadata
    ) -> tuple[Any, Any]:
        """Load a large model using the streaming safetensors loader.

        Bypasses from_pretrained() entirely to avoid OOM. Loads weights
        directly from safetensors files, sharding as they're read.

        Returns:
            Tuple of (TensorParallelShard, tokenizer)
        """
        from exo.worker.engines.pytorch_xpu.streaming_loader import load_sharded_from_safetensors

        # Get model config for TPShardConfig
        config = self._transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if hasattr(config, 'text_config') and config.text_config is not None:
            text_config = config.text_config
        else:
            text_config = config

        hidden_size = text_config.hidden_size
        num_attention_heads = text_config.num_attention_heads
        intermediate_size = text_config.intermediate_size
        num_key_value_heads = getattr(text_config, "num_key_value_heads", num_attention_heads)
        head_dim = getattr(text_config, "head_dim", hidden_size // num_attention_heads)

        tp_config = TPShardConfig(
            rank=shard_metadata.device_rank,
            world_size=shard_metadata.world_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            num_key_value_heads=num_key_value_heads,
        )

        logger.info(
            f"Streaming load: rank={shard_metadata.device_rank}/{shard_metadata.world_size}, "
            f"hidden_size={hidden_size}, heads={num_attention_heads}, "
            f"head_dim={head_dim}, intermediate={intermediate_size}"
        )

        model_path = self._resolve_model_path(model_id)

        sharded_state_dict, native_layers, tokenizer = load_sharded_from_safetensors(
            model_path=model_path,
            config=tp_config,
            device=str(device),
            model_id=model_id,
        )

        # Create TensorParallelShard from pre-sharded state dict
        shard = TensorParallelShard(
            model=sharded_state_dict,
            config=tp_config,
            device=str(device),
        )

        # Attach native linear_attn layers if any
        if native_layers:
            shard._native_linear_attn_layers = native_layers
            logger.info(f"Attached {len(native_layers)} native linear_attn layers")

        return shard, tokenizer

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

    def _create_tensor_parallel_shard(
        self, model: Any, shard_metadata: ShardMetadata, device: Any
    ) -> TensorParallelShard:
        """
        Create a TensorParallelShard for tensor-parallel distributed inference.

        This wraps the model in a TensorParallelShard that holds ALL layers
        but with sharded weights within each layer. Each rank holds 1/world_size
        of the attention heads and MLP intermediate dimension, with all-reduce
        synchronization after row-parallel layers.

        Args:
            model: Full model instance (on device)
            shard_metadata: Metadata describing the shard (must have world_size > 1,
                start_layer=0, end_layer=n_layers)
            device: Torch device object

        Returns:
            TensorParallelShard instance with sharded weights

        Requirements: 1.5, 2.4, 3.1
        """
        # Extract model config from HuggingFace model
        config = model.config
        # For vision-language models (Qwen3.5/3.6), the text config is nested
        if hasattr(config, 'text_config') and config.text_config is not None:
            text_config = config.text_config
        else:
            text_config = config
        hidden_size = text_config.hidden_size
        num_attention_heads = text_config.num_attention_heads
        intermediate_size = text_config.intermediate_size
        num_key_value_heads = getattr(
            text_config, "num_key_value_heads", num_attention_heads
        )
        head_dim = getattr(text_config, "head_dim", hidden_size // num_attention_heads)

        # Construct TPShardConfig
        tp_config = TPShardConfig(
            rank=shard_metadata.device_rank,
            world_size=shard_metadata.world_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            num_key_value_heads=num_key_value_heads,
        )

        logger.info(
            f"Creating TensorParallelShard: "
            f"rank={shard_metadata.device_rank}/{shard_metadata.world_size}, "
            f"hidden_size={hidden_size}, "
            f"num_attention_heads={num_attention_heads}, "
            f"head_dim={head_dim}, "
            f"intermediate_size={intermediate_size}, "
            f"num_key_value_heads={num_key_value_heads}"
        )

        # Check model size — use streaming loader for large models (>20GB)
        # to avoid OOM from loading full model + state dict copy
        model_size_bytes = shard_metadata.model_card.storage_size.in_bytes if hasattr(shard_metadata, 'model_card') else 0
        use_streaming = model_size_bytes > 20_000_000_000  # 20GB threshold

        if use_streaming:
            logger.info(
                f"Using streaming safetensors loader for large model "
                f"({model_size_bytes / 1e9:.1f}GB > 20GB threshold)"
            )
            # Free the HuggingFace model to reclaim memory
            del model
            import gc
            gc.collect()

            from exo.worker.engines.pytorch_xpu.streaming_loader import load_sharded_from_safetensors

            # Resolve model path on disk
            model_id_str = str(shard_metadata.model_card.model_id)
            model_path = self._resolve_model_path(model_id_str)

            sharded_state_dict, native_layers, _ = load_sharded_from_safetensors(
                model_path=model_path,
                config=tp_config,
                device=str(device),
                model_id=str(shard_metadata.model_card.model_id),
            )

            # Create TensorParallelShard from pre-sharded state dict
            shard = TensorParallelShard(
                model=sharded_state_dict,
                config=tp_config,
                device=str(device),
            )

            # Attach native linear_attn layers if any
            if native_layers:
                shard._native_linear_attn_layers = native_layers

            return shard

        # Standard path for smaller models: load full model, extract state dict
        shard = TensorParallelShard(
            model=model,
            config=tp_config,
            device=str(device),
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
            # Rotary embeddings for position encoding (required by newer transformers)
            self.rotary_emb = getattr(model.model, "rotary_emb", None)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            self.layers = model.transformer.h[start_layer:end_layer]
            self.embed_tokens = (
                model.transformer.wte if is_first_layer else None
            )  # word token embeddings
            self.norm = model.transformer.ln_f if is_last_layer else None  # final norm
            self.lm_head = model.lm_head if is_last_layer else None
            self.rotary_emb = None
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
        import torch

        # Detect Qwen3.5 architecture (uses DynamicCache and 4D position IDs)
        is_qwen3_5 = hasattr(self.model, 'model') and type(self.model.model).__name__ == 'Qwen3_5TextModel'

        if is_qwen3_5:
            return self._forward_qwen3_5(input_data, attention_mask, past_key_values)

        # If first layer, embed tokens
        if self.is_first_layer and self.embed_tokens is not None:
            hidden_states = self.embed_tokens(input_data)
        else:
            hidden_states = input_data

        # Compute position embeddings if rotary_emb is available
        # (required by newer transformers for Qwen, Llama, etc.)
        position_embeddings = None
        if self.rotary_emb is not None:
            seq_len = hidden_states.shape[1]
            # Compute position IDs based on sequence length and past KV cache
            past_len = 0
            if past_key_values and len(past_key_values) > 0 and past_key_values[0] is not None:
                # past_key_values[0] is a tuple (key, value) for the first layer
                if isinstance(past_key_values[0], tuple) and len(past_key_values[0]) >= 1:
                    past_len = past_key_values[0][0].shape[2]
                elif hasattr(past_key_values[0], 'key_cache'):
                    past_len = past_key_values[0].key_cache.shape[2]
            position_ids = torch.arange(past_len, past_len + seq_len, device=hidden_states.device).unsqueeze(0)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Process through layers
        new_past_key_values = []
        for i, layer in enumerate(self.layers):
            # Get past KV for this layer if available
            layer_past = past_key_values[i] if past_key_values else None

            # Forward through layer
            layer_kwargs: dict[str, Any] = {
                "attention_mask": attention_mask,
                "past_key_value": layer_past,
                "use_cache": True,
            }
            if position_embeddings is not None:
                layer_kwargs["position_embeddings"] = position_embeddings

            layer_outputs = layer(hidden_states, **layer_kwargs)

            # Extract hidden states and new KV
            hidden_states = layer_outputs[0]
            if isinstance(layer_outputs, tuple) and len(layer_outputs) > 1 and layer_outputs[1] is not None:
                new_past_key_values.append(layer_outputs[1])

        # If last layer, apply norm and lm_head
        if self.is_last_layer:
            if self.norm is not None:
                hidden_states = self.norm(hidden_states)
            if self.lm_head is not None:
                hidden_states = self.lm_head(hidden_states)

        return hidden_states, new_past_key_values if new_past_key_values else None

    def _forward_qwen3_5(
        self,
        input_data: Any,
        attention_mask: Optional[Any] = None,
        past_key_values: Optional[Any] = None,
    ) -> tuple[Any, Optional[Any]]:
        """
        Forward pass for Qwen3.5 architecture.

        Qwen3.5 uses DynamicCache, 4D position IDs, and layers that mutate
        the cache in-place. We delegate to the Qwen3_5TextModel's forward
        method which handles all the internal complexity.
        """
        import torch
        from transformers import DynamicCache  # pyright: ignore[reportMissingImports]

        text_model = self.model.model  # Qwen3_5TextModel

        # For first shard: embed tokens
        if self.is_first_layer and self.embed_tokens is not None:
            inputs_embeds = self.embed_tokens(input_data)
        else:
            inputs_embeds = input_data

        # Create or reuse DynamicCache — pass config so it knows about linear_attention layers
        if past_key_values is None:
            cache = DynamicCache(config=text_model.config)
        else:
            cache = past_key_values  # Already a DynamicCache from previous call

        # Compute position IDs (4D for Qwen3.5: text, temporal, height, width)
        seq_len = inputs_embeds.shape[1]
        batch_size = inputs_embeds.shape[0]
        past_seen_tokens = cache.get_seq_length() if cache is not None else 0
        position_ids = torch.arange(seq_len, device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.view(1, 1, -1).expand(4, batch_size, -1)

        # Split into text_position_ids and spatial position_ids
        text_position_ids = position_ids[0]
        spatial_position_ids = position_ids[1:]

        # Compute position embeddings using rotary_emb
        position_embeddings = text_model.rotary_emb(inputs_embeds, spatial_position_ids)

        # Compute causal mask
        try:
            from transformers.modeling_utils import create_causal_mask  # pyright: ignore[reportMissingImports]
            causal_mask = create_causal_mask(
                config=text_model.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=cache,
                position_ids=text_position_ids,
            )
        except (ImportError, Exception):
            causal_mask = None

        # Compute linear attention mask
        linear_attn_mask = text_model._update_linear_attn_mask(attention_mask, cache)

        # Process through our assigned layers only
        hidden_states = inputs_embeds
        layer_types = text_model.config.layer_types

        for i, layer in enumerate(self.layers):
            # Determine the global layer index
            global_layer_idx = self.start_layer + i
            layer_type = layer_types[global_layer_idx] if global_layer_idx < len(layer_types) else "full_attention"
            layer_mask = linear_attn_mask if layer_type == "linear_attention" else causal_mask

            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=cache,
                use_cache=True,
            )

        # If last layer, apply norm and lm_head
        if self.is_last_layer:
            if self.norm is not None:
                hidden_states = self.norm(hidden_states)
            if self.lm_head is not None:
                hidden_states = self.lm_head(hidden_states)

        return hidden_states, cache

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

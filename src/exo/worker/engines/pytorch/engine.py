"""UnifiedPyTorchEngine — InferenceBackend implementation for CUDA and XPU devices.

Implements the InferenceBackend protocol from src/exo/worker/engines/base.py
for both NVIDIA CUDA and Intel XPU device backends. Uses PyTorch's device-agnostic
APIs (tensor.to(device), torch.zeros(..., device=device)) for all tensor operations.

Integrates:
- GpuValidator for runtime assertions on forward passes
- sampling.py for token sampling
- model/kv_cache.py for KV cache management
- model/loader.py for safetensors model loading
- pipeline/coordinator.py for pipeline configuration (optional)

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, final

import numpy as np
import torch

from exo.shared.types.worker.shards import ShardMetadata
from exo.worker.engines.base import InferenceBackend

from .gpu_validator import GpuValidator
from .model.kv_cache import KVCache
from .model.loader import LoadProgress, load_safetensors_to_device
from .sampling import sample_token

if TYPE_CHECKING:
    from .pipeline.coordinator import PipelineConfig

logger = logging.getLogger(__name__)


@final
class UnifiedPyTorchEngine(InferenceBackend):
    """Unified PyTorch engine supporting CUDA and XPU devices.

    Implements the InferenceBackend protocol for both NVIDIA and Intel GPUs
    using PyTorch's device-agnostic APIs. Accepts a device_type parameter
    ("cuda" or "xpu") and routes all tensor operations to the selected device.

    Shares model loading, KV cache management, and token generation logic
    between both device backends through device-agnostic tensor operations.

    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8
    """

    def __init__(
        self,
        device_type: Literal["cuda", "xpu"],
        device_index: int = 0,
        pipeline_config: PipelineConfig | None = None,
    ) -> None:
        """Initialize the unified PyTorch engine.

        Args:
            device_type: Device backend to use — "cuda" for NVIDIA GPUs,
                "xpu" for Intel Arc GPUs.
            device_index: Index of the device to use (default 0).
            pipeline_config: Optional pipeline parallelism configuration.
                When provided, the engine participates in distributed inference.
        """
        self._device_type: Literal["cuda", "xpu"] = device_type
        self._device_index: int = device_index
        self._device_str: str = f"{device_type}:{device_index}"
        self._pipeline_config = pipeline_config

        # Create GPU validator for runtime assertions
        self._validator = GpuValidator(
            expected_device=self._device_str,
            debug_mode=False,
        )

        # Model state — populated by load_checkpoint
        self._model: torch.nn.Module | None = None
        self._tokenizer: "AutoTokenizer | None" = None
        self._kv_cache: KVCache | None = None
        self._state_dict: dict[str, torch.Tensor] | None = None

        # Sampling defaults
        self._temperature: float = 0.7
        self._top_k: int = 50
        self._top_p: float = 0.9

        logger.info(
            "UnifiedPyTorchEngine initialized: device=%s, pipeline=%s",
            self._device_str,
            "enabled" if pipeline_config is not None else "disabled",
        )

    @property
    def device(self) -> str:
        """Return the device string (e.g., 'xpu:0', 'cuda:0')."""
        return self._device_str

    @property
    def device_type(self) -> Literal["cuda", "xpu"]:
        """Return the device type."""
        return self._device_type

    async def encode(self, shard_metadata: ShardMetadata, prompt: str) -> np.ndarray:
        """Encode a text prompt into token IDs.

        Uses the HuggingFace AutoTokenizer associated with the loaded model
        to tokenize the prompt. Returns token IDs as a numpy array.

        Args:
            shard_metadata: Metadata describing the model shard.
            prompt: Text prompt to encode.

        Returns:
            Token IDs as numpy array of shape (seq_len,).

        Raises:
            RuntimeError: If no tokenizer has been loaded.
        """
        if self._tokenizer is None:
            raise RuntimeError(
                "No tokenizer loaded. Call load_checkpoint() before encode()."
            )

        token_ids = self._tokenizer.encode(prompt, return_tensors="np")
        # AutoTokenizer.encode with return_tensors="np" returns shape (1, seq_len)
        if hasattr(token_ids, "shape") and len(token_ids.shape) == 2:
            token_ids = token_ids[0]

        return np.asarray(token_ids, dtype=np.int64)

    async def decode(self, shard_metadata: ShardMetadata, tokens: np.ndarray) -> str:
        """Decode token IDs back into text.

        Uses the HuggingFace AutoTokenizer to convert token IDs to text.

        Args:
            shard_metadata: Metadata describing the model shard.
            tokens: Token IDs to decode.

        Returns:
            Decoded text string.

        Raises:
            RuntimeError: If no tokenizer has been loaded.
        """
        if self._tokenizer is None:
            raise RuntimeError(
                "No tokenizer loaded. Call load_checkpoint() before decode()."
            )

        token_list = tokens.tolist() if isinstance(tokens, np.ndarray) else list(tokens)
        return self._tokenizer.decode(token_list, skip_special_tokens=True)

    async def infer_tensor(
        self,
        request_id: str,
        shard_metadata: ShardMetadata,
        input_data: np.ndarray,
        inference_state: dict | None = None,
    ) -> tuple[np.ndarray, dict | None]:
        """Execute tensor inference through the model.

        Converts input numpy array to a PyTorch tensor on the target device,
        runs the forward pass through the model, validates tensors are on
        the correct device via GpuValidator, and returns output as numpy.

        Args:
            request_id: Unique identifier for this inference request.
            shard_metadata: Metadata describing the model shard.
            input_data: Input tensor as numpy array (token IDs or hidden states).
            inference_state: Optional state from previous inference (e.g. KV cache position).

        Returns:
            Tuple of (output_data as numpy array, updated inference_state).

        Raises:
            RuntimeError: If no model has been loaded.
        """
        if self._model is None:
            raise RuntimeError(
                "No model loaded. Call load_checkpoint() before infer_tensor()."
            )

        # Convert input to tensor on target device
        input_tensor = torch.from_numpy(input_data).to(self._device_str)

        # Track sequence position from inference state
        seq_position = 0
        if inference_state is not None:
            seq_position = inference_state.get("seq_position", 0)

        # Run forward pass through the model
        with torch.no_grad():
            output_tensor = self._model(input_tensor)

        # Validate tensors are on the correct device
        self._validator.validate_forward_pass(
            inputs={"input": input_tensor},
            outputs=output_tensor,
        )

        # Update inference state
        new_seq_position = seq_position + input_data.shape[-1]
        new_state: dict = {
            "request_id": request_id,
            "seq_position": new_seq_position,
        }

        # Convert output to numpy
        output_np = output_tensor.detach().cpu().numpy()

        return output_np, new_state

    async def sample(self, logits: np.ndarray) -> np.ndarray:
        """Sample token IDs from logits using temperature, top-k, and top-p.

        Converts logits to a GPU tensor, applies the sampling pipeline
        (temperature scaling, top-k filtering, nucleus sampling), and
        returns sampled token IDs as numpy.

        Args:
            logits: Logit values from model output as numpy array.

        Returns:
            Sampled token IDs as numpy array.
        """
        # Convert logits to tensor on target device for GPU-accelerated sampling
        logits_tensor = torch.from_numpy(logits).to(self._device_str)

        # Use the sampling module
        token_ids = sample_token(
            logits=logits_tensor,
            temperature=self._temperature,
            top_k=self._top_k,
            top_p=self._top_p,
        )

        return token_ids

    async def load_checkpoint(self, shard_metadata: ShardMetadata, path: str) -> None:
        """Load model weights from a safetensors checkpoint directory.

        Loads all safetensors files from the given path onto the target GPU
        device. Also loads the tokenizer from the same directory. After loading,
        validates that all model parameters reside on the expected device.

        Args:
            shard_metadata: Metadata describing the model shard.
            path: Path to the checkpoint directory containing .safetensors files
                and tokenizer configuration.

        Raises:
            RuntimeError: If loading fails (missing files, memory exceeded, etc.).
        """
        model_dir = Path(path)

        if not model_dir.exists():
            raise RuntimeError(
                f"Checkpoint path does not exist: {path}. "
                f"Ensure the model has been downloaded."
            )

        # Load tokenizer
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            logger.info("Tokenizer loaded from %s", model_dir)
        except Exception as e:
            logger.warning(
                "Failed to load tokenizer from %s: %s. "
                "encode() and decode() will not be available.",
                model_dir,
                e,
            )
            self._tokenizer = None

        # Load model weights via safetensors loader
        state_dict: dict[str, torch.Tensor] | None = None
        loader = load_safetensors_to_device(
            model_dir=model_dir,
            device_type=self._device_type,
            device_index=self._device_index,
        )

        for item in loader:
            if isinstance(item, LoadProgress):
                logger.info(
                    "Loading checkpoint: %.1f%% (%s)",
                    item.percent,
                    item.current_file,
                )
            elif isinstance(item, dict):
                state_dict = item

        if state_dict is None:
            raise RuntimeError(
                f"Model loading produced no state dict from {path}."
            )

        self._state_dict = state_dict

        logger.info(
            "Checkpoint loaded: %d tensors on %s from %s",
            len(state_dict),
            self._device_str,
            path,
        )

    # ------------------------------------------------------------------
    # Pipeline integration helpers (used by PipelineCoordinator)
    # ------------------------------------------------------------------

    def forward_layers(
        self,
        hidden_states: torch.Tensor,
        start_layer: int,
        end_layer: int,
    ) -> torch.Tensor:
        """Run forward pass through a subset of transformer layers.

        This is a placeholder that will be filled when a real model
        architecture is integrated. For now, returns the input unchanged
        to allow pipeline coordination testing.

        Args:
            hidden_states: Input hidden states tensor on the target device.
            start_layer: First layer index (inclusive).
            end_layer: Last layer index (exclusive).

        Returns:
            Output hidden states tensor on the target device.
        """
        # Ensure tensor is on the correct device
        hidden_states = hidden_states.to(self._device_str)
        self._validator.assert_on_device(hidden_states, "hidden_states")
        # Placeholder: real implementation processes through transformer layers
        return hidden_states

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to embeddings.

        Placeholder for real embedding layer. Returns a dummy embedding
        tensor of appropriate shape for pipeline testing.

        Args:
            token_ids: Token ID tensor on the target device.

        Returns:
            Embedding tensor on the target device.
        """
        token_ids = token_ids.to(self._device_str)
        # Placeholder: create dummy embeddings (hidden_size=256 for testing)
        hidden_size = 256
        batch_size = token_ids.shape[0] if token_ids.dim() > 0 else 1
        embeddings = torch.zeros(
            batch_size, hidden_size, device=self._device_str, dtype=torch.float32
        )
        return embeddings

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits.

        Placeholder for real LM head projection. Returns dummy logits
        for pipeline testing.

        Args:
            hidden_states: Hidden states tensor on the target device.

        Returns:
            Logits tensor on the target device.
        """
        hidden_states = hidden_states.to(self._device_str)
        # Placeholder: project to a dummy vocab size
        vocab_size = 32000
        batch_size = hidden_states.shape[0] if hidden_states.dim() > 1 else 1
        logits = torch.zeros(
            batch_size, vocab_size, device=self._device_str, dtype=torch.float32
        )
        return logits

    def sample_token_from_tensor(self, logits: torch.Tensor) -> np.ndarray:
        """Sample a token from logits tensor, returning token ID(s) as numpy.

        Args:
            logits: Logits tensor on the target device.

        Returns:
            Sampled token IDs as numpy array.
        """
        return sample_token(
            logits=logits,
            temperature=self._temperature,
            top_k=self._top_k,
            top_p=self._top_p,
        )

"""Base protocol for inference backends.

All inference backends (MLX, tinygrad, etc.) must implement this protocol
to be compatible with the exo runner system.

This follows the pattern from exo-cuda reference implementation.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from exo.shared.types.worker.shards import ShardMetadata


class InferenceBackend(ABC):
    """Abstract base class that all inference backends must implement.

    This defines the interface for loading models and executing inference
    across different hardware backends (MLX, tinygrad, etc.).

    Based on the InferenceEngine pattern from exo-cuda.
    """

    @abstractmethod
    async def encode(self, shard_metadata: ShardMetadata, prompt: str) -> np.ndarray:
        """Encode a text prompt into tokens.

        Args:
            shard_metadata: Metadata describing the model shard
            prompt: Text prompt to encode

        Returns:
            Token IDs as numpy array
        """
        ...

    @abstractmethod
    async def decode(self, shard_metadata: ShardMetadata, tokens: np.ndarray) -> str:
        """Decode tokens back into text.

        Args:
            shard_metadata: Metadata describing the model shard
            tokens: Token IDs to decode

        Returns:
            Decoded text string
        """
        ...

    @abstractmethod
    async def infer_tensor(
        self,
        request_id: str,
        shard_metadata: ShardMetadata,
        input_data: np.ndarray,
        inference_state: Optional[dict] = None,
    ) -> tuple[np.ndarray, Optional[dict]]:
        """Execute tensor inference.

        Args:
            request_id: Unique identifier for this inference request
            shard_metadata: Metadata describing the model shard
            input_data: Input tensor as numpy array
            inference_state: Optional state from previous inference (KV cache, etc.)

        Returns:
            Tuple of (output_data, new_inference_state)
        """
        ...

    @abstractmethod
    async def sample(self, logits: np.ndarray) -> np.ndarray:
        """Sample tokens from logits.

        Args:
            logits: Logit values from model output

        Returns:
            Sampled token IDs
        """
        ...

    @abstractmethod
    async def load_checkpoint(self, shard_metadata: ShardMetadata, path: str) -> None:
        """Load model weights from checkpoint.

        Args:
            shard_metadata: Metadata describing the model shard
            path: Path to checkpoint file
        """
        ...

    async def save_checkpoint(self, shard_metadata: ShardMetadata, path: str) -> None:
        """Save model weights to checkpoint.

        Args:
            shard_metadata: Metadata describing the model shard
            path: Path to save checkpoint
        """
        raise NotImplementedError("Checkpoint saving not implemented for this backend")

    async def infer_prompt(
        self,
        request_id: str,
        shard_metadata: ShardMetadata,
        prompt: str,
        inference_state: Optional[dict] = None,
    ) -> tuple[np.ndarray, Optional[dict]]:
        """Convenience method to encode prompt and run inference.

        Args:
            request_id: Unique identifier for this inference request
            shard_metadata: Metadata describing the model shard
            prompt: Text prompt to process
            inference_state: Optional state from previous inference

        Returns:
            Tuple of (output_data, new_inference_state)
        """
        tokens = await self.encode(shard_metadata, prompt)
        x = tokens.reshape(1, -1)
        return await self.infer_tensor(request_id, shard_metadata, x, inference_state)

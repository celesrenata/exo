"""Tinygrad inference backend implementation.

This module implements the InferenceBackend protocol using tinygrad,
with support for Intel Arc GPU acceleration via Level Zero and OpenCL.
"""

import os
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
from exo.worker.engines.base import InferenceBackend
from exo.worker.engines.metrics import BackendMetricsCollector
from exo.worker.engines.tinygrad.device_config import (
    DeviceCapabilities,
    detect_capabilities,
)


class TinygradBackend(InferenceBackend):
    """Tinygrad-based inference backend with Intel Arc GPU support.

    This backend uses tinygrad for model inference and supports multiple
    hardware targets including Intel Arc iGPUs via Level Zero and OpenCL.

    Attributes:
        shard_downloader: Downloader for fetching model weights
        device_capabilities: Detected hardware capabilities
        device: Configured device type (CPU, GPU, METAL)
        runtime: Active GPU runtime (LEVEL_ZERO, OPENCL, etc.)
        model: Loaded model instance
        tokenizer: Loaded tokenizer instance

    Example:
        >>> from exo.download.shard_download import ShardDownloader
        >>> downloader = ShardDownloader()
        >>> backend = TinygradBackend(downloader)
        >>> # Backend will auto-detect Intel Arc and configure Level Zero
    """

    def __init__(self, shard_downloader: Any):
        """Initialize tinygrad backend.

        Args:
            shard_downloader: ShardDownloader instance for fetching weights
        """
        self.shard_downloader = shard_downloader
        self.device_capabilities: DeviceCapabilities | None = None
        self.device: str = "CPU"
        self.runtime: str | None = None
        self.model: Any = None
        self.tokenizer: Any = None
        self.metrics_collector: BackendMetricsCollector | None = None

        logger.info(
            "TinygradBackend initialized",
            backend_type="tinygrad",
            status="initializing",
        )

    def _configure_device(self) -> None:
        """Configure tinygrad device based on available hardware.

        This method detects available hardware (Intel Arc GPU, CPU, etc.)
        and configures tinygrad to use the most capable device. It sets
        appropriate environment variables for tinygrad's device selection.

        The configuration priority is:
        1. Intel Arc GPU with Level Zero
        2. GPU with OpenCL
        3. NVIDIA GPU with CUDA
        4. Apple Metal (on macOS)
        5. CPU (fallback)
        """
        # Detect hardware capabilities
        self.device_capabilities = detect_capabilities()

        logger.info(
            "Device capabilities detected",
            backend_type="tinygrad",
            device_name=self.device_capabilities.device_name,
            device_type=self.device_capabilities.device_type,
            memory_gb=round(self.device_capabilities.memory_gb, 2),
            compute_units=self.device_capabilities.compute_units,
            runtime=self.device_capabilities.runtime,
        )

        runtime = self.device_capabilities.runtime
        if runtime:
            logger.info(
                "Runtime selected",
                backend_type="tinygrad",
                runtime=runtime,
                device_name=self.device_capabilities.device_name,
            )

        # Configure tinygrad based on detected capabilities
        if self.device_capabilities.device_type == "GPU":
            self._configure_gpu()
        elif self.device_capabilities.device_type == "METAL":
            self._configure_metal()
        else:
            self._configure_cpu()

        logger.info(
            "Tinygrad device configured",
            backend_type="tinygrad",
            device=self.device,
            runtime=self.runtime,
            device_name=self.device_capabilities.device_name,
            memory_gb=round(self.device_capabilities.memory_gb, 1),
        )

    def _configure_gpu(self) -> None:
        """Configure tinygrad for GPU execution.

        Sets environment variables to enable GPU execution with the
        appropriate runtime (Level Zero, OpenCL, or CUDA).
        """
        if not self.device_capabilities:
            return

        self.device = "GPU"
        self.runtime = self.device_capabilities.runtime

        # Enable GPU in tinygrad
        os.environ["GPU"] = "1"

        # Configure specific runtime
        if self.runtime == "LEVEL_ZERO":
            os.environ["LEVEL_ZERO"] = "1"
            logger.info(
                "GPU runtime configured: Level Zero (optimal for Intel Arc)",
                backend_type="tinygrad",
                device="GPU",
                runtime="LEVEL_ZERO",
                device_name=self.device_capabilities.device_name,
                recommendation="Level Zero provides optimal performance for Intel Arc GPUs",
            )
        elif self.runtime == "OPENCL":
            os.environ["OPENCL"] = "1"
            if "Intel Arc" in self.device_capabilities.device_name:
                logger.info(
                    "GPU runtime configured: OpenCL (fallback for Intel Arc)",
                    backend_type="tinygrad",
                    device="GPU",
                    runtime="OPENCL",
                    device_name=self.device_capabilities.device_name,
                    recommendation="Level Zero is recommended for better performance on Intel Arc",
                )
            else:
                logger.info(
                    "GPU runtime configured: OpenCL",
                    backend_type="tinygrad",
                    device="GPU",
                    runtime="OPENCL",
                    device_name=self.device_capabilities.device_name,
                )
        elif self.runtime == "CUDA":
            os.environ["CUDA"] = "1"
            logger.info(
                "GPU runtime configured: CUDA",
                backend_type="tinygrad",
                device="GPU",
                runtime="CUDA",
                device_name=self.device_capabilities.device_name,
            )

    def _configure_metal(self) -> None:
        """Configure tinygrad for Apple Metal execution."""
        if not self.device_capabilities:
            return

        self.device = "METAL"
        self.runtime = "METAL"
        os.environ["METAL"] = "1"
        logger.info(
            "GPU runtime configured: Metal",
            backend_type="tinygrad",
            device="METAL",
            runtime="METAL",
            device_name=self.device_capabilities.device_name,
        )

    def _configure_cpu(self) -> None:
        """Configure tinygrad for CPU execution."""
        if not self.device_capabilities:
            return

        self.device = "CPU"
        self.runtime = None
        # Ensure GPU is disabled
        os.environ.pop("GPU", None)
        os.environ.pop("METAL", None)
        logger.info(
            "CPU execution configured (GPU unavailable or not requested)",
            backend_type="tinygrad",
            device="CPU",
            device_name=self.device_capabilities.device_name,
        )

    async def encode(
        self, shard_metadata: ShardMetadata, prompt: str
    ) -> "np.ndarray[Any, Any]":
        """Encode a text prompt into tokens.

        Args:
            shard_metadata: Metadata describing the model shard
            prompt: Text prompt to encode

        Returns:
            Token IDs as numpy array

        Raises:
            RuntimeError: If tokenizer is not loaded
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_checkpoint first.")

        # Use tokenizer to encode prompt
        # This will be implemented in model_loader.py
        from exo.worker.engines.tinygrad.model_loader import encode_prompt

        return await encode_prompt(self.tokenizer, prompt)

    async def decode(
        self, shard_metadata: ShardMetadata, tokens: "np.ndarray[Any, Any]"
    ) -> str:
        """Decode tokens back into text.

        Args:
            shard_metadata: Metadata describing the model shard
            tokens: Token IDs to decode

        Returns:
            Decoded text string

        Raises:
            RuntimeError: If tokenizer is not loaded
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_checkpoint first.")

        # Use tokenizer to decode tokens
        from exo.worker.engines.tinygrad.model_loader import decode_tokens

        return await decode_tokens(self.tokenizer, tokens)

    async def infer_tensor(
        self,
        request_id: str,
        shard_metadata: ShardMetadata,
        input_data: "np.ndarray[Any, Any]",
        inference_state: dict[str, Any] | None = None,
    ) -> tuple["np.ndarray[Any, Any]", dict[str, Any] | None]:
        """Execute tensor inference.

        Args:
            request_id: Unique identifier for this inference request
            shard_metadata: Metadata describing the model shard
            input_data: Input tensor as numpy array
            inference_state: Optional state from previous inference (KV cache, etc.)

        Returns:
            Tuple of (output_data, new_inference_state)

        Raises:
            RuntimeError: If model is not loaded
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint first.")

        # Execute inference using tinygrad
        # This will be implemented in generator.py
        from exo.worker.engines.tinygrad.generator import infer_tensor

        return await infer_tensor(
            self.model,
            input_data,
            inference_state,
            self.device,
        )

    async def sample(self, logits: "np.ndarray[Any, Any]") -> "np.ndarray[Any, Any]":
        """Sample tokens from logits.

        Args:
            logits: Logit values from model output

        Returns:
            Sampled token IDs
        """
        # Simple argmax sampling for now
        # Can be extended with temperature, top-k, top-p, etc.
        return np.argmax(logits, axis=-1)

    async def load_checkpoint(self, shard_metadata: ShardMetadata, path: str) -> None:
        """Load model weights from checkpoint.

        This method loads model weights and tokenizer from the specified path,
        converting them to tinygrad format if necessary.

        Args:
            shard_metadata: Metadata describing the model shard
            path: Path to checkpoint directory

        Raises:
            FileNotFoundError: If checkpoint path does not exist
            RuntimeError: If model loading fails
        """
        logger.info(f"Loading checkpoint from {path}")

        # Configure device before loading model
        if self.device_capabilities is None:
            self._configure_device()

        # Load model and tokenizer
        # This will be implemented in model_loader.py
        from exo.worker.engines.tinygrad.model_loader import load_tinygrad_model

        self.model, self.tokenizer = await load_tinygrad_model(
            shard_metadata,
            path,
            self.device,
        )

        logger.info(f"Successfully loaded model on {self.device}")

    def initialize_metrics_collector(self, runner_id: Any) -> None:
        """Initialize metrics collector for this backend.

        Args:
            runner_id: RunnerId for the runner using this backend

        Example:
            >>> backend.initialize_metrics_collector(runner_id)
            >>> # Metrics will now be collected during inference
        """
        if self.device_capabilities is None:
            self._configure_device()

        self.metrics_collector = BackendMetricsCollector(
            runner_id=runner_id,
            backend_type="tinygrad",
            device_type=self.device,
        )

        logger.info(
            "Metrics collector initialized",
            runner_id=str(runner_id),
            backend_type="tinygrad",
            device_type=self.device,
        )

    def get_metrics_stats(self) -> dict[str, Any]:
        """Get current metrics statistics.

        Returns:
            Dictionary of aggregated metrics statistics

        Example:
            >>> stats = backend.get_metrics_stats()
            >>> print(f"Avg throughput: {stats['avg_tokens_per_second']:.2f}")
        """
        if self.metrics_collector is None:
            return {}

        return self.metrics_collector.get_stats()

    async def save_checkpoint(self, shard_metadata: ShardMetadata, path: str) -> None:
        """Save model weights to checkpoint.

        Args:
            shard_metadata: Metadata describing the model shard
            path: Path to save checkpoint

        Raises:
            RuntimeError: If model is not loaded
        """
        if self.model is None:
            raise RuntimeError("No model loaded to save")

        logger.info(f"Saving checkpoint to {path}")

        # Save model weights
        # This will be implemented in model_loader.py
        from exo.worker.engines.tinygrad.model_loader import save_model

        await save_model(self.model, path)

        logger.info("Checkpoint saved successfully")

    def collect_metrics(self) -> dict[str, Any]:
        """Collect current GPU performance metrics.

        Returns:
            Dictionary containing GPU metrics (memory, utilization, etc.)

        Example:
            >>> backend = TinygradBackend(downloader)
            >>> backend._configure_device()
            >>> metrics = backend.collect_metrics()
            >>> print(f"Memory: {metrics['memory_used_mb']}/{metrics['memory_total_mb']} MB")
        """
        if self.device != "GPU" and self.device != "METAL":
            # No metrics for CPU
            return {}

        if not self.device_capabilities or not self.runtime:
            return {}

        from exo.worker.engines.tinygrad.metrics import collect_gpu_metrics

        try:
            metrics = collect_gpu_metrics(
                self.device_capabilities.device_name,
                self.runtime,  # type: ignore
            )

            return {
                "device_name": metrics.device_name,
                "runtime": metrics.runtime,
                "memory_used_mb": metrics.memory_used_mb,
                "memory_total_mb": metrics.memory_total_mb,
                "utilization_percent": metrics.utilization_percent,
                "timestamp": metrics.timestamp,
            }
        except Exception as e:
            logger.debug(f"Failed to collect GPU metrics: {e}")
            return {}

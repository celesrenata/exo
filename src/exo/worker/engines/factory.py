"""Factory for creating inference backend instances.

This module provides a factory function to instantiate the appropriate
backend based on configuration. Adapted from exo-cuda reference implementation.
"""

from typing import Any

from exo.worker.engines.base import InferenceBackend

# Type alias for backend names
BackendName = str  # "mlx", "tinygrad", etc.


class BackendNotAvailableError(Exception):
    """Raised when a requested backend is not available on this platform."""

    def __init__(self, backend_name: str, reason: str):
        self.backend_name = backend_name
        self.reason = reason
        super().__init__(f"Backend '{backend_name}' not available: {reason}")


def get_inference_backend(
    backend_name: str,
    shard_downloader: Any,
) -> InferenceBackend:
    """Factory function to create appropriate inference backend.

    This follows the pattern from exo-cuda's get_inference_engine function.

    Args:
        backend_name: Name of backend to create ("mlx", "tinygrad", "dummy")
        shard_downloader: ShardDownloader instance for downloading model weights

    Returns:
        An instance implementing the InferenceBackend interface

    Raises:
        BackendNotAvailableError: If the requested backend is not available
        ValueError: If backend_name is unknown

    Example:
        >>> from exo.download.shard_downloader import ShardDownloader
        >>> downloader = ShardDownloader()
        >>> backend = get_inference_backend("tinygrad", downloader)
    """
    if backend_name == "mlx":
        try:
            from exo.worker.engines.mlx.mlx_backend import MLXBackend

            return MLXBackend(shard_downloader)
        except ImportError as e:
            raise BackendNotAvailableError(
                "mlx", f"MLX backend not available: {e}"
            ) from e

    elif backend_name == "pytorch_xpu":
        try:
            from exo.worker.engines.pytorch_xpu.pytorch_xpu_backend import (
                PyTorchXPUBackend,
            )

            return PyTorchXPUBackend(shard_downloader)
        except ImportError as e:
            raise BackendNotAvailableError(
                "pytorch_xpu", f"PyTorch XPU backend not available: {e}"
            ) from e

    elif backend_name == "pytorch":
        try:
            from exo.worker.engines.pytorch.device_detector import select_primary_device
            from exo.worker.engines.pytorch.engine import UnifiedPyTorchEngine

            device = select_primary_device()
            return UnifiedPyTorchEngine(
                device_type=device.device_type,
                device_index=device.device_index,
            )
        except ImportError as e:
            raise BackendNotAvailableError(
                "pytorch", f"Unified PyTorch backend not available: {e}"
            ) from e

    elif backend_name == "dummy":
        try:
            from exo.worker.engines.dummy import DummyBackend

            return DummyBackend()
        except ImportError as e:
            raise BackendNotAvailableError(
                "dummy", f"Dummy backend not available: {e}"
            ) from e

    else:
        raise ValueError(f"Unknown backend: {backend_name}")


# Registry of available backends
BACKEND_REGISTRY = {
    "mlx": "MLXBackend",
    "pytorch": "UnifiedPyTorchEngine",
    "pytorch_xpu": "PyTorchXPUBackend",
    "dummy": "DummyBackend",
}

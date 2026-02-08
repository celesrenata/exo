"""Factory for creating inference backend instances.

This module provides a factory function to instantiate the appropriate
backend based on configuration. Adapted from exo-cuda reference implementation.
"""

import os
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

    elif backend_name == "tinygrad":
        try:
            import tinygrad.helpers

            from exo.worker.engines.tinygrad.tinygrad_backend import (
                TinygradBackend,
            )

            # Set tinygrad debug level from environment
            tinygrad.helpers.DEBUG.value = int(os.getenv("TINYGRAD_DEBUG", default="0"))

            return TinygradBackend(shard_downloader)
        except ImportError as e:
            raise BackendNotAvailableError(
                "tinygrad", f"tinygrad backend not available: {e}"
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
    "tinygrad": "TinygradBackend",
    "dummy": "DummyBackend",
}

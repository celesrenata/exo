"""Backend selection and fallback logic.

This module provides functions to select the appropriate inference backend
based on configuration, environment variables, and hardware availability.
It implements a fallback chain to ensure inference works even when the
preferred backend is unavailable.
"""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from exo.download.shard_downloader import ShardDownloader
else:
    ShardDownloader = Any

from exo.shared.constants import EXO_TINYGRAD_ENABLED
from exo.shared.types.worker.shards import ShardMetadata
from exo.worker.engines.base import InferenceBackend
from exo.worker.engines.factory import BackendNotAvailableError, get_inference_backend

logger = logging.getLogger(__name__)


class BackendInitializationError(Exception):
    """Raised when backend fails to initialize.

    Attributes:
        backend_type: Name of the backend that failed
        reason: Description of why initialization failed
        can_fallback: Whether fallback to another backend is possible
    """

    def __init__(
        self, backend_type: str, reason: str, can_fallback: bool = True
    ) -> None:
        self.backend_type = backend_type
        self.reason = reason
        self.can_fallback = can_fallback
        super().__init__(f"{backend_type} initialization failed: {reason}")


def get_fallback_chain(preferred: str) -> list[str]:
    """Get ordered list of backends to try.

    The fallback chain ensures that inference can proceed even if the
    preferred backend is unavailable. The chain is:
    - tinygrad → mlx (on macOS/Apple Silicon)
    - npu → tinygrad → mlx
    - mlx (no fallback, it's the baseline)

    Args:
        preferred: Preferred backend name ("mlx", "tinygrad", "npu")

    Returns:
        List of backend names to try in order

    Example:
        >>> get_fallback_chain("tinygrad")
        ["tinygrad", "mlx"]
        >>> get_fallback_chain("mlx")
        ["mlx"]
    """
    if preferred == "tinygrad":
        return ["tinygrad", "mlx"]  # Try tinygrad, fall back to MLX
    elif preferred == "npu":
        return ["npu", "tinygrad", "mlx"]  # NPU → tinygrad → MLX
    else:
        return [preferred]  # No fallback for MLX (it's the baseline)


def select_backend_from_config(
    shard_metadata: ShardMetadata,
) -> str:
    """Select backend based on shard metadata and environment variables.

    This function determines which backend to use based on:
    1. Shard metadata (if backend is specified)
    2. Environment variables (EXO_TINYGRAD_ENABLED)
    3. Platform defaults (MLX on macOS, tinygrad elsewhere if enabled)

    Args:
        shard_metadata: Metadata describing the model shard

    Returns:
        Backend name to use ("mlx", "tinygrad", "npu")

    Example:
        >>> # With EXO_TINYGRAD_ENABLED=true
        >>> select_backend_from_config(shard_metadata)
        "tinygrad"
        >>> # Without EXO_TINYGRAD_ENABLED (default)
        >>> select_backend_from_config(shard_metadata)
        "mlx"
    """
    # Check if backend is specified in shard metadata
    # Use hasattr since backend attribute may not exist
    if hasattr(shard_metadata, "backend"):
        backend_attr = getattr(shard_metadata, "backend", None)
        if backend_attr:
            logger.info(f"Using backend from shard metadata: {backend_attr}")
            return str(backend_attr)  # pyright: ignore[reportAny]

    # Check environment variable directly (not cached constant)
    # This allows runtime changes to the environment variable
    import os
    if os.environ.get("EXO_TINYGRAD_ENABLED", "false").lower() == "true":
        logger.info("EXO_TINYGRAD_ENABLED=true, using tinygrad backend")
        return "tinygrad"

    # Default to MLX (current behavior)
    logger.info("Using default MLX backend")
    return "mlx"


def initialize_backend_with_fallback(
    preferred_backend: str,
    shard_downloader: "ShardDownloader",
) -> tuple[InferenceBackend, str]:
    """Initialize backend with automatic fallback.

    This function attempts to initialize the preferred backend, and if that
    fails, tries fallback backends in order until one succeeds.

    Args:
        preferred_backend: Preferred backend name
        shard_downloader: ShardDownloader instance for fetching weights

    Returns:
        Tuple of (initialized_backend, backend_name)

    Raises:
        RuntimeError: If all backends in the fallback chain fail

    Example:
        >>> backend, name = initialize_backend_with_fallback("tinygrad", downloader)
        >>> print(f"Using {name} backend")
        Using tinygrad backend
    """
    fallback_chain = get_fallback_chain(preferred_backend)
    last_error: Exception | None = None

    for backend_type in fallback_chain:
        try:
            logger.info(f"Attempting to initialize {backend_type} backend")
            backend = get_inference_backend(backend_type, shard_downloader)
            logger.info(f"Successfully initialized {backend_type} backend")
            return backend, backend_type
        except BackendNotAvailableError as e:
            logger.warning(
                f"Backend {backend_type} not available: {e.reason}. "
                f"Trying next backend in fallback chain."
            )
            last_error = e
            continue
        except Exception as e:
            logger.error(
                f"Unexpected error initializing {backend_type} backend: {e}",
                exc_info=True,
            )
            last_error = e
            continue

    # All backends failed
    error_msg = (
        f"All backends in fallback chain {fallback_chain} failed to initialize. "
        f"Last error: {last_error}"
    )
    logger.error(error_msg)
    raise RuntimeError(error_msg)


def get_backend_device_info(backend: InferenceBackend) -> dict[str, str | None]:
    """Extract device information from backend for observability.

    This function extracts device information from the backend to be
    included in BackendInitialized events for dashboard visibility.

    Args:
        backend: Initialized backend instance

    Returns:
        Dictionary with device_type, device_name, and runtime keys

    Example:
        >>> info = get_backend_device_info(tinygrad_backend)
        >>> print(info)
        {
            "device_type": "GPU",
            "device_name": "Intel Arc Graphics",
            "runtime": "LEVEL_ZERO"
        }
    """
    # Try to get device info from tinygrad backend
    # Use hasattr to check for attributes since they're not in the protocol
    if hasattr(backend, "device_capabilities"):
        caps = getattr(backend, "device_capabilities", None)
        if caps is not None:
            return {
                "device_type": getattr(caps, "device_type", "Unknown"),  # pyright: ignore[reportAny]
                "device_name": getattr(caps, "device_name", "Unknown"),  # pyright: ignore[reportAny]
                "runtime": getattr(caps, "runtime", None),  # pyright: ignore[reportAny]
            }

    # Try to get device info from tinygrad backend attributes
    if hasattr(backend, "device") and hasattr(backend, "runtime"):
        device_name = "Unknown"
        if hasattr(backend, "device_capabilities"):
            caps = getattr(backend, "device_capabilities", None)
            if caps is not None:
                device_name = getattr(caps, "device_name", "Unknown")  # pyright: ignore[reportAny]

        return {
            "device_type": str(getattr(backend, "device", "Unknown")),
            "device_name": device_name,
            "runtime": str(getattr(backend, "runtime", None))
            if getattr(backend, "runtime", None)
            else None,
        }

    # Default for MLX or other backends
    return {
        "device_type": "METAL",  # MLX uses Metal on macOS
        "device_name": "Apple Silicon",
        "runtime": "METAL",
    }

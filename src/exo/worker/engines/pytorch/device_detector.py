"""GPU detection and selection for CUDA and XPU device backends.

Detects available GPUs using native PyTorch APIs and selects the primary
inference device. Prefers CUDA over XPU when both are available. Raises
a fatal error if no GPU is detected — CPU inference is never acceptable.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

# Keywords in Intel XPU device names that indicate integrated (shared-memory) GPUs.
_INTEL_INTEGRATED_KEYWORDS: tuple[str, ...] = (
    "integrated",
    "iris",
    "uhd",
    "core ultra",
    "meteor lake",
)


@dataclass(frozen=True)
class DetectedDevice:
    """A detected GPU device with its properties."""

    device_type: Literal["cuda", "xpu"]
    device_index: int
    device_name: str
    memory_bytes: int
    memory_architecture: Literal["shared", "discrete"]


class NoGpuError(RuntimeError):
    """Raised when no GPU device is available and inference cannot proceed."""


def _classify_xpu_memory_architecture(device_name: str) -> Literal["shared", "discrete"]:
    """Classify an Intel XPU device as shared or discrete based on its name.

    Intel Meteor Lake-P Arc iGPUs use shared system memory.
    Intel discrete GPUs (Arc A-series, Arc B-series) have dedicated VRAM.
    """
    name_lower = device_name.lower()
    for keyword in _INTEL_INTEGRATED_KEYWORDS:
        if keyword in name_lower:
            return "shared"
    # Discrete Intel GPUs have "Arc A" or "Arc B" in their name
    if "arc a" in name_lower or "arc b" in name_lower:
        return "discrete"
    # Default to shared for unrecognized Intel XPU devices since the gremlin
    # cluster uses integrated GPUs and over-estimating memory is worse than
    # under-estimating.
    return "shared"


def _detect_cuda_devices() -> list[DetectedDevice]:
    """Detect NVIDIA GPUs using torch.cuda."""
    try:
        import torch
    except ImportError:
        logger.debug("PyTorch not available, skipping CUDA detection")
        return []

    if not torch.cuda.is_available():
        return []

    devices: list[DetectedDevice] = []
    device_count = torch.cuda.device_count()

    for i in range(device_count):
        try:
            props = torch.cuda.get_device_properties(i)
            total_memory: int = props.total_memory

            device = DetectedDevice(
                device_type="cuda",
                device_index=i,
                device_name=props.name,
                memory_bytes=total_memory,
                memory_architecture="discrete",
            )
            devices.append(device)
            logger.info(
                "Detected CUDA device %d: %s (%.2f GiB)",
                i,
                props.name,
                total_memory / (1024**3),
            )
        except Exception:
            logger.warning("Failed to query CUDA device %d", i, exc_info=True)

    return devices


def _detect_xpu_devices() -> list[DetectedDevice]:
    """Detect Intel Arc GPUs using torch.xpu (PyTorch 2.11+)."""
    try:
        import torch
    except ImportError:
        logger.debug("PyTorch not available, skipping XPU detection")
        return []

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        return []

    devices: list[DetectedDevice] = []
    device_count: int = torch.xpu.device_count()

    for i in range(device_count):
        try:
            props = torch.xpu.get_device_properties(i)
            device_name: str = props.name
            architecture = _classify_xpu_memory_architecture(device_name)

            if architecture == "shared":
                # Shared-memory GPU uses system RAM as its memory pool.
                import psutil

                total_memory = psutil.virtual_memory().total
            else:
                total_memory = props.total_memory

            device = DetectedDevice(
                device_type="xpu",
                device_index=i,
                device_name=device_name,
                memory_bytes=total_memory,
                memory_architecture=architecture,
            )
            devices.append(device)
            logger.info(
                "Detected XPU device %d: %s (architecture=%s, %.2f GiB)",
                i,
                device_name,
                architecture,
                total_memory / (1024**3),
            )
        except Exception:
            logger.warning("Failed to query XPU device %d", i, exc_info=True)

    return devices


def detect_devices() -> list[DetectedDevice]:
    """Detect all available GPU devices on this node.

    Queries torch.cuda and torch.xpu for available devices. Returns a list
    of all detected devices (CUDA and XPU combined).

    Returns:
        List of detected GPU devices. May be empty if no GPUs are found.
    """
    all_devices: list[DetectedDevice] = []

    try:
        cuda_devices = _detect_cuda_devices()
        all_devices.extend(cuda_devices)
    except Exception:
        logger.warning("CUDA device detection failed", exc_info=True)

    try:
        xpu_devices = _detect_xpu_devices()
        all_devices.extend(xpu_devices)
    except Exception:
        logger.warning("XPU device detection failed", exc_info=True)

    logger.info(
        "Device detection complete: %d CUDA, %d XPU device(s) found",
        sum(1 for d in all_devices if d.device_type == "cuda"),
        sum(1 for d in all_devices if d.device_type == "xpu"),
    )

    return all_devices


def select_primary_device() -> DetectedDevice:
    """Select the primary GPU device for inference.

    Prefers CUDA over XPU when both are available. Raises NoGpuError if
    no GPU device is detected — CPU inference is never acceptable.

    Returns:
        The selected primary device.

    Raises:
        NoGpuError: If no CUDA or XPU device is available.
    """
    devices = detect_devices()

    if not devices:
        raise NoGpuError(
            "No GPU devices detected. Neither torch.cuda nor torch.xpu reported "
            "available devices. CPU inference is not supported — a CUDA or XPU "
            "GPU is required."
        )

    # Prefer CUDA over XPU
    cuda_devices = [d for d in devices if d.device_type == "cuda"]
    if cuda_devices:
        primary = cuda_devices[0]
    else:
        primary = devices[0]

    logger.info(
        "Selected primary device: %s — %s (index=%d, memory=%.2f GiB, architecture=%s)",
        primary.device_type.upper(),
        primary.device_name,
        primary.device_index,
        primary.memory_bytes / (1024**3),
        primary.memory_architecture,
    )

    return primary

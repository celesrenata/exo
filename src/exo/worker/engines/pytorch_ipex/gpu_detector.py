"""
GPU Detector for Distributed Inference

Detects NVIDIA and Intel Arc GPUs using native PyTorch APIs (no IPEX dependency).
Classifies memory architecture as Shared (Intel integrated GPU sharing system RAM)
or Discrete (NVIDIA or Intel discrete GPU with dedicated VRAM).

Falls back to CPU-only reporting if no GPU is detected or detection fails.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.8, 11.1, 11.2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Literal

logger = logging.getLogger(__name__)

# Keywords in Intel XPU device names that indicate integrated (shared-memory) GPUs.
# Intel discrete GPUs typically contain "Arc A" series names (e.g., "Arc A770").
_INTEL_INTEGRATED_KEYWORDS: tuple[str, ...] = (
    "integrated",
    "iris",
    "uhd",
    "core ultra",
)


class GpuMemoryArchitecture(str, Enum):
    """Memory architecture classification for GPU devices."""

    Shared = "Shared"  # Intel integrated GPU — shares system RAM
    Discrete = "Discrete"  # NVIDIA or Intel discrete — dedicated VRAM


@dataclass(frozen=True)
class GpuInfo:
    """Detected GPU information for a single device."""

    name: str
    device_type: Literal["cuda", "xpu"]
    device_index: int
    memory_architecture: GpuMemoryArchitecture
    total_memory_bytes: int
    available_memory_bytes: int


@dataclass(frozen=True)
class NodeGpuReport:
    """Complete GPU report for a node."""

    gpus: list[GpuInfo]
    has_gpu: bool
    primary_device_type: Literal["cuda", "xpu", "cpu"]


def _cpu_only_report() -> NodeGpuReport:
    """Return a CPU-only report with no GPUs detected."""
    return NodeGpuReport(gpus=[], has_gpu=False, primary_device_type="cpu")


def _classify_xpu_memory_architecture(device_name: str) -> GpuMemoryArchitecture:
    """Classify an Intel XPU device as Shared or Discrete based on its name.

    Intel integrated GPUs (UHD, Iris, Core Ultra integrated) share system RAM.
    Intel discrete GPUs (Arc A-series) have dedicated VRAM.
    """
    name_lower = device_name.lower()
    for keyword in _INTEL_INTEGRATED_KEYWORDS:
        if keyword in name_lower:
            return GpuMemoryArchitecture.Shared
    # If the device name doesn't match any integrated keyword, check for
    # the absence of discrete indicators. For safety, default to Shared
    # for unrecognized Intel XPU devices since the gremlin cluster uses
    # integrated GPUs and over-estimating available memory is worse than
    # under-estimating.
    if "arc a" in name_lower or "arc b" in name_lower:
        return GpuMemoryArchitecture.Discrete
    return GpuMemoryArchitecture.Shared


def _detect_cuda_gpus() -> list[GpuInfo]:
    """Detect NVIDIA GPUs using torch.cuda."""
    import torch

    if not torch.cuda.is_available():
        return []

    gpus: list[GpuInfo] = []
    device_count = torch.cuda.device_count()

    for i in range(device_count):
        try:
            props = torch.cuda.get_device_properties(i)
            total_memory: int = props.total_mem
            allocated: int = torch.cuda.memory_allocated(i)
            available_memory = total_memory - allocated

            gpu = GpuInfo(
                name=props.name,
                device_type="cuda",
                device_index=i,
                memory_architecture=GpuMemoryArchitecture.Discrete,
                total_memory_bytes=total_memory,
                available_memory_bytes=available_memory,
            )
            gpus.append(gpu)
            logger.info(
                "Detected NVIDIA GPU %d: %s, %.2f GiB total, %.2f GiB available",
                i,
                props.name,
                total_memory / (1024**3),
                available_memory / (1024**3),
            )
        except Exception:
            logger.warning("Failed to get properties for CUDA device %d", i, exc_info=True)

    return gpus


def _detect_xpu_gpus() -> list[GpuInfo]:
    """Detect Intel Arc GPUs using native torch.xpu (PyTorch 2.11+, no IPEX)."""
    import torch

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        return []

    import psutil

    system_ram_bytes = psutil.virtual_memory().total

    gpus: list[GpuInfo] = []
    device_count: int = torch.xpu.device_count()

    for i in range(device_count):
        try:
            props = torch.xpu.get_device_properties(i)
            device_name: str = props.name
            architecture = _classify_xpu_memory_architecture(device_name)

            if architecture == GpuMemoryArchitecture.Shared:
                # Shared-memory GPU uses system RAM as its memory pool.
                total_memory = system_ram_bytes
                vm = psutil.virtual_memory()
                available_memory = vm.available
            else:
                # Discrete Intel GPU has its own VRAM.
                total_memory = props.total_memory
                allocated: int = torch.xpu.memory_allocated(i)
                available_memory = total_memory - allocated

            gpu = GpuInfo(
                name=device_name,
                device_type="xpu",
                device_index=i,
                memory_architecture=architecture,
                total_memory_bytes=total_memory,
                available_memory_bytes=available_memory,
            )
            gpus.append(gpu)
            logger.info(
                "Detected Intel XPU %d: %s, architecture=%s, %.2f GiB total, %.2f GiB available",
                i,
                device_name,
                architecture.value,
                total_memory / (1024**3),
                available_memory / (1024**3),
            )
        except Exception:
            logger.warning("Failed to get properties for XPU device %d", i, exc_info=True)

    return gpus


def detect_gpus() -> NodeGpuReport:
    """Detect all GPUs on this Linux node using native PyTorch APIs.

    Uses torch.cuda for NVIDIA, torch.xpu for Intel Arc.
    Classifies Intel integrated GPUs as Shared memory architecture.
    Falls back to CPU-only if no GPU detected or detection fails.

    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.8, 11.1, 11.2
    """
    all_gpus: list[GpuInfo] = []

    # Detect NVIDIA GPUs
    try:
        cuda_gpus = _detect_cuda_gpus()
        all_gpus.extend(cuda_gpus)
    except Exception:
        logger.warning("NVIDIA GPU detection failed", exc_info=True)

    # Detect Intel XPU GPUs (native PyTorch 2.11+, no IPEX)
    try:
        xpu_gpus = _detect_xpu_gpus()
        all_gpus.extend(xpu_gpus)
    except Exception:
        logger.warning("Intel XPU GPU detection failed", exc_info=True)

    if not all_gpus:
        logger.info("No GPUs detected, reporting CPU-only node")
        return _cpu_only_report()

    # Determine primary device type: prefer CUDA over XPU if both present.
    cuda_present = any(gpu.device_type == "cuda" for gpu in all_gpus)
    primary_device_type: Literal["cuda", "xpu"] = "cuda" if cuda_present else "xpu"

    return NodeGpuReport(
        gpus=all_gpus,
        has_gpu=True,
        primary_device_type=primary_device_type,
    )

"""Device detection and configuration for tinygrad backend.

This module handles hardware capability detection and device configuration
for tinygrad, with special support for Intel Arc GPUs.
"""

import logging
import platform
import subprocess
from dataclasses import dataclass
from typing import Literal

from loguru import logger


@dataclass
class DeviceCapabilities:
    """Hardware device capabilities detected on the system.

    Attributes:
        device_type: Type of device (CPU, GPU, METAL)
        runtime: GPU runtime if applicable (LEVEL_ZERO, OPENCL, None)
        memory_gb: Available memory in gigabytes
        compute_units: Number of compute units (GPU cores, CPU cores, etc.)
        device_name: Human-readable device name
    """

    device_type: Literal["CPU", "GPU", "METAL"]
    runtime: str | None
    memory_gb: float
    compute_units: int | None
    device_name: str


def detect_capabilities() -> DeviceCapabilities:
    """Detect available hardware capabilities for tinygrad.

    This function checks for GPU availability (Intel Arc, NVIDIA, AMD),
    determines the best runtime (Level Zero, OpenCL, CUDA, Metal),
    and falls back to CPU if no GPU is available.

    Returns:
        DeviceCapabilities describing the detected hardware

    Example:
        >>> caps = detect_capabilities()
        >>> print(f"Device: {caps.device_name} ({caps.device_type})")
        >>> if caps.runtime:
        ...     print(f"Runtime: {caps.runtime}")
    """
    system = platform.system()

    # Check for macOS Metal
    if system == "Darwin":
        return _detect_metal_capabilities()

    # Check for GPU on Linux/Windows
    if _has_gpu():
        gpu_caps = _detect_gpu_capabilities()
        if gpu_caps:
            return gpu_caps

    # Fall back to CPU
    return _detect_cpu_capabilities()


def _has_gpu() -> bool:
    """Check if a GPU is available on the system.

    Returns:
        True if GPU hardware is detected
    """
    try:
        # Try lspci on Linux
        result = subprocess.run(
            ["lspci"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = result.stdout.lower()

        # Check for common GPU vendors
        return any(
            keyword in output
            for keyword in ["vga", "3d", "display", "nvidia", "amd", "intel arc"]
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        # lspci not available or failed
        return False


def _detect_metal_capabilities() -> DeviceCapabilities:
    """Detect Metal GPU capabilities on macOS.

    Returns:
        DeviceCapabilities for Metal device
    """
    logger.info("Detected macOS system, using Metal backend")

    # Get system memory as proxy for GPU memory on unified memory systems
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        memory_bytes = int(result.stdout.strip())
        memory_gb = memory_bytes / (1024**3)
    except (
        subprocess.TimeoutExpired,
        FileNotFoundError,
        subprocess.SubprocessError,
        ValueError,
    ):
        memory_gb = 8.0  # Default fallback

    return DeviceCapabilities(
        device_type="METAL",
        runtime="METAL",
        memory_gb=memory_gb,
        compute_units=None,
        device_name="Apple Metal GPU",
    )


def _detect_gpu_capabilities() -> DeviceCapabilities | None:
    """Detect GPU capabilities and select appropriate runtime.

    Tries Level Zero first (for Intel Arc), then OpenCL, then CUDA.

    Returns:
        DeviceCapabilities if GPU is available, None otherwise
    """
    # Import Intel Arc detection functions
    from exo.worker.engines.tinygrad.intel_arc import (
        detect_intel_arc,
        select_runtime,
    )

    # Check for Intel Arc GPU specifically
    if detect_intel_arc():
        logger.info(
            "Intel Arc GPU detected",
            backend_type="tinygrad",
            gpu_vendor="Intel",
        )
        runtime = select_runtime()

        if runtime == "LEVEL_ZERO":
            logger.info(
                "Runtime selection: Level Zero (optimal for Intel Arc)",
                backend_type="tinygrad",
                runtime="LEVEL_ZERO",
                gpu_vendor="Intel",
                reason="Level Zero provides best performance for Intel Arc",
            )
            return DeviceCapabilities(
                device_type="GPU",
                runtime="LEVEL_ZERO",
                memory_gb=_estimate_gpu_memory(),
                compute_units=None,
                device_name="Intel Arc GPU (Level Zero)",
            )
        elif runtime == "OPENCL":
            logger.warning(
                "Fallback to OpenCL runtime (Level Zero unavailable)",
                backend_type="tinygrad",
                runtime="OPENCL",
                gpu_vendor="Intel",
                requested_runtime="LEVEL_ZERO",
                fallback_runtime="OPENCL",
                reason="Level Zero runtime not available, using OpenCL fallback",
            )
            return DeviceCapabilities(
                device_type="GPU",
                runtime="OPENCL",
                memory_gb=_estimate_gpu_memory(),
                compute_units=None,
                device_name="Intel Arc GPU (OpenCL)",
            )
        else:
            logger.warning(
                "GPU fallback to CPU (no runtime available)",
                backend_type="tinygrad",
                gpu_vendor="Intel",
                requested_device="GPU",
                fallback_device="CPU",
                reason="Intel Arc GPU detected but no runtime (Level Zero or OpenCL) available",
            )
            return None

    # Try OpenCL for other GPUs
    if _check_opencl_available():
        device_name = _get_opencl_device_name()
        logger.info(
            "OpenCL runtime detected for GPU",
            backend_type="tinygrad",
            runtime="OPENCL",
            device_name=device_name,
        )
        return DeviceCapabilities(
            device_type="GPU",
            runtime="OPENCL",
            memory_gb=_estimate_gpu_memory(),
            compute_units=None,
            device_name=device_name,
        )

    # Try CUDA (NVIDIA GPUs)
    if _check_cuda_available():
        logger.info(
            "CUDA runtime detected for NVIDIA GPU",
            backend_type="tinygrad",
            runtime="CUDA",
            gpu_vendor="NVIDIA",
        )
        return DeviceCapabilities(
            device_type="GPU",
            runtime="CUDA",
            memory_gb=_estimate_gpu_memory(),
            compute_units=None,
            device_name="NVIDIA GPU (CUDA)",
        )

    logger.warning(
        "GPU fallback to CPU (no runtime available)",
        backend_type="tinygrad",
        requested_device="GPU",
        fallback_device="CPU",
        reason="GPU hardware detected but no compatible runtime available",
    )
    return None


def _check_opencl_available() -> bool:
    """Check if OpenCL runtime is available.

    Returns:
        True if OpenCL is installed and has devices
    """
    try:
        import pyopencl as cl

        platforms = cl.get_platforms()
        return len(platforms) > 0 and any(len(p.get_devices()) > 0 for p in platforms)
    except (ImportError, Exception):
        return False


def _check_cuda_available() -> bool:
    """Check if CUDA runtime is available.

    Returns:
        True if CUDA is installed and functional
    """
    try:
        # Check for nvidia-smi
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False


def _get_opencl_device_name() -> str:
    """Get the name of the first OpenCL device.

    Returns:
        Device name or generic fallback
    """
    try:
        import pyopencl as cl

        platforms = cl.get_platforms()
        if platforms:
            devices = platforms[0].get_devices()
            if devices:
                return devices[0].name
    except Exception:
        pass

    return "GPU (OpenCL)"


def _estimate_gpu_memory() -> float:
    """Estimate available GPU memory in GB.

    This is a rough estimate since we can't always query GPU memory directly.

    Returns:
        Estimated GPU memory in gigabytes
    """
    # Try to get actual GPU memory if possible
    try:
        import pyopencl as cl

        platforms = cl.get_platforms()
        if platforms:
            devices = platforms[0].get_devices()
            if devices:
                # Get global memory size in bytes
                mem_bytes = devices[0].global_mem_size
                return mem_bytes / (1024**3)
    except Exception:
        pass

    # Default estimate for integrated GPUs
    return 4.0


def _detect_cpu_capabilities() -> DeviceCapabilities:
    """Detect CPU capabilities as fallback.

    Returns:
        DeviceCapabilities for CPU device
    """
    logger.info(
        "CPU backend selected (GPU unavailable)",
        backend_type="tinygrad",
        device="CPU",
        reason="No GPU hardware or runtime available",
    )

    # Get CPU count
    import multiprocessing

    cpu_count = multiprocessing.cpu_count()

    # Get system memory
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                meminfo = f.read()
                for line in meminfo.split("\n"):
                    if line.startswith("MemTotal:"):
                        mem_kb = int(line.split()[1])
                        memory_gb = mem_kb / (1024**2)
                        break
                else:
                    memory_gb = 8.0
        else:
            memory_gb = 8.0  # Default fallback
    except Exception:
        memory_gb = 8.0

    return DeviceCapabilities(
        device_type="CPU",
        runtime=None,
        memory_gb=memory_gb,
        compute_units=cpu_count,
        device_name=f"CPU ({cpu_count} cores)",
    )

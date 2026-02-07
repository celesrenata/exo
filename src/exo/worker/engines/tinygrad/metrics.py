"""GPU performance monitoring and metrics collection for tinygrad backend.

This module provides functions to collect GPU utilization, memory usage,
and other performance metrics during inference.
"""

import logging
import time
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class GPUMetrics:
    """GPU performance metrics collected during inference.

    Attributes:
        device_name: Human-readable name of the GPU device
        runtime: GPU runtime being used (LEVEL_ZERO, OPENCL, CUDA)
        memory_used_mb: Current GPU memory usage in megabytes
        memory_total_mb: Total GPU memory available in megabytes
        utilization_percent: GPU utilization percentage (0-100), None if unavailable
        timestamp: Unix timestamp when metrics were collected
    """

    device_name: str
    runtime: Literal["LEVEL_ZERO", "OPENCL", "CUDA", "METAL"]
    memory_used_mb: float
    memory_total_mb: float
    utilization_percent: float | None
    timestamp: float


def collect_gpu_metrics(
    device_name: str,
    runtime: Literal["LEVEL_ZERO", "OPENCL", "CUDA", "METAL"],
) -> GPUMetrics:
    """Collect current GPU performance metrics.

    This function queries the GPU device for current utilization and memory
    usage. The method used depends on the runtime (Level Zero, OpenCL, etc.).

    Args:
        device_name: Name of the GPU device
        runtime: GPU runtime being used

    Returns:
        GPUMetrics object with current performance data

    Example:
        >>> metrics = collect_gpu_metrics("Intel Arc GPU", "LEVEL_ZERO")
        >>> print(f"GPU Memory: {metrics.memory_used_mb:.0f}/{metrics.memory_total_mb:.0f} MB")
        >>> if metrics.utilization_percent:
        ...     print(f"GPU Utilization: {metrics.utilization_percent:.1f}%")
    """
    timestamp = time.time()

    # Try to collect metrics based on runtime
    if runtime == "LEVEL_ZERO":
        return _collect_level_zero_metrics(device_name, runtime, timestamp)
    elif runtime == "OPENCL":
        return _collect_opencl_metrics(device_name, runtime, timestamp)
    elif runtime == "CUDA":
        return _collect_cuda_metrics(device_name, runtime, timestamp)
    elif runtime == "METAL":
        return _collect_metal_metrics(device_name, runtime, timestamp)
    else:
        # Fallback with no metrics
        return GPUMetrics(
            device_name=device_name,
            runtime=runtime,
            memory_used_mb=0.0,
            memory_total_mb=0.0,
            utilization_percent=None,
            timestamp=timestamp,
        )


def _collect_level_zero_metrics(
    device_name: str,
    runtime: Literal["LEVEL_ZERO"],
    timestamp: float,
) -> GPUMetrics:
    """Collect metrics from Level Zero runtime.

    Args:
        device_name: Name of the GPU device
        runtime: Runtime type (LEVEL_ZERO)
        timestamp: Current timestamp

    Returns:
        GPUMetrics with Level Zero data
    """
    # Level Zero metrics collection would require ctypes bindings to Level Zero API
    # For now, we'll try to get basic info from sysfs
    memory_used_mb = 0.0
    memory_total_mb = 0.0
    utilization_percent = None

    try:
        # Try to read Intel GPU memory info from sysfs
        import glob

        # Look for Intel GPU memory info
        drm_devices = glob.glob("/sys/class/drm/card*/device/vendor")
        for vendor_file in drm_devices:
            with open(vendor_file) as f:
                vendor_id = f.read().strip()
                if vendor_id == "0x8086":  # Intel vendor ID
                    # Try to get memory info
                    base_path = vendor_file.replace("/vendor", "")

                    # Try to read memory regions
                    try:
                        mem_info_path = f"{base_path}/mem_info_vram_total"
                        if glob.glob(mem_info_path):
                            with open(mem_info_path) as mf:
                                memory_total_mb = int(mf.read().strip()) / (1024 * 1024)
                    except (FileNotFoundError, ValueError, PermissionError):
                        pass

                    try:
                        mem_used_path = f"{base_path}/mem_info_vram_used"
                        if glob.glob(mem_used_path):
                            with open(mem_used_path) as mf:
                                memory_used_mb = int(mf.read().strip()) / (1024 * 1024)
                    except (FileNotFoundError, ValueError, PermissionError):
                        pass

                    break
    except Exception as e:
        logger.debug(f"Could not collect Level Zero metrics from sysfs: {e}")

    # If we couldn't get memory info, provide estimates
    if memory_total_mb == 0.0:
        # Estimate based on typical Intel Arc iGPU configurations
        memory_total_mb = 4096.0  # 4GB typical for integrated graphics
        logger.debug("Using estimated memory total for Intel Arc")

    return GPUMetrics(
        device_name=device_name,
        runtime=runtime,
        memory_used_mb=memory_used_mb,
        memory_total_mb=memory_total_mb,
        utilization_percent=utilization_percent,
        timestamp=timestamp,
    )


def _collect_opencl_metrics(
    device_name: str,
    runtime: Literal["OPENCL"],
    timestamp: float,
) -> GPUMetrics:
    """Collect metrics from OpenCL runtime.

    Args:
        device_name: Name of the GPU device
        runtime: Runtime type (OPENCL)
        timestamp: Current timestamp

    Returns:
        GPUMetrics with OpenCL data
    """
    memory_used_mb = 0.0
    memory_total_mb = 0.0
    utilization_percent = None

    try:
        import pyopencl as cl  # type: ignore

        # Get first platform and device
        platforms = cl.get_platforms()  # type: ignore
        if platforms:
            devices = platforms[0].get_devices()  # type: ignore
            if devices:
                device = devices[0]  # type: ignore

                # Get total memory
                memory_total_bytes = device.global_mem_size  # type: ignore
                memory_total_mb = float(memory_total_bytes) / (1024 * 1024)

                # OpenCL doesn't provide used memory directly
                # We can only get total memory
                logger.debug(f"OpenCL device memory: {memory_total_mb:.0f} MB")

    except ImportError:
        logger.debug("pyopencl not available for metrics collection")
    except Exception as e:
        logger.debug(f"Could not collect OpenCL metrics: {e}")

    return GPUMetrics(
        device_name=device_name,
        runtime=runtime,
        memory_used_mb=memory_used_mb,
        memory_total_mb=memory_total_mb,
        utilization_percent=utilization_percent,
        timestamp=timestamp,
    )


def _collect_cuda_metrics(
    device_name: str,
    runtime: Literal["CUDA"],
    timestamp: float,
) -> GPUMetrics:
    """Collect metrics from CUDA runtime.

    Args:
        device_name: Name of the GPU device
        runtime: Runtime type (CUDA)
        timestamp: Current timestamp

    Returns:
        GPUMetrics with CUDA data
    """
    memory_used_mb = 0.0
    memory_total_mb = 0.0
    utilization_percent = None

    try:
        # Try using nvidia-smi to get metrics
        import subprocess  # noqa: F401

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            # Parse output: "memory_used, memory_total, utilization"
            parts = result.stdout.strip().split(",")
            if len(parts) >= 3:
                memory_used_mb = float(parts[0].strip())
                memory_total_mb = float(parts[1].strip())
                utilization_percent = float(parts[2].strip())

    except (
        subprocess.TimeoutExpired,
        FileNotFoundError,
        subprocess.SubprocessError,
        ValueError,
    ) as e:
        logger.debug(f"Could not collect CUDA metrics: {e}")

    return GPUMetrics(
        device_name=device_name,
        runtime=runtime,
        memory_used_mb=memory_used_mb,
        memory_total_mb=memory_total_mb,
        utilization_percent=utilization_percent,
        timestamp=timestamp,
    )


def _collect_metal_metrics(
    device_name: str,
    runtime: Literal["METAL"],
    timestamp: float,
) -> GPUMetrics:
    """Collect metrics from Metal runtime.

    Args:
        device_name: Name of the GPU device
        runtime: Runtime type (METAL)
        timestamp: Current timestamp

    Returns:
        GPUMetrics with Metal data
    """
    memory_used_mb = 0.0
    memory_total_mb = 0.0
    utilization_percent = None

    try:
        # Try to get system memory on macOS (unified memory architecture)
        import subprocess  # noqa: F401

        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            memory_bytes = int(result.stdout.strip())
            memory_total_mb = memory_bytes / (1024 * 1024)

    except (
        subprocess.TimeoutExpired,
        FileNotFoundError,
        subprocess.SubprocessError,
        ValueError,
    ) as e:
        logger.debug(f"Could not collect Metal metrics: {e}")

    return GPUMetrics(
        device_name=device_name,
        runtime=runtime,
        memory_used_mb=memory_used_mb,
        memory_total_mb=memory_total_mb,
        utilization_percent=utilization_percent,
        timestamp=timestamp,
    )


def track_inference_memory(
    runtime: Literal["LEVEL_ZERO", "OPENCL", "CUDA", "METAL"],
) -> tuple[float, float]:
    """Track memory usage during inference.

    This function should be called before and after inference to measure
    memory consumption.

    Args:
        runtime: GPU runtime being used

    Returns:
        Tuple of (memory_used_mb, memory_available_mb)

    Example:
        >>> mem_before = track_inference_memory("LEVEL_ZERO")
        >>> # ... run inference ...
        >>> mem_after = track_inference_memory("LEVEL_ZERO")
        >>> mem_delta = mem_after[0] - mem_before[0]
        >>> print(f"Inference used {mem_delta:.0f} MB")
    """
    try:
        if runtime == "OPENCL":
            import pyopencl as cl  # type: ignore

            platforms = cl.get_platforms()  # type: ignore
            if platforms:
                devices = platforms[0].get_devices()  # type: ignore
                if devices:
                    device = devices[0]  # type: ignore
                    memory_total = float(device.global_mem_size) / (1024 * 1024)  # type: ignore
                    # OpenCL doesn't provide used memory
                    return (0.0, memory_total)

        elif runtime == "CUDA":
            import subprocess  # noqa: F401

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                if len(parts) >= 2:
                    memory_used = float(parts[0].strip())
                    memory_free = float(parts[1].strip())
                    return (memory_used, memory_free)

    except Exception as e:
        logger.debug(f"Could not track inference memory: {e}")

    return (0.0, 0.0)

"""Intel Arc GPU detection and runtime selection.

This module provides functions to detect Intel Arc integrated GPUs and
select the appropriate runtime (Level Zero or OpenCL) for optimal performance.
"""

import os
import subprocess
from typing import Literal

from loguru import logger


def detect_intel_arc() -> bool:
    """Check if Intel Arc iGPU is available on the system.

    This function checks for Intel Arc graphics hardware by examining
    system information via lspci or similar tools.

    Returns:
        True if Intel Arc GPU is detected, False otherwise

    Example:
        >>> if detect_intel_arc():
        ...     print("Intel Arc GPU found!")
        ...     runtime = select_runtime()
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

        # Check for Intel Arc or Intel Graphics
        # Intel Arc devices typically show up as "Intel Corporation Device"
        # with specific device IDs, or as "Intel Arc" in newer drivers
        if "intel" in output and any(
            keyword in output for keyword in ["arc", "dg2", "alchemist"]
        ):
            logger.info("Intel Arc GPU detected via lspci")
            return True

        # Also check for Intel integrated graphics that might support Arc
        if "intel" in output and "vga" in output:
            logger.debug("Intel graphics detected, may support Arc features")
            # Additional check could be done here for specific device IDs
            return True

    except (
        subprocess.TimeoutExpired,
        FileNotFoundError,
        subprocess.SubprocessError,
    ) as e:
        logger.debug(f"lspci check failed: {e}")

    # Check for Intel GPU via /sys on Linux
    try:
        import glob

        # Check for Intel GPU in /sys/class/drm
        drm_devices = glob.glob("/sys/class/drm/card*/device/vendor")
        for vendor_file in drm_devices:
            with open(vendor_file) as f:
                vendor_id = f.read().strip()
                # 0x8086 is Intel's PCI vendor ID
                if vendor_id == "0x8086":
                    logger.info("Intel GPU detected via /sys/class/drm")
                    return True
    except (FileNotFoundError, PermissionError, OSError) as e:
        logger.debug(f"/sys check failed: {e}")

    logger.debug("No Intel Arc GPU detected")
    return False


def check_level_zero_available() -> bool:
    """Verify that Intel Level Zero runtime is available.

    Level Zero is Intel's low-level API for GPU programming and is the
    preferred runtime for Intel Arc GPUs due to better performance.

    Returns:
        True if Level Zero runtime is installed and functional

    Example:
        >>> if check_level_zero_available():
        ...     os.environ["LEVEL_ZERO"] = "1"
        ...     os.environ["GPU"] = "1"
    """
    # Check environment variable override
    if os.getenv("LEVEL_ZERO") == "1":
        logger.debug("LEVEL_ZERO environment variable is set")
        return True

    # Check for Level Zero loader library on Linux
    try:
        result = subprocess.run(
            ["ldconfig", "-p"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if "libze_loader.so" in result.stdout:
            logger.info("Level Zero loader library (libze_loader.so) found")
            return True

        logger.debug("Level Zero loader library not found in ldconfig")

    except (
        subprocess.TimeoutExpired,
        FileNotFoundError,
        subprocess.SubprocessError,
    ) as e:
        logger.debug(f"ldconfig check failed: {e}")

    # Check for library file directly
    try:
        import ctypes.util

        lib_path = ctypes.util.find_library("ze_loader")
        if lib_path:
            logger.info(f"Level Zero library found at: {lib_path}")
            return True

    except Exception as e:
        logger.debug(f"ctypes library search failed: {e}")

    # Check common installation paths
    common_paths = [
        "/usr/lib/x86_64-linux-gnu/libze_loader.so",
        "/usr/lib64/libze_loader.so",
        "/usr/local/lib/libze_loader.so",
        "/opt/intel/oneapi/compiler/latest/linux/lib/libze_loader.so",
    ]

    for path in common_paths:
        if os.path.exists(path):
            logger.info(f"Level Zero library found at: {path}")
            return True

    logger.debug("Level Zero runtime not available")
    return False


def check_opencl_available() -> bool:
    """Verify that OpenCL runtime is available.

    OpenCL is a cross-platform API for heterogeneous computing and serves
    as a fallback runtime when Level Zero is not available.

    Returns:
        True if OpenCL runtime is installed and functional

    Example:
        >>> if check_opencl_available():
        ...     os.environ["OPENCL"] = "1"
        ...     os.environ["GPU"] = "1"
    """
    # Check environment variable override
    if os.getenv("OPENCL") == "1":
        logger.debug("OPENCL environment variable is set")
        return True

    # Try importing pyopencl to verify OpenCL availability
    try:
        import pyopencl as cl  # type: ignore

        platforms = cl.get_platforms()  # type: ignore
        if platforms:
            logger.info(f"OpenCL available with {len(platforms)} platform(s)")  # type: ignore

            # Log platform details for debugging
            for i, platform in enumerate(platforms):  # type: ignore
                logger.debug(f"OpenCL Platform {i}: {platform.name}")  # type: ignore
                try:
                    devices = platform.get_devices()  # type: ignore
                    for j, device in enumerate(devices):  # type: ignore
                        logger.debug(f"  Device {j}: {device.name}")  # type: ignore
                except Exception as e:
                    logger.debug(f"  Could not enumerate devices: {e}")

            return True
        else:
            logger.debug("OpenCL platforms found but no devices available")
            return False

    except ImportError as e:
        logger.warning(f"pyopencl not available: {e}")
    except Exception as e:
        logger.warning(f"OpenCL check failed: {e}", exc_info=True)

    # Check for OpenCL library directly
    try:
        import ctypes.util

        lib_path = ctypes.util.find_library("OpenCL")
        if lib_path:
            logger.info(f"OpenCL library found at: {lib_path}")
            return True

    except Exception as e:
        logger.debug(f"ctypes library search failed: {e}")

    logger.debug("OpenCL runtime not available")
    return False


def select_runtime() -> Literal["LEVEL_ZERO", "OPENCL"] | None:
    """Select the best available runtime for Intel Arc GPU.

    This function checks for available GPU runtimes in order of preference:
    1. Level Zero (best performance for Intel Arc)
    2. OpenCL (fallback with broader compatibility)

    Returns:
        Runtime name if available, None if no GPU runtime is available

    Example:
        >>> runtime = select_runtime()
        >>> if runtime == "LEVEL_ZERO":
        ...     print("Using Level Zero for optimal performance")
        >>> elif runtime == "OPENCL":
        ...     print("Using OpenCL fallback")
        >>> else:
        ...     print("No GPU runtime available, falling back to CPU")
    """
    # Try Level Zero first (preferred for Intel Arc)
    if check_level_zero_available():
        logger.info(
            "Runtime selection: Level Zero (optimal)",
            backend_type="tinygrad",
            runtime="LEVEL_ZERO",
            gpu_vendor="Intel",
            reason="Level Zero provides best performance for Intel Arc",
        )
        return "LEVEL_ZERO"

    # Fall back to OpenCL
    if check_opencl_available():
        logger.warning(
            "Runtime selection: OpenCL (fallback)",
            backend_type="tinygrad",
            runtime="OPENCL",
            gpu_vendor="Intel",
            requested_runtime="LEVEL_ZERO",
            fallback_runtime="OPENCL",
            reason="Level Zero unavailable, using OpenCL fallback",
        )
        return "OPENCL"

    # No GPU runtime available
    logger.warning(
        "No GPU runtime available",
        backend_type="tinygrad",
        gpu_vendor="Intel",
        checked_runtimes=["LEVEL_ZERO", "OPENCL"],
        reason="Neither Level Zero nor OpenCL runtime is available",
    )
    return None

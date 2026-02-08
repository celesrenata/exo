"""Intel NPU hardware discovery and capability detection.

This module detects Intel NPU (Neural Processing Unit) hardware and determines
available software stacks for NPU execution.
"""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NPUCapabilities:
    """Capabilities and status of Intel NPU hardware."""

    available: bool
    device_path: str | None  # /dev/accel/accel0 or similar
    driver_version: str | None
    kernel_modules: list[str]  # Loaded kernel modules (intel_vpu, ivpu)
    software_stack: Literal["OPENVINO", "NONE"]
    openvino_version: str | None
    supported_model_types: list[str]  # vision, audio, embeddings, etc.
    error_message: str | None  # Error if NPU unavailable


def discover_npu() -> NPUCapabilities:
    """Discover Intel NPU hardware and determine capabilities.

    This function checks for:
    1. NPU device nodes (/dev/accel/accel*)
    2. Loaded kernel modules (intel_vpu, ivpu)
    3. OpenVINO availability and version
    4. Supported model types

    Returns:
        NPUCapabilities object with discovery results
    """
    logger.info("Starting Intel NPU discovery")

    # Check for NPU device nodes
    device_path = _find_npu_device()
    if device_path is None:
        logger.info("No NPU device node found")
        return NPUCapabilities(
            available=False,
            device_path=None,
            driver_version=None,
            kernel_modules=[],
            software_stack="NONE",
            openvino_version=None,
            supported_model_types=[],
            error_message="No NPU device node found (/dev/accel/accel* or /dev/dri/renderD*)",
        )

    logger.info(f"Found NPU device at {device_path}")

    # Check for loaded kernel modules
    kernel_modules = _check_kernel_modules()
    if not kernel_modules:
        logger.warning("No Intel NPU kernel modules loaded")
        return NPUCapabilities(
            available=False,
            device_path=device_path,
            driver_version=None,
            kernel_modules=[],
            software_stack="NONE",
            openvino_version=None,
            supported_model_types=[],
            error_message="Intel NPU kernel modules (intel_vpu, ivpu) not loaded",
        )

    logger.info(f"Found kernel modules: {', '.join(kernel_modules)}")

    # Get driver version
    driver_version = _get_driver_version(kernel_modules)
    if driver_version:
        logger.info(f"NPU driver version: {driver_version}")

    # Check OpenVINO availability
    openvino_available, openvino_version = _check_openvino()
    software_stack: Literal["OPENVINO", "NONE"] = (
        "OPENVINO" if openvino_available else "NONE"
    )

    if openvino_available:
        logger.info(f"OpenVINO available: {openvino_version}")
    else:
        logger.warning("OpenVINO not available - NPU cannot be used")

    # Determine supported model types
    supported_model_types = _determine_supported_model_types(openvino_available)

    # NPU is available if we have device, drivers, and software stack
    available = (
        device_path is not None and len(kernel_modules) > 0 and openvino_available
    )

    if available:
        logger.info(
            f"Intel NPU is available with {len(supported_model_types)} supported model types"
        )
    else:
        logger.info("Intel NPU hardware detected but not fully functional")

    return NPUCapabilities(
        available=available,
        device_path=device_path,
        driver_version=driver_version,
        kernel_modules=kernel_modules,
        software_stack=software_stack,
        openvino_version=openvino_version,
        supported_model_types=supported_model_types,
        error_message=None
        if available
        else "NPU hardware present but OpenVINO not available",
    )


def _find_npu_device() -> str | None:
    """Find NPU device node in /dev.

    Intel NPU may appear as:
    - /dev/accel/accel0 (newer kernel)
    - /dev/dri/renderD* (older kernel, shared with GPU)

    Returns:
        Path to NPU device or None if not found
    """
    # Check for dedicated NPU device node
    accel_devices = (
        list(Path("/dev/accel").glob("accel*")) if Path("/dev/accel").exists() else []
    )
    if accel_devices:
        # Return first accel device
        return str(accel_devices[0])

    # Check for render nodes (may be shared with GPU)
    # We'll need to verify it's actually NPU later
    render_devices = (
        list(Path("/dev/dri").glob("renderD*")) if Path("/dev/dri").exists() else []
    )
    if render_devices:
        # Check if any render device is associated with NPU
        for device in render_devices:
            if _is_npu_render_device(device):
                return str(device)

    return None


def _is_npu_render_device(device_path: Path) -> bool:
    """Check if a render device is associated with Intel NPU.

    Args:
        device_path: Path to render device

    Returns:
        True if device is NPU, False otherwise
    """
    try:
        # Check sysfs for device information
        # renderD128 -> /sys/class/drm/renderD128/device/
        device_name = device_path.name
        sysfs_path = Path(f"/sys/class/drm/{device_name}/device/")

        if not sysfs_path.exists():
            return False

        # Check vendor and device IDs
        vendor_path = sysfs_path / "vendor"
        if vendor_path.exists():
            vendor_id = vendor_path.read_text().strip()
            # Intel vendor ID is 0x8086
            if vendor_id != "0x8086":
                return False

        # Check if device is NPU by looking at driver
        driver_path = sysfs_path / "driver"
        if driver_path.exists() and driver_path.is_symlink():
            driver_name = driver_path.resolve().name
            # NPU uses intel_vpu or ivpu driver
            if driver_name in ["intel_vpu", "ivpu"]:
                return True

        return False
    except Exception as e:
        logger.debug(f"Error checking render device {device_path}: {e}")
        return False


def _check_kernel_modules() -> list[str]:
    """Check for loaded Intel NPU kernel modules.

    Returns:
        List of loaded NPU kernel modules
    """
    npu_modules = ["intel_vpu", "ivpu"]
    loaded_modules = []

    try:
        # Read /proc/modules to check loaded modules
        with open("/proc/modules", "r") as f:
            modules_text = f.read()

        for module in npu_modules:
            if module in modules_text:
                loaded_modules.append(module)

    except Exception as e:
        logger.debug(f"Error checking kernel modules: {e}")

    return loaded_modules


def _get_driver_version(kernel_modules: list[str]) -> str | None:
    """Get NPU driver version from kernel module info.

    Args:
        kernel_modules: List of loaded kernel modules

    Returns:
        Driver version string or None
    """
    if not kernel_modules:
        return None

    try:
        # Try to get version from modinfo
        module = kernel_modules[0]
        result = subprocess.run(
            ["modinfo", module],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            # Parse version from modinfo output
            for line in result.stdout.splitlines():
                if line.startswith("version:"):
                    return line.split(":", 1)[1].strip()

    except Exception as e:
        logger.debug(f"Error getting driver version: {e}")

    return None


def _check_openvino() -> tuple[bool, str | None]:
    """Check if OpenVINO is available on the system.

    Returns:
        Tuple of (available, version)
    """
    try:
        # Try to import openvino
        import openvino as ov

        version = ov.__version__
        logger.debug(f"OpenVINO version {version} found")

        # Verify NPU device is available in OpenVINO
        core = ov.Core()
        available_devices = core.available_devices()

        # NPU device should appear as "NPU" or "NPU.0"
        has_npu = any("NPU" in device for device in available_devices)

        if has_npu:
            logger.debug(f"OpenVINO NPU device available: {available_devices}")
            return True, version
        else:
            logger.debug(
                f"OpenVINO available but no NPU device found. Available: {available_devices}"
            )
            return False, version

    except ImportError:
        logger.debug("OpenVINO not installed")
        return False, None
    except Exception as e:
        logger.debug(f"Error checking OpenVINO: {e}")
        return False, None


def _determine_supported_model_types(openvino_available: bool) -> list[str]:
    """Determine which model types are supported on NPU.

    Intel NPU is optimized for:
    - Vision models (image classification, object detection)
    - Audio models (speech recognition, audio classification)
    - Embedding models (text embeddings, sentence transformers)
    - Small transformer models

    NOT suitable for:
    - Large LLM decode (use GPU/CPU instead)
    - High-throughput streaming inference

    Args:
        openvino_available: Whether OpenVINO is available

    Returns:
        List of supported model type strings
    """
    if not openvino_available:
        return []

    # These are the model types that typically benefit from NPU
    return [
        "vision",  # Image classification, object detection
        "audio",  # Speech recognition, audio processing
        "embeddings",  # Text embeddings, sentence transformers
        "vad",  # Voice activity detection
        "small_transformers",  # Small transformer models (<1B params)
    ]

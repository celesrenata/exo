#!/usr/bin/env python3
"""
Intel Arc GPU Detection Script for PyTorch

This script detects Intel Arc GPUs using native torch.xpu and logs device capabilities.
It verifies that PyTorch is properly installed with XPU support and can access Intel Arc GPUs.

Requirements:
- PyTorch 2.11+ (native XPU support, no IPEX needed)
- Intel compute-runtime and level-zero drivers
"""

import sys
from typing import Final

try:
    import torch  # type: ignore
except ImportError:
    torch = None  # type: ignore

try:
    from loguru import logger  # type: ignore
except ImportError:
    import logging

    logger = logging.getLogger(__name__)  # type: ignore

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="DEBUG",
)


def detect_intel_arc_gpu() -> None:
    """
    Detect Intel Arc GPUs using torch.xpu and log device capabilities.

    This function:
    1. Checks if PyTorch is installed
    2. Checks if IPEX is installed
    3. Detects XPU devices (Intel Arc GPUs)
    4. Logs device properties (memory, compute units, etc.)
    """
    logger.info("Starting Intel Arc GPU detection...")

    # Step 1: Check PyTorch installation
    try:
        import torch

        logger.info(f"PyTorch version: {torch.__version__}")
    except ImportError as e:
        logger.error(f"PyTorch not found: {e}")
        logger.error("Please install PyTorch 2.0+ for Intel Arc support")
        sys.exit(1)

    # Step 2: Check XPU availability (native PyTorch 2.11+, no IPEX needed)
    if not hasattr(torch, "xpu"):
        logger.error("torch.xpu module not found")
        logger.error("PyTorch may not be built with Intel GPU support")
        sys.exit(1)

    if not torch.xpu.is_available():
        logger.warning("Intel XPU (Arc GPU) is not available")
        logger.warning("Possible reasons:")
        logger.warning("  - Intel Arc GPU not present in system")
        logger.warning("  - Intel compute-runtime or level-zero drivers not installed")
        logger.warning("  - Drivers not properly configured")
        sys.exit(1)

    logger.info("Intel XPU (Arc GPU) is available!")

    # Step 4: Enumerate XPU devices
    device_count: Final[int] = torch.xpu.device_count()
    logger.info(f"Found {device_count} Intel XPU device(s)")

    if device_count == 0:
        logger.warning("No Intel XPU devices found")
        sys.exit(1)

    # Step 5: Log device properties for each device
    for device_id in range(device_count):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Device {device_id} Properties:")
        logger.info(f"{'=' * 60}")

        try:
            # Get device properties
            props = torch.xpu.get_device_properties(device_id)

            # Log basic properties
            logger.info(f"  Name: {props.name}")
            logger.info(f"  Type: {props.type}")
            logger.info(f"  Platform: {props.platform_name}")
            logger.info(f"  Driver Version: {props.driver_version}")

            # Log memory information
            total_memory_gb: Final[float] = props.total_memory / (1024**3)
            logger.info(f"  Total Memory: {total_memory_gb:.2f} GB")

            # Get current memory usage
            allocated_memory: Final[int] = torch.xpu.memory_allocated(device_id)
            allocated_memory_gb: Final[float] = allocated_memory / (1024**3)
            free_memory_gb: Final[float] = total_memory_gb - allocated_memory_gb

            logger.info(f"  Allocated Memory: {allocated_memory_gb:.2f} GB")
            logger.info(f"  Free Memory: {free_memory_gb:.2f} GB")

            # Log compute capabilities
            logger.info(f"  Max Compute Units: {props.max_compute_units}")
            logger.info(f"  Max Work Group Size: {props.max_work_group_size}")
            logger.info(f"  Max Num Sub Groups: {props.max_num_sub_groups}")

            # Log additional properties if available
            if hasattr(props, "sub_group_sizes"):
                logger.info(f"  Sub Group Sizes: {props.sub_group_sizes}")

            if hasattr(props, "gpu_eu_count"):
                logger.info(f"  GPU EU Count: {props.gpu_eu_count}")

        except Exception as e:
            logger.error(f"Error getting properties for device {device_id}: {e}")

    logger.info(f"\n{'=' * 60}")
    logger.info("Intel Arc GPU detection completed successfully!")
    logger.info(f"{'=' * 60}")


def main() -> None:
    """Main entry point for the detection script."""
    try:
        detect_intel_arc_gpu()
    except KeyboardInterrupt:
        logger.info("\nDetection interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error during detection: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

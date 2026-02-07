"""Generate NPU capability report for Intel Core Ultra processors.

This script generates a detailed report of NPU hardware capabilities,
driver status, and supported model types.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from exo.worker.engines.npu.discovery import discover_npu

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_capability_report() -> dict[str, Any]:
    """Generate comprehensive NPU capability report.

    Returns:
        Dictionary containing capability report data
    """
    logger.info("Generating Intel NPU capability report")

    # Discover NPU hardware
    capabilities = discover_npu()

    # Build report
    report = {
        "npu_available": capabilities.available,
        "device_path": capabilities.device_path,
        "driver_version": capabilities.driver_version,
        "kernel_modules": capabilities.kernel_modules,
        "software_stack": capabilities.software_stack,
        "openvino_version": capabilities.openvino_version,
        "supported_model_types": capabilities.supported_model_types,
        "error_message": capabilities.error_message,
        "hardware_details": _get_hardware_details(),
        "device_nodes": _list_device_nodes(),
        "kernel_info": _get_kernel_info(),
        "driver_details": _get_driver_details(capabilities.kernel_modules),
    }

    return report


def _get_hardware_details() -> dict[str, Any]:
    """Get CPU and hardware information.

    Returns:
        Dictionary with hardware details
    """
    details = {}

    try:
        # Get CPU model
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    details["cpu_model"] = line.split(":", 1)[1].strip()
                    break

        # Check if this is a Core Ultra processor
        if "cpu_model" in details:
            details["is_core_ultra"] = "Core Ultra" in details["cpu_model"]
        else:
            details["is_core_ultra"] = False

    except Exception as e:
        logger.debug(f"Error getting hardware details: {e}")
        details["error"] = str(e)

    return details


def _list_device_nodes() -> dict[str, list[str]]:
    """List all relevant device nodes.

    Returns:
        Dictionary with device node information
    """
    nodes = {
        "accel_devices": [],
        "render_devices": [],
        "card_devices": [],
    }

    # Check /dev/accel
    accel_path = Path("/dev/accel")
    if accel_path.exists():
        nodes["accel_devices"] = [str(p) for p in accel_path.glob("accel*")]

    # Check /dev/dri
    dri_path = Path("/dev/dri")
    if dri_path.exists():
        nodes["render_devices"] = [str(p) for p in dri_path.glob("renderD*")]
        nodes["card_devices"] = [str(p) for p in dri_path.glob("card*")]

    return nodes


def _get_kernel_info() -> dict[str, Any]:
    """Get kernel version and configuration.

    Returns:
        Dictionary with kernel information
    """
    info = {}

    try:
        # Get kernel version
        result = subprocess.run(
            ["uname", "-r"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["kernel_version"] = result.stdout.strip()

        # Check if NPU-related kernel configs are enabled
        config_path = Path(f"/boot/config-{info.get('kernel_version', '')}")
        if config_path.exists():
            with open(config_path, "r") as f:
                config_text = f.read()
                info["config_drm_accel"] = "CONFIG_DRM_ACCEL=y" in config_text
                info["config_intel_vpu"] = "CONFIG_DRM_INTEL_VPU" in config_text

    except Exception as e:
        logger.debug(f"Error getting kernel info: {e}")
        info["error"] = str(e)

    return info


def _get_driver_details(kernel_modules: list[str]) -> dict[str, Any]:
    """Get detailed driver information.

    Args:
        kernel_modules: List of loaded kernel modules

    Returns:
        Dictionary with driver details
    """
    details = {}

    for module in kernel_modules:
        try:
            result = subprocess.run(
                ["modinfo", module],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                module_info = {}
                for line in result.stdout.splitlines():
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip()
                        value = value.strip()
                        if key in ["version", "description", "author", "license", "firmware"]:
                            module_info[key] = value

                details[module] = module_info

        except Exception as e:
            logger.debug(f"Error getting driver details for {module}: {e}")
            details[module] = {"error": str(e)}

    return details


def print_report(report: dict[str, Any]) -> None:
    """Print capability report in human-readable format.

    Args:
        report: Report dictionary from generate_capability_report()
    """
    print("\n" + "=" * 80)
    print("Intel NPU Capability Report")
    print("=" * 80)

    # NPU Status
    print("\n## NPU Status")
    print(f"Available: {report['npu_available']}")
    if report["error_message"]:
        print(f"Error: {report['error_message']}")

    # Hardware Details
    print("\n## Hardware Details")
    hw = report["hardware_details"]
    if "cpu_model" in hw:
        print(f"CPU Model: {hw['cpu_model']}")
        print(f"Core Ultra Processor: {hw['is_core_ultra']}")

    # Device Information
    print("\n## Device Information")
    print(f"Device Path: {report['device_path'] or 'Not found'}")
    print(f"Driver Version: {report['driver_version'] or 'Unknown'}")
    print(f"Kernel Modules: {', '.join(report['kernel_modules']) or 'None loaded'}")

    # Device Nodes
    print("\n## Device Nodes")
    nodes = report["device_nodes"]
    print(f"Accel Devices: {', '.join(nodes['accel_devices']) or 'None'}")
    print(f"Render Devices: {', '.join(nodes['render_devices']) or 'None'}")
    print(f"Card Devices: {', '.join(nodes['card_devices']) or 'None'}")

    # Kernel Information
    print("\n## Kernel Information")
    kernel = report["kernel_info"]
    if "kernel_version" in kernel:
        print(f"Kernel Version: {kernel['kernel_version']}")
    if "config_drm_accel" in kernel:
        print(f"DRM Accel Support: {kernel['config_drm_accel']}")
    if "config_intel_vpu" in kernel:
        print(f"Intel VPU Config: {kernel['config_intel_vpu']}")

    # Driver Details
    print("\n## Driver Details")
    if report["driver_details"]:
        for module, info in report["driver_details"].items():
            print(f"\n### {module}")
            for key, value in info.items():
                print(f"  {key}: {value}")
    else:
        print("No driver details available")

    # Software Stack
    print("\n## Software Stack")
    print(f"Software Stack: {report['software_stack']}")
    print(f"OpenVINO Version: {report['openvino_version'] or 'Not installed'}")

    # Supported Model Types
    print("\n## Supported Model Types")
    if report["supported_model_types"]:
        for model_type in report["supported_model_types"]:
            print(f"  - {model_type}")
    else:
        print("  None (OpenVINO not available)")

    print("\n" + "=" * 80)


def main() -> int:
    """Main entry point for capability report generation.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        report = generate_capability_report()

        # Print human-readable report
        print_report(report)

        # Also save JSON report
        output_path = Path("npu_capability_report.json")
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"\nJSON report saved to: {output_path}")

        # Return success if NPU is available
        return 0 if report["npu_available"] else 1

    except Exception as e:
        logger.error(f"Error generating capability report: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

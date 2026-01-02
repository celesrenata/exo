import socket
import sys
from subprocess import CalledProcessError
from typing import Any

import psutil
from anyio import run_process

from exo.shared.types.profiling import NetworkInterfaceInfo


async def get_friendly_name() -> str:
    """
    Asynchronously gets the 'Computer Name' (friendly name) of a Mac.
    e.g., "John's MacBook Pro"
    Returns the name as a string, or None if an error occurs or not on macOS.
    """
    hostname = socket.gethostname()

    # TODO: better non mac support
    if sys.platform != "darwin":  # 'darwin' is the platform name for macOS
        return hostname

    try:
        process = await run_process(["scutil", "--get", "ComputerName"])
    except CalledProcessError:
        return hostname

    return process.stdout.decode("utf-8", errors="replace").strip() or hostname


def get_network_interfaces() -> list[NetworkInterfaceInfo]:
    """
    Retrieves detailed network interface information on macOS.
    Parses output from 'networksetup -listallhardwareports' and 'ifconfig'
    to determine interface names, IP addresses, and types (ethernet, wifi, vpn, other).
    Returns a list of NetworkInterfaceInfo objects.
    """
    interfaces_info: list[NetworkInterfaceInfo] = []

    for iface, services in psutil.net_if_addrs().items():
        for service in services:
            match service.family:
                case socket.AF_INET | socket.AF_INET6:
                    interfaces_info.append(
                        NetworkInterfaceInfo(name=iface, ip_address=service.address)
                    )
                case _:
                    pass

    return interfaces_info


async def get_model_and_chip() -> tuple[str, str]:
    """Get system information for both Mac and Linux."""
    model = "Unknown Model"
    chip = "Unknown Chip"

    if sys.platform == "darwin":
        # macOS detection
        try:
            process = await run_process(
                [
                    "system_profiler",
                    "SPHardwareDataType",
                ]
            )
            output = process.stdout.decode().strip()

            model_line = next(
                (line for line in output.split("\n") if "Model Name" in line), None
            )
            model = model_line.split(": ")[1] if model_line else "Unknown Model"

            chip_line = next((line for line in output.split("\n") if "Chip" in line), None)
            chip = chip_line.split(": ")[1] if chip_line else "Unknown Chip"
        except CalledProcessError:
            pass
    else:
        # Linux detection
        try:
            # Get CPU model from /proc/cpuinfo
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        chip = line.split(": ")[1].strip()
                        break
            
            # Get system model from DMI
            try:
                with open("/sys/class/dmi/id/product_name", "r") as f:
                    model = f.read().strip()
                if not model or model in ["System Product Name", "To be filled by O.E.M."]:
                    # Try board name as fallback
                    with open("/sys/class/dmi/id/board_name", "r") as f:
                        board = f.read().strip()
                    if board and board not in ["System Board Name", "To be filled by O.E.M."]:
                        model = f"Linux System ({board})"
                    else:
                        model = "Linux System"
            except (FileNotFoundError, PermissionError):
                model = "Linux System"
                
        except (FileNotFoundError, PermissionError):
            # Fallback to basic Linux info
            model = "Linux System"
            chip = "Unknown CPU"

    return (model, chip)


async def get_intel_gpu_info() -> dict[str, Any]:
    """Get Intel GPU information."""
    info = {
        "intel_gpu_available": False,
        "intel_gpu_count": 0,
        "intel_gpu_memory": 0,
        "intel_gpu_devices": [],
        "ipex_version": None,
        "intel_gpu_driver_version": None,
    }
    
    try:
        import intel_extension_for_pytorch as ipex
        import torch
        
        info["ipex_version"] = ipex.__version__
        
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            info["intel_gpu_available"] = True
            info["intel_gpu_count"] = torch.xpu.device_count()
            
            # Get information for each Intel GPU device
            for i in range(info["intel_gpu_count"]):
                try:
                    props = torch.xpu.get_device_properties(i)
                    device_info = {
                        "device_id": i,
                        "name": props.name if hasattr(props, 'name') else f"Intel GPU {i}",
                        "total_memory": props.total_memory if hasattr(props, 'total_memory') else 0,
                        "max_compute_units": props.max_compute_units if hasattr(props, 'max_compute_units') else 0,
                    }
                    info["intel_gpu_devices"].append(device_info)
                    
                    # Use first device memory as primary memory info
                    if i == 0 and hasattr(props, 'total_memory'):
                        info["intel_gpu_memory"] = props.total_memory
                        
                except Exception:
                    # If we can't get device properties, still record the device exists
                    info["intel_gpu_devices"].append({
                        "device_id": i,
                        "name": f"Intel GPU {i}",
                        "total_memory": 0,
                        "max_compute_units": 0,
                    })
                    
    except ImportError:
        # IPEX not available
        pass
    except Exception:
        # Other errors in Intel GPU detection
        pass
    
    # Try to get Intel GPU driver version from system
    if info["intel_gpu_available"]:
        try:
            if sys.platform == "linux":
                # Try to get Intel GPU driver version from modinfo
                process = await run_process(["modinfo", "i915"])
                output = process.stdout.decode().strip()
                for line in output.split("\n"):
                    if line.startswith("version:"):
                        info["intel_gpu_driver_version"] = line.split(": ")[1].strip()
                        break
        except (CalledProcessError, FileNotFoundError):
            # Driver version detection failed, not critical
            pass
    
    return info

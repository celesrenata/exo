"""
Device Manager for PyTorch + IPEX Backend

This module provides device detection, selection, and monitoring for Intel Arc GPUs,
NVIDIA GPUs, and CPU fallback. It implements the device abstraction layer for the
PyTorch + IPEX inference engine.

Requirements addressed:
- 1.1: Device detection and enumeration
- 1.2: Intel Arc GPU prioritization
- 1.3: Multi-device selection
- 1.4: Fallback mechanism
- 1.5: Device health monitoring
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

logger = logging.getLogger(__name__)


class DeviceType(str, Enum):
    """Enumeration of supported device types."""

    INTEL_ARC = "intel_arc"
    NVIDIA_GPU = "nvidia_gpu"
    CPU = "cpu"


@dataclass(frozen=True)
class DeviceInfo:
    """Information about a compute device."""

    device_type: DeviceType
    device_id: int
    name: str
    total_memory_bytes: int
    free_memory_bytes: int
    compute_units: Optional[int] = None
    driver_version: Optional[str] = None


class DeviceManager:
    """
    Manages device detection, selection, and monitoring for PyTorch + IPEX.

    This class provides:
    - Automatic detection of Intel Arc, NVIDIA, and CPU devices
    - Device selection with priority logic (Intel Arc > NVIDIA > CPU)
    - Memory monitoring and health checks
    - Device abstraction layer for inference engine

    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
    """

    def __init__(self) -> None:
        """Initialize the DeviceManager."""
        self._torch_available: bool = False
        self._ipex_available: bool = False
        self._xpu_available: bool = False
        self._cuda_available: bool = False

        # Try to import torch and check availability
        try:
            import torch  # type: ignore

            self._torch_available = True
            self._torch = torch

            # Check for IPEX
            try:
                import intel_extension_for_pytorch as ipex  # type: ignore

                self._ipex_available = True
                self._ipex = ipex
                logger.info(f"Intel Extension for PyTorch (IPEX) available: {ipex.__version__}")
            except ImportError:
                logger.debug("Intel Extension for PyTorch (IPEX) not available")

            # Check for XPU (Intel Arc)
            if hasattr(torch, "xpu"):
                self._xpu_available = torch.xpu.is_available()  # type: ignore
                if self._xpu_available:
                    logger.info("Intel XPU (Arc GPU) available")
            else:
                logger.debug("torch.xpu module not found")

            # Check for CUDA (NVIDIA)
            self._cuda_available = torch.cuda.is_available()
            if self._cuda_available:
                logger.info("NVIDIA CUDA available")

        except ImportError:
            logger.warning("PyTorch not available - CPU-only mode")

    def detect_devices(self) -> list[DeviceInfo]:
        """
        Detect all available compute devices.

        Returns:
            List of DeviceInfo objects for all detected devices.

        Requirements: 1.1
        """
        devices: list[DeviceInfo] = []

        if not self._torch_available:
            logger.warning("PyTorch not available - returning CPU device only")
            devices.append(
                DeviceInfo(
                    device_type=DeviceType.CPU,
                    device_id=0,
                    name="CPU",
                    total_memory_bytes=0,
                    free_memory_bytes=0,
                )
            )
            return devices

        # Detect Intel Arc GPUs
        if self._xpu_available:
            xpu_devices = self._detect_intel_arc_devices()
            devices.extend(xpu_devices)
            logger.info(f"Detected {len(xpu_devices)} Intel Arc GPU(s)")

        # Detect NVIDIA GPUs
        if self._cuda_available:
            cuda_devices = self._detect_nvidia_devices()
            devices.extend(cuda_devices)
            logger.info(f"Detected {len(cuda_devices)} NVIDIA GPU(s)")

        # Always add CPU as fallback
        devices.append(
            DeviceInfo(
                device_type=DeviceType.CPU,
                device_id=0,
                name="CPU",
                total_memory_bytes=0,
                free_memory_bytes=0,
            )
        )

        return devices

    def _detect_intel_arc_devices(self) -> list[DeviceInfo]:
        """
        Detect Intel Arc GPU devices using torch.xpu.

        Returns:
            List of DeviceInfo objects for Intel Arc devices.

        Requirements: 1.2
        """
        devices: list[DeviceInfo] = []

        if not self._xpu_available:
            return devices

        try:
            device_count: int = self._torch.xpu.device_count()  # type: ignore

            for device_id in range(device_count):
                try:
                    props = self._torch.xpu.get_device_properties(device_id)  # type: ignore
                    allocated = self._torch.xpu.memory_allocated(device_id)  # type: ignore

                    total_memory: int = props.total_memory  # type: ignore
                    free_memory: int = total_memory - allocated

                    device_info = DeviceInfo(
                        device_type=DeviceType.INTEL_ARC,
                        device_id=device_id,
                        name=props.name,  # type: ignore
                        total_memory_bytes=total_memory,
                        free_memory_bytes=free_memory,
                        compute_units=props.max_compute_units,  # type: ignore
                        driver_version=props.driver_version,  # type: ignore
                    )
                    devices.append(device_info)

                    logger.debug(
                        f"Intel Arc device {device_id}: {props.name}, "  # type: ignore
                        f"{total_memory / (1024**3):.2f} GB total, "
                        f"{free_memory / (1024**3):.2f} GB free"
                    )

                except Exception as e:
                    logger.error(f"Error detecting Intel Arc device {device_id}: {e}")

        except Exception as e:
            logger.error(f"Error enumerating Intel Arc devices: {e}")

        return devices

    def _detect_nvidia_devices(self) -> list[DeviceInfo]:
        """
        Detect NVIDIA GPU devices using torch.cuda.

        Returns:
            List of DeviceInfo objects for NVIDIA devices.

        Requirements: 1.4
        """
        devices: list[DeviceInfo] = []

        if not self._cuda_available:
            return devices

        try:
            device_count: int = self._torch.cuda.device_count()

            for device_id in range(device_count):
                try:
                    props = self._torch.cuda.get_device_properties(device_id)
                    allocated = self._torch.cuda.memory_allocated(device_id)

                    total_memory: int = props.total_memory
                    free_memory: int = total_memory - allocated

                    device_info = DeviceInfo(
                        device_type=DeviceType.NVIDIA_GPU,
                        device_id=device_id,
                        name=props.name,
                        total_memory_bytes=total_memory,
                        free_memory_bytes=free_memory,
                        compute_units=props.multi_processor_count,
                    )
                    devices.append(device_info)

                    logger.debug(
                        f"NVIDIA device {device_id}: {props.name}, "
                        f"{total_memory / (1024**3):.2f} GB total, "
                        f"{free_memory / (1024**3):.2f} GB free"
                    )

                except Exception as e:
                    logger.error(f"Error detecting NVIDIA device {device_id}: {e}")

        except Exception as e:
            logger.error(f"Error enumerating NVIDIA devices: {e}")

        return devices

    def select_device(
        self, preference: Optional[DeviceType] = None
    ) -> tuple[Literal["xpu", "cuda", "cpu"], int]:
        """
        Select the optimal device for inference.

        Selection priority (unless overridden by preference):
        1. Intel Arc GPU with most free memory
        2. NVIDIA GPU with most free memory
        3. CPU

        Args:
            preference: Optional device type preference to override default priority.

        Returns:
            Tuple of (device_type_string, device_id) suitable for torch.device().

        Requirements: 1.2, 1.3, 1.4
        """
        devices = self.detect_devices()

        # If preference specified, try to honor it
        if preference is not None:
            preferred_devices = [d for d in devices if d.device_type == preference]
            if preferred_devices:
                # Select device with most free memory
                best_device = max(preferred_devices, key=lambda d: d.free_memory_bytes)
                device_str = self._device_type_to_string(best_device.device_type)
                logger.info(
                    f"Selected preferred device: {device_str}:{best_device.device_id} "
                    f"({best_device.name})"
                )
                return (device_str, best_device.device_id)
            else:
                logger.warning(
                    f"Preferred device type {preference} not available, using default priority"
                )

        # Default priority: Intel Arc > NVIDIA > CPU
        intel_arc_devices = [d for d in devices if d.device_type == DeviceType.INTEL_ARC]
        if intel_arc_devices:
            best_device = max(intel_arc_devices, key=lambda d: d.free_memory_bytes)
            logger.info(
                f"Selected Intel Arc GPU: xpu:{best_device.device_id} ({best_device.name}), "
                f"{best_device.free_memory_bytes / (1024**3):.2f} GB free"
            )
            return ("xpu", best_device.device_id)

        nvidia_devices = [d for d in devices if d.device_type == DeviceType.NVIDIA_GPU]
        if nvidia_devices:
            best_device = max(nvidia_devices, key=lambda d: d.free_memory_bytes)
            logger.info(
                f"Selected NVIDIA GPU: cuda:{best_device.device_id} ({best_device.name}), "
                f"{best_device.free_memory_bytes / (1024**3):.2f} GB free"
            )
            return ("cuda", best_device.device_id)

        # Fallback to CPU
        logger.info("No GPU available, falling back to CPU")
        return ("cpu", 0)

    def _device_type_to_string(self, device_type: DeviceType) -> Literal["xpu", "cuda", "cpu"]:
        """Convert DeviceType enum to torch device string."""
        if device_type == DeviceType.INTEL_ARC:
            return "xpu"
        elif device_type == DeviceType.NVIDIA_GPU:
            return "cuda"
        else:
            return "cpu"

    def get_device_memory(self, device_type_str: str, device_id: int) -> tuple[int, int]:
        """
        Get memory information for a specific device.

        Args:
            device_type_str: Device type string ("xpu", "cuda", or "cpu").
            device_id: Device ID.

        Returns:
            Tuple of (total_memory_bytes, free_memory_bytes).

        Requirements: 1.5, 9.2
        """
        if not self._torch_available:
            return (0, 0)

        try:
            if device_type_str == "xpu" and self._xpu_available:
                props = self._torch.xpu.get_device_properties(device_id)  # type: ignore
                allocated = self._torch.xpu.memory_allocated(device_id)  # type: ignore
                total: int = props.total_memory  # type: ignore
                free: int = total - allocated
                return (total, free)

            elif device_type_str == "cuda" and self._cuda_available:
                props = self._torch.cuda.get_device_properties(device_id)
                allocated = self._torch.cuda.memory_allocated(device_id)
                total: int = props.total_memory
                free: int = total - allocated
                return (total, free)

            else:
                # CPU has no meaningful memory limit
                return (0, 0)

        except Exception as e:
            logger.error(f"Error getting memory for {device_type_str}:{device_id}: {e}")
            return (0, 0)

    def is_device_available(self, device_type_str: str, device_id: int) -> bool:
        """
        Check if a device is available and healthy.

        Args:
            device_type_str: Device type string ("xpu", "cuda", or "cpu").
            device_id: Device ID.

        Returns:
            True if device is available and healthy, False otherwise.

        Requirements: 1.5, 10.5
        """
        if not self._torch_available:
            return device_type_str == "cpu"

        try:
            if device_type_str == "xpu":
                if not self._xpu_available:
                    return False
                device_count: int = self._torch.xpu.device_count()  # type: ignore
                if device_id >= device_count:
                    return False
                # Try a simple operation to verify device health
                test_tensor = self._torch.tensor([1.0], device=f"xpu:{device_id}")  # type: ignore
                _ = test_tensor + 1.0
                return True

            elif device_type_str == "cuda":
                if not self._cuda_available:
                    return False
                device_count: int = self._torch.cuda.device_count()
                if device_id >= device_count:
                    return False
                # Try a simple operation to verify device health
                test_tensor = self._torch.tensor([1.0], device=f"cuda:{device_id}")
                _ = test_tensor + 1.0
                return True

            elif device_type_str == "cpu":
                return True

            else:
                return False

        except Exception as e:
            logger.error(f"Device health check failed for {device_type_str}:{device_id}: {e}")
            return False

    def get_device_stats(self, device_type_str: str, device_id: int) -> dict[str, object]:
        """
        Get detailed statistics for a device.

        Args:
            device_type_str: Device type string ("xpu", "cuda", or "cpu").
            device_id: Device ID.

        Returns:
            Dictionary containing device statistics.

        Requirements: 9.2
        """
        stats: dict[str, object] = {
            "device_type": device_type_str,
            "device_id": device_id,
            "available": self.is_device_available(device_type_str, device_id),
        }

        total_memory, free_memory = self.get_device_memory(device_type_str, device_id)
        stats["total_memory_bytes"] = total_memory
        stats["free_memory_bytes"] = free_memory
        stats["used_memory_bytes"] = total_memory - free_memory

        if total_memory > 0:
            stats["memory_utilization_percent"] = (
                (total_memory - free_memory) / total_memory * 100
            )
        else:
            stats["memory_utilization_percent"] = 0.0

        logger.debug(f"Device stats for {device_type_str}:{device_id}: {stats}")

        return stats

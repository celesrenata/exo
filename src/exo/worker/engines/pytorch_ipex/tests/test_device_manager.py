"""
Tests for DeviceManager

These tests validate device detection, selection, and monitoring functionality.
"""

import pytest

from exo.worker.engines.pytorch_ipex.device_manager import (
    DeviceInfo,
    DeviceManager,
    DeviceType,
)


def test_device_manager_initialization() -> None:
    """Test that DeviceManager initializes correctly."""
    manager = DeviceManager()
    assert manager is not None


def test_detect_devices() -> None:
    """Test device detection returns at least CPU."""
    manager = DeviceManager()
    devices = manager.detect_devices()

    # Should always have at least CPU
    assert len(devices) > 0

    # Should have at least one CPU device
    cpu_devices = [d for d in devices if d.device_type == DeviceType.CPU]
    assert len(cpu_devices) > 0


def test_select_device() -> None:
    """Test device selection returns valid device."""
    manager = DeviceManager()
    device_str, device_id = manager.select_device()

    # Should return valid device type
    assert device_str in ["xpu", "cuda", "cpu"]

    # Device ID should be non-negative
    assert device_id >= 0


def test_select_device_with_cpu_preference() -> None:
    """Test device selection with CPU preference."""
    manager = DeviceManager()
    device_str, device_id = manager.select_device(preference=DeviceType.CPU)

    # Should return CPU
    assert device_str == "cpu"
    assert device_id == 0


def test_is_device_available_cpu() -> None:
    """Test that CPU is always available."""
    manager = DeviceManager()
    assert manager.is_device_available("cpu", 0) is True


def test_get_device_memory_cpu() -> None:
    """Test getting memory for CPU device."""
    manager = DeviceManager()
    total, free = manager.get_device_memory("cpu", 0)

    # CPU returns (0, 0) as it has no meaningful memory limit
    assert total == 0
    assert free == 0


def test_get_device_stats_cpu() -> None:
    """Test getting stats for CPU device."""
    manager = DeviceManager()
    stats = manager.get_device_stats("cpu", 0)

    assert stats["device_type"] == "cpu"
    assert stats["device_id"] == 0
    assert stats["available"] is True
    assert stats["total_memory_bytes"] == 0
    assert stats["free_memory_bytes"] == 0


def test_device_info_immutable() -> None:
    """Test that DeviceInfo is immutable (frozen dataclass)."""
    device = DeviceInfo(
        device_type=DeviceType.CPU,
        device_id=0,
        name="CPU",
        total_memory_bytes=0,
        free_memory_bytes=0,
    )

    # Should not be able to modify frozen dataclass
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        device.device_id = 1  # type: ignore


def test_device_type_enum() -> None:
    """Test DeviceType enum values."""
    assert DeviceType.INTEL_ARC.value == "intel_arc"
    assert DeviceType.NVIDIA_GPU.value == "nvidia_gpu"
    assert DeviceType.CPU.value == "cpu"

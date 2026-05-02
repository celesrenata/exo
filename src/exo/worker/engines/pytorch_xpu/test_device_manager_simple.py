#!/usr/bin/env python3
"""
Simple validation script for DeviceManager.

This script tests the DeviceManager functionality without requiring pytest.
"""

import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Test DeviceManager functionality."""
    try:
        from exo.worker.engines.pytorch_xpu.device_manager import (
            DeviceManager,
            DeviceType,
        )

        logger.info("=" * 60)
        logger.info("Testing DeviceManager")
        logger.info("=" * 60)

        # Test 1: Initialize DeviceManager
        logger.info("\nTest 1: Initialize DeviceManager")
        manager = DeviceManager()
        logger.info("✓ DeviceManager initialized successfully")

        # Test 2: Detect devices
        logger.info("\nTest 2: Detect devices")
        devices = manager.detect_devices()
        logger.info(f"✓ Detected {len(devices)} device(s)")
        for device in devices:
            logger.info(
                f"  - {device.device_type.value}: {device.name} "
                f"(ID: {device.device_id}, "
                f"Memory: {device.total_memory_bytes / (1024**3):.2f} GB)"
            )

        # Test 3: Select device
        logger.info("\nTest 3: Select device")
        device_str, device_id = manager.select_device()
        logger.info(f"✓ Selected device: {device_str}:{device_id}")

        # Test 4: Select device with CPU preference
        logger.info("\nTest 4: Select device with CPU preference")
        device_str, device_id = manager.select_device(preference=DeviceType.CPU)
        logger.info(f"✓ Selected CPU device: {device_str}:{device_id}")
        assert device_str == "cpu", "CPU preference should return CPU device"

        # Test 5: Check device availability
        logger.info("\nTest 5: Check device availability")
        is_available = manager.is_device_available("cpu", 0)
        logger.info(f"✓ CPU device available: {is_available}")
        assert is_available, "CPU should always be available"

        # Test 6: Get device memory
        logger.info("\nTest 6: Get device memory")
        total, free = manager.get_device_memory("cpu", 0)
        logger.info(f"✓ CPU memory: total={total}, free={free}")

        # Test 7: Get device stats
        logger.info("\nTest 7: Get device stats")
        stats = manager.get_device_stats("cpu", 0)
        logger.info(f"✓ CPU stats: {stats}")

        # Test 8: Test with selected device
        logger.info("\nTest 8: Test with selected device")
        device_str, device_id = manager.select_device()
        is_available = manager.is_device_available(device_str, device_id)
        logger.info(f"✓ Selected device {device_str}:{device_id} available: {is_available}")

        if is_available:
            total, free = manager.get_device_memory(device_str, device_id)
            logger.info(
                f"✓ Device memory: total={total / (1024**3):.2f} GB, "
                f"free={free / (1024**3):.2f} GB"
            )

            stats = manager.get_device_stats(device_str, device_id)
            logger.info(f"✓ Device stats: {stats}")

        logger.info("\n" + "=" * 60)
        logger.info("All tests passed!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

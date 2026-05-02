#!/usr/bin/env python3
"""
Simple test for PyTorchIPEXBackend

This script tests basic functionality of the PyTorch XPU backend:
- Backend initialization
- Device detection
- Component integration
"""

import sys


def test_backend_initialization() -> None:
    """Test that the backend can be initialized."""
    print("Testing PyTorchIPEXBackend initialization...")

    try:
        from exo.worker.engines.pytorch_ipex import PyTorchIPEXBackend

        backend = PyTorchIPEXBackend()
        print("✓ Backend initialized successfully")
        print(f"  Device: {backend._device_type}:{backend._device_id}")

        # Get stats
        stats = backend.get_stats()
        print("✓ Backend stats retrieved:")
        print(f"  Device type: {stats['device_type']}")
        print(f"  Device ID: {stats['device_id']}")
        print(f"  Loaded models: {stats['loaded_models']}")

        # Cleanup
        backend.cleanup()
        print("✓ Backend cleanup successful")

        return True

    except Exception as e:
        print(f"✗ Backend initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_device_manager() -> None:
    """Test device manager functionality."""
    print("\nTesting DeviceManager...")

    try:
        from exo.worker.engines.pytorch_ipex import DeviceManager

        manager = DeviceManager()
        print("✓ DeviceManager initialized")

        # Detect devices
        devices = manager.detect_devices()
        print(f"✓ Detected {len(devices)} device(s):")
        for device in devices:
            print(f"  - {device.device_type}: {device.name}")

        # Select device
        device_type, device_id = manager.select_device()
        print(f"✓ Selected device: {device_type}:{device_id}")

        # Check device availability
        is_available = manager.is_device_available(device_type, device_id)
        print(f"✓ Device available: {is_available}")

        return True

    except Exception as e:
        print(f"✗ DeviceManager test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_error_classes() -> None:
    """Test custom error classes."""
    print("\nTesting error classes...")

    try:
        from exo.worker.engines.pytorch_ipex import (
            CacheError,
            DeviceError,
            InferenceError,
            ModelError,
        )

        # Test DeviceError
        try:
            raise DeviceError(
                message="Test device error",
                device_type="xpu",
                device_id=0,
            )
        except DeviceError as e:
            print(f"✓ DeviceError: {e}")

        # Test ModelError
        try:
            raise ModelError(
                message="Test model error",
                model_id="test-model",
                shard_info="layers 0-10",
            )
        except ModelError as e:
            print(f"✓ ModelError: {e}")

        # Test InferenceError
        try:
            raise InferenceError(
                message="Test inference error",
                request_id="test-request",
                model_id="test-model",
            )
        except InferenceError as e:
            print(f"✓ InferenceError: {e}")

        # Test CacheError
        try:
            raise CacheError(
                message="Test cache error",
                request_id="test-request",
            )
        except CacheError as e:
            print(f"✓ CacheError: {e}")

        return True

    except Exception as e:
        print(f"✗ Error classes test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main() -> int:
    """Run all tests."""
    print("=" * 60)
    print("PyTorchIPEXBackend Simple Test")
    print("=" * 60)

    results = []

    # Test device manager first (doesn't require PyTorch)
    results.append(("DeviceManager", test_device_manager()))

    # Test error classes
    results.append(("Error Classes", test_error_classes()))

    # Test backend initialization (requires PyTorch)
    results.append(("Backend Initialization", test_backend_initialization()))

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(passed for _, passed in results)
    print("=" * 60)

    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

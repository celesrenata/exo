#!/usr/bin/env python3
"""
Simple test to verify the enhanced runner supervisor functionality.
This test focuses on the core enhancements without requiring full system setup.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def test_error_handler_creation():
    """Test that the error handler can be created and configured."""
    try:
        from exo.worker.runner.error_handler import (
            ErrorHandler,
            RetryConfig,
        )

        # Create error handler
        handler = ErrorHandler()

        # Test retry config registration
        config = RetryConfig(max_attempts=3, base_delay=1.0)
        handler.register_retry_config("test_operation", config)

        # Test error handling (with a mock error context)
        print("✓ ErrorHandler creation and configuration successful")
        return True

    except Exception as e:
        print(f"✗ ErrorHandler test failed: {e}")
        return False


def test_resource_manager_creation():
    """Test that the resource manager can be created."""
    try:
        from exo.worker.runner.resource_manager import (
            ResourceManager,
            ResourceState,
            ResourceType,
        )

        # Create resource manager
        manager = ResourceManager()

        # Test resource registration (with a mock resource)
        class MockResource:
            def close(self):
                pass

        mock_resource = MockResource()
        handle = manager.register_resource(
            resource=mock_resource,
            resource_type=ResourceType.CUSTOM,
            cleanup_order=10,
            cleanup_func=mock_resource.close,
        )

        # Test resource state tracking
        assert handle.state == ResourceState.ACTIVE

        print("✓ ResourceManager creation and registration successful")
        return True

    except Exception as e:
        print(f"✗ ResourceManager test failed: {e}")
        return False


def test_shutdown_coordinator_creation():
    """Test that the shutdown coordinator can be created."""
    try:
        from exo.worker.runner.shutdown_coordinator import (
            ShutdownCoordinator,
        )

        # Create shutdown coordinator
        coordinator = ShutdownCoordinator()

        # Test handler registration
        def mock_handler(runner_id: str):
            pass

        coordinator.register_shutdown_handler(mock_handler)

        print("✓ ShutdownCoordinator creation and handler registration successful")
        return True

    except Exception as e:
        print(f"✗ ShutdownCoordinator test failed: {e}")
        return False


async def test_integration():
    """Test that all components can work together."""
    try:
        from exo.worker.runner.error_handler import get_error_handler
        from exo.worker.runner.resource_manager import get_resource_manager
        from exo.worker.runner.shutdown_coordinator import get_shutdown_coordinator

        # Get global instances
        error_handler = get_error_handler()
        resource_manager = get_resource_manager()
        shutdown_coordinator = get_shutdown_coordinator()

        # Test that they're properly initialized
        assert error_handler is not None
        assert resource_manager is not None
        assert shutdown_coordinator is not None

        # Test error handling
        recovery_action = await error_handler.handle_error(
            error=ValueError("Test error"),
            component="TestComponent",
            operation="test_operation",
        )

        assert recovery_action is not None

        print("✓ Component integration test successful")
        return True

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing Enhanced Runner Supervisor Components...")
    print("=" * 50)

    tests = [
        test_error_handler_creation,
        test_resource_manager_creation,
        test_shutdown_coordinator_creation,
    ]

    async_tests = [
        test_integration,
    ]

    # Run synchronous tests
    sync_results = []
    for test in tests:
        result = test()
        sync_results.append(result)

    # Run asynchronous tests
    async_results = []
    for test in async_tests:
        try:
            result = asyncio.run(test())
            async_results.append(result)
        except Exception as e:
            print(f"✗ Async test {test.__name__} failed: {e}")
            async_results.append(False)

    # Summary
    all_results = sync_results + async_results
    passed = sum(all_results)
    total = len(all_results)

    print("=" * 50)
    print(f"Test Results: {passed}/{total} passed")

    if passed == total:
        print(
            "🎉 All tests passed! Enhanced runner supervisor components are working correctly."
        )
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

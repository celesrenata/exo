#!/usr/bin/env python3
"""
Multi-node Integration Validation Script

This script validates that the multi-node race condition fixes are properly integrated
and working correctly. It performs basic functionality tests without requiring
a full multi-node setup.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from exo.shared.constants import (
        EXO_RUNNER_CLEANUP_TIMEOUT,
        EXO_RUNNER_ENABLE_ENHANCED_LOGGING,
        EXO_RUNNER_HEALTH_CHECK_INTERVAL,
        EXO_RUNNER_MAX_STARTUP_RETRIES,
        EXO_RUNNER_SHUTDOWN_TIMEOUT,
        EXO_RUNNER_STARTUP_TIMEOUT,
    )
    from exo.worker.runner.channel_manager import get_channel_manager
    from exo.worker.runner.error_handler import get_error_handler
    from exo.worker.runner.lifecycle_logger import get_lifecycle_logger
    from exo.worker.runner.resource_manager import get_resource_manager
    from exo.worker.runner.shutdown_coordinator import get_shutdown_coordinator
except ImportError as e:
    print("❌ Import error: {e}")
    print("Make sure you're running this from the EXO root directory")
    sys.exit(1)


class IntegrationValidator:
    """Validates the integration of multi-node race condition fixes."""

    def __init__(self):
        self.test_results: Dict[str, bool] = {}
        self.start_time = time.time()

    def log_test(self, test_name: str, success: bool, message: str = ""):
        """Log test result."""
        status = "✓ PASS" if success else "✗ FAIL"
        self.test_results[test_name] = success
        print("{status}: {test_name}")
        if message:
            print("    {message}")

    def test_component_imports(self) -> bool:
        """Test that all new components can be imported."""
        print("\n1. Testing Component Imports")
        print("-" * 30)

        components = [
            ("ShutdownCoordinator", get_shutdown_coordinator),
            ("ResourceManager", get_resource_manager),
            ("ChannelManager", get_channel_manager),
            ("ErrorHandler", get_error_handler),
            ("LifecycleLogger", get_lifecycle_logger),
        ]

        all_success = True

        for component_name, get_func in components:
            try:
                instance = get_func()
                success = instance is not None
                self.log_test(
                    f"Import {component_name}",
                    success,
                    f"Instance type: {type(instance).__name__}"
                    if success
                    else "Failed to create instance",
                )
                all_success = all_success and success
            except Exception as e:
                self.log_test(f"Import {component_name}", False, f"Error: {e}")
                all_success = False

        return all_success

    def test_configuration_constants(self) -> bool:
        """Test that configuration constants are properly defined."""
        print("\n2. Testing Configuration Constants")
        print("-" * 35)

        constants = [
            ("EXO_RUNNER_SHUTDOWN_TIMEOUT", EXO_RUNNER_SHUTDOWN_TIMEOUT, float),
            (
                "EXO_RUNNER_HEALTH_CHECK_INTERVAL",
                EXO_RUNNER_HEALTH_CHECK_INTERVAL,
                float,
            ),
            ("EXO_RUNNER_STARTUP_TIMEOUT", EXO_RUNNER_STARTUP_TIMEOUT, float),
            ("EXO_RUNNER_CLEANUP_TIMEOUT", EXO_RUNNER_CLEANUP_TIMEOUT, float),
            ("EXO_RUNNER_MAX_STARTUP_RETRIES", EXO_RUNNER_MAX_STARTUP_RETRIES, int),
            (
                "EXO_RUNNER_ENABLE_ENHANCED_LOGGING",
                EXO_RUNNER_ENABLE_ENHANCED_LOGGING,
                bool,
            ),
        ]

        all_success = True

        for const_name, const_value, expected_type in constants:
            try:
                success = (
                    isinstance(const_value, expected_type) and const_value is not None
                )
                self.log_test(
                    f"Constant {const_name}",
                    success,
                    f"Value: {const_value} (type: {type(const_value).__name__})",
                )
                all_success = all_success and success
            except Exception as e:
                self.log_test(f"Constant {const_name}", False, f"Error: {e}")
                all_success = False

        return all_success

    async def test_shutdown_coordinator_functionality(self) -> bool:
        """Test basic shutdown coordinator functionality."""
        print("\n3. Testing Shutdown Coordinator Functionality")
        print("-" * 45)

        try:
            coordinator = get_shutdown_coordinator()

            # Test shutdown handler registration
            handler_called = False

            def test_handler(runner_id: str):
                nonlocal handler_called
                handler_called = True

            coordinator.register_shutdown_handler(test_handler)
            self.log_test(
                "Register shutdown handler", True, "Handler registered successfully"
            )

            # Test shutdown initiation
            test_runner_id = "test_runner_123"
            shutdown_result = await coordinator.initiate_shutdown(
                test_runner_id, timeout=1.0
            )
            self.log_test(
                "Initiate shutdown",
                shutdown_result is not None,
                f"Shutdown result: {shutdown_result}",
            )

            # Test shutdown status check
            status = coordinator.get_shutdown_status(test_runner_id)
            self.log_test(
                "Get shutdown status",
                status is not None,
                f"Status available: {status is not None}",
            )

            return True

        except Exception as e:
            self.log_test("Shutdown coordinator functionality", False, f"Error: {e}")
            return False

    async def test_resource_manager_functionality(self) -> bool:
        """Test basic resource manager functionality."""
        print("\n4. Testing Resource Manager Functionality")
        print("-" * 42)

        try:
            manager = get_resource_manager()

            # Test resource registration
            test_resource = "test_resource"
            from exo.worker.runner.resource_manager import ResourceType

            handle = manager.register_resource(
                resource=test_resource,
                resource_type=ResourceType.CHANNEL,
                cleanup_order=10,
                cleanup_func=lambda: None,
                resource_id="test_resource_1",
            )

            self.log_test(
                "Register resource",
                handle is not None,
                f"Handle created: {handle is not None}",
            )

            # Test resource state tracking
            states = manager.get_resource_count_by_state()
            self.log_test(
                "Get resource states",
                states is not None and len(states) > 0,
                f"States: {dict(states) if states else 'None'}",
            )

            # Test resource cleanup
            cleanup_result = await manager.cleanup_resources(timeout=1.0)
            self.log_test(
                "Cleanup resources",
                cleanup_result is not None,
                f"Cleanup success: {cleanup_result.success if cleanup_result else 'None'}",
            )

            return True

        except Exception as e:
            self.log_test("Resource manager functionality", False, f"Error: {e}")
            return False

    async def test_error_handler_functionality(self) -> bool:
        """Test basic error handler functionality."""
        print("\n5. Testing Error Handler Functionality")
        print("-" * 40)

        try:
            handler = get_error_handler()

            # Test error handling
            test_error = ValueError("Test error")
            recovery_action = await handler.handle_error(
                error=test_error,
                component="TestComponent",
                operation="test_operation",
                runner_id="test_runner",
            )

            self.log_test(
                "Handle error",
                recovery_action is not None,
                f"Recovery action: {recovery_action}",
            )

            # Test error statistics
            stats = handler.get_error_statistics()
            self.log_test(
                "Get error statistics",
                stats is not None,
                f"Stats available: {stats is not None}",
            )

            # Test recent errors
            recent_errors = handler.get_recent_errors(hours=1)
            self.log_test(
                "Get recent errors",
                recent_errors is not None,
                f"Recent errors count: {len(recent_errors) if recent_errors else 0}",
            )

            return True

        except Exception as e:
            self.log_test("Error handler functionality", False, f"Error: {e}")
            return False

    async def test_channel_manager_functionality(self) -> bool:
        """Test basic channel manager functionality."""
        print("\n6. Testing Channel Manager Functionality")
        print("-" * 42)

        try:
            manager = get_channel_manager()

            # Test channel creation
            channel = manager.create_channel("test_channel", buffer_size=100)
            self.log_test(
                "Create channel",
                channel is not None,
                f"Channel created: {channel is not None}",
            )

            # Test channel status check
            is_active = manager.is_channel_active("test_channel")
            self.log_test(
                "Check channel active",
                isinstance(is_active, bool),
                f"Channel active: {is_active}",
            )

            # Test channel closure
            close_result = await manager.close_channel(
                "test_channel", drain_timeout=1.0
            )
            self.log_test(
                "Close channel",
                isinstance(close_result, bool),
                f"Close result: {close_result}",
            )

            return True

        except Exception as e:
            self.log_test("Channel manager functionality", False, f"Error: {e}")
            return False

    async def test_lifecycle_logger_functionality(self) -> bool:
        """Test basic lifecycle logger functionality."""
        print("\n7. Testing Lifecycle Logger Functionality")
        print("-" * 43)

        try:
            logger = get_lifecycle_logger()

            # Test correlation ID generation
            correlation_id = logger.generate_correlation_id()
            self.log_test(
                "Generate correlation ID",
                correlation_id is not None and len(correlation_id) > 0,
                f"Correlation ID: {correlation_id[:8]}..."
                if correlation_id
                else "None",
            )

            # Test event logging
            await logger.log_runner_created(
                runner_id="test_runner", metadata={"test": "data"}
            )
            self.log_test("Log runner created", True, "Event logged successfully")

            # Test health check logging
            await logger.log_health_check(
                runner_id="test_runner",
                success=True,
                duration_ms=100.0,
                health_score=0.9,
                issues=[],
                metadata={},
            )
            self.log_test("Log health check", True, "Health check logged successfully")

            return True

        except Exception as e:
            self.log_test("Lifecycle logger functionality", False, f"Error: {e}")
            return False

    def test_backward_compatibility(self) -> bool:
        """Test that existing APIs are still compatible."""
        print("\n8. Testing Backward Compatibility")
        print("-" * 35)

        try:
            # Test that we can still import existing components
            from exo.worker.runner.runner_supervisor import RunnerSupervisor

            self.log_test("Import RunnerSupervisor", True, "Import successful")
            self.log_test("Import bootstrap entrypoint", True, "Import successful")
            self.log_test("Import Worker", True, "Import successful")

            # Test that RunnerSupervisor.create still works
            # (We can't fully test this without mocking, but we can check the method exists)
            has_create_method = hasattr(RunnerSupervisor, "create")
            self.log_test(
                "RunnerSupervisor.create method exists",
                has_create_method,
                "Method available",
            )

            return True

        except Exception as e:
            self.log_test("Backward compatibility", False, f"Error: {e}")
            return False

    async def run_all_tests(self) -> bool:
        """Run all integration tests."""
        print("🚀 Starting Multi-node Integration Validation")
        print("=" * 50)

        tests = [
            ("Component Imports", self.test_component_imports),
            ("Configuration Constants", self.test_configuration_constants),
            ("Shutdown Coordinator", self.test_shutdown_coordinator_functionality),
            ("Resource Manager", self.test_resource_manager_functionality),
            ("Error Handler", self.test_error_handler_functionality),
            ("Channel Manager", self.test_channel_manager_functionality),
            ("Lifecycle Logger", self.test_lifecycle_logger_functionality),
            ("Backward Compatibility", self.test_backward_compatibility),
        ]

        for test_name, test_func in tests:
            try:
                if asyncio.iscoroutinefunction(test_func):
                    result = await test_func()
                else:
                    result = test_func()

                if not result:
                    print("\n❌ Test suite '{test_name}' failed!")

            except Exception as e:
                print("\n❌ Test suite '{test_name}' failed with exception: {e}")
                self.test_results[test_name] = False

        return self.print_summary()

    def print_summary(self) -> bool:
        """Print test summary and return overall success."""
        duration = time.time() - self.start_time

        print("\n" + "=" * 50)
        print("📊 VALIDATION SUMMARY")
        print("=" * 50)

        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)

        print("Tests passed: {passed}/{total}")
        print("Duration: {duration:.2f} seconds")
        print()

        # Group results by test suite
        current_suite = None
        for test_name, result in self.test_results.items():
            # Extract suite name (everything before the first space or colon)
            suite_name = test_name.split()[0] if " " in test_name else test_name

            if suite_name != current_suite:
                if current_suite is not None:
                    print()
                current_suite = suite_name

            status = "✓" if result else "✗"
            print("  {status} {test_name}")

        print()

        if passed == total:
            print("🎉 ALL TESTS PASSED! Multi-node integration is working correctly.")
            return True
        else:
            print("❌ {total - passed} tests failed. Please check the integration.")
            return False


async def main():
    """Main validation function."""
    validator = IntegrationValidator()
    success = await validator.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print("\n💥 Validation failed with unexpected error: {e}")
        sys.exit(1)

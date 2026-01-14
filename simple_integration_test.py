#!/usr/bin/env python3
"""
Simple Integration Test

Tests basic integration without external dependencies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_imports():
    """Test that we can import the new components."""
    print("Testing imports...")

    try:
        # Test constants import
        from exo.shared.constants import (
            EXO_RUNNER_CLEANUP_TIMEOUT,
            EXO_RUNNER_ENABLE_ENHANCED_LOGGING,
            EXO_RUNNER_HEALTH_CHECK_INTERVAL,
            EXO_RUNNER_MAX_STARTUP_RETRIES,
            EXO_RUNNER_SHUTDOWN_TIMEOUT,
            EXO_RUNNER_STARTUP_TIMEOUT,
        )

        print("✓ Constants imported successfully")
        print("  - Shutdown timeout: {EXO_RUNNER_SHUTDOWN_TIMEOUT}")
        print("  - Health check interval: {EXO_RUNNER_HEALTH_CHECK_INTERVAL}")
        print("  - Enhanced logging: {EXO_RUNNER_ENABLE_ENHANCED_LOGGING}")

        # Test component imports
        from exo.worker.runner.channel_manager import ChannelManager
        from exo.worker.runner.error_handler import ErrorHandler
        from exo.worker.runner.lifecycle_logger import LifecycleLogger
        from exo.worker.runner.resource_manager import ResourceManager
        from exo.worker.runner.shutdown_coordinator import ShutdownCoordinator

        print("✓ All new components imported successfully")

        # Test existing components still work
        from exo.worker.runner.bootstrap import entrypoint
        from exo.worker.runner.runner_supervisor import RunnerSupervisor

        print("✓ Existing components still importable")

        return True

    except ImportError as e:
        print("✗ Import failed: {e}")
        return False
    except Exception as e:
        print("✗ Unexpected error: {e}")
        return False


def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")

    required_files = [
        "src/exo/shared/constants.py",
        "src/exo/worker/main.py",
        "src/exo/worker/engines/engine_init.py",
        "src/exo/worker/runner/runner_supervisor.py",
        "src/exo/worker/runner/bootstrap.py",
        "src/exo/worker/runner/shutdown_coordinator.py",
        "src/exo/worker/runner/resource_manager.py",
        "src/exo/worker/runner/channel_manager.py",
        "src/exo/worker/runner/error_handler.py",
        "src/exo/worker/runner/lifecycle_logger.py",
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print("✗ Missing files:")
        for file_path in missing_files:
            print("  - {file_path}")
        return False
    else:
        print("✓ All required files exist")
        return True


def test_configuration():
    """Test configuration values."""
    print("\nTesting configuration...")

    try:
        from exo.shared.constants import (
            EXO_RUNNER_CLEANUP_TIMEOUT,
            EXO_RUNNER_ENABLE_ENHANCED_LOGGING,
            EXO_RUNNER_HEALTH_CHECK_INTERVAL,
            EXO_RUNNER_MAX_STARTUP_RETRIES,
            EXO_RUNNER_SHUTDOWN_TIMEOUT,
            EXO_RUNNER_STARTUP_TIMEOUT,
        )

        # Check types and reasonable values
        checks = [
            ("EXO_RUNNER_SHUTDOWN_TIMEOUT", EXO_RUNNER_SHUTDOWN_TIMEOUT, float, 0.0),
            (
                "EXO_RUNNER_HEALTH_CHECK_INTERVAL",
                EXO_RUNNER_HEALTH_CHECK_INTERVAL,
                float,
                0.0,
            ),
            ("EXO_RUNNER_STARTUP_TIMEOUT", EXO_RUNNER_STARTUP_TIMEOUT, float, 0.0),
            ("EXO_RUNNER_CLEANUP_TIMEOUT", EXO_RUNNER_CLEANUP_TIMEOUT, float, 0.0),
            ("EXO_RUNNER_MAX_STARTUP_RETRIES", EXO_RUNNER_MAX_STARTUP_RETRIES, int, 0),
            (
                "EXO_RUNNER_ENABLE_ENHANCED_LOGGING",
                EXO_RUNNER_ENABLE_ENHANCED_LOGGING,
                bool,
                None,
            ),
        ]

        all_good = True
        for name, value, expected_type, min_value in checks:
            if not isinstance(value, expected_type):
                print(
                    f"✗ {name} has wrong type: {type(value)} (expected {expected_type})"
                )
                all_good = False
            elif min_value is not None and value < min_value:
                print("✗ {name} has invalid value: {value} (should be >= {min_value})")
                all_good = False
            else:
                print("✓ {name}: {value}")

        return all_good

    except Exception as e:
        print("✗ Configuration test failed: {e}")
        return False


def test_integration_points():
    """Test key integration points."""
    print("\nTesting integration points...")

    try:
        # Test that worker main imports the constants
        with open("src/exo/worker/main.py", "r") as f:
            worker_content = f.read()

        if "EXO_RUNNER_SHUTDOWN_TIMEOUT" in worker_content:
            print("✓ Worker main.py uses new constants")
        else:
            print("✗ Worker main.py doesn't use new constants")
            return False

        # Test that engine init has enhanced logging
        with open("src/exo/worker/engines/engine_init.py", "r") as f:
            engine_content = f.read()

        if "EXO_RUNNER_ENABLE_ENHANCED_LOGGING" in engine_content:
            print("✓ Engine init has enhanced logging")
        else:
            print("✗ Engine init doesn't have enhanced logging")
            return False

        # Test that constants file has new constants
        with open("src/exo/shared/constants.py", "r") as f:
            constants_content = f.read()

        required_constants = [
            "EXO_RUNNER_SHUTDOWN_TIMEOUT",
            "EXO_RUNNER_HEALTH_CHECK_INTERVAL",
            "EXO_RUNNER_STARTUP_TIMEOUT",
        ]

        for const in required_constants:
            if const not in constants_content:
                print("✗ Missing constant: {const}")
                return False

        print("✓ All required constants present")
        return True

    except Exception as e:
        print("✗ Integration points test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Simple Integration Test")
    print("=" * 30)

    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Integration Points", test_integration_points),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print("✗ {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 30)
    print("📊 SUMMARY")
    print("=" * 30)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print("{status}: {test_name}")

    print("\nResult: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 Integration test PASSED!")
        return 0
    else:
        print("❌ Integration test FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

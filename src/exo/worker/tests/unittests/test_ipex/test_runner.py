#!/usr/bin/env python3
"""
Test runner for IPEX engine tests.

This script provides a convenient way to run all IPEX-related tests
with proper configuration and reporting.
"""

import sys
from pathlib import Path

import pytest


def run_ipex_tests():
    """Run all IPEX engine tests."""

    # Get the directory containing this script
    test_dir = Path(__file__).parent

    # Configure pytest arguments
    pytest_args = [
        str(test_dir),  # Run tests in this directory
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker checking
        "-x",  # Stop on first failure (optional)
        "--disable-warnings",  # Disable warnings for cleaner output
    ]

    # Add coverage if available
    try:
        import coverage

        pytest_args.extend(
            [
                "--cov=exo.worker.engines.ipex",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/ipex",
            ]
        )
        print("Running with coverage reporting...")
    except ImportError:
        print("Coverage not available, running without coverage...")

    print("=" * 60)
    print("Running Intel IPEX Engine Tests")
    print("=" * 60)
    print(f"Test directory: {test_dir}")
    print(f"Python version: {sys.version}")
    print("=" * 60)

    # Run the tests
    exit_code = pytest.main(pytest_args)

    print("=" * 60)
    if exit_code == 0:
        print("✅ All IPEX tests passed!")
    else:
        print("❌ Some IPEX tests failed!")
    print("=" * 60)

    return exit_code


def run_specific_test_module(module_name):
    """Run tests from a specific module."""

    test_dir = Path(__file__).parent
    module_path = test_dir / f"{module_name}.py"

    if not module_path.exists():
        print(f"❌ Test module not found: {module_path}")
        return 1

    pytest_args = [str(module_path), "-v", "--tb=short", "--disable-warnings"]

    print(f"Running tests from: {module_name}")
    return pytest.main(pytest_args)


def list_available_tests():
    """List all available test modules."""

    test_dir = Path(__file__).parent
    test_files = list(test_dir.glob("test_*.py"))

    print("Available IPEX test modules:")
    print("=" * 40)

    for test_file in sorted(test_files):
        if test_file.name != "test_runner.py":
            module_name = test_file.stem
            print(f"  {module_name}")

            # Try to extract test classes/functions for preview
            try:
                with open(test_file, "r") as f:
                    content = f.read()

                # Find test classes
                import re

                classes = re.findall(r"class (Test\w+)", content)
                if classes:
                    for cls in classes[:3]:  # Show first 3 classes
                        print(f"    - {cls}")
                    if len(classes) > 3:
                        print(f"    - ... and {len(classes) - 3} more")

            except Exception:
                pass

            print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "list":
            list_available_tests()
            sys.exit(0)
        elif command.startswith("test_"):
            # Run specific test module
            exit_code = run_specific_test_module(command)
            sys.exit(exit_code)
        elif command == "help":
            print("IPEX Test Runner")
            print("================")
            print()
            print("Usage:")
            print("  python test_runner.py              # Run all tests")
            print("  python test_runner.py list         # List available test modules")
            print("  python test_runner.py test_<name>  # Run specific test module")
            print("  python test_runner.py help         # Show this help")
            print()
            print("Examples:")
            print("  python test_runner.py test_ipex_engine_detection")
            print("  python test_runner.py test_ipex_inference")
            print("  python test_runner.py test_ipex_dashboard_integration")
            sys.exit(0)
        else:
            print(f"Unknown command: {command}")
            print("Use 'python test_runner.py help' for usage information")
            sys.exit(1)
    else:
        # Run all tests
        exit_code = run_ipex_tests()
        sys.exit(exit_code)

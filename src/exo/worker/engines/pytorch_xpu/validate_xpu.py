#!/usr/bin/env python3
"""
XPU Functionality Validation Script

This script validates that native PyTorch XPU support (2.11+) is working correctly
by performing basic tensor operations on Intel Arc GPUs. No IPEX dependency.

Tests include:
1. Basic tensor operations (creation, movement, arithmetic)
2. Matrix multiplication (matmul) performance
3. Softmax operations
4. bfloat16 precision support

Requirements:
- PyTorch 2.11+ (native XPU support)
- Intel Arc GPU with drivers installed
"""

import sys
import time
from typing import Final

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore

try:
    from loguru import logger  # type: ignore
except ImportError:
    import logging

    logger = logging.getLogger(__name__)  # type: ignore

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="DEBUG",
)


def validate_basic_operations(device: "torch.device") -> bool:
    """
    Validate basic tensor operations on the specified device.

    Args:
        device: The torch device to test (e.g., xpu:0)

    Returns:
        True if all operations succeed, False otherwise
    """
    logger.info("\n" + "=" * 60)
    logger.info("Test 1: Basic Tensor Operations")
    logger.info("=" * 60)

    try:
        import torch

        # Create tensors on device
        logger.info("Creating tensors on device...")
        a = torch.randn(100, 100, device=device)
        b = torch.randn(100, 100, device=device)

        # Basic arithmetic
        logger.info("Testing addition...")
        c = a + b
        assert c.device.type == device.type, "Result tensor not on correct device"

        logger.info("Testing multiplication...")
        d = a * b
        assert d.device.type == device.type, "Result tensor not on correct device"

        # Move tensor to CPU and back
        logger.info("Testing device transfer (XPU -> CPU -> XPU)...")
        cpu_tensor = a.cpu()
        assert cpu_tensor.device.type == "cpu", "Tensor not moved to CPU"

        xpu_tensor = cpu_tensor.to(device)
        assert xpu_tensor.device.type == device.type, "Tensor not moved back to XPU"

        logger.info("✓ Basic tensor operations passed")
        return True

    except Exception as e:
        logger.error(f"✗ Basic tensor operations failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def benchmark_matmul(device: "torch.device", size: int = 2048) -> bool:
    """
    Benchmark matrix multiplication performance.

    Args:
        device: The torch device to test
        size: Size of square matrices to multiply

    Returns:
        True if benchmark succeeds, False otherwise
    """
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Matrix Multiplication (matmul) Benchmark")
    logger.info("=" * 60)

    try:
        import torch

        logger.info(f"Creating {size}x{size} matrices...")
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)

        # Warm-up
        logger.info("Warming up...")
        for _ in range(5):
            _ = torch.matmul(a, b)

        # Benchmark
        logger.info("Running benchmark (10 iterations)...")
        start_time = time.perf_counter()

        for _ in range(10):
            torch.matmul(a, b)

        # Ensure computation is complete
        if device.type == "xpu":
            torch.xpu.synchronize()

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        avg_time = elapsed_time / 10

        # Calculate GFLOPS
        # Matrix multiplication: 2 * size^3 FLOPs
        flops = 2 * (size**3)
        gflops = (flops / avg_time) / 1e9

        logger.info(f"Average time per matmul: {avg_time * 1000:.2f} ms")
        logger.info(f"Performance: {gflops:.2f} GFLOPS")
        logger.info("✓ Matrix multiplication benchmark passed")
        return True

    except Exception as e:
        logger.error(f"✗ Matrix multiplication benchmark failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def test_softmax(device: "torch.device") -> bool:
    """
    Test softmax operations.

    Args:
        device: The torch device to test

    Returns:
        True if test succeeds, False otherwise
    """
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Softmax Operations")
    logger.info("=" * 60)

    try:
        import torch

        logger.info("Creating test tensor...")
        x = torch.randn(128, 512, device=device)

        logger.info("Applying softmax...")
        y = torch.nn.functional.softmax(x, dim=-1)

        # Verify softmax properties
        logger.info("Verifying softmax properties...")
        sums = y.sum(dim=-1)

        # Check that all rows sum to approximately 1.0
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
            "Softmax rows don't sum to 1"
        )

        # Check that all values are in [0, 1]
        assert (y >= 0).all() and (y <= 1).all(), "Softmax values out of range [0, 1]"

        logger.info("✓ Softmax operations passed")
        return True

    except Exception as e:
        logger.error(f"✗ Softmax operations failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def test_bfloat16(device: "torch.device") -> bool:
    """
    Test bfloat16 precision support.

    Args:
        device: The torch device to test

    Returns:
        True if test succeeds, False otherwise
    """
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: bfloat16 Precision Support")
    logger.info("=" * 60)

    try:
        import torch

        logger.info("Creating bfloat16 tensors...")
        a = torch.randn(100, 100, dtype=torch.bfloat16, device=device)
        b = torch.randn(100, 100, dtype=torch.bfloat16, device=device)

        logger.info("Testing bfloat16 operations...")
        c = a + b
        assert c.dtype == torch.bfloat16, "Result tensor not in bfloat16"

        d = torch.matmul(a, b)
        assert d.dtype == torch.bfloat16, "Matmul result not in bfloat16"

        logger.info("✓ bfloat16 support passed")
        return True

    except Exception as e:
        logger.error(f"✗ bfloat16 support failed: {e}")
        logger.error("Note: bfloat16 may not be supported on all Intel Arc GPUs")
        import traceback

        logger.error(traceback.format_exc())
        return False


def validate_xpu_functionality() -> None:
    """
    Run all XPU validation tests.

    This function runs a comprehensive suite of tests to validate that
    native PyTorch XPU support is working correctly on Intel Arc GPUs.
    """
    logger.info("Starting XPU functionality validation (native PyTorch, no IPEX)...")

    # Check PyTorch installation
    try:
        import torch

        logger.info(f"PyTorch version: {torch.__version__}")
    except ImportError as e:
        logger.error(f"PyTorch not found: {e}")
        sys.exit(1)

    # Check XPU availability via native torch.xpu
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        logger.error("Intel XPU (Arc GPU) not available via native torch.xpu")
        sys.exit(1)

    device_count: Final[int] = torch.xpu.device_count()
    logger.info(f"Found {device_count} Intel XPU device(s)")

    if device_count == 0:
        logger.error("No Intel XPU devices found")
        sys.exit(1)

    # Use first device for testing
    device: Final["torch.device"] = torch.device("xpu:0")
    logger.info(f"Using device: {device}")

    # Run all tests
    results: dict[str, bool] = {}

    results["basic_operations"] = validate_basic_operations(device)
    results["matmul_benchmark"] = benchmark_matmul(device)
    results["softmax"] = test_softmax(device)
    results["bfloat16"] = test_bfloat16(device)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Validation Summary")
    logger.info("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name}: {status}")

    # Check if all tests passed
    all_passed = all(results.values())

    if all_passed:
        logger.info("\n" + "=" * 60)
        logger.info("All XPU validation tests PASSED!")
        logger.info("Intel Arc GPU is ready for native PyTorch inference")
        logger.info("=" * 60)
        sys.exit(0)
    else:
        logger.error("\n" + "=" * 60)
        logger.error("Some XPU validation tests FAILED")
        logger.error("Please check the errors above and fix any issues")
        logger.error("=" * 60)
        sys.exit(1)


def main() -> None:
    """Main entry point for the validation script."""
    try:
        validate_xpu_functionality()
    except KeyboardInterrupt:
        logger.info("\nValidation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

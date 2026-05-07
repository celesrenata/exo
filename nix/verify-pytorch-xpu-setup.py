#!/usr/bin/env python3
"""
Verification script for PyTorch with Intel XPU support.

This script verifies that PyTorch is correctly installed with
Intel Arc GPU (XPU) support using native torch.xpu (PyTorch 2.11+).
It performs a series of checks and reports the results.

Note: IPEX (Intel Extension for PyTorch) is discontinued.
PyTorch 2.11+ includes native XPU support.

Usage:
    python nix/verify-pytorch-xpu-setup.py [--strict]

Options:
    --strict    Require GPU to be present (fail if no GPU detected)

Exit codes:
    0 - All checks passed
    1 - Critical failure (imports failed, etc.)
    2 - GPU not available (only in strict mode)
    3 - GPU tests failed
"""

import sys
import argparse
from typing import Tuple, List


def check_imports() -> Tuple[bool, str]:
    """Check if PyTorch can be imported."""
    try:
        import torch
        return True, f"✓ PyTorch {torch.__version__} imported successfully"
    except ImportError as e:
        return False, f"✗ Failed to import PyTorch: {e}"


def check_xpu_available() -> Tuple[bool, str, bool]:
    """Check if XPU is available. Returns (success, message, gpu_present)."""
    try:
        import torch
        available = torch.xpu.is_available()
        if available:
            return True, "✓ XPU is available", True
        else:
            return True, "⚠ XPU is not available (no Intel Arc GPU detected)", False
    except Exception as e:
        return False, f"✗ Failed to check XPU availability: {e}", False


def check_xpu_device_count() -> Tuple[bool, str]:
    """Check XPU device count."""
    try:
        import torch
        if not torch.xpu.is_available():
            return True, "⊘ Skipped (no GPU)"
        
        count = torch.xpu.device_count()
        if count > 0:
            return True, f"✓ XPU device count: {count}"
        else:
            return False, "✗ XPU available but device count is 0"
    except Exception as e:
        return False, f"✗ Failed to get XPU device count: {e}"


def check_xpu_device_name() -> Tuple[bool, str]:
    """Check XPU device name."""
    try:
        import torch
        if not torch.xpu.is_available():
            return True, "⊘ Skipped (no GPU)"
        
        name = torch.xpu.get_device_name(0)
        return True, f"✓ XPU device name: {name}"
    except Exception as e:
        return False, f"✗ Failed to get XPU device name: {e}"


def check_tensor_creation() -> Tuple[bool, str]:
    """Test basic tensor creation on XPU."""
    try:
        import torch
        if not torch.xpu.is_available():
            return True, "⊘ Skipped (no GPU)"
        
        x = torch.randn(3, 3).to('xpu')
        if x.device.type == 'xpu':
            return True, f"✓ Tensor created on XPU: shape {x.shape}"
        else:
            return False, f"✗ Tensor not on XPU device: {x.device}"
    except Exception as e:
        return False, f"✗ Failed to create tensor on XPU: {e}"


def check_tensor_operations() -> Tuple[bool, str]:
    """Test basic tensor operations on XPU."""
    try:
        import torch
        if not torch.xpu.is_available():
            return True, "⊘ Skipped (no GPU)"
        
        x = torch.randn(3, 3).to('xpu')
        y = torch.randn(3, 3).to('xpu')
        z = torch.matmul(x, y)
        
        if z.device.type == 'xpu' and z.shape == (3, 3):
            return True, f"✓ Matrix multiplication on XPU successful: {z.shape}"
        else:
            return False, f"✗ Matrix multiplication failed: device={z.device}, shape={z.shape}"
    except Exception as e:
        return False, f"✗ Failed tensor operations on XPU: {e}"


def check_xpu_model_inference() -> Tuple[bool, str]:
    """Test model inference on XPU."""
    try:
        import torch
        
        if not torch.xpu.is_available():
            return True, "⊘ Skipped (no GPU)"
        
        # Create a simple model and move to XPU
        model = torch.nn.Linear(10, 10).to('xpu')
        model.eval()
        
        # Test inference
        x = torch.randn(1, 10).to('xpu')
        with torch.no_grad():
            y = model(x)
        
        if y.device.type == 'xpu':
            return True, "✓ XPU model inference successful"
        else:
            return False, f"✗ XPU inference failed: output device={y.device}"
    except Exception as e:
        return False, f"✗ Failed XPU model inference: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Verify PyTorch with Intel XPU support"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require GPU to be present (fail if no GPU detected)"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("PyTorch XPU Verification")
    print("=" * 70)
    print("Note: Using native torch.xpu support (IPEX is discontinued)")
    print()
    
    checks: List[Tuple[str, callable]] = [
        ("Import PyTorch", check_imports),
        ("Check XPU availability", check_xpu_available),
        ("Check XPU device count", check_xpu_device_count),
        ("Check XPU device name", check_xpu_device_name),
        ("Test tensor creation on XPU", check_tensor_creation),
        ("Test tensor operations on XPU", check_tensor_operations),
        ("Test XPU model inference", check_xpu_model_inference),
    ]
    
    results = []
    gpu_present = False
    critical_failure = False
    gpu_tests_failed = False
    
    for name, check_func in checks:
        print(f"Checking: {name}")
        result = check_func()
        
        if len(result) == 3:
            # Special case for XPU availability check
            success, message, gpu_detected = result
            gpu_present = gpu_detected
        else:
            success, message = result
        
        print(f"  {message}")
        results.append((name, success, message))
        
        # Check for critical failures
        if not success and name in ["Import PyTorch"]:
            critical_failure = True
        
        # Check for GPU test failures
        if not success and "XPU" in name and "⊘" not in message:
            gpu_tests_failed = True
        
        print()
    
    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"Checks passed: {passed}/{total}")
    print()
    
    if critical_failure:
        print("❌ CRITICAL FAILURE: PyTorch import failed")
        print("   Please install PyTorch following nix/PYTORCH_XPU_INSTALLATION.md")
        return 1
    
    if not gpu_present:
        if args.strict:
            print("❌ FAILURE: No Intel Arc GPU detected (strict mode)")
            print("   XPU is not available on this system")
            return 2
        else:
            print("⚠️  WARNING: No Intel Arc GPU detected")
            print("   PyTorch is installed correctly")
            print("   XPU features will not be available without Intel Arc GPU")
            print()
            print("   To test on a system with Intel Arc GPU:")
            print("   - Ensure Intel GPU drivers are installed")
            print("   - Check 'clinfo' shows Intel GPU")
            print("   - Check 'sycl-ls' shows Level Zero devices")
            return 0
    
    if gpu_tests_failed:
        print("❌ FAILURE: GPU tests failed")
        print("   Intel Arc GPU is detected but some operations failed")
        print("   Check the error messages above for details")
        return 3
    
    print("✅ SUCCESS: All checks passed")
    print("   PyTorch is correctly installed with XPU support")
    print("   Intel Arc GPU is available and working")
    return 0


if __name__ == "__main__":
    sys.exit(main())

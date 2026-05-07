#!/usr/bin/env python3
"""
Verification script for PyTorch with native XPU support.

This script verifies that:
1. PyTorch with XPU support is available
2. XPU devices are detected (if Intel Arc GPU present)
3. Basic tensor operations work on XPU
4. Model optimization works on XPU

Requirements:
- PyTorch 2.11+ with native XPU support
- Intel Arc GPU (optional, for full verification)

Note: IPEX (Intel Extension for PyTorch) is discontinued.
PyTorch 2.11+ includes native XPU support via torch.xpu.
"""

import sys
from typing import Optional


def check_pytorch_xpu() -> bool:
    """Verify PyTorch with XPU support is available."""
    print("=" * 60)
    print("1. Checking PyTorch XPU Support")
    print("=" * 60)
    
    try:
        import torch
        print("✓ PyTorch imported successfully")
        print(f"  Version: {torch.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import PyTorch: {e}")
        return False
    
    # Check if XPU is available
    try:
        xpu_available = torch.xpu.is_available()
        print(f"  XPU available: {xpu_available}")
        
        if xpu_available:
            device_count = torch.xpu.device_count()
            print(f"  XPU device count: {device_count}")
            
            for i in range(device_count):
                props = torch.xpu.get_device_properties(i)
                print(f"  Device {i}: {props.name}")
                print(f"    Total memory: {props.total_memory / 1024**3:.2f} GB")
        else:
            print("  Note: XPU not available (no Intel Arc GPU detected)")
            print("  This is expected if running without Intel Arc GPU hardware")
    except Exception as e:
        print(f"✗ Error checking XPU availability: {e}")
        return False
    
    print()
    return True


def check_xpu_tensor_ops() -> Optional[bool]:
    """Verify basic tensor operations on XPU."""
    print("=" * 60)
    print("2. Checking XPU Tensor Operations")
    print("=" * 60)
    
    try:
        import torch
        
        if not torch.xpu.is_available():
            print("⊘ Skipping tensor operations test (no Intel Arc GPU)")
            print("  This is expected if running without Intel Arc GPU hardware")
            print()
            return None
        
        # Test tensor creation
        x = torch.randn(3, 3).to('xpu')
        print(f"✓ Tensor created on XPU: shape {x.shape}, device {x.device}")
        
        # Test matrix multiplication
        y = torch.randn(3, 3).to('xpu')
        z = torch.matmul(x, y)
        print(f"✓ Matrix multiplication on XPU: shape {z.shape}")
        
    except Exception as e:
        print(f"✗ XPU tensor operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def check_xpu_model_optimization() -> Optional[bool]:
    """Verify model optimization on XPU (requires Intel Arc GPU)."""
    print("=" * 60)
    print("3. Checking XPU Model Optimization")
    print("=" * 60)
    
    try:
        import torch
        
        if not torch.xpu.is_available():
            print("⊘ Skipping XPU model optimization test (no Intel Arc GPU)")
            print("  This is expected if running without Intel Arc GPU hardware")
            print()
            return None
        
        # Create a simple model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        device = torch.device("xpu:0")
        model = model.to(device)
        print(f"✓ Moved model to {device}")
        
        # Test inference on XPU
        model.eval()
        x = torch.randn(2, 10, device=device)
        with torch.no_grad():
            output = model(x)
        print(f"✓ XPU inference succeeded, output shape: {output.shape}")
        print(f"  Output device: {output.device}")
        
    except Exception as e:
        print(f"✗ XPU model optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("PyTorch XPU Verification Script")
    print("=" * 60)
    print("Note: IPEX is discontinued. Using native torch.xpu support.")
    print()
    
    results = {
        "PyTorch XPU": check_pytorch_xpu(),
        "XPU Tensor Operations": check_xpu_tensor_ops(),
        "XPU Model Optimization": check_xpu_model_optimization(),
    }
    
    # Print summary
    print("=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    for check, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⊘ SKIP"
        print(f"{status:8} {check}")
    
    print()
    
    # Determine overall result
    required_checks = ["PyTorch XPU"]
    required_passed = all(results[check] for check in required_checks)
    
    if required_passed:
        print("✓ All required checks passed!")
        print()
        print("PyTorch XPU build is functional.")
        
        if results["XPU Tensor Operations"] is None:
            print()
            print("Note: XPU-specific tests were skipped (no Intel Arc GPU detected).")
            print("This is expected if building without Intel Arc GPU hardware.")
            print("Deploy to a machine with Intel Arc GPU to test XPU functionality.")
        elif results["XPU Tensor Operations"] is True:
            print()
            print("✓ XPU-specific tests also passed!")
            print("PyTorch XPU is fully functional with Intel Arc GPU.")
        
        return 0
    else:
        print("✗ Some required checks failed.")
        print()
        print("PyTorch XPU build has issues. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

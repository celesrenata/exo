#!/usr/bin/env python3
"""
Verification script for Intel Extension for PyTorch (IPEX) with XPU support.

This script verifies that:
1. PyTorch with XPU support is available
2. IPEX imports successfully
3. IPEX can optimize models for XPU
4. Basic IPEX operations work

Requirements:
- PyTorch 2.5+ with XPU support
- Intel Extension for PyTorch 2.5+ with XPU support
- Intel Arc GPU (optional, for full verification)
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


def check_ipex_import() -> bool:
    """Verify IPEX imports successfully."""
    print("=" * 60)
    print("2. Checking IPEX Import")
    print("=" * 60)
    
    try:
        import intel_extension_for_pytorch as ipex
        print("✓ IPEX imported successfully")
        print(f"  Version: {ipex.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import IPEX: {e}")
        print(f"  Error details: {type(e).__name__}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error importing IPEX: {e}")
        return False
    
    print()
    return True


def check_ipex_optimization() -> bool:
    """Verify IPEX can optimize models."""
    print("=" * 60)
    print("3. Checking IPEX Optimization")
    print("=" * 60)
    
    try:
        import intel_extension_for_pytorch as ipex
        import torch
        
        # Create a simple model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        print("✓ Created simple test model")
        
        # Try to optimize with IPEX
        model.eval()
        optimized_model = ipex.optimize(model, dtype=torch.float32)
        print("✓ IPEX optimization succeeded")
        
        # Test inference
        x = torch.randn(2, 10)
        with torch.no_grad():
            output = optimized_model(x)
        print(f"✓ Inference succeeded, output shape: {output.shape}")
        
    except Exception as e:
        print(f"✗ IPEX optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def check_ipex_xpu_optimization() -> Optional[bool]:
    """Verify IPEX can optimize models for XPU (requires Intel Arc GPU)."""
    print("=" * 60)
    print("4. Checking IPEX XPU Optimization (Optional)")
    print("=" * 60)
    
    try:
        import intel_extension_for_pytorch as ipex
        import torch
        
        if not torch.xpu.is_available():
            print("⊘ Skipping XPU optimization test (no Intel Arc GPU)")
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
        
        # Optimize with IPEX for XPU
        model.eval()
        optimized_model = ipex.optimize(model, dtype=torch.bfloat16)
        print("✓ IPEX XPU optimization succeeded")
        
        # Test inference on XPU
        x = torch.randn(2, 10, device=device)
        with torch.no_grad():
            output = optimized_model(x)
        print(f"✓ XPU inference succeeded, output shape: {output.shape}")
        print(f"  Output device: {output.device}")
        
    except Exception as e:
        print(f"✗ IPEX XPU optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("IPEX XPU Verification Script")
    print("=" * 60)
    print()
    
    results = {
        "PyTorch XPU": check_pytorch_xpu(),
        "IPEX Import": check_ipex_import(),
        "IPEX Optimization": check_ipex_optimization(),
        "IPEX XPU Optimization": check_ipex_xpu_optimization(),
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
    required_checks = ["PyTorch XPU", "IPEX Import", "IPEX Optimization"]
    required_passed = all(results[check] for check in required_checks)
    
    if required_passed:
        print("✓ All required checks passed!")
        print()
        print("IPEX XPU build is functional.")
        
        if results["IPEX XPU Optimization"] is None:
            print()
            print("Note: XPU-specific tests were skipped (no Intel Arc GPU detected).")
            print("This is expected if building without Intel Arc GPU hardware.")
            print("Deploy to a machine with Intel Arc GPU to test XPU functionality.")
        elif results["IPEX XPU Optimization"] is True:
            print()
            print("✓ XPU-specific tests also passed!")
            print("IPEX is fully functional with Intel Arc GPU.")
        
        return 0
    else:
        print("✗ Some required checks failed.")
        print()
        print("IPEX XPU build has issues. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

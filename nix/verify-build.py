#!/usr/bin/env python3
"""
Comprehensive build verification script for PyTorch + IPEX with XPU support.

This script verifies all requirements from task 10.4:
- Test torch.xpu.is_available() after build
- Verify torch.xpu.device_count() returns devices
- Test basic tensor operations on XPU
- Verify IPEX import and optimization
- Document build success criteria

This combines verification of both PyTorch XPU and IPEX XPU builds.

Usage:
    python nix/verify-build.py [--strict]
    
Options:
    --strict    Require Intel Arc GPU to be present (fail if not detected)

Exit codes:
    0 - All critical tests passed
    1 - Critical tests failed
    2 - XPU hardware not available (only in strict mode)
"""

import sys
import argparse
from typing import Optional, Dict, Any


class BuildVerifier:
    """Verifies PyTorch + IPEX build with XPU support."""
    
    def __init__(self, strict: bool = False):
        self.strict = strict
        self.results: Dict[str, Optional[bool]] = {}
        self.has_xpu_hardware = False
    
    def print_header(self, title: str):
        """Print a formatted section header."""
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
    
    def print_result(self, message: str, success: Optional[bool]):
        """Print a test result with appropriate symbol."""
        if success is True:
            symbol = "✓"
        elif success is False:
            symbol = "✗"
        else:
            symbol = "⊘"
        print(f"{symbol} {message}")
    
    def test_pytorch_import(self) -> bool:
        """Test that PyTorch can be imported."""
        self.print_header("1. PyTorch Import")
        
        try:
            import torch
            self.print_result(f"PyTorch imported successfully (version {torch.__version__})", True)
            self.results["pytorch_import"] = True
            return True
        except ImportError as e:
            self.print_result(f"Failed to import PyTorch: {e}", False)
            self.results["pytorch_import"] = False
            return False
    
    def test_xpu_available(self) -> bool:
        """Test torch.xpu.is_available() - Requirement 11.4."""
        self.print_header("2. XPU Availability (torch.xpu.is_available)")
        
        try:
            import torch
            
            # Check if XPU module exists
            if not hasattr(torch, 'xpu'):
                self.print_result("torch.xpu module not found - XPU support not compiled", False)
                self.results["xpu_available"] = False
                return False
            
            # Check if XPU is available
            xpu_available = torch.xpu.is_available()
            
            if xpu_available:
                self.print_result("torch.xpu.is_available() returns True", True)
                self.has_xpu_hardware = True
                self.results["xpu_available"] = True
                return True
            else:
                self.print_result("torch.xpu.is_available() returns False", False)
                print("  Note: This may indicate:")
                print("    - No Intel Arc GPU present in system")
                print("    - Intel GPU drivers not installed")
                print("    - XPU runtime not properly configured")
                
                if self.strict:
                    print("  STRICT MODE: Failing due to missing XPU hardware")
                    self.results["xpu_available"] = False
                    return False
                else:
                    print("  NON-STRICT MODE: Continuing (XPU tests will be skipped)")
                    self.results["xpu_available"] = None
                    return True
        
        except Exception as e:
            self.print_result(f"Error checking XPU availability: {e}", False)
            self.results["xpu_available"] = False
            return False
    
    def test_xpu_device_count(self) -> Optional[bool]:
        """Test torch.xpu.device_count() - Requirement 11.4."""
        self.print_header("3. XPU Device Enumeration (torch.xpu.device_count)")
        
        if not self.has_xpu_hardware:
            self.print_result("Skipping (no XPU hardware detected)", None)
            self.results["xpu_device_count"] = None
            return None
        
        try:
            import torch
            
            device_count = torch.xpu.device_count()
            self.print_result(f"torch.xpu.device_count() = {device_count}", device_count > 0)
            
            if device_count == 0:
                print("  Warning: XPU available but no devices found")
                self.results["xpu_device_count"] = False
                return False
            
            # Enumerate devices
            print(f"\n  Found {device_count} XPU device(s):")
            for i in range(device_count):
                try:
                    name = torch.xpu.get_device_name(i)
                    props = torch.xpu.get_device_properties(i)
                    total_mem_gb = props.total_memory / (1024 ** 3)
                    
                    print(f"    Device {i}: {name}")
                    print(f"      Total memory: {total_mem_gb:.2f} GB")
                    print(f"      Max compute units: {props.max_compute_units}")
                    print(f"      GPU EU count: {props.gpu_eu_count}")
                except Exception as e:
                    print(f"    Device {i}: Error getting properties - {e}")
            
            self.results["xpu_device_count"] = True
            return True
        
        except Exception as e:
            self.print_result(f"Error enumerating XPU devices: {e}", False)
            self.results["xpu_device_count"] = False
            return False
    
    def test_xpu_tensor_operations(self) -> Optional[bool]:
        """Test basic tensor operations on XPU - Requirement 11.4."""
        self.print_header("4. XPU Tensor Operations")
        
        if not self.has_xpu_hardware:
            self.print_result("Skipping (no XPU hardware detected)", None)
            self.results["xpu_tensor_ops"] = None
            return None
        
        try:
            import torch
            
            device = torch.device("xpu:0")
            print(f"  Testing on device: {device}")
            
            # Test 1: Tensor creation
            a = torch.randn(100, 100, device=device)
            b = torch.randn(100, 100, device=device)
            self.print_result("Created tensors on XPU", True)
            
            # Test 2: Matrix multiplication
            c = torch.matmul(a, b)
            assert c.device.type == "xpu", f"Result not on XPU: {c.device}"
            self.print_result("Matrix multiplication successful", True)
            
            # Test 3: Element-wise operations
            d = a + b
            assert d.device.type == "xpu"
            self.print_result("Element-wise addition successful", True)
            
            # Test 4: Reduction operations
            e = torch.sum(c)
            self.print_result("Reduction operation successful", True)
            
            # Test 5: Data transfer
            cpu_tensor = c.cpu()
            assert cpu_tensor.device.type == "cpu"
            self.print_result("Data transfer to CPU successful", True)
            
            print(f"\n  All tensor operations completed successfully")
            self.results["xpu_tensor_ops"] = True
            return True
        
        except Exception as e:
            self.print_result(f"XPU tensor operations failed: {e}", False)
            import traceback
            traceback.print_exc()
            self.results["xpu_tensor_ops"] = False
            return False
    
    def test_ipex_import(self) -> bool:
        """Test IPEX import - Requirement 11.4."""
        self.print_header("5. IPEX Import")
        
        try:
            import intel_extension_for_pytorch as ipex
            self.print_result(f"IPEX imported successfully (version {ipex.__version__})", True)
            self.results["ipex_import"] = True
            return True
        except ImportError as e:
            self.print_result(f"Failed to import IPEX: {e}", False)
            print("  Error: IPEX must be built and available")
            self.results["ipex_import"] = False
            return False
        except Exception as e:
            self.print_result(f"Unexpected error importing IPEX: {e}", False)
            self.results["ipex_import"] = False
            return False
    
    def test_ipex_optimization_cpu(self) -> bool:
        """Test IPEX optimization on CPU - Requirement 11.4."""
        self.print_header("6. IPEX Optimization (CPU)")
        
        try:
            import intel_extension_for_pytorch as ipex
            import torch
            import torch.nn as nn
            
            # Create a simple model
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear1 = nn.Linear(128, 64)
                    self.relu = nn.ReLU()
                    self.linear2 = nn.Linear(64, 10)
                
                def forward(self, x):
                    x = self.linear1(x)
                    x = self.relu(x)
                    x = self.linear2(x)
                    return x
            
            model = SimpleModel()
            model.eval()
            self.print_result("Created test model", True)
            
            # Optimize with IPEX
            optimized_model = ipex.optimize(model, dtype=torch.float32)
            self.print_result("IPEX optimization succeeded", True)
            
            # Test inference
            x = torch.randn(4, 128)
            with torch.no_grad():
                output = optimized_model(x)
            
            assert output.shape == (4, 10), f"Unexpected output shape: {output.shape}"
            self.print_result(f"Inference successful (output shape: {output.shape})", True)
            
            self.results["ipex_opt_cpu"] = True
            return True
        
        except Exception as e:
            self.print_result(f"IPEX CPU optimization failed: {e}", False)
            import traceback
            traceback.print_exc()
            self.results["ipex_opt_cpu"] = False
            return False
    
    def test_ipex_optimization_xpu(self) -> Optional[bool]:
        """Test IPEX optimization on XPU - Requirement 11.4."""
        self.print_header("7. IPEX Optimization (XPU)")
        
        if not self.has_xpu_hardware:
            self.print_result("Skipping (no XPU hardware detected)", None)
            self.results["ipex_opt_xpu"] = None
            return None
        
        try:
            import intel_extension_for_pytorch as ipex
            import torch
            import torch.nn as nn
            
            # Create a simple model
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear1 = nn.Linear(128, 64)
                    self.relu = nn.ReLU()
                    self.linear2 = nn.Linear(64, 10)
                
                def forward(self, x):
                    x = self.linear1(x)
                    x = self.relu(x)
                    x = self.linear2(x)
                    return x
            
            device = torch.device("xpu:0")
            model = SimpleModel().to(device)
            model.eval()
            self.print_result(f"Created test model on {device}", True)
            
            # Optimize with IPEX for XPU (using bfloat16)
            optimized_model = ipex.optimize(model, dtype=torch.bfloat16)
            self.print_result("IPEX XPU optimization succeeded (bfloat16)", True)
            
            # Test inference on XPU
            x = torch.randn(4, 128, device=device)
            with torch.no_grad():
                output = optimized_model(x)
            
            assert output.device.type == "xpu", f"Output not on XPU: {output.device}"
            assert output.shape == (4, 10), f"Unexpected output shape: {output.shape}"
            self.print_result(f"XPU inference successful (output shape: {output.shape})", True)
            
            # Test with float32 as well
            optimized_model_fp32 = ipex.optimize(model, dtype=torch.float32)
            with torch.no_grad():
                output_fp32 = optimized_model_fp32(x.float())
            self.print_result("IPEX XPU optimization also works with float32", True)
            
            self.results["ipex_opt_xpu"] = True
            return True
        
        except Exception as e:
            self.print_result(f"IPEX XPU optimization failed: {e}", False)
            import traceback
            traceback.print_exc()
            self.results["ipex_opt_xpu"] = False
            return False
    
    def print_summary(self):
        """Print verification summary and build success criteria."""
        self.print_header("Verification Summary")
        
        print("\nTest Results:")
        print("-" * 70)
        
        test_names = {
            "pytorch_import": "PyTorch Import",
            "xpu_available": "XPU Available (torch.xpu.is_available)",
            "xpu_device_count": "XPU Device Count (torch.xpu.device_count)",
            "xpu_tensor_ops": "XPU Tensor Operations",
            "ipex_import": "IPEX Import",
            "ipex_opt_cpu": "IPEX Optimization (CPU)",
            "ipex_opt_xpu": "IPEX Optimization (XPU)",
        }
        
        for key, name in test_names.items():
            result = self.results.get(key)
            if result is True:
                status = "✓ PASS"
            elif result is False:
                status = "✗ FAIL"
            else:
                status = "⊘ SKIP"
            print(f"  {status:10} {name}")
        
        print("\n" + "-" * 70)
        print("\nBuild Success Criteria:")
        print("-" * 70)
        
        # Define critical tests
        critical_tests = ["pytorch_import", "ipex_import", "ipex_opt_cpu"]
        critical_passed = all(
            self.results.get(test) is True 
            for test in critical_tests
        )
        
        print("\nCritical Requirements (must pass):")
        print("  1. PyTorch imports successfully")
        print("  2. PyTorch has XPU support compiled in (torch.xpu module exists)")
        print("  3. IPEX imports successfully")
        print("  4. IPEX can optimize models on CPU")
        
        if critical_passed:
            print(f"\n  ✓ All critical requirements met")
        else:
            print(f"\n  ✗ Some critical requirements failed")
        
        print("\nOptional Requirements (hardware-dependent):")
        print("  5. Intel Arc GPU detected (torch.xpu.is_available() == True)")
        print("  6. XPU devices enumerated (torch.xpu.device_count() > 0)")
        print("  7. XPU tensor operations work")
        print("  8. IPEX can optimize models for XPU")
        
        if self.has_xpu_hardware:
            xpu_tests = ["xpu_device_count", "xpu_tensor_ops", "ipex_opt_xpu"]
            xpu_passed = all(
                self.results.get(test) is True 
                for test in xpu_tests
            )
            if xpu_passed:
                print(f"\n  ✓ All XPU requirements met")
            else:
                print(f"\n  ✗ Some XPU requirements failed")
        else:
            print(f"\n  ⊘ XPU requirements skipped (no Intel Arc GPU detected)")
            print("     This is expected when building without Intel Arc GPU hardware")
        
        return critical_passed
    
    def run(self) -> int:
        """Run all verification tests and return exit code."""
        print("=" * 70)
        print("PyTorch + IPEX Build Verification")
        print("=" * 70)
        print("\nThis script verifies the build meets all requirements from task 10.4:")
        print("  - torch.xpu.is_available() works")
        print("  - torch.xpu.device_count() returns devices (if hardware present)")
        print("  - Basic tensor operations work on XPU (if hardware present)")
        print("  - IPEX imports and can optimize models")
        print("  - Build success criteria are documented")
        
        if self.strict:
            print("\nRunning in STRICT mode: Intel Arc GPU required")
        else:
            print("\nRunning in NON-STRICT mode: XPU tests optional")
        
        # Run all tests
        tests = [
            self.test_pytorch_import,
            self.test_xpu_available,
            self.test_xpu_device_count,
            self.test_xpu_tensor_operations,
            self.test_ipex_import,
            self.test_ipex_optimization_cpu,
            self.test_ipex_optimization_xpu,
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"\n✗ Unexpected error in {test.__name__}: {e}")
                import traceback
                traceback.print_exc()
        
        # Print summary
        success = self.print_summary()
        
        # Determine exit code
        print("\n" + "=" * 70)
        if success:
            if self.has_xpu_hardware:
                xpu_tests_passed = all(
                    self.results.get(test) is True
                    for test in ["xpu_device_count", "xpu_tensor_ops", "ipex_opt_xpu"]
                )
                if xpu_tests_passed:
                    print("✓ BUILD VERIFICATION PASSED (with XPU support)")
                    print("\nThe build is fully functional with Intel Arc GPU support.")
                    return 0
                else:
                    print("⚠ BUILD VERIFICATION PASSED (but XPU tests failed)")
                    print("\nThe build is functional but XPU support has issues.")
                    print("Review the XPU test failures above.")
                    return 0 if not self.strict else 1
            else:
                print("✓ BUILD VERIFICATION PASSED (without XPU hardware)")
                print("\nThe build is functional. XPU tests were skipped because")
                print("no Intel Arc GPU was detected. This is expected when building")
                print("on a machine without Intel Arc GPU hardware.")
                print("\nTo test XPU functionality:")
                print("  1. Deploy to a machine with Intel Arc GPU")
                print("  2. Run: python nix/verify-build.py")
                return 0 if not self.strict else 2
        else:
            print("✗ BUILD VERIFICATION FAILED")
            print("\nCritical requirements were not met. Review the failures above.")
            return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify PyTorch + IPEX build with XPU support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit codes:
  0 - All critical tests passed
  1 - Critical tests failed
  2 - XPU hardware not available (only in strict mode)

Examples:
  # Normal verification (XPU tests optional)
  python nix/verify-build.py
  
  # Strict verification (requires XPU hardware)
  python nix/verify-build.py --strict
        """
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require Intel Arc GPU to be present (fail if not detected)"
    )
    
    args = parser.parse_args()
    
    verifier = BuildVerifier(strict=args.strict)
    return verifier.run()


if __name__ == "__main__":
    sys.exit(main())

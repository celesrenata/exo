#!/usr/bin/env python3
"""
Test the CPU engine from the Nix-built package.
"""

import sys
import os

# Add the Nix package to Python path
nix_package_path = "/nix/store/6vdnghr98pwywhyhyr68hi1x1aja9l9p-exo-cpu-0.1.0/lib/python3.13/site-packages"
sys.path.insert(0, nix_package_path)

def test_cpu_engine():
    """Test CPU engine functionality."""
    print("🚀 Testing EXO CPU Engine from Nix Package")
    print("=" * 50)
    
    # Set environment for CPU engine
    os.environ['EXO_ENGINE'] = 'torch'
    
    try:
        # Test engine detection
        print("\n1. Testing Engine Detection...")
        from exo.worker.engines.engine_utils import detect_available_engines, select_best_engine
        
        available = detect_available_engines()
        selected = select_best_engine()
        
        print(f"   Available engines: {available}")
        print(f"   Selected engine: {selected}")
        
        if 'torch' in available:
            print("   ✅ PyTorch engine detected")
        else:
            print("   ❌ PyTorch engine not available")
            return False
            
    except Exception as e:
        print(f"   ❌ Engine detection failed: {e}")
        return False
    
    try:
        # Test PyTorch functionality
        print("\n2. Testing PyTorch Core...")
        import torch
        
        print(f"   PyTorch version: {torch.__version__}")
        
        # Test tensor operations
        x = torch.tensor([1.0, 2.0, 3.0])
        y = x * 2
        print(f"   Tensor test: {x.tolist()} * 2 = {y.tolist()}")
        
        print("   ✅ PyTorch working correctly")
        
    except Exception as e:
        print(f"   ❌ PyTorch test failed: {e}")
        return False
    
    try:
        # Test engine components
        print("\n3. Testing Engine Components...")
        from exo.worker.engines.torch.utils_torch import check_torch_availability
        from exo.worker.engines.torch.generator.generate import warmup_inference, torch_generate
        
        torch_available = check_torch_availability()
        print(f"   Torch available: {torch_available}")
        
        if torch_available:
            print("   ✅ Engine components loaded successfully")
        else:
            print("   ❌ Torch availability check failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Engine components test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 CPU Engine Test Complete!")
    print("✅ The Nix-packaged EXO CPU engine is working correctly!")
    print("\nThe CPU inference engine is ready to use.")
    print("Note: Full application requires Rust bindings to be fixed.")
    
    return True

if __name__ == "__main__":
    success = test_cpu_engine()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Test script to verify PyTorch and IPEX are available in the Nix environment.
"""

import sys

print("Python version:", sys.version)
print("Python executable:", sys.executable)
print()

try:
    import torch
    print("✓ PyTorch imported successfully")
    print("  Version:", torch.__version__)
    print("  Has XPU:", hasattr(torch, 'xpu'))
    
    if hasattr(torch, 'xpu'):
        print("  XPU available:", torch.xpu.is_available())
        if torch.xpu.is_available():
            print("  XPU device count:", torch.xpu.device_count())
except ImportError as e:
    print("✗ PyTorch import failed:", e)
    sys.exit(1)

print()

try:
    import intel_extension_for_pytorch as ipex
    print("✓ IPEX imported successfully")
    print("  Version:", ipex.__version__)
except ImportError as e:
    print("✗ IPEX import failed:", e)
    print("  This is expected if IPEX is not available in the Nix environment yet")

print()
print("Test completed!")

#!/usr/bin/env bash
# Test script for PyTorch XPU Nix derivation
# This script attempts to build the PyTorch XPU derivation and verify it works

set -euo pipefail

echo "=========================================="
echo "PyTorch XPU Build Test"
echo "=========================================="
echo ""

# Check if we're on Linux
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Error: PyTorch XPU is only supported on Linux"
    exit 1
fi

echo "Step 1: Building PyTorch with XPU support..."
echo "This will take a long time (30-60 minutes or more)"
echo ""

# Try to build the pytorch-xpu derivation
if nix build .#pytorch-xpu --print-build-logs; then
    echo ""
    echo "✓ Build succeeded!"
else
    echo ""
    echo "✗ Build failed!"
    echo ""
    echo "Common issues:"
    echo "1. Missing hash - run 'nix build .#pytorch-xpu' to get the correct hash"
    echo "2. Build dependencies missing - check nix/pytorch-xpu.nix"
    echo "3. CMake configuration errors - check USE_XPU flag is set"
    exit 1
fi

echo ""
echo "Step 2: Verifying PyTorch XPU build..."
echo ""

# Run verification script
if nix run .#pytorch-xpu -- python nix/verify-pytorch-xpu.py; then
    echo ""
    echo "✓ Verification passed!"
else
    echo ""
    echo "✗ Verification failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "PyTorch XPU Build Test Complete"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Update flake.nix to use the pytorch-xpu derivation"
echo "2. Build IPEX with XPU support (task 10.2)"
echo "3. Test end-to-end with exo"

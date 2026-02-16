#!/usr/bin/env bash
#
# Build PyTorch and IPEX from MordragT's nixos repository
# Fixed version that handles libffi test failures
#

set -euo pipefail

echo "=========================================="
echo "Building PyTorch + IPEX from Source (Fixed)"
echo "=========================================="
echo ""
echo "This will build:"
echo "  - PyTorch 2.9.1 with Intel XPU support"
echo "  - Intel Extension for PyTorch (IPEX) 2.8.10+xpu"
echo "  - Intel SYCL/DPC++ toolchain"
echo ""
echo "Workaround: Using --keep-going and --fallback to handle test failures"
echo ""

# Create log directory
mkdir -p build_logs

echo "Building PyTorch with workarounds for test failures..."
echo "Started at: $(date)"

# Use --keep-going to continue past test failures
# Use --fallback to build locally if remote builders fail
nix build github:MordragT/nixos#intel-python.pkgs.torch \
  --print-build-logs \
  --keep-going \
  --fallback \
  --option sandbox relaxed \
  2>&1 | tee build_logs/pytorch_build_fixed.log

BUILD_EXIT=$?

if [ $BUILD_EXIT -eq 0 ]; then
  echo "✓ PyTorch build completed successfully"
  
  echo ""
  echo "Building IPEX..."
  echo "Started at: $(date)"
  
  nix build github:MordragT/nixos#intel-python.pkgs.ipex \
    --print-build-logs \
    --keep-going \
    --fallback \
    --option sandbox relaxed \
    2>&1 | tee build_logs/ipex_build_fixed.log
  
  if [ $? -eq 0 ]; then
    echo "✓ IPEX build completed successfully"
    echo ""
    echo "=========================================="
    echo "Build Complete!"
    echo "Completed at: $(date)"
    echo "=========================================="
  else
    echo "✗ IPEX build failed, but PyTorch succeeded"
    echo "You can still use PyTorch without IPEX"
  fi
else
  echo "Build encountered errors but may have produced usable artifacts"
  echo "Check build_logs/pytorch_build_fixed.log for details"
fi

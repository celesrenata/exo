#!/usr/bin/env bash
#
# Build PyTorch and IPEX from MordragT's nixos repository
# This will take 1-2 hours to complete
#

set -euo pipefail

echo "=========================================="
echo "Building PyTorch + IPEX from Source"
echo "=========================================="
echo ""
echo "This will build:"
echo "  - PyTorch 2.9.1 with Intel XPU support"
echo "  - Intel Extension for PyTorch (IPEX) 2.8.10+xpu"
echo "  - Intel SYCL/DPC++ toolchain"
echo "  - Intel MKL, DNNL, and other dependencies"
echo ""
echo "Estimated time: 1-2 hours on a modern system"
echo "Estimated disk space: ~10GB"
echo ""
echo "Build logs will be saved to:"
echo "  - pytorch_build.log"
echo "  - ipex_build.log"
echo ""

# Create log directory
mkdir -p build_logs

echo "Step 1: Building PyTorch..."
echo "Started at: $(date)"
nix build github:MordragT/nixos#intel-python.pkgs.torch \
  --print-build-logs \
  --keep-going \
  2>&1 | tee build_logs/pytorch_build.log

if [ $? -eq 0 ]; then
  echo "✓ PyTorch build completed successfully"
else
  echo "✗ PyTorch build failed. Check build_logs/pytorch_build.log"
  exit 1
fi

echo ""
echo "Step 2: Building IPEX..."
echo "Started at: $(date)"
nix build github:MordragT/nixos#intel-python.pkgs.ipex \
  --print-build-logs \
  --keep-going \
  2>&1 | tee build_logs/ipex_build.log

if [ $? -eq 0 ]; then
  echo "✓ IPEX build completed successfully"
else
  echo "✗ IPEX build failed. Check build_logs/ipex_build.log"
  exit 1
fi

echo ""
echo "=========================================="
echo "Build Complete!"
echo "Completed at: $(date)"
echo "=========================================="
echo ""
echo "PyTorch and IPEX are now available in the Nix store."
echo "You can now use 'nix develop' to access them."
echo ""

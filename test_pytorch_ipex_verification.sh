#!/usr/bin/env bash
#
# Test script for PyTorch + IPEX verification
#
# This script runs the verification checks for PyTorch and IPEX with
# Intel XPU support. It's designed to work both on systems with and
# without Intel Arc GPUs.
#
# Usage:
#   ./test_pytorch_ipex_verification.sh [--strict]
#
# Options:
#   --strict    Require GPU to be present (fail if no GPU detected)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "PyTorch + IPEX Verification Test"
echo "=========================================="
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

echo "Python version: $(python3 --version)"
echo

# Check if PyTorch is installed
if ! python3 -c "import torch" 2>/dev/null; then
    echo "Error: PyTorch is not installed"
    echo
    echo "Please install PyTorch + IPEX following the guide:"
    echo "  cat nix/PYTORCH_IPEX_INSTALLATION.md"
    echo
    echo "Quick install:"
    echo "  pip install torch==2.5.1+xpu torchvision==0.20.1+xpu \\"
    echo "    --index-url https://download.pytorch.org/whl/xpu"
    echo "  pip install intel-extension-for-pytorch==2.5.10+xpu \\"
    echo "    --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"
    exit 1
fi

# Run the verification script
echo "Running verification checks..."
echo

python3 "${SCRIPT_DIR}/nix/verify-pytorch-ipex-xpu.py" "$@"
exit_code=$?

echo
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "✅ Verification completed successfully"
elif [ $exit_code -eq 2 ]; then
    echo "⚠️  Verification completed (no GPU detected)"
else
    echo "❌ Verification failed"
fi
echo "=========================================="

exit $exit_code

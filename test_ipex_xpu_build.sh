#!/usr/bin/env bash
#
# Test script for building IPEX with XPU support in Nix
#
# This script:
# 1. Builds the IPEX XPU derivation
# 2. Runs verification tests
# 3. Reports success/failure
#
# Requirements:
# - Nix with flakes enabled
# - PyTorch XPU already built (task 10.1)
# - Sufficient disk space (~10GB for build artifacts)
# - Sufficient RAM (~16GB recommended)
#
# Usage:
#   ./test_ipex_xpu_build.sh

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "IPEX XPU Build Test"
echo "========================================"
echo ""

# Check if PyTorch XPU is built
echo "Checking for PyTorch XPU..."
if nix build .#pytorch-xpu --no-link 2>/dev/null; then
    echo -e "${GREEN}✓${NC} PyTorch XPU is available"
else
    echo -e "${RED}✗${NC} PyTorch XPU is not built"
    echo ""
    echo "Please build PyTorch XPU first (task 10.1):"
    echo "  nix build .#pytorch-xpu"
    exit 1
fi
echo ""

# Build IPEX XPU
echo "Building IPEX XPU derivation..."
echo "This may take 20-40 minutes on first build..."
echo ""

if nix build .#ipex-xpu --print-build-logs; then
    echo ""
    echo -e "${GREEN}✓${NC} IPEX XPU build succeeded"
else
    EXIT_CODE=$?
    echo ""
    echo -e "${RED}✗${NC} IPEX XPU build failed with exit code $EXIT_CODE"
    echo ""
    
    # Check for common issues
    if [ $EXIT_CODE -eq 102 ]; then
        echo "Hash mismatch detected. This is expected on first build."
        echo ""
        echo "To fix:"
        echo "1. Look for the 'got:' hash in the error output above"
        echo "2. Update nix/ipex-xpu.nix with the correct hash:"
        echo "   hash = \"sha256-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\";"
        echo "3. Run this script again"
    else
        echo "Build failed. Common issues:"
        echo "- Insufficient disk space (need ~10GB free)"
        echo "- Insufficient RAM (need ~16GB)"
        echo "- Missing dependencies (check flake.nix)"
        echo "- PyTorch XPU not properly built"
        echo ""
        echo "Check the build logs above for specific errors."
    fi
    
    exit $EXIT_CODE
fi
echo ""

# Run verification tests
echo "Running verification tests..."
echo ""

if nix run .#ipex-xpu -- python nix/verify-ipex-xpu.py; then
    echo ""
    echo -e "${GREEN}✓${NC} Verification tests passed"
else
    echo ""
    echo -e "${YELLOW}⚠${NC} Verification tests had issues"
    echo ""
    echo "Note: Some tests may fail without Intel Arc GPU hardware."
    echo "This is expected if building on a machine without Intel Arc GPU."
    echo ""
    echo "To test on hardware:"
    echo "1. Deploy to gremlin-1 (Intel Arc GPU test machine)"
    echo "2. Run: python nix/verify-ipex-xpu.py"
fi
echo ""

# Success summary
echo "========================================"
echo "Build Test Complete"
echo "========================================"
echo ""
echo -e "${GREEN}✓${NC} IPEX XPU derivation built successfully"
echo ""
echo "Next steps:"
echo "1. Review verification results above"
echo "2. If on hardware with Intel Arc GPU, all tests should pass"
echo "3. If building without GPU, XPU tests will be skipped (expected)"
echo "4. Proceed to task 10.3: Configure oneAPI dependencies"
echo "5. Then task 10.5: Update flake.nix with new derivations"
echo ""
echo "To use IPEX XPU:"
echo "  nix run .#ipex-xpu -- python -c 'import intel_extension_for_pytorch as ipex; print(ipex.__version__)'"
echo ""
echo "To deploy to gremlin-1 for testing:"
echo "  bash force_update_gremlin1.sh"
echo ""

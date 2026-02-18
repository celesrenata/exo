#!/usr/bin/env bash
#
# Build Verification Test Script
#
# This script runs comprehensive verification of PyTorch + IPEX build with XPU support.
# It tests all requirements from task 10.4:
# - torch.xpu.is_available() after build
# - torch.xpu.device_count() returns devices
# - Basic tensor operations on XPU
# - IPEX import and optimization
# - Documents build success criteria
#
# Usage:
#   ./test_build_verification.sh [--strict]
#
# Options:
#   --strict    Require Intel Arc GPU to be present (fail if not detected)
#
# Exit codes:
#   0 - All critical tests passed
#   1 - Critical tests failed
#   2 - XPU hardware not available (only in strict mode)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
STRICT_MODE=""
if [[ "${1:-}" == "--strict" ]]; then
    STRICT_MODE="--strict"
fi

echo -e "${BLUE}========================================"
echo "Build Verification Test"
echo -e "========================================${NC}"
echo ""

# Check if we're on Linux
if [[ "$(uname -s)" != "Linux" ]]; then
    echo -e "${RED}Error: PyTorch + IPEX XPU is only supported on Linux${NC}"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

# Check if verification script exists
if [[ ! -f "nix/verify-build.py" ]]; then
    echo -e "${RED}Error: nix/verify-build.py not found${NC}"
    echo "Make sure you're running this from the repository root"
    exit 1
fi

# Make verification script executable
chmod +x nix/verify-build.py

echo "Running comprehensive build verification..."
echo ""

if [[ -n "$STRICT_MODE" ]]; then
    echo -e "${YELLOW}Running in STRICT mode: Intel Arc GPU required${NC}"
else
    echo "Running in NON-STRICT mode: XPU tests optional"
fi
echo ""

# Run verification
if python3 nix/verify-build.py $STRICT_MODE; then
    EXIT_CODE=0
    echo ""
    echo -e "${GREEN}========================================"
    echo "✓ Build Verification Passed"
    echo -e "========================================${NC}"
else
    EXIT_CODE=$?
    echo ""
    
    if [[ $EXIT_CODE -eq 2 ]]; then
        echo -e "${YELLOW}========================================"
        echo "⚠ XPU Hardware Not Available"
        echo -e "========================================${NC}"
        echo ""
        echo "The build is functional but no Intel Arc GPU was detected."
        echo "This is expected when building on a machine without Intel Arc GPU."
        echo ""
        echo "To test XPU functionality:"
        echo "  1. Deploy to gremlin-1 (Intel Arc GPU test machine)"
        echo "  2. Run: bash force_update_gremlin1.sh"
        echo "  3. SSH to gremlin-1: ssh root@10.1.1.12"
        echo "  4. Run: python3 /root/exo/nix/verify-build.py"
    else
        echo -e "${RED}========================================"
        echo "✗ Build Verification Failed"
        echo -e "========================================${NC}"
        echo ""
        echo "Critical tests failed. Review the errors above."
        echo ""
        echo "Common issues:"
        echo "  1. PyTorch not built with XPU support"
        echo "     - Check nix/pytorch-xpu.nix has USE_XPU=ON"
        echo "     - Rebuild: nix build .#pytorch-xpu"
        echo ""
        echo "  2. IPEX not built correctly"
        echo "     - Check nix/ipex-xpu.nix configuration"
        echo "     - Rebuild: nix build .#ipex-xpu"
        echo ""
        echo "  3. Missing dependencies"
        echo "     - Check flake.nix has all required packages"
        echo "     - Update: nix flake update"
        echo ""
        echo "  4. Driver issues (if XPU tests fail)"
        echo "     - Check Intel GPU drivers: intel-compute-runtime, level-zero"
        echo "     - Verify: clinfo | grep -i intel"
    fi
fi

echo ""
exit $EXIT_CODE

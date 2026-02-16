#!/usr/bin/env bash
#
# Test harness for PyTorch + IPEX Intel Arc GPU setup
#
# This script runs the detection and validation scripts to verify that
# PyTorch and IPEX are properly configured for Intel Arc GPU support.
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "PyTorch + IPEX Setup Test Harness"
echo "=========================================="
echo ""

# Check if running on Linux
if [[ "$(uname -s)" != "Linux" ]]; then
  echo -e "${RED}Error: This script is designed for Linux systems${NC}"
  echo "PyTorch + IPEX Intel Arc support is only available on Linux"
  exit 1
fi

# Check if uv is available
if ! command -v uv &>/dev/null; then
  echo -e "${RED}Error: uv not found${NC}"
  echo "Please install uv: https://docs.astral.sh/uv/"
  exit 1
fi

echo -e "${YELLOW}Step 1: Running Intel Arc GPU detection...${NC}"
echo ""

if uv run python src/exo/worker/engines/pytorch_ipex/detect_intel_arc.py; then
  echo ""
  echo -e "${GREEN}✓ Intel Arc GPU detection passed${NC}"
else
  echo ""
  echo -e "${RED}✗ Intel Arc GPU detection failed${NC}"
  echo "Please check that:"
  echo "  - Intel Arc GPU is present in the system"
  echo "  - Intel compute-runtime and level-zero drivers are installed"
  echo "  - PyTorch and IPEX are properly installed"
  exit 1
fi

echo ""
echo "=========================================="
echo ""

echo -e "${YELLOW}Step 2: Running IPEX functionality validation...${NC}"
echo ""

if uv run python src/exo/worker/engines/pytorch_ipex/validate_ipex.py; then
  echo ""
  echo -e "${GREEN}✓ IPEX functionality validation passed${NC}"
else
  echo ""
  echo -e "${RED}✗ IPEX functionality validation failed${NC}"
  echo "Please check the error messages above"
  exit 1
fi

echo ""
echo "=========================================="
echo -e "${GREEN}All tests passed!${NC}"
echo "PyTorch + IPEX is properly configured for Intel Arc GPU"
echo "=========================================="

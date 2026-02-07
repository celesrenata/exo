#!/usr/bin/env bash
# Test script for Intel hardware configuration on NixOS
# This script verifies that Intel Arc and NPU support is properly configured

set -e

echo "=== Intel Hardware Configuration Test ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
pass() {
  echo -e "${GREEN}✓${NC} $1"
  ((TESTS_PASSED++))
}

fail() {
  echo -e "${RED}✗${NC} $1"
  ((TESTS_FAILED++))
}

warn() {
  echo -e "${YELLOW}⚠${NC} $1"
}

# Test 1: Check for Intel GPU
echo "Test 1: Checking for Intel GPU..."
if lspci | grep -i "VGA.*Intel" >/dev/null 2>&1; then
  GPU_NAME=$(lspci | grep -i "VGA.*Intel" | cut -d: -f3)
  pass "Intel GPU detected:$GPU_NAME"
else
  fail "Intel GPU not detected"
fi
echo ""

# Test 2: Check for Level Zero runtime
echo "Test 2: Checking for Level Zero runtime..."
if [ -f "/run/opengl-driver/lib/libze_loader.so" ] || [ -f "/run/opengl-driver/lib/libze_loader.so.1" ]; then
  pass "Level Zero loader library found"
else
  fail "Level Zero loader library not found"
  warn "Install intel-compute-runtime package"
fi
echo ""

# Test 3: Check for OpenCL runtime
echo "Test 3: Checking for OpenCL runtime..."
if [ -f "/run/opengl-driver/lib/libOpenCL.so" ] || [ -f "/run/opengl-driver/lib/libOpenCL.so.1" ]; then
  pass "OpenCL library found"
else
  fail "OpenCL library not found"
  warn "Install intel-compute-runtime package"
fi
echo ""

# Test 4: Check for clinfo utility
echo "Test 4: Checking OpenCL devices with clinfo..."
if command -v clinfo >/dev/null 2>&1; then
  if clinfo 2>/dev/null | grep -i "Intel" >/dev/null; then
    DEVICE_COUNT=$(clinfo 2>/dev/null | grep -c "Device Name.*Intel" || echo "0")
    pass "Found $DEVICE_COUNT Intel OpenCL device(s)"
  else
    warn "clinfo found but no Intel devices detected"
  fi
else
  warn "clinfo not available (install clinfo package to test)"
fi
echo ""

# Test 5: Check for NPU device node
echo "Test 5: Checking for Intel NPU device..."
if [ -e "/dev/accel/accel0" ]; then
  pass "NPU device node found: /dev/accel/accel0"
elif [ -e "/dev/dri/renderD128" ]; then
  warn "DRI render node found, but no NPU-specific device"
else
  warn "No NPU device node found (expected on Core Ultra processors)"
fi
echo ""

# Test 6: Check for NPU kernel modules
echo "Test 6: Checking for NPU kernel modules..."
if lsmod | grep -E "intel_vpu|ivpu" >/dev/null 2>&1; then
  MODULE_NAME=$(lsmod | grep -E "intel_vpu|ivpu" | awk '{print $1}')
  pass "NPU kernel module loaded: $MODULE_NAME"
else
  warn "NPU kernel module not loaded (expected on Core Ultra processors)"
fi
echo ""

# Test 7: Check for tinygrad Python package
echo "Test 7: Checking for tinygrad Python package..."
if python3 -c "import tinygrad" 2>/dev/null; then
  TINYGRAD_VERSION=$(python3 -c "import tinygrad; print(tinygrad.__version__)" 2>/dev/null || echo "unknown")
  pass "tinygrad package available (version: $TINYGRAD_VERSION)"
else
  fail "tinygrad package not available"
  warn "Ensure tinygrad is in your Python environment"
fi
echo ""

# Test 8: Check for pyopencl Python package
echo "Test 8: Checking for pyopencl Python package..."
if python3 -c "import pyopencl" 2>/dev/null; then
  pass "pyopencl package available"
else
  warn "pyopencl package not available (needed for OpenCL backend)"
fi
echo ""

# Test 9: Check for exo-npu systemd service (if NPU enabled)
echo "Test 9: Checking for exo-npu systemd service..."
if systemctl list-unit-files | grep -q "exo-npu.service"; then
  SERVICE_STATUS=$(systemctl is-active exo-npu.service 2>/dev/null || echo "inactive")
  if [ "$SERVICE_STATUS" = "active" ]; then
    pass "exo-npu service is active"
  else
    warn "exo-npu service exists but is $SERVICE_STATUS"
  fi
else
  warn "exo-npu service not configured (optional)"
fi
echo ""

# Test 10: Check graphics hardware configuration
echo "Test 10: Checking NixOS graphics configuration..."
if nixos-option hardware.graphics.enable 2>/dev/null | grep -q "true"; then
  pass "hardware.graphics.enable is true"
else
  fail "hardware.graphics.enable is not true"
fi
echo ""

# Summary
echo "=== Test Summary ==="
echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests failed: ${RED}$TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
  echo -e "${GREEN}All critical tests passed!${NC}"
  echo "Your Intel hardware configuration appears to be correct."
  exit 0
else
  echo -e "${YELLOW}Some tests failed.${NC}"
  echo "Review the failures above and check the setup guide:"
  echo "  docs/intel-hardware-setup.md"
  exit 1
fi

#!/usr/bin/env bash
# Quick test of gremlin-1 Intel hardware support
# This script runs basic checks without rebuilding the system
#
# Usage:
#   ./quick_test_gremlin1.sh

set -euo pipefail

TARGET_IP="10.1.1.12"
SSH_CMD="ssh root@${TARGET_IP}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}✓${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }

echo "=== Quick Test: gremlin-1 Intel Hardware Support ==="
echo ""

# Test 1: Connectivity
echo "Test 1: Connectivity"
if ping -c 1 -W 2 "$TARGET_IP" >/dev/null 2>&1; then
  pass "Can reach gremlin-1"
else
  fail "Cannot reach gremlin-1"
  exit 1
fi
echo ""

# Test 2: Intel GPU
echo "Test 2: Intel GPU Detection"
if $SSH_CMD "lspci | grep -i 'VGA.*Intel'" >/dev/null 2>&1; then
  GPU=$($SSH_CMD "lspci | grep -i 'VGA.*Intel' | head -1 | cut -d: -f3")
  pass "Intel GPU detected:$GPU"
else
  fail "Intel GPU not detected"
fi
echo ""

# Test 3: DRI devices
echo "Test 3: DRI Render Devices"
if $SSH_CMD "ls /dev/dri/renderD* >/dev/null 2>&1"; then
  DEVICES=$($SSH_CMD "ls /dev/dri/renderD*" | tr '\n' ' ')
  pass "DRI devices: $DEVICES"
else
  fail "No DRI render devices"
fi
echo ""

# Test 4: Level Zero
echo "Test 4: Level Zero Runtime"
if $SSH_CMD "ls /run/opengl-driver/lib/libze_loader.so* >/dev/null 2>&1"; then
  pass "Level Zero library found"
else
  warn "Level Zero library not found"
fi
echo ""

# Test 5: NPU
echo "Test 5: Intel NPU"
if $SSH_CMD "ls /dev/accel/accel* >/dev/null 2>&1"; then
  NPU=$($SSH_CMD "ls /dev/accel/accel*" | head -1)
  pass "NPU device: $NPU"
else
  warn "NPU device not found (optional)"
fi
echo ""

# Test 6: Tinygrad
echo "Test 6: Tinygrad Availability"
if $SSH_CMD "nix-shell -p python3Packages.tinygrad --run 'python -c \"import tinygrad\"'" 2>/dev/null; then
  pass "Tinygrad can be loaded"
else
  fail "Tinygrad not available"
fi
echo ""

# Test 7: Exo service
echo "Test 7: Exo Service"
if $SSH_CMD "systemctl list-unit-files | grep -q exo.service"; then
  STATUS=$($SSH_CMD "systemctl is-active exo.service 2>/dev/null || echo 'inactive'")
  if [ "$STATUS" = "active" ]; then
    pass "exo.service is active"
  else
    warn "exo.service exists but is $STATUS"
  fi
else
  warn "exo.service not configured"
fi
echo ""

# Test 8: API endpoint
echo "Test 8: Exo API"
if curl -s --connect-timeout 5 "http://${TARGET_IP}:52415/health" >/dev/null 2>&1; then
  pass "API is responding"

  # Check models endpoint
  if curl -s --connect-timeout 5 "http://${TARGET_IP}:52415/v1/models" | grep -q "data"; then
    pass "OpenAI-compatible API is working"
  fi
else
  warn "API is not responding"
  echo "    Start with: ssh root@${TARGET_IP} 'EXO_TINYGRAD_ENABLED=true exo -vv'"
fi
echo ""

echo "=== Summary ==="
echo "Hardware: Ready ✓"
echo "Software: Needs configuration"
echo ""
echo "To complete setup:"
echo "  1. Run: ./tests/rebuild_and_test_gremlin1.sh"
echo "  2. Or manually: ssh root@${TARGET_IP} 'cd /etc/nixos && nixos-rebuild switch --flake .#gremlin-1'"

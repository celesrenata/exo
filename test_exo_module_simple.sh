#!/usr/bin/env bash
# Simple test to verify exo module works on gremlin-1
# This doesn't modify the system, just tests if exo is available

set -euo pipefail

GREMLIN1_HOST="root@10.1.1.12"

echo "=== Testing exo on gremlin-1 ==="
echo ""

# Check if exo is installed
echo "1. Checking if exo is installed..."
if ssh "$GREMLIN1_HOST" "which exo"; then
    echo "✓ exo is installed"
else
    echo "✗ exo is not installed"
    exit 1
fi

# Check GPU detection
echo ""
echo "2. Checking GPU detection..."
ssh "$GREMLIN1_HOST" "lspci | grep -i 'vga.*intel'"

# Check if tinygrad is available
echo ""
echo "3. Checking if tinygrad is available..."
if ssh "$GREMLIN1_HOST" "python3 -c 'import tinygrad; print(f\"tinygrad {tinygrad.__version__}\")'" 2>/dev/null; then
    echo "✓ tinygrad is available"
else
    echo "✗ tinygrad is not available"
fi

# Check OpenCL
echo ""
echo "4. Checking OpenCL..."
if ssh "$GREMLIN1_HOST" "which clinfo" >/dev/null 2>&1; then
    ssh "$GREMLIN1_HOST" "clinfo | grep -A 3 'Platform Name'"
else
    echo "⚠ clinfo not installed"
fi

# Try to start exo manually
echo ""
echo "5. Testing exo startup (will kill after 5 seconds)..."
ssh "$GREMLIN1_HOST" "timeout 5 env EXO_TINYGRAD_ENABLED=true TINYGRAD_BACKEND=GPU exo -vv 2>&1 || true" | head -50

echo ""
echo "=== Test complete ==="

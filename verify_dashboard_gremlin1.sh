#!/usr/bin/env bash
set -euo pipefail

echo "=== Verifying Dashboard on gremlin-1 ==="
echo ""

TARGET="gremlin-1"

echo "1. Finding exo installation..."
EXO_PATH=$(ssh $TARGET 'readlink -f $(which exo)')
PACKAGE_DIR=$(ssh $TARGET "dirname \$(dirname $EXO_PATH)")
echo "Package: $PACKAGE_DIR"

echo ""
echo "2. Checking dashboard content..."
DASHBOARD_DIR="$PACKAGE_DIR/dashboard"

# Check if PyTorch+IPEX is in the dashboard
if ssh $TARGET "grep -r 'PyTorch.*IPEX.*Ring' $DASHBOARD_DIR 2>/dev/null | head -1"; then
    echo ""
    echo "✓ Dashboard DOES contain PyTorch+IPEX Ring"
else
    echo ""
    echo "✗ Dashboard DOES NOT contain PyTorch+IPEX Ring"
    echo ""
    echo "The dashboard needs to be rebuilt. Run:"
    echo "  ./force_rebuild_gremlin1.sh"
fi

echo ""
echo "3. Checking git commit..."
ssh $TARGET "cd /tmp && nix flake metadata 2>/dev/null | grep -A 3 'exo'" || echo "Could not get flake metadata"

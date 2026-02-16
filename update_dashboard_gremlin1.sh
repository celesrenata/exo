#!/usr/bin/env bash
# Update dashboard on gremlin-1 with the newly built version
#
# This script copies the locally built dashboard to gremlin-1

set -euo pipefail

TARGET_IP="10.1.1.12"

echo "=== Updating Dashboard on gremlin-1 ==="
echo ""

# Check if dashboard is built
if [ ! -d "dashboard/build" ]; then
  echo "Error: dashboard/build directory not found"
  echo "Run: npm run build --prefix dashboard"
  exit 1
fi

# Find where exo is installed on gremlin-1
echo "Finding exo installation path on gremlin-1..."
EXO_PATH=$(ssh root@${TARGET_IP} 'readlink -f $(which exo)')
echo "Exo binary: $EXO_PATH"

# Get the package directory (should be something like /nix/store/xxx-exo-0.3.0)
PACKAGE_DIR=$(ssh root@${TARGET_IP} "dirname \$(dirname $EXO_PATH)")
echo "Package directory: $PACKAGE_DIR"

# Check if dashboard exists in package
DASHBOARD_PATH="${PACKAGE_DIR}/dashboard/build"
echo "Expected dashboard path: $DASHBOARD_PATH"

if ssh root@${TARGET_IP} "[ -d '$DASHBOARD_PATH' ]"; then
  echo "✓ Dashboard directory exists"
else
  echo "✗ Dashboard directory not found at $DASHBOARD_PATH"
  echo ""
  echo "The dashboard may be in a different location."
  echo "Searching for dashboard..."
  ssh root@${TARGET_IP} "find $PACKAGE_DIR -name 'build' -type d 2>/dev/null | grep dashboard" || echo "No dashboard build directory found"
  exit 1
fi

# Backup existing dashboard
echo ""
echo "Creating backup of existing dashboard..."
ssh root@${TARGET_IP} "cp -r $DASHBOARD_PATH ${DASHBOARD_PATH}.backup-\$(date +%Y%m%d-%H%M%S)"
echo "✓ Backup created"

# Copy new dashboard
echo ""
echo "Copying new dashboard to gremlin-1..."
rsync -avz --delete dashboard/build/ root@${TARGET_IP}:${DASHBOARD_PATH}/
echo "✓ Dashboard copied"

# Restart exo service if it's running
echo ""
echo "Checking if exo service needs restart..."
if ssh root@${TARGET_IP} "pgrep -f 'exo -vv' >/dev/null"; then
  echo "Exo is running. Restart it to see changes:"
  echo "  ssh root@${TARGET_IP} 'pkill -f \"exo -vv\" && nohup env EXO_TINYGRAD_ENABLED=true exo -vv > /tmp/exo.log 2>&1 &'"
else
  echo "Exo is not running. Start it with:"
  echo "  ssh root@${TARGET_IP} 'EXO_TINYGRAD_ENABLED=true exo -vv'"
fi

echo ""
echo "=== Dashboard Update Complete ==="
echo "Access dashboard at: http://${TARGET_IP}:52415"
echo ""
echo "Note: You may need to hard refresh your browser (Ctrl+Shift+R or Cmd+Shift+R)"

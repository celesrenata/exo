#!/usr/bin/env bash
set -e

echo "=== Fixing apply.py bug on gremlin-1 ==="

# Copy the fixed file to gremlin-1
echo "Copying fixed apply.py to gremlin-1..."
scp src/exo/shared/apply.py root@gremlin-1:/tmp/apply.py

# SSH into gremlin-1 and apply the fix
ssh root@gremlin-1 << 'EOF'
set -e

echo "Stopping exo service..."
systemctl stop exo-user.service || true

echo "Backing up original apply.py..."
cp /nix/store/r907wq8wm8ci9d3z7b2bsc0y76vj1kcd-exo-0.3.0/lib/python3.13/site-packages/exo/shared/apply.py \
   /tmp/apply.py.backup || true

echo "Finding exo installation path..."
EXO_PATH=$(find /nix/store -name "exo-0.3.0" -type d | grep "lib/python3.13/site-packages" | head -1)
if [ -z "$EXO_PATH" ]; then
    echo "ERROR: Could not find exo installation"
    exit 1
fi

echo "Found exo at: $EXO_PATH"
APPLY_PATH="$EXO_PATH/exo/shared/apply.py"

echo "Replacing apply.py..."
cp /tmp/apply.py "$APPLY_PATH"

echo "Clearing log file..."
> /tmp/exo.log

echo "Starting exo service..."
systemctl start exo-user.service

echo "Waiting for service to start..."
sleep 5

echo "Checking service status..."
systemctl status exo-user.service --no-pager || true

echo "Checking logs..."
tail -50 /tmp/exo.log

echo ""
echo "=== Fix applied! ==="
echo "Monitor logs with: ssh root@gremlin-1 'tail -f /tmp/exo.log'"
EOF

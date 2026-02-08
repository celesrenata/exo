#!/usr/bin/env bash
set -euo pipefail

echo "=== Building and deploying exo to gremlin-1 (user service) ==="

# Build the exo package locally for Linux
echo "Building exo package for Linux..."
nix build .#exo --system x86_64-linux

# Copy the built package to gremlin-1
echo "Copying exo package to gremlin-1..."
nix copy --to ssh://gremlin-1 ./result

# Get the store path
STORE_PATH=$(readlink -f ./result)
echo "Store path: $STORE_PATH"

# Create a user systemd service on gremlin-1
echo "Creating user systemd service on gremlin-1..."
ssh gremlin-1 bash <<EOF
set -euo pipefail

# Create user systemd directory
mkdir -p ~/.config/systemd/user

# Stop existing service if running
systemctl --user stop exo.service 2>/dev/null || true

# Create service file
cat > ~/.config/systemd/user/exo.service <<'SERVICE'
[Unit]
Description=EXO distributed AI inference system
After=network.target

[Service]
Type=simple
ExecStart=$STORE_PATH/bin/exo
Restart=always
RestartSec=10
Environment="EXO_TINYGRAD_ENABLED=true"
Environment="TINYGRAD_BACKEND=CLANG"
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
SERVICE

# Reload systemd and start service
systemctl --user daemon-reload
systemctl --user enable exo.service
systemctl --user start exo.service

echo "Service started. Checking status..."
sleep 2
systemctl --user status exo.service --no-pager || true
EOF

echo ""
echo "=== Deployment complete ==="
echo "Check service status: ssh gremlin-1 'systemctl --user status exo.service'"
echo "View logs: ssh gremlin-1 'journalctl --user -u exo.service -f'"
echo "Stop service: ssh gremlin-1 'systemctl --user stop exo.service'"
echo "Test API: curl http://gremlin-1:52415/health"

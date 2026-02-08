#!/usr/bin/env bash
set -euo pipefail

echo "=== Building and deploying exo to gremlin-1 ==="

# Build the exo package locally for Linux
echo "Building exo package for Linux..."
nix build .#exo --system x86_64-linux

# Copy the built package to gremlin-1
echo "Copying exo package to gremlin-1..."
nix copy --to ssh://gremlin-1 ./result

# Get the store path
STORE_PATH=$(readlink -f ./result)
echo "Store path: $STORE_PATH"

# Create a systemd service on gremlin-1 that uses this package
echo "Creating systemd service on gremlin-1..."
ssh -t gremlin-1 bash <<EOF
set -euo pipefail

# Stop existing service if running
sudo systemctl stop exo.service 2>/dev/null || true

# Create service file
sudo tee /etc/systemd/system/exo.service > /dev/null <<'SERVICE'
[Unit]
Description=EXO distributed AI inference system
After=network.target

[Service]
Type=simple
ExecStart=$STORE_PATH/bin/exo
Restart=always
RestartSec=10
User=root
Environment="EXO_TINYGRAD_ENABLED=true"
Environment="TINYGRAD_BACKEND=CLANG"
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVICE

# Reload systemd and start service
sudo systemctl daemon-reload
sudo systemctl enable exo.service
sudo systemctl start exo.service

echo "Service started. Checking status..."
sleep 2
sudo systemctl status exo.service --no-pager || true
EOF

echo ""
echo "=== Deployment complete ==="
echo "Check service status: ssh gremlin-1 'sudo systemctl status exo.service'"
echo "View logs: ssh gremlin-1 'sudo journalctl -u exo.service -f'"
echo "Test API: curl http://gremlin-1:52415/health"

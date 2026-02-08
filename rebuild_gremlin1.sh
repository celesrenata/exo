#!/usr/bin/env bash
set -euo pipefail

echo "=== Rebuilding gremlin-1 NixOS configuration ==="

# Copy the flake to gremlin-1
echo "Copying flake configuration to gremlin-1..."
scp -r gremlin-1-flake.nix gremlin-1:/tmp/flake.nix

# SSH into gremlin-1 and rebuild
echo "Rebuilding NixOS configuration on gremlin-1..."
ssh gremlin-1 'sudo nixos-rebuild switch --flake /tmp#gremlin-1'

echo "=== Rebuild complete ==="
echo "Check service status with: ssh gremlin-1 'systemctl status exo.service'"
echo "View logs with: ssh gremlin-1 'journalctl -u exo.service -f'"

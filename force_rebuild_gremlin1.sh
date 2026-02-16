#!/usr/bin/env bash
set -euo pipefail

echo "=== Force Rebuilding gremlin-1 with Latest Changes ==="
echo ""

# Copy the flake to gremlin-1
echo "1. Copying flake configuration to gremlin-1..."
scp gremlin-1-flake.nix gremlin-1:/tmp/flake.nix

# SSH into gremlin-1 and rebuild with updated flake
echo ""
echo "2. Updating flake inputs and rebuilding on gremlin-1..."
ssh gremlin-1 << 'EOF'
  cd /tmp
  
  # Update the flake lock to get latest exo from GitHub
  echo "Updating flake inputs to get latest exo commit..."
  nix flake update
  
  # Show what commit we're using
  echo ""
  echo "Exo commit being used:"
  nix flake metadata | grep -A 5 "exo"
  
  # Rebuild with the updated flake
  echo ""
  echo "Rebuilding NixOS configuration..."
  sudo nixos-rebuild switch --flake /tmp#gremlin-1
EOF

echo ""
echo "=== Rebuild complete ==="
echo ""
echo "The dashboard should now show 'PyTorch+IPEX Ring' option"
echo ""
echo "Check service status with: ssh gremlin-1 'systemctl status exo.service'"
echo "View logs with: ssh gremlin-1 'journalctl -u exo.service -f'"
echo "Access dashboard at: http://10.1.1.12:52415"
echo ""
echo "Note: You may need to hard refresh your browser (Ctrl+Shift+R)"

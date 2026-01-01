#!/usr/bin/env bash

# Script to update NixOS flake to use local EXO with MLX fixes

set -e

NIXOS_DIR="$HOME/sources/nixos"
EXO_DIR="/home/celes/sources/celesrenata/exo"

echo "🔄 Updating NixOS flake to use local EXO with MLX fixes..."

# Check if NixOS directory exists
if [[ ! -d "$NIXOS_DIR" ]]; then
    echo "❌ NixOS directory not found: $NIXOS_DIR"
    echo "Please update the NIXOS_DIR variable in this script"
    exit 1
fi

# Check if EXO directory exists
if [[ ! -d "$EXO_DIR" ]]; then
    echo "❌ EXO directory not found: $EXO_DIR"
    echo "Please update the EXO_DIR variable in this script"
    exit 1
fi

cd "$NIXOS_DIR"

echo "📁 Working in: $(pwd)"

# Backup the current flake.nix
echo "💾 Creating backup of flake.nix..."
cp flake.nix flake.nix.backup

# Update the EXO input to use local path
echo "🔧 Updating EXO input to use local path..."
sed -i 's|exo\.url = "github:celesrenata/exo";|exo.url = "path:/home/celes/sources/celesrenata/exo";|' flake.nix

# Verify the change
if grep -q 'exo.url = "path:/home/celes/sources/celesrenata/exo"' flake.nix; then
    echo "✅ Successfully updated EXO input to local path"
else
    echo "❌ Failed to update EXO input. Please check flake.nix manually"
    exit 1
fi

# Update the flake lock for EXO
echo "🔄 Updating flake lock for EXO..."
nix flake update exo

echo "🏗️  Rebuilding NixOS system..."
sudo nixos-rebuild switch --flake .#esnixi

echo "✅ NixOS rebuild complete!"
echo "🔄 Checking EXO service status..."
systemctl status exo --no-pager -l

echo ""
echo "🎯 EXO should now be running with MLX fixes!"
echo "🌐 Dashboard available at: http://localhost:52415"
echo "📊 Monitor logs with: sudo journalctl -u exo -f"
echo ""
echo "Expected behavior:"
echo "  ✅ No more 'ModuleNotFoundError: No module named mlx' errors"
echo "  ✅ Service shows 'MLX not available' and continues with CPU inference"
echo "  ✅ Dashboard is accessible and functional"
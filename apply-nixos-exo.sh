#!/usr/bin/env bash

# Script to apply EXO configuration to NixOS system

set -e

echo "🔄 Applying EXO configuration to NixOS..."

# Check if we're running as root or with sudo access
if [[ $EUID -eq 0 ]]; then
    SUDO=""
elif sudo -n true 2>/dev/null; then
    SUDO="sudo"
else
    echo "❌ This script requires sudo access to modify NixOS configuration"
    exit 1
fi

# Get the absolute path to the EXO configuration
EXO_CONFIG_PATH=$(realpath nixos-exo-config.nix)
echo "📁 EXO config path: $EXO_CONFIG_PATH"

# Check if the configuration file exists
if [[ ! -f "$EXO_CONFIG_PATH" ]]; then
    echo "❌ EXO configuration file not found: $EXO_CONFIG_PATH"
    exit 1
fi

# Copy the configuration to /etc/nixos/
echo "📋 Copying EXO configuration to /etc/nixos/..."
$SUDO cp "$EXO_CONFIG_PATH" /etc/nixos/exo.nix

# Check if the main configuration.nix imports our EXO config
if ! $SUDO grep -q "exo.nix" /etc/nixos/configuration.nix; then
    echo "🔧 Adding EXO import to configuration.nix..."
    
    # Create a backup
    $SUDO cp /etc/nixos/configuration.nix /etc/nixos/configuration.nix.backup
    
    # Add the import (assuming there's already an imports section)
    if $SUDO grep -q "imports.*=.*\[" /etc/nixos/configuration.nix; then
        # Add to existing imports
        $SUDO sed -i '/imports.*=.*\[/a\    ./exo.nix' /etc/nixos/configuration.nix
    else
        # Add imports section at the top
        $SUDO sed -i '1i\  imports = [\n    ./exo.nix\n  ];\n' /etc/nixos/configuration.nix
    fi
    
    echo "✅ Added EXO import to configuration.nix"
else
    echo "ℹ️  EXO import already exists in configuration.nix"
fi

echo "🔄 Rebuilding NixOS system..."
$SUDO nixos-rebuild switch

echo "✅ NixOS rebuild complete!"
echo "🔄 Checking EXO service status..."
systemctl status exo --no-pager -l

echo ""
echo "🎯 EXO service should now be running with CPU inference!"
echo "🌐 Dashboard available at: http://localhost:52415"
echo "📊 Monitor logs with: sudo journalctl -u exo -f"
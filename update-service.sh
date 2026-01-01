#!/usr/bin/env bash

# Script to update the EXO systemd service with the new build

set -e

echo "🔄 Updating EXO systemd service..."

# Get the new package path
NEW_PACKAGE=$(readlink -f ./result)
echo "📦 New package: $NEW_PACKAGE"

# Check if service exists
if ! systemctl list-unit-files | grep -q "exo.service"; then
    echo "❌ EXO service not found. Please install it first."
    exit 1
fi

# Update the service file to use the new package
SERVICE_FILE="/etc/systemd/system/exo.service"

echo "🛠️  Updating service file..."
sudo sed -i "s|ExecStart=.*|ExecStart=$NEW_PACKAGE/bin/exo|" "$SERVICE_FILE"

# Add environment variables for MLX disable and CPU inference
echo "🔧 Adding environment variables..."
if ! grep -q "MLX_DISABLE" "$SERVICE_FILE"; then
    sudo sed -i '/\[Service\]/a Environment="MLX_DISABLE=1"' "$SERVICE_FILE"
fi

if ! grep -q "EXO_INFERENCE_ENGINE" "$SERVICE_FILE"; then
    sudo sed -i '/\[Service\]/a Environment="EXO_INFERENCE_ENGINE=cpu"' "$SERVICE_FILE"
fi

if ! grep -q "DASHBOARD_DIR" "$SERVICE_FILE"; then
    sudo sed -i "/\[Service\]/a Environment=\"DASHBOARD_DIR=$NEW_PACKAGE/share/exo/dashboard\"" "$SERVICE_FILE"
fi

# Reload systemd and restart service
echo "🔄 Reloading systemd..."
sudo systemctl daemon-reload

echo "🔄 Restarting EXO service..."
sudo systemctl restart exo

echo "✅ Service updated successfully!"
echo "📊 Checking service status..."
sudo systemctl status exo --no-pager -l

echo ""
echo "🎯 To monitor logs: sudo journalctl -u exo -f"
echo "🌐 Dashboard should be available at: http://localhost:52415"
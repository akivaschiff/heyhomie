#!/bin/bash
# Install Homie as a systemd service

set -e

echo "Installing Homie systemd service..."

# Copy service file to systemd directory
sudo cp homie.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable homie.service

echo "✓ Service installed and enabled"
echo ""
echo "Commands:"
echo "  sudo systemctl start homie    # Start now"
echo "  sudo systemctl stop homie     # Stop"
echo "  sudo systemctl status homie   # Check status"
echo "  journalctl -u homie -f        # View logs"

#!/bin/bash

# Simple MPTCP scheduler install script
set -e

echo "=== Installing MPTCP ReLes Scheduler ==="

# Clean and compile
echo "Cleaning..."
make clean

echo "Compiling..."
make

# Remove old module if loaded
echo "Removing old module..."
sudo rmmod mptcp_reles 2>/dev/null || echo "Module not loaded"

# Copy to system directory
echo "Installing module..."
sudo cp mptcp_reles.ko /lib/modules/$(uname -r)/kernel/net/mptcp/

# Update dependencies
echo "Updating module dependencies..."
sudo depmod

# Load new module
echo "Loading module..."
sudo modprobe mptcp_reles

# Set scheduler
echo "Setting scheduler..."
echo "reles" | sudo tee /proc/sys/net/mptcp/mptcp_scheduler

echo "=== Done! ==="
echo "Current scheduler: $(cat /proc/sys/net/mptcp/mptcp_scheduler)"
echo "Module version: $(modinfo mptcp_reles | grep version)"
sudo sysctl -w net.mptcp.mptcp_enabled=1
sudo sysctl net.mptcp.mptcp_scheduler=reles

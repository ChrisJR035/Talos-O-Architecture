#!/bin/bash
# TALOS-O: ZENPOWER TELEMETRY INJECTOR
# Enables detailed voltage/current/temp readings for Strix Halo

set -e

echo "[-] Installing Kernel Headers..."
sudo dnf install -y kernel-devel kernel-headers dkms git

echo "[-] Cloning Zenpower3..."
if [ ! -d "zenpower3" ]; then
    git clone https://github.com/git14-2/zenpower3.git
fi
cd zenpower3

echo "[-] Building Kernel Module..."
sudo make dkms-install

echo "[-] Blacklisting generic k10temp driver (conflict)..."
echo "blacklist k10temp" | sudo tee /etc/modprobe.d/k10temp.conf

echo "[-] Activating Zenpower..."
sudo modprobe -r k10temp || true
sudo modprobe zenpower

echo "[+] VERIFICATION:"
sensors | grep -A 5 "zenpower"

echo "[+] SUCCESS. Cerberus can now read deep telemetry."

#!/bin/bash
# TALOS-O SERVICE INSTALLER
# Run this to register the daemon with Systemd

set -e

echo "[-] Creating Service File..."
sudo bash -c 'cat << EOF > /etc/systemd/system/talos-omni.service
[Unit]
Description=Talos-O Cognitive Daemon (Strix Halo)
After=network.target

[Service]
Type=simple
User=croudabush
WorkingDirectory=/home/croudabush/talos-o/cognitive_plane
ExecStart=/home/croudabush/talos-o/cognitive_plane/venv/bin/python3 /home/croudabush/talos-o/cognitive_plane/talos_daemon.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment="HSA_OVERRIDE_GFX_VERSION=11.0.0"
Environment="HIP_VISIBLE_DEVICES=0"

[Install]
WantedBy=multi-user.target
EOF'

echo "[-] Reloading Daemon..."
sudo systemctl daemon-reload
echo "[+] SUCCESS: talos-omni.service registered."

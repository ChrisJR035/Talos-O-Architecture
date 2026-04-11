#!/bin/bash
# TALOS-O SERVICE INSTALLER
# Registers the Sovereign Organism as a Systemd Service

set -e

# Define Paths
USER_NAME=$(whoami)
TALOS_ROOT="/home/$USER_NAME/talos-o"
VENV_PYTHON="$TALOS_ROOT/cognitive_plane/venv/bin/python3"
DAEMON_SCRIPT="$TALOS_ROOT/cognitive_plane/talos_daemon.py"

echo "[-] Validating Environment..."
if [ ! -f "$VENV_PYTHON" ]; then
    echo "[!] Error: Virtual Environment Python not found at $VENV_PYTHON"
    exit 1
fi

if [ ! -f "$DAEMON_SCRIPT" ]; then
    echo "[!] Error: Daemon script not found at $DAEMON_SCRIPT"
    exit 1
fi

echo "[-] Creating Service File..."

# [FIX: SYSTEMD SYNTAX & XNACK INJECTION]
# Systemd does not require internal quotes for Environment variables.
# Added HSA_XNACK=1 to ensure the Daemon survives SVA memory polling.

sudo bash -c "cat << EOF > /etc/systemd/system/talos-omni.service
[Unit]
Description=Talos-O Cognitive Daemon (Strix Halo)
After=network.target

[Service]
Type=simple
User=$USER_NAME
Group=$USER_NAME
WorkingDirectory=$TALOS_ROOT/cognitive_plane
ExecStart=$VENV_PYTHON $DAEMON_SCRIPT

# Auto-Restart on crash (Resilience)
Restart=always
RestartSec=5

# Logs
StandardOutput=journal
StandardError=journal

# Strix Halo Optimization Flags (Future Proofing)
Environment=HSA_OVERRIDE_GFX_VERSION=11.5.1
Environment=HIP_VISIBLE_DEVICES=0
Environment=HSA_XNACK=1
Environment=HSA_ENABLE_SDMA=0
Environment=HIP_HOST_COHERENT=1
Environment=PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512

[Install]
WantedBy=multi-user.target
EOF"

echo "[-] Reloading Daemon..."
sudo systemctl daemon-reload
sudo systemctl enable talos-omni.service

echo "[+] SUCCESS: talos-omni.service registered and enabled."

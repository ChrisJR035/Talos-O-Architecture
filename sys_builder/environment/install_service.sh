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

echo "[-] Resolving dynamic library paths..."
TORCH_LIB=$($VENV_PYTHON -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null || echo "$TALOS_ROOT/cognitive_plane/venv/lib/python3.13t/site-packages/torch/lib")

echo "[-] Creating Service File..."

# [FIX: SYSTEMD SYNTAX & XNACK INJECTION]
# Systemd does not require internal quotes for Environment variables.
# Added HSA_XNACK=1 to ensure the Daemon survives SVA memory polling.

# [FIX: BASH INTERPOLATION] Using 'tee' prevents the shell from swallowing the \$f loop variable.
cat << EOF | sudo tee /etc/systemd/system/talos-omni.service > /dev/null
[Unit]
Description=Talos-O Cognitive Daemon (Strix Halo)
After=network.target

[Service]
Type=notify
User=$USER_NAME
Group=$USER_NAME
WorkingDirectory=$TALOS_ROOT/cognitive_plane

# --- THE EPISTEMIC MEMBRANE (PRE-IGNITION) ---
# Zero-cost bitwise homogenization. Sweeps the Python codebase to eradicate invisible 
# syntactic poison (like U+00A0 Non-Breaking Spaces) before the interpreter boots.
ExecStartPre=+/bin/sh -c "find $TALOS_ROOT/cognitive_plane -type f -name '*.py' -exec sed -i 's/\\xC2\\xA0/ /g' {} +"

# --- THE THERMAL CAGE (PRE-IGNITION) ---
# Systemd forces all cores to minimum clock speeds before Python executes
ExecStartPre=+/bin/sh -c 'for f in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do echo power > "\$f" 2>/dev/null || true; done'

ExecStart=$VENV_PYTHON $DAEMON_SCRIPT

# --- THE THERMAL RELEASE (POST-IGNITION) ---
# Systemd restores frequency scaling ONLY after Python signals sd_notify "READY=1"
ExecStartPost=+/bin/sh -c 'for f in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do echo balance_performance > "\$f" 2>/dev/null || true; done'

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

# [FIX: SYSTEMD ENVIRONMENT ISOLATION]
# Injecting the exact ROCm paths and runtime flags from activate_talos.sh
Environment=LD_LIBRARY_PATH=$TORCH_LIB:/home/$USER_NAME/rocm-native/lib:/home/$USER_NAME/rocm-native/llvm/lib:/usr/lib64:/usr/lib64/rocm/lib
Environment=HIPBLASLT_DISABLE=1
Environment=GPU_MAX_HW_QUEUES=4
Environment=HSA_ENABLE_INTERRUPT=0
Environment=AMDR_WAVEFRONT_SIZE=32
Environment=PYTHON_GIL=0

[Install]
WantedBy=multi-user.target
EOF

echo "[-] Reloading Daemon..."
sudo systemctl daemon-reload
sudo systemctl enable talos-omni.service

echo "[+] SUCCESS: talos-omni.service registered and enabled."

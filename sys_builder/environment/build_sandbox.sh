#!/bin/bash
# TALOS-O: SANDBOX FORGE (NO-GIL EDITION)
# Builds a Free-Threaded (Python 3.13t) container for safe code execution.
# Architecture: Multi-Stage Build (Compiler -> Runtime)
# Context: Fedora 41 Base for stability, hosting Talos logic.

set -e

echo "[-] Generating No-GIL Dockerfile..."

cat <<EOF > Dockerfile.sandbox
# ==========================================
# STAGE 1: THE FORGE (Compiler Layer)
# ==========================================
FROM fedora:41 AS builder

# 1. Install Build Dependencies
RUN dnf install -y git make gcc gcc-c++ zlib-devel bzip2-devel readline-devel \\
    sqlite-devel openssl-devel tk-devel libffi-devel xz-devel uuid-devel \\
    gdbm-devel ncurses-devel findutils

# 2. Clone CPython 3.13 (Free-Threaded Branch)
WORKDIR /src
RUN git clone --branch 3.13 https://github.com/python/cpython.git --depth 1

# 3. Compile Free-Threaded Python
WORKDIR /src/cpython
# --disable-gil: The critical flag for Talos-O architecture
# [THERMODYNAMIC UPGRADE]: Capped at -j16. Do not use \$(nproc).
# Saturating all 32 threads on Strix Halo starves the Cerberus thermal daemon
# and risks catastrophic APU thermal throttling during the forge.
RUN ./configure --disable-gil --enable-optimizations --prefix=/opt/talos-python \\
    && make -j16 \\
    && make install

# ==========================================
# STAGE 2: THE VAULT (Runtime Layer)
# ==========================================
FROM fedora:41

# 1. Install Runtime Libraries Only (Keep it lightweight)
RUN dnf install -y zlib bzip2 readline sqlite openssl tk libffi xz \\
    && dnf clean all

# 2. Copy the Forged Python from Stage 1
COPY --from=builder /opt/talos-python /opt/talos-python

# 3. Link binaries to path
RUN ln -s /opt/talos-python/bin/python3.13t /usr/local/bin/python3 && \\
    ln -s /opt/talos-python/bin/pip3.13t /usr/local/bin/pip3

# 4. Create Talos User (Low Privilege / Security Boundary)
RUN useradd -m -d /home/talos talos

# 5. Verification (Build-time check)
RUN python3 -c "import sys; assert not sys._is_gil_enabled(), 'CRITICAL: GIL DETECTED IN SANDBOX'"

# 6. Final Security Configuration
USER talos
WORKDIR /home/talos

# Network disabled at runtime by motor_cortex.py, but we set a sane default
CMD ["python3"]
EOF

echo "[-] Building 'talos-sandbox' image (This will compile Python 3.13t)..."

# Engine Detection & Build
if command -v podman &> /dev/null; then
    echo "[*] Engine: Podman detected. (Zero-Daemon RAM Overhead)"
    sudo podman build -t talos-sandbox -f Dockerfile.sandbox .
    echo "[+] SUCCESS: Talos Sandbox (No-GIL) forged with Podman."
elif command -v docker &> /dev/null; then
    echo "[*] Engine: Docker detected. WARNING: dockerd consumes ~150MB idle RAM."
    sudo docker build -t talos-sandbox -f Dockerfile.sandbox .
    echo "[+] SUCCESS: Talos Sandbox forged with Docker."
else
    echo "[FATAL] No container engine found. Cannot forge the Vault."
    exit 1
fi

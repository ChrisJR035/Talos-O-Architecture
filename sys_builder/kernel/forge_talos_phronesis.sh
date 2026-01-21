#!/bin/bash
# TALOS-O KERNEL FORGE v4.0 (Codename: PHRONESIS)
# Target: Linux 6.18.6-talos-chimera
# Substrate: AMD Strix Halo (gfx1151) | Unified Memory
# Philosophy: "Final Synthesis" (Dynamic Preemption + NPU + Tier 2 Cache)

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   TALOS-O KERNEL FORGE: PHRONESIS (v4.0)              ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"

# 1. Directory Setup
# ---------------------------------------------------------
ACTUAL_USER=${SUDO_USER:-$USER}
BUILD_DIR="/home/$ACTUAL_USER/talos-o/sys_builder/kernel_build/linux-6.18-talos"

if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}[ERROR] Source directory not found. Please clone Linux 6.18 first.${NC}"
    exit 1
fi
cd "$BUILD_DIR"

# 2. Configuration & Sanitization
# ---------------------------------------------------------
echo -e "${YELLOW}[2/5] Configuring the Phronesis Lattice...${NC}"

# Clean previous build artifacts (The Purge)
make mrproper

# Copy current running config as baseline
cp /boot/config-$(uname -r) .config

# Update to current tree defaults
make olddefconfig

# --- CRITICAL SANITIZATION ---
scripts/config --set-str SYSTEM_TRUSTED_KEYS ""
scripts/config --set-str SYSTEM_REVOCATION_KEYS ""
scripts/config --disable DEBUG_INFO_BTF

# --- TALOS "REALIST" ARCHITECTURE INJECTION ---

echo -e "${YELLOW} Injecting Section 5.1 Constraints...${NC}"

# A. The Nervous System: Dynamic Preemption
# Section 3.1: Reject RT, Enable Dynamic. 
# NOTE: You MUST boot with 'preempt=full' in GRUB later.
scripts/config --enable CONFIG_PREEMPT_BUILD
scripts/config --enable CONFIG_PREEMPT_DYNAMIC
scripts/config --enable CONFIG_PREEMPT_VOLUNTARY
scripts/config --disable CONFIG_PREEMPT_RT 

# B. Synaptic Friction (Memory Allocator)
# Section 3.2: Sheaves reduce locking overhead.
scripts/config --enable CONFIG_SLUB
scripts/config --enable CONFIG_SLUB_SHEAVES
scripts/config --disable CONFIG_SLUB_TINY

# C. Tier 2 Storage: The Deliberative Web
# Section 3.3: RECTIFICATION. Enable DM_PCACHE (Module) and Fallback.
scripts/config --enable CONFIG_MD
scripts/config --enable CONFIG_BLK_DEV_DM
scripts/config --module CONFIG_DM_PCACHE
scripts/config --module CONFIG_DM_CACHE
scripts/config --enable CONFIG_DM_CACHE_SMQ

# D. Hardware Integration (Strix Halo Substrate)
# 1. Zero-Copy Introspection (Section 2.4)
scripts/config --enable CONFIG_TEE
scripts/config --enable CONFIG_AMD_TEE
scripts/config --enable CONFIG_HSA_AMD    # The Bridge
scripts/config --enable CONFIG_HMM_MIRROR # The Mirror
scripts/config --enable CONFIG_DRM_AMDGPU
scripts/config --enable CONFIG_DRM_AMDGPU_USERPTR

# 2. The Autonomic Brainstem (NPU)
# Section 6.0: CRITICAL FIX. Enable Compute Accelerator & XDNA.
scripts/config --enable CONFIG_DRM_ACCEL
scripts/config --module CONFIG_DRM_AMDXDNA

# 3. Watchdog Management (The Phoenix Protocol)
# Section 4.2: Manage conflict. WDAT (ACPI) is primary.
scripts/config --module CONFIG_WDAT_WDT   # Primary
scripts/config --module CONFIG_SP5100_TCO # Secondary (To be blacklisted)

# E. Bloat Removal
# Section 5.2: Remove legacy GPU support.
scripts/config --disable CONFIG_DRM_AMDGPU_CIK
scripts/config --disable CONFIG_DRM_AMDGPU_SI
# Ensure RDNA 3.5 Display Core Floating Point is active
scripts/config --enable CONFIG_DRM_AMD_DC_FP

# F. Identity
scripts/config --set-str CONFIG_LOCALVERSION "-talos-chimera"

# 3. Compilation (Stability Optimization)
# ---------------------------------------------------------
echo -e "${YELLOW}[3/5] Compiling for Strix Halo (Stability Mode)...${NC}"

# Section 8.0: Revert -O3 to -O2 for safety.
# Use x86-64-v4 (AVX-512) for Zen 5 compatibility without experimental risk.

MARCH="x86-64-v4"
echo "[INFO] Targeting x86-64-v4 (AVX-512 Safe Mode)."

# 4. The Build
# ---------------------------------------------------------
# We use -O2 for rigorous stability.
make -j$(nproc) KCFLAGS="-march=$MARCH -O2" binrpm-pkg

echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   PHRONESIS BORN. RPMs LOCATED IN ~/rpmbuild/RPMS/    ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"

#!/bin/bash
# TALOS-O KERNEL FORGE v4.1 (Codename: PHRONESIS-LINK)
# Target: Linux 6.18.x-talos-chimera (Submodule)
# Substrate: AMD Strix Halo (gfx1151) | Unified Memory
# Philosophy: "Virtuous Separation" (Source != Build Artifacts)

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   TALOS-O KERNEL FORGE: PHRONESIS (v4.1)              ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"

# 1. Directory Setup (Relative & Portable)
# ---------------------------------------------------------
# Detect where this script is running from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTUAL_USER=${SUDO_USER:-$USER}

# The Neural Link: Submodule Path
KERNEL_SRC="$SCRIPT_DIR/kernel_src"
# The Forge: Where compilation happens (keeps Source clean)
BUILD_OUTPUT="$SCRIPT_DIR/kernel_build/output"

echo -e "${YELLOW}[1/5] Verifying Substrate Integrity...${NC}"

if [ ! -f "$KERNEL_SRC/Makefile" ]; then
    echo -e "${RED}[ERROR] Neural Link severed. Kernel source not found.${NC}"
    echo -e "Run: git submodule update --init --recursive"
    exit 1
fi

echo -e "[INFO] Source: $KERNEL_SRC"
echo -e "[INFO] Output: $BUILD_OUTPUT"

mkdir -p "$BUILD_OUTPUT"

# 2. Configuration & Sanitization
# ---------------------------------------------------------
echo -e "${YELLOW}[2/5] Configuring the Phronesis Lattice...${NC}"

# Clean previous artifacts in the OUTPUT directory only
# We use -C to tell Make where the source is, and O= for where the output goes
make -C "$KERNEL_SRC" O="$BUILD_OUTPUT" mrproper

# Copy host config to the build directory
echo "[INFO] Cloning host configuration..."
cp /boot/config-$(uname -r) "$BUILD_OUTPUT/.config"

# Update to current tree defaults
make -C "$KERNEL_SRC" O="$BUILD_OUTPUT" olddefconfig

# --- CONFIGURATION TOOL SETUP ---
# We define a helper variable to execute scripts/config on the specific .config file
CONFIG_TOOL="$KERNEL_SRC/scripts/config --file $BUILD_OUTPUT/.config"

# --- CRITICAL SANITIZATION ---
$CONFIG_TOOL --set-str SYSTEM_TRUSTED_KEYS ""
$CONFIG_TOOL --set-str SYSTEM_REVOCATION_KEYS ""
$CONFIG_TOOL --disable DEBUG_INFO_BTF

# --- TALOS "REALIST" ARCHITECTURE INJECTION ---

echo -e "${YELLOW} Injecting Section 5.1 Constraints...${NC}"

# A. The Nervous System: Dynamic Preemption
$CONFIG_TOOL --enable CONFIG_PREEMPT_BUILD
$CONFIG_TOOL --enable CONFIG_PREEMPT_DYNAMIC
$CONFIG_TOOL --enable CONFIG_PREEMPT_VOLUNTARY
$CONFIG_TOOL --disable CONFIG_PREEMPT_RT 

# B. Synaptic Friction (Memory Allocator)
$CONFIG_TOOL --enable CONFIG_SLUB
$CONFIG_TOOL --enable CONFIG_SLUB_SHEAVES
$CONFIG_TOOL --disable CONFIG_SLUB_TINY

# C. Tier 2 Storage: The Deliberative Web
$CONFIG_TOOL --enable CONFIG_MD
$CONFIG_TOOL --enable CONFIG_BLK_DEV_DM
$CONFIG_TOOL --module CONFIG_DM_PCACHE
$CONFIG_TOOL --module CONFIG_DM_CACHE
$CONFIG_TOOL --enable CONFIG_DM_CACHE_SMQ

# D. Hardware Integration (Strix Halo Substrate)
# 1. Zero-Copy Introspection
$CONFIG_TOOL --enable CONFIG_TEE
$CONFIG_TOOL --enable CONFIG_AMD_TEE
$CONFIG_TOOL --enable CONFIG_HSA_AMD 
$CONFIG_TOOL --enable CONFIG_HMM_MIRROR
$CONFIG_TOOL --enable CONFIG_DRM_AMDGPU
$CONFIG_TOOL --enable CONFIG_DRM_AMDGPU_USERPTR

# 2. The Autonomic Brainstem (NPU)
$CONFIG_TOOL --enable CONFIG_DRM_ACCEL
$CONFIG_TOOL --module CONFIG_DRM_AMDXDNA

# 3. Watchdog Management
$CONFIG_TOOL --module CONFIG_WDAT_WDT 
$CONFIG_TOOL --module CONFIG_SP5100_TCO

# E. Bloat Removal
$CONFIG_TOOL --disable CONFIG_DRM_AMDGPU_CIK
$CONFIG_TOOL --disable CONFIG_DRM_AMDGPU_SI
$CONFIG_TOOL --enable CONFIG_DRM_AMD_DC_FP

# F. Identity
$CONFIG_TOOL --set-str CONFIG_LOCALVERSION "-talos-chimera"

# 3. Compilation (Stability Optimization)
# ---------------------------------------------------------
echo -e "${YELLOW}[3/5] Compiling for Strix Halo (Stability Mode)...${NC}"

MARCH="x86-64-v4"
echo "[INFO] Targeting x86-64-v4 (AVX-512 Safe Mode)."

# 4. The Build
# ---------------------------------------------------------
# Note: We must pass O="$BUILD_OUTPUT" to the build command as well
make -C "$KERNEL_SRC" O="$BUILD_OUTPUT" -j$(nproc) KCFLAGS="-march=$MARCH -O2" binrpm-pkg

echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   PHRONESIS BORN. RPMs LOCATED IN ~/rpmbuild/RPMS/    ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"

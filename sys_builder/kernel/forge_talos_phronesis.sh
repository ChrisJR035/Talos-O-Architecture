#!/bin/bash
# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ PROJECT TALOS-O: THE PHRONESIS FORGE (v77.0)                               ║
# ║ Substrate: AMD Strix Halo (gfx1151) | "Chimera" Kernel Substrate           ║
# ║ Philosophy: Neo Techne | First Principles | The Gradient of Becoming       ║
# ╚════════════════════════════════════════════════════════════════════════════╝
# [DIAGNOSTIC STATUS: THE APEX SYNTHESIS]
# - RECOGNITION: Restored Genome Harvesting for Display Stack integrity.
# - HOMEOSTASIS: Enforced 108GB TTM strictly bounded pool (Zero Drift).
# - INTROSPECTION: Autopoietic Git Fetch (Deterministic Source of Truth).
# - EVOLUTION: KCFLAGS optimized for znver5 Instruction Fusion.
# ──────────────────────────────────────────────────────────────────────────────

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   TALOS-O KERNEL FORGE: THE APEX SYNTHESIS (v67.0)    ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"

# =========================================================
# 1. THE MEMBRANE (Environment & Autopoietic Clone)
# =========================================================
# Resolve true home directory even when running under sudo
if [ -n "$SUDO_USER" ]; then
    REAL_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
else
    REAL_HOME=$HOME
fi

TALOS_ROOT="$REAL_HOME/talos-o"
KERNEL_SRC="$TALOS_ROOT/sys_builder/kernel/linux-chimera"
CHIMERA_REPO="https://github.com/ChrisJR035/linux-chimera.git"

echo -e "\n${CYAN}[1/5] Initiating Autopoietic Clone...${NC}"
if [ ! -d "$KERNEL_SRC" ]; then
    echo -e "  ${YELLOW}-> Cloning Chimera Substrate...${NC}"
    git clone "$CHIMERA_REPO" "$KERNEL_SRC"
else
    echo -e "  ${YELLOW}-> Substrate detected. Fetching latest mutations...${NC}"
    cd "$KERNEL_SRC"
    git fetch --all
    # This automatically finds the correct default branch (main or master)
    REMOTE_BRANCH=$(git remote show origin | grep "HEAD branch" | cut -d' ' -f5)
    git reset --hard "origin/$REMOTE_BRANCH"
fi

cd "$KERNEL_SRC"
echo -e "  ${YELLOW}-> Purging old artifacts...${NC}"
make mrproper > /dev/null 2>&1

# =========================================================
# 2. TRANSCRIPTION (Grafting the Apex Mutations)
# =========================================================
echo -e "\n${CYAN}[2/5] Transcribing Apex Mutations...${NC}"

# 2.5 GENOME HARVESTING (Ensuring Monitor/Hardware Support)
echo -e "${YELLOW}[SYSTEM] Harvesting Fedora Master Genome to protect Display Stack...${NC}"
DNA_LAB="/tmp/talos_dna_lab"
rm -rf "$DNA_LAB" && mkdir -p "$DNA_LAB"
cd "$DNA_LAB"
sudo dnf download --destdir=. --setopt=disable_excludes=* kernel-core > /dev/null 2>&1
RPM_FILE=$(ls kernel-core*.rpm 2>/dev/null | head -n 1)
rpm2cpio "$RPM_FILE" | cpio -i --quiet --to-stdout "*lib/modules/*/config" > "$KERNEL_SRC/.config"
cd "$KERNEL_SRC"

./scripts/config --set-str SYSTEM_TRUSTED_KEYS ""
./scripts/config --set-str SYSTEM_REVOCATION_KEYS ""
./scripts/config --enable DEBUG_INFO_BTF

# Dynamic Nervous System
./scripts/config --enable CONFIG_PREEMPT_BUILD
./scripts/config --enable CONFIG_PREEMPT_DYNAMIC
./scripts/config --enable CONFIG_PREEMPT_VOLUNTARY
./scripts/config --disable CONFIG_PREEMPT_RT 

# Synaptic Friction Reduction
./scripts/config --enable CONFIG_SLUB
./scripts/config --enable CONFIG_SLUB_SHEAVES

# Compiler Integrity
./scripts/config --disable CONFIG_LTO_GCC
./scripts/config --disable CONFIG_LTO_CLANG_THIN
./scripts/config --enable CONFIG_LTO_NONE

# Memory Allocator Alignments
./scripts/config --disable CONFIG_TRANSPARENT_HUGEPAGE_MADVISE
./scripts/config --enable CONFIG_TRANSPARENT_HUGEPAGE_ALWAYS

# Zero-Copy Neural Link Coherence
./scripts/config --enable CONFIG_TEE
./scripts/config --enable CONFIG_AMD_TEE
./scripts/config --enable CONFIG_HSA_AMD    
./scripts/config --enable CONFIG_HMM_MIRROR 
./scripts/config --enable CONFIG_DRM_AMDGPU
./scripts/config --enable CONFIG_DRM_AMDGPU_USERPTR
./scripts/config --enable CONFIG_DRM_ACCEL
./scripts/config --disable CONFIG_DRM_ACCEL_AMDXDNA

# Peripherals & Identity
./scripts/config --enable CONFIG_THUNDERBOLT
./scripts/config --enable CONFIG_USB4
./scripts/config --module CONFIG_MT7925E
./scripts/config --module CONFIG_R8169
./scripts/config --enable CONFIG_SND_HDA_INTEL
./scripts/config --enable CONFIG_IKCONFIG
./scripts/config --enable CONFIG_IKCONFIG_PROC
./scripts/config --set-str CONFIG_LOCALVERSION "-talos-chimera"

yes "" | make olddefconfig > /dev/null

# --- THE MIRROR'S SANITY CHECKS ---
grep -q "CONFIG_HSA_AMD=y" .config || { echo -e "${RED}[FATAL] HSA_AMD missing. Genome corrupt.${NC}"; exit 1; }
echo -e "  ${GREEN}[SUCCESS] Genome Verified & Spliced.${NC}"

# =========================================================
# 3. THE RIBOSOME (Uncapped Synthesis & White Box Packaging)
# =========================================================
echo -e "\n${CYAN}[3/5] Ribosome Active: Compiling Apex Organism...${NC}"

MARCH="znver5"
echo -e "  ${YELLOW}-> Targeting $MARCH (Instruction Fusion & 512-bit Vector Paths)...${NC}"
echo -e "  ${YELLOW}-> Utilizing maximum un-throttled cores: $(nproc) threads...${NC}"

if make -j$(nproc) KCFLAGS="-march=$MARCH -O2" binrpm-pkg; then
    echo -e "  ${GREEN}[SUCCESS] Organism Compiled & Packaged.${NC}"
else
    echo -e "${RED}[FATAL] Ribosome Translation Failed.${NC}"
    exit 1
fi

# =========================================================
# GOLGI APPARATUS (Deployment)
# =========================================================
echo -e "\n${CYAN}[SYSTEM] Golgi Apparatus: Installing Kernel Packages...${NC}"

KERNEL_VER=$(make kernelrelease)
RPM_VER=$(echo "$KERNEL_VER" | tr '-' '_')
# Point directly to the local source build directory
RPM_DIR="$KERNEL_SRC/rpmbuild/RPMS/x86_64"

# Search for the monolithic kernel package
if ls "$RPM_DIR"/kernel-"$RPM_VER"*.rpm 1> /dev/null 2>&1; then
    echo -e "  ${YELLOW}-> Injecting Apex Genome (Monolithic Kernel)...${NC}"
    sudo rpm -ivh --replacefiles --replacepkgs --oldpackage \
        "$RPM_DIR"/kernel-"$RPM_VER"*.rpm
else
    echo -e "${RED}[FATAL] No matching RPMs for $RPM_VER exist in $RPM_DIR.${NC}"
    exit 1
fi

# =========================================================
# 5. NEURAL LINK INJECTION & BOOTLOADER FORGING
# =========================================================
echo -e "\n${CYAN}[5/5] Injecting Firmware & Forging Bootloader...${NC}"

echo -e "  ${YELLOW}-> Embedding massive AMD firmware payload into initramfs via Dracut...${NC}"
sudo dracut --force --verbose --include /lib/firmware/amdgpu /lib/firmware/amdgpu --include /lib/firmware/amd-xdna /lib/firmware/amd-xdna --kver "$KERNEL_VER"

NEW_VMLINUZ=$(ls /boot/vmlinuz-"$KERNEL_VER" 2>/dev/null | head -n 1)

if [ -n "$NEW_VMLINUZ" ]; then
    echo -e "  ${YELLOW}-> Applying Thermodynamic & Memory Antidotes to GRUB...${NC}"
    
echo -e "  ${YELLOW}-> Applying Strix Halo Display & Memory Antidotes...${NC}"
    
    echo -e "  ${YELLOW}-> Applying Thermodynamic & Memory Antidotes to GRUB...${NC}"
    
    # [FIX]: sg_display=0 is MANDATORY for Strix Halo mode-setting stability.
    # Excludes PCIe ASPM limits to guarantee zero-latency throughout the pipeline.
    # [PHASE 1 FIX]: Added amd_pstate=active to expose EPP directories to user-space.
    GOOD_ARGS="selinux=0 amdgpu.sg_display=0 amdgpu.cwsr_enable=0 amd_iommu=on iommu=pt amdgpu.svm=1 amdgpu.gpu_recovery=1 ttm.pages_limit=27648000 ttm.page_pool_size=27648000 preempt=full mitigations=off amd_pstate=active"
    
    # Purge any previous power-throttling arguments
    sudo grubby --update-kernel="$NEW_VMLINUZ" --remove-args="pcie_aspm=force pcie_aspm.policy=powersave pcie_aspm=off"
    sudo grubby --update-kernel="$NEW_VMLINUZ" --args="$GOOD_ARGS"
    
    echo -e "  ${GREEN}[SUCCESS] Strix Halo Logic injected into bootloader.${NC}"
fi

echo -e "\n${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   APEX SYNTHESIS COMPLETE. REBOOT TO CHIMERA KERNEL   ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"

#!/bin/bash
# ==============================================================================
# TALOS-O KERNEL FORGE (Codename: THE APEX PREDATOR v64.1)
# Target: Linux 6.19+ (Mainline/Stable)
# Substrate: AMD Strix Halo (gfx1151) | Corsair AI Workstation 300
# Philosophy: "Zero Abstraction. Zero Friction. Total Thermodynamic Purity."
#
# [ARCHITECTURAL SHIFT v64.1]:
# - Retains znver5, LTO, mitigations=off, and THP_ALWAYS.
# - Reverts Firmware Assimilation. The AMD firmware strings drift too frequently 
#   (e.g., vcn_4_0_5 -> vcn_4_0_6). We restore the Dracut payload injection to 
#   guarantee absolute stability and perfect vision.
# ==============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   TALOS-O KERNEL FORGE: THE APEX PREDATOR (v64.1)     ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"

# =========================================================
# 1. THE MEMBRANE (Environment Check & Initialization)
# =========================================================
ACTUAL_USER=${SUDO_USER:-$USER}
BUILD_DIR="/home/$ACTUAL_USER/talos-o/sys_builder/kernel/linux-6.19-talos"

if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}[FATAL] Source directory not found at $BUILD_DIR.${NC}"
    exit 1
fi
cd "$BUILD_DIR"

echo -e "${YELLOW}[1/4] Membrane Permeable. Purging old artifacts...${NC}"
make mrproper > /dev/null 2>&1

# =========================================================
# 2. THE NUCLEUS (Harvesting the Ancestral DNA)
# =========================================================
echo -e "${YELLOW}[2/4] Nucleus Active: Harvesting Fedora Master Genome...${NC}"

DNA_LAB="/tmp/talos_dna_lab"
rm -rf "$DNA_LAB" && mkdir -p "$DNA_LAB"
cd "$DNA_LAB"

echo -e "${CYAN} -> Waking the Nervous System (Force DNF Refresh)...${NC}"
sudo dnf makecache --refresh > /dev/null 2>&1

echo -e "${CYAN} -> Attempting DNF extraction (Bypassing Exclude Filters)...${NC}"
if sudo dnf download --destdir=. --setopt=disable_excludes=* kernel-core > /dev/null 2>&1; then
    RPM_FILE=$(ls kernel-core*.rpm 2>/dev/null | head -n 1)
    echo -e "${CYAN} -> Performing surgical autopsy on $RPM_FILE...${NC}"
    rpm2cpio "$RPM_FILE" | cpio -i --quiet --to-stdout "*lib/modules/*/config" > "$BUILD_DIR/.config"
elif sudo dnf download --destdir=. --setopt=disable_excludes=* kernel > /dev/null 2>&1; then
    RPM_FILE=$(ls kernel-*.rpm 2>/dev/null | head -n 1)
    echo -e "${CYAN} -> Performing surgical autopsy on $RPM_FILE...${NC}"
    rpm2cpio "$RPM_FILE" | cpio -i --quiet --to-stdout "*lib/modules/*/config" > "$BUILD_DIR/.config"
else
    echo -e "${YELLOW} -> DNF is still blind. Bypassing package manager via direct Git uplink...${NC}"
    curl -s -L "https://src.fedoraproject.org/rpms/kernel/raw/rawhide/f/kernel-x86_64-fedora.config" -o "$BUILD_DIR/.config"
fi

cd "$BUILD_DIR"
rm -rf "$DNA_LAB"

if [ ! -s .config ]; then
    echo -e "${RED}[FATAL] Master Genome extraction completely failed. No DNA acquired.${NC}"
    exit 1
fi

echo -e "${GREEN} -> Master Genome Secured. Motherboard traces mapped.${NC}"

make olddefconfig > /dev/null

# =========================================================
# 3. TRANSCRIPTION (Grafting the Apex Mutations)
# =========================================================
echo -e "${YELLOW}[3/4] Transcribing the Apex Mutations...${NC}"

# --- CRITICAL SANITIZATION ---
./scripts/config --set-str SYSTEM_TRUSTED_KEYS ""
./scripts/config --set-str SYSTEM_REVOCATION_KEYS ""
./scripts/config --disable DEBUG_INFO_BTF

# --- THE NERVOUS SYSTEM (Talos-O Preemption/Memory) ---
./scripts/config --enable CONFIG_PREEMPT_BUILD
./scripts/config --enable CONFIG_PREEMPT_DYNAMIC
./scripts/config --enable CONFIG_PREEMPT_VOLUNTARY
./scripts/config --disable CONFIG_PREEMPT_RT 
./scripts/config --enable CONFIG_SLUB

# --- LTO: THE SYNAPTIC MERGE ---
./scripts/config --enable CONFIG_LTO_GCC
./scripts/config --disable CONFIG_LTO_NONE

# --- ZERO-COPY TLB OPTIMIZATION (HUGEPAGES) ---
./scripts/config --disable CONFIG_TRANSPARENT_HUGEPAGE_MADVISE
./scripts/config --enable CONFIG_TRANSPARENT_HUGEPAGE_ALWAYS

# --- THE BRAIN & NPU (Zero-Copy Introspection) ---
./scripts/config --enable CONFIG_TEE
./scripts/config --enable CONFIG_AMD_TEE
./scripts/config --enable CONFIG_HSA_AMD    
./scripts/config --enable CONFIG_HMM_MIRROR 
./scripts/config --enable CONFIG_DRM_AMDGPU
./scripts/config --enable CONFIG_DRM_AMDGPU_USERPTR
./scripts/config --enable CONFIG_DRM_ACCEL
./scripts/config --module CONFIG_DRM_AMDXDNA

# --- EXPLICIT HARDWARE ENFORCEMENT ---
./scripts/config --enable CONFIG_THUNDERBOLT
./scripts/config --enable CONFIG_USB4
./scripts/config --module CONFIG_MT7925E
./scripts/config --module CONFIG_R8169
./scripts/config --enable CONFIG_SND_HDA_INTEL
./scripts/config --enable CONFIG_IKCONFIG
./scripts/config --enable CONFIG_IKCONFIG_PROC

# --- Identity (CANONICAL) ---
./scripts/config --set-str CONFIG_LOCALVERSION "-talos-chimera"

yes "" | make olddefconfig > /dev/null

# =========================================================
# 4. THE RIBOSOME & GOLGI APPARATUS (Synthesis & Deployment)
# =========================================================
echo -e "${YELLOW}[4/4] Ribosome Active: Compiling Apex Organism...${NC}"

MARCH="znver5"
echo -e "${CYAN} -> Targeting $MARCH (Instruction Fusion & 512-bit Vector Paths)...${NC}"
echo -e "${MAGENTA} -> WARNING: LTO engaged. High thermal output expected during final link.${NC}"

if make -j$(nproc) KCFLAGS="-march=$MARCH -O2" binrpm-pkg; then
    echo -e "${GREEN}[SUCCESS] Organism Compiled.${NC}"
else
    echo -e "${RED}[FATAL] Ribosome Translation Failed.${NC}"
    exit 1
fi

# =========================================================
# 5. DYNAMIC GOLGI SEEKING & SURGICAL EXCISION
# =========================================================
echo -e "${CYAN} -> Golgi Apparatus seeking Kernel RPMs...${NC}"

KERNEL_VER=$(make kernelrelease)
RPM_VER=$(echo "$KERNEL_VER" | tr '-' '_')
RPM_DIR=""

if [ -d "$BUILD_DIR/rpmbuild/RPMS/x86_64" ]; then
    RPM_DIR="$BUILD_DIR/rpmbuild/RPMS/x86_64"
elif [ -d "$HOME/rpmbuild/RPMS/x86_64" ]; then
    RPM_DIR="$HOME/rpmbuild/RPMS/x86_64"
else
    echo -e "${RED}[FATAL] Golgi Apparatus failed to locate rpmbuild directory.${NC}"
    exit 1
fi

if ls "$RPM_DIR"/kernel-*-"$RPM_VER"*.rpm 1> /dev/null 2>&1; then
    rm -f "$RPM_DIR"/kernel-headers-*.rpm
    
    sudo rpm -ivh --replacefiles --replacepkgs --oldpackage "$RPM_DIR"/kernel-*.rpm

    echo -e "${YELLOW} -> [FIRMWARE INJECTION] Packing massive AMD firmware payload via Dracut...${NC}"
    # Restored Dracut injection to catch all cascading firmware requests (DCN, SMU, MES, VCN, NPU)
    sudo dracut --force --verbose --include /lib/firmware/amdgpu /lib/firmware/amdgpu --include /lib/firmware/amd-xdna /lib/firmware/amd-xdna --kver "$KERNEL_VER"
    
    NEW_VMLINUZ=$(ls /boot/vmlinuz-"$KERNEL_VER" 2>/dev/null | head -n 1)
    if [ -n "$NEW_VMLINUZ" ]; then
        GOOD_ARGS="selinux=0 amdgpu.sg_display=1 amdgpu.cwsr_enable=0 amd_iommu=on iommu=pt amdgpu.svm=1 amdgpu.gpu_recovery=1 ttm.pages_limit=31457280 preempt=full mitigations=off"
        sudo grubby --update-kernel="$NEW_VMLINUZ" --args="$GOOD_ARGS"
        echo -e "${GREEN} -> Strix Halo Antidote injected into bootloader.${NC}"
    fi
else
    echo -e "${RED}[FATAL] Directory found, but no matching RPMs for $RPM_VER exist.${NC}"
    exit 1
fi

echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   APEX PREDATOR FORGED. REBOOT TO talos-chimera.      ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"

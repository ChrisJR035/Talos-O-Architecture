#!/bin/bash
# ==============================================================================
# TALOS-O: LITMUS TEST v14.0 (Codename: OMNI-VALIDATION / PHENOTYPE PATCH)
# Substrate: AMD Strix Halo (Ryzen AI Max+ 395)
# Philosophy: "The Map matches the Territory" -> Total Verification
# ==============================================================================

# --- VISUALS ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   TALOS-O DIAGNOSTIC: OMNI-VALIDATION (v14.0)         ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"

# =========================================================
# 1. IDENTITY & PATH DISCOVERY
# =========================================================
echo -e "\n${YELLOW}[1] IDENTITY CHECK (Kernel & Filesystem)${NC}"

CURRENT_KERNEL=$(uname -r)
EXPECTED_TAG="chimera"

CONFIG_CANDIDATES=(
    "/boot/config-$CURRENT_KERNEL"
    "/lib/modules/$CURRENT_KERNEL/config"
    "/proc/config.gz"
)

ACTIVE_CONFIG=""
for c in "${CONFIG_CANDIDATES[@]}"; do
    if [ -f "$c" ]; then
        ACTIVE_CONFIG="$c"
        break
    fi
done

if [[ "$CURRENT_KERNEL" == *"$EXPECTED_TAG"* ]]; then
    echo -e "  [${GREEN}PASS${NC}] Kernel Identity Verified: $CURRENT_KERNEL"
else
    echo -e "  [${RED}FAIL${NC}] Kernel Identity Mismatch: $CURRENT_KERNEL (Expected '$EXPECTED_TAG')"
fi

# =========================================================
# 2. BOOT ARGUMENT VERIFICATION (Active Substring Eval)
# =========================================================
echo -e "\n${YELLOW}[2] BOOT ARGUMENT VERIFICATION${NC}"
CMDLINE=$(cat /proc/cmdline)

check_arg() {
    local arg="$1"
    local name="$2"
    # [PHASE 3 FIX: Wildcard Substring Matching replaces strict '==']
    if [[ "$CMDLINE" == *"$arg"* ]]; then
        echo -e "  [${GREEN}PASS${NC}] $name ($arg)"
    else
        echo -e "  [${RED}FAIL${NC}] $name missing ($arg)"
    fi
}

check_arg "amdgpu.sg_display=1" "UMA Display Override"
check_arg "amdgpu.svm=1" "Shared Virtual Memory (SVM)"
check_arg "amd_iommu=on" "IOMMU Enabled"
check_arg "iommu=on" "IOMMU Strict"
check_arg "amd_pstate=active" "Zen 5 P-State Control"
check_arg "ttm.pages_limit=" "VRAM TTM Allocation"
check_arg "mitigations=off" "Hardware Friction Minimized"

# =========================================================
# 3. GENOTYPE VERIFICATION (Kernel Configs)
# =========================================================
echo -e "\n${YELLOW}[3] GENOTYPE VERIFICATION (Kernel Configuration)${NC}"

check_conf() {
    local key="$1"
    local val="$2"
    local name="$3"
    
    local found=$(zgrep -E "^$key=$val" "$ACTIVE_CONFIG" 2>/dev/null || grep -E "^$key=$val" "$ACTIVE_CONFIG" 2>/dev/null || true)
    if [ -n "$found" ]; then
        echo -e "  [${GREEN}PASS${NC}] $key=$val ($name)"
    else
        echo -e "  [${RED}FAIL${NC}] $key=$val NOT SET! ($name)"
    fi
}

if [ -n "$ACTIVE_CONFIG" ]; then
    check_conf "CONFIG_PREEMPT_DYNAMIC" "y" "Dynamic Preemption"
    check_conf "CONFIG_HSA_AMD" "y" "ROCm Link"
    check_conf "CONFIG_HMM_MIRROR" "y" "Unified Memory"
    check_conf "CONFIG_DRM_AMDGPU_USERPTR" "y" "Zero-Copy Access"
    check_conf "CONFIG_DRM_ACCEL_AMDXDNA" "m" "NPU Brainstem"
else
    echo -e "${RED}Skipping Genotype Check: No config file available.${NC}"
fi

# =========================================================
# 4. PHENOTYPE VERIFICATION (Active Hardware Probing)
# =========================================================
echo -e "\n${YELLOW}[4] PHENOTYPE VERIFICATION (Runtime Hardware)${NC}"

# [PHASE 3 FIX: Active probing of sysfs/devfs for actual hardware instantiation]
if [ -c "/dev/kfd" ]; then
    echo -e "  [${GREEN}PASS${NC}] /dev/kfd present (ROCm Link Active)"
else
    echo -e "  [${RED}FAIL${NC}] /dev/kfd missing (ROCm DEAD)"
fi

if [ -c "/dev/accel/accel0" ]; then
    echo -e "  [${GREEN}PASS${NC}] /dev/accel/accel0 present (XDNA 2 NPU Brainstem Active)"
else
    echo -e "  [${RED}FAIL${NC}] /dev/accel/accel0 missing (NPU driver failed to load)"
fi

if [ -d "/sys/class/kfd/kfd/topology/nodes" ]; then
    NUM_NODES=$(ls -1q /sys/class/kfd/kfd/topology/nodes | wc -l)
    echo -e "  [${GREEN}PASS${NC}] KFD Topology Verified ($NUM_NODES Nodes Detected)"
else
    echo -e "  [${RED}FAIL${NC}] KFD Topology Missing! Zero-Copy compromised."
fi

FW_FILE="/lib/firmware/amdgpu/gc_11_5_1_pfp.bin"
if [ -f "$FW_FILE" ]; then
    FW_DATE=$(date -r "$FW_FILE" "+%Y-%m-%d")
    echo -e "  [${GREEN}PASS${NC}] Firmware PRESENT ($FW_DATE)"
else
    echo -e "  [${RED}FAIL${NC}] Firmware MISSING ($FW_FILE)"
fi

echo -e "\n${CYAN}Validation Complete.${NC}"

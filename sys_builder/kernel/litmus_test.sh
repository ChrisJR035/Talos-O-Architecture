#!/bin/bash
# ==============================================================================
# TALOS-O: LITMUS TEST v13.1 (Codename: THE PHENOTYPE PATCH)
# Substrate: AMD Strix Halo (Ryzen AI Max+ 395)
# Philosophy: "The Map matches the Territory" -> Total Verification
#
# [ARCHITECTURAL SHIFT v13.1 - The Skeptical Mirror Patch]:
# - D1: Fixed Identity Tag to target 'talos-chimera'.
# - D2: Upgraded check_erased() to safely handle gzipped kernel configs.
# - D3: Added strict FAIL branches for missing Strix Halo firmware.
# - D4: Enforced verification of ttm.pages_limit and mitigations=off.
# - D5: Introduced PHENOTYPE verification (Live checking of /dev/kfd, /dev/dri, amdxdna).
# ==============================================================================

# --- VISUALS ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   TALOS-O DIAGNOSTIC: THE PHENOTYPE PATCH (v13.1)     ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"

# =========================================================
# 1. IDENTITY & PATH DISCOVERY
# =========================================================
echo -e "\n${YELLOW}[1] IDENTITY CHECK (Kernel & Filesystem)${NC}"

CURRENT_KERNEL=$(uname -r)
EXPECTED_TAG="talos-chimera"

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
    echo -e " -> Kernel Version: ${GREEN}$CURRENT_KERNEL (VERIFIED)${NC}"
else
    echo -e " -> Kernel Version: ${RED}$CURRENT_KERNEL (EXPECTED $EXPECTED_TAG)${NC}"
fi

if [ -n "$ACTIVE_CONFIG" ]; then
    echo -e " -> Kernel Config:  ${GREEN}FOUND ($ACTIVE_CONFIG)${NC}"
else
    echo -e " -> Kernel Config:  ${RED}NOT FOUND${NC}"
fi

# =========================================================
# 2. PROPRIOCEPTION (Runtime Reality)
# =========================================================
echo -e "\n${YELLOW}[2] PROPRIOCEPTION (Runtime Reality)${NC}"
CMDLINE=$(cat /proc/cmdline)

check_arg() {
    if [[ "$CMDLINE" == *"$1"* ]]; then
        echo -e "  [${GREEN}PASS${NC}] Boot Arg: $1"
    else
        echo -e "  [${RED}FAIL${NC}] Boot Arg: $1 is MISSING"
    fi
}

check_arg "iommu=pt"
check_arg "amd_iommu=on"
check_arg "amdgpu.svm=1"
check_arg "preempt=full"
check_arg "amdgpu.cwsr_enable=0"
check_arg "ttm.pages_limit=29360128"
check_arg "mitigations=off"

# =========================================================
# 3. GENOTYPE VERIFICATION (Kernel Configuration)
# =========================================================
echo -e "\n${YELLOW}[3] GENOTYPE VERIFICATION (Kernel Configuration)${NC}"

check_conf() {
    local key=$1
    local expected=$2
    local name=$3
    
    if [[ "$ACTIVE_CONFIG" == *.gz ]]; then
        val=$(zcat "$ACTIVE_CONFIG" | grep "^$key=" | cut -d'=' -f2)
    else
        val=$(grep "^$key=" "$ACTIVE_CONFIG" | cut -d'=' -f2)
    fi
    
    if [ -z "$val" ]; then
        val=$(grep "$key is not set" "$ACTIVE_CONFIG" | head -n 1 || echo "NOT_SET")
    fi

    if [ "$val" == "$expected" ]; then
        echo -e "  [${GREEN}PASS${NC}] $key = $val ($name)"
    else
        echo -e "  [${RED}FAIL${NC}] $key ($val != $expected) ($name)"
    fi
}

check_erased() {
    local key=$1
    local name=$2
    local found
    
    if [[ "$ACTIVE_CONFIG" == *.gz ]]; then
        found=$(zcat "$ACTIVE_CONFIG" | grep "^$key=" || true)
    else
        found=$(grep "^$key=" "$ACTIVE_CONFIG" || true)
    fi
    
    if [ -z "$found" ]; then
        echo -e "  [${GREEN}PASS${NC}] $key ERADICATED ($name)"
    else
        echo -e "  [${RED}FAIL${NC}] $key still exists! ($name)"
    fi
}

if [ -n "$ACTIVE_CONFIG" ]; then
    check_conf "CONFIG_PREEMPT_DYNAMIC" "y" "Dynamic Preemption"
    check_conf "CONFIG_HSA_AMD" "y" "ROCm Link"
    check_conf "CONFIG_HMM_MIRROR" "y" "Unified Memory"
    check_conf "CONFIG_DRM_AMDGPU_USERPTR" "y" "Zero-Copy Access"
    check_conf "CONFIG_DRM_ACCEL_AMDXDNA" "m" "NPU Brainstem"
    
    # The Linux 6.19 Sheaves Evolution Check
    check_erased "CONFIG_SLUB_CPU_PARTIAL" "Native Sheaves Verification"
    check_erased "CONFIG_SLUB_SHEAVES" "Deprecated Flag Cleaned"
else
    echo -e "${RED}Skipping Genotype Check: No config file available.${NC}"
fi

# =========================================================
# 4. SENSORY & METABOLISM
# =========================================================
echo -e "\n${YELLOW}[4] SENSORY & METABOLISM (Hardware & Math)${NC}"

FW_FILE="/lib/firmware/amdgpu/gc_11_5_1_pfp.bin"
if [ -f "$FW_FILE" ]; then
    FW_DATE=$(date -r "$FW_FILE" "+%Y-%m-%d")
    echo -e " -> Firmware: ${GREEN}PRESENT ($FW_DATE)${NC}"
else
    echo -e " -> Firmware: ${RED}MISSING ($FW_FILE)${NC}"
fi

if command -v python3 &> /dev/null; then
    python3 -c "
import time
import sys
try:
    start = time.time()
    for _ in range(100000): pass
    elapsed = time.time() - start
    print(f' -> Metabolic Pulse: \033[0;32m{elapsed:.4f}s (CORTEX IS ALIVE)\033[0m')
except:
    pass
"
fi

# =========================================================
# 5. PHENOTYPE VERIFICATION (Runtime Hardware)
# =========================================================
echo -e "\n${YELLOW}[5] PHENOTYPE VERIFICATION (Runtime Hardware)${NC}"

# Is the GPU accessible?
if [ -c "/dev/kfd" ]; then
    echo -e "  [${GREEN}PASS${NC}] /dev/kfd present (ROCm Link Active)"
else
    echo -e "  [${RED}FAIL${NC}] /dev/kfd missing (ROCm DEAD)"
fi

if [ -c "/dev/dri/renderD128" ]; then
    echo -e "  [${GREEN}PASS${NC}] /dev/dri/renderD128 present (GPU Online)"
else
    echo -e "  [${RED}FAIL${NC}] /dev/dri/renderD128 missing (GPU DARK)"
fi

# Is the NPU module actually loaded?
if lsmod | grep -q "amdxdna"; then
    echo -e "  [${GREEN}PASS${NC}] amdxdna module loaded (NPU Brainstem Active)"
else
    echo -e "  [${RED}FAIL${NC}] amdxdna not loaded (NPU SILENT)"
fi

echo -e "\n${CYAN}=== DIAGNOSTIC PHENOTYPE PATCH COMPLETE ===${NC}"

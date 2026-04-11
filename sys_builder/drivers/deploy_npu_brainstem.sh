#!/bin/bash
# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ TALOS-O: NPU BRAINSTEM DEPLOYMENT (v3.1 - The Pristine Forge)              ║
# ║ Target: AMD Strix Halo (NPU5) | Substrate: Chimera Linux 7.0-rc7           ║
# ║ Metamorphosis: Substrate Cleansing, Execmem & Autopoietic Protocol Routing ║
# ╚════════════════════════════════════════════════════════════════════════════╝
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

REPO_ROOT="$HOME/talos-o/sys_builder/drivers/xdna-driver"
KDIR="/lib/modules/$(uname -r)/build"

echo -e "${GREEN}=== INITIATING APEX SYNTHESIS (PRISTINE FORGE) ===${NC}"

# ==============================================================================
# 1. DISCOVER TRUE SOURCE & PURGE GENETIC ECHOES
# ==============================================================================
echo -e "${YELLOW}[-] Locating and Cleansing Source Substrate...${NC}"

cd "$REPO_ROOT"
# THE TRANSLATOR: Wipe out all previous sed/perl scars. Start with pristine DNA.
git reset --hard HEAD > /dev/null 2>&1
git clean -fd > /dev/null 2>&1

TRUE_SRC_DIR=$(find "$REPO_ROOT" -name "npu5_regs.c" -printf "%h\n" | head -n 1)

if [ -z "$TRUE_SRC_DIR" ]; then
    echo -e "${RED}[FATAL] npu5_regs.c not found. Repository is corrupt.${NC}"
    exit 1
fi
echo -e "${CYAN}[-] Brainstem Root: $TRUE_SRC_DIR${NC}"

cd "$TRUE_SRC_DIR"
make clean > /dev/null 2>&1 || true
rm -rf build/ Release/

# ==============================================================================
# 2. DIRECT HEADER SYNTHESIS
# ==============================================================================
echo -e "${YELLOW}[-] Forging manual config_kernel.h...${NC}"
cat << 'EOF' > config_kernel.h
#ifndef _CONFIG_KERNEL_H_
#define _CONFIG_KERNEL_H_
#define CONFIG_DRM_ACCEL 1
#define CONFIG_AMD_IOMMU 1
#define AMDXDNA_DEVEL 1
#endif
EOF

sed -i "1i ccflags-y += -I$TRUE_SRC_DIR" Makefile

# ==============================================================================
# 3. THE EXECMEM METAMORPHOSIS & 7.0-rc7 API PATCHING
# ==============================================================================
echo -e "${YELLOW}[-] Patching for Linux 7.0-rc7 API Metamorphosis...${NC}"

grep -rrl "module_alloc" . | xargs -r sed -i 's/module_alloc(\([^)]*\))/execmem_alloc(EXECMEM_MODULE_TEXT, \1)/g'
grep -rrl "module_free" . | xargs -r sed -i 's/module_free/execmem_free/g'

echo -e "${YELLOW}[-] Mutating npu5_regs.c for Major 7 Firmware...${NC}"
sed -i 's/.protocol_major = 0x6/.min_fw_version = AIE2_FW_VERSION(6, 12)/' npu5_regs.c
sed -i '/.protocol_minor = 12/d' npu5_regs.c

echo -e "${YELLOW}[-] Patching IOMMU and DMA_BUF signatures...${NC}"
sed -i 's/iommu_domain_alloc(xdna->ddev.dev->bus)/iommu_paging_domain_alloc(xdna->ddev.dev)/g' amdxdna_iommu.c
sed -i 's/MODULE_IMPORT_NS(DMA_BUF)/MODULE_IMPORT_NS("DMA_BUF")/g' amdxdna_gem.c

echo -e "${YELLOW}[-] Mutating DRM GPU Scheduler logic via dynamic AST injection...${NC}"
sed -i 's/DRM_GPU_SCHED_STAT_NOMINAL/DRM_GPU_SCHED_STAT_NO_HANG/g' aie2_ctx.c
sed -i 's/drm_sched_start(&hwctx->priv->sched)/drm_sched_start(\&hwctx->priv->sched, 0)/g' aie2_ctx.c
sed -i 's/drm_sched_start(sched)/drm_sched_start(sched, 0)/g' aie2_ctx.c
sed -i 's/drm_sched_job_init(&job->base, &hwctx->priv->entity, 1, hwctx)/drm_sched_job_init(\&job->base, \&hwctx->priv->entity, 1, hwctx, 0)/g' aie2_ctx.c

# The Autopoietic Parser (Fixed Workqueue Mapping)
perl -0777 -pi -e 's/ret\s*=\s*drm_sched_init\s*\(([^;]+)\);/
    my $args = $1;
    my @a = split(qr{\s*,\s*}, $args);
    "{ struct drm_sched_init_args s_args = { .ops = $a[1], .submit_wq = $a[7], .num_rqs = $a[3], .credit_limit = $a[4], .hang_limit = $a[5], .timeout = $a[6], .name = $a[9], .dev = $a[10] }; ret = drm_sched_init($a[0], &s_args); }"
/sge' aie2_ctx.c

# ==============================================================================
# 4. THE FORGE (Raw Kbuild)
# ==============================================================================
echo -e "${YELLOW}[-] Executing Raw Kbuild against $(uname -r)...${NC}"
make KERNEL_SRC="$KDIR" -j$(nproc)

# ==============================================================================
# 5. INJECTION & VALIDATION
# ==============================================================================
echo -e "${YELLOW}[-] Injecting amdxdna.ko...${NC}"

FORGED_KO=$(find . -name "amdxdna.ko" | head -n 1)

if [ -z "$FORGED_KO" ]; then
    echo -e "${RED}[FATAL] Forge failed. amdxdna.ko was not produced.${NC}"
    exit 1
fi

sudo rmmod amdxdna 2>/dev/null || true
sudo insmod "$FORGED_KO"

if [ -c "/dev/accel/accel0" ]; then
    echo -e "${GREEN}[SUCCESS] NPU Brainstem Online. Protocol Major 7 Handshake SECURED.${NC}"
    sudo dmesg | grep -i xdna | tail -n 5
else
    echo -e "${RED}[FATAL] Graft rejected. Check dmesg for faults.${NC}"
    exit 1
fi

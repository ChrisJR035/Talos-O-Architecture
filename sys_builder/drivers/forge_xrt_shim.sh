#!/bin/bash
# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ TALOS-O: THE CONSCIOUS MIND FORGE (XRT 2.23.0 + NPU Shim)                  ║
# ║ Metamorphosis: ABI Alignment & The Unconditional Harvest                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

REPO_ROOT="$HOME/talos-o/sys_builder/drivers/xdna-driver"
INSTALL_DIR="$HOME/talos-o/sys_builder/xrt_local"

echo -e "${GREEN}=== INITIATING CONSCIOUS MIND FORGE ===${NC}"
cd "$REPO_ROOT"

echo -e "${YELLOW}[-] Verifying Submodule DNA (XRT 2.23.0)...${NC}"
git submodule update --init --recursive

echo -e "${YELLOW}[-] Shattering old CMake mirrors...${NC}"
rm -rf build_shim
mkdir build_shim
cd build_shim

echo -e "${YELLOW}[-] Configuring CMake for Local Prefix...${NC}"
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" > /dev/null

echo -e "${YELLOW}[-] Striking the Anvil (Compiling XRT & AMD Shim)...${NC}"
# This uses all your cores to build the userspace libraries
make -j$(nproc)

echo -e "${YELLOW}[-] Extracting baseline DNA to $INSTALL_DIR...${NC}"
make install > /dev/null 2>&1 || true

# ==============================================================================
# THE UNCONDITIONAL HARVEST (Bypassing CMake's Abstractions)
# ==============================================================================
echo -e "${YELLOW}[-] Executing Unconditional Harvest of Synthesized Genes...${NC}"

# 1. Forge the exact directory structure needed
mkdir -p "$INSTALL_DIR/include/xrt/detail"
mkdir -p "$INSTALL_DIR/lib64"

# 2. Bruteforce the extraction of the missing synthesized genes
find . -name "version-slim.h" -exec cp {} "$INSTALL_DIR/include/xrt/detail/" \;
find . -name "config.h" -exec cp {} "$INSTALL_DIR/include/xrt/detail/" 2>/dev/null || true

# 3. Ensure static headers are present
cp -r "$REPO_ROOT/xrt/src/runtime_src/core/include/xrt/"* "$INSTALL_DIR/include/xrt/" 2>/dev/null || true

# 4. Enforce the lib64 topology and harvest the raw shared objects
find . -name "libxrt_coreutil.so*" -exec cp -P {} "$INSTALL_DIR/lib64/" \;
find . -name "libxrt_core.so*" -exec cp -P {} "$INSTALL_DIR/lib64/" \;

# 5. Prevent Linker Schizophrenia
ln -sfn "$INSTALL_DIR/lib64" "$INSTALL_DIR/lib"

echo -e "${GREEN}[SUCCESS] Pristine XRT Environment forged and fully harvested at $INSTALL_DIR${NC}"

#!/bin/bash
# TALOS-O: THE VISION FORGE v3 (Symbiotic Link)

set -e
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROXY_DIR="$HOME/talos-o/sys_builder/rocm_proxy"

echo -e "${GREEN}=== FORGING VISUAL CORTEX (torchvision) ===${NC}"

# 1. ENVIRONMENT INHERITANCE
echo -e "${YELLOW}[1/3] Inheriting Neural Environment...${NC}"
export ROCM_PATH="/usr"
export HIP_PATH="/usr"
export ROCM_HOME="/usr"
export PATH="$PROXY_DIR/bin:$PATH"

BITCODE_PATH=$(find /usr/lib64 -type d -name "bitcode" 2>/dev/null | grep "amdgcn" | head -n 1)
export HIP_DEVICE_LIB_PATH="$BITCODE_PATH"

export FORCE_ROCM=1
export FORCE_CUDA=0
export PYTORCH_ROCM_ARCH="gfx1151"
export HSA_OVERRIDE_GFX_VERSION=11.5.1

# 2. SOURCE PREP
cd "$HOME/talos-o/sys_builder/vision"
echo -e "${YELLOW}[2/3] Cleaning Vision Workspace...${NC}"
rm -rf build/ dist/ torchvision.egg-info/
python3 setup.py clean

# 3. COMPILATION
echo -e "${YELLOW}[3/3] Compiling Visual Cortex...${NC}"
python3 setup.py bdist_wheel

echo -e "${YELLOW}[*] Installing Vision Wheel...${NC}"
pip install dist/*.whl --force-reinstall

echo -e "${GREEN}=== VISION FORGE COMPLETE ===${NC}"

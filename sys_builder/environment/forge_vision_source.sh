#!/bin/bash
# =============================================================================
# TALOS-O: THE VISION FORGE v5.3 (Pure C++ / LibTorch Titan Alignment)
# ARCHITECTURE: AMD Strix Halo (gfx1151)
# FIXES: Injected FORCE_CUDA=1 and hwmon thermal targeting.
# =============================================================================
set -e
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== FORGING VISUAL CORTEX (torchvision) v5.3: PURE C++ ===${NC}"

# 0. EPISTEMIC SOVEREIGNTY (Version Pinning)
SRC_DIR="$HOME/talos-o/sys_builder/vision"
VERSION_TAG="v0.18.0"

if [ ! -d "$SRC_DIR" ]; then
    echo -e "${YELLOW}[0/3] Anchoring to Verified Release: $VERSION_TAG...${NC}"
    git clone --branch $VERSION_TAG https://github.com/pytorch/vision.git "$SRC_DIR" --depth 1
else
    echo -e "${YELLOW}[0/3] Sovereign Source detected. Cleaning...${NC}"
    cd "$SRC_DIR"
    rm -rf build/
    git clean -fdx
fi

# 1. ENVIRONMENT INHERITANCE
echo -e "${YELLOW}[1/3] Inheriting Neural Environment...${NC}"
export ROCM_PATH="/usr"
export HIP_PATH="/usr"

if [ -d "/usr/lib64/amdgcn/bitcode" ]; then
    export HIP_DEVICE_LIB_PATH="/usr/lib64/amdgcn/bitcode"
elif [ -d "/usr/amdgcn/bitcode" ]; then
    export HIP_DEVICE_LIB_PATH="/usr/amdgcn/bitcode"
else
    export HIP_DEVICE_LIB_PATH=$(find /usr -type d -name "bitcode" | grep "amdgcn" | head -n 1)
fi
echo -e "${GREEN}[DONE] Bitcode Targeted: $HIP_DEVICE_LIB_PATH${NC}"

# 2. DYNAMIC THERMODYNAMIC THROTTLING
# [FIX: HW_MON TARGETING] Hunts for the actual APU die temperature (k10temp/zenpower)
T_MAX=75
N_THREADS=32
K_MULT=3
K_DIV=2

T_DIE_MILLI=$(grep -l "Tctl" /sys/class/hwmon/hwmon*/temp*_label 2>/dev/null | sed 's/_label/_input/' | xargs cat 2>/dev/null | head -n 1)

if [ -n "$T_DIE_MILLI" ]; then
    T_DIE=$((T_DIE_MILLI / 1000))
else
    # Fallback to standard thermal zone if hwmon Tctl is hidden
    if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
        T_DIE_MILLI=$(cat /sys/class/thermal/thermal_zone0/temp)
        T_DIE=$((T_DIE_MILLI / 1000))
    else
        T_DIE=45
    fi
fi

T_DIFF=$((T_MAX - T_DIE))
J_CALC=$(( (T_DIFF * K_MULT) / K_DIV ))
if [ $J_CALC -lt 1 ]; then J_CALC=1; fi
if [ $J_CALC -gt $N_THREADS ]; then J_CALC=$N_THREADS; fi

export MAX_JOBS=$J_CALC
echo -e "${YELLOW}[THERMODYNAMICS] T_die: ${T_DIE}C | Allocated Threads: $MAX_JOBS${NC}"

# 3. CONFIGURE CMAKE ALIGNMENT
export CXXFLAGS="-I/usr/include -I/usr/include/rocm -fPIC -D__HIP_PLATFORM_AMD__=1"
export HIP_CLANG_FLAGS=$CXXFLAGS
export PYTORCH_ROCM_ARCH="gfx1151"

export CMAKE_PREFIX_PATH="$ROCM_PATH"
export CMAKE_MODULE_PATH="$ROCM_PATH/lib/cmake/hip"

# PyTorch TorchConfig Bridge
TORCH_CMAKE=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'share/cmake/Torch'))")
export Torch_DIR="$TORCH_CMAKE"

# [FIX: THE VISION LOBOTOMY CURE]
# Forces setup.py to bypass auto-detection and explicitly compile the HIP C++ extensions.
export FORCE_CUDA=1

# 4. IGNITE THE FORGE
echo -e "${YELLOW}[3/3] Igniting the Vision Forge...${NC}"
cd "$SRC_DIR"
python3 setup.py bdist_wheel

WHEEL_FILE=$(ls dist/*.whl | head -n 1)
echo -e "${GREEN}=== VISUAL CORTEX SYNTHESIZED SUCCESSFULLY ===${NC}"
echo -e "Wheel Object: ${CYAN}$WHEEL_FILE${NC}"
echo -e "To implant into Talos-O, run: ${YELLOW}pip install $WHEEL_FILE${NC}"

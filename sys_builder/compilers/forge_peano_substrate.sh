#!/bin/bash
# ==============================================================================
# TALOS-O: THE PEANO FORGE (Sovereign LLVM AIE Backend)
# Target: AMD XDNA 2 (aie2p)
# ==============================================================================
set -e
set -o pipefail

export BASE_DIR="$HOME/talos-o/sys_builder/compilers"
export PEANO_DIR="$BASE_DIR/llvm-aie"
export PEANO_BUILD_DIR="$PEANO_DIR/build"

# Thermodynamic Guardband
CORES=$(( $(nproc) / 2 ))
if [ "$CORES" -gt 16 ]; then CORES=16; fi

echo -e "\033[96m[SYSTEM] Initiating Sovereign Peano Substrate Forge...\033[0m"

if [ ! -d "$PEANO_DIR" ]; then
    echo -e "\033[93m[SYSTEM] Cloning Xilinx llvm-aie (Peano)...\033[0m"
    git clone https://github.com/Xilinx/llvm-aie.git "$PEANO_DIR"
fi

mkdir -p "$PEANO_BUILD_DIR"
cd "$PEANO_BUILD_DIR"

echo -e "\033[93m[SYSTEM] Configuring Peano CMake Substrate...\033[0m"
# Peano is an LLVM fork. We MUST explicitly tell it to build the custom AIE backend
# alongside the X86 host tools, or 'llc' will not recognize the 'aie2p' target triple.
cmake -G Ninja -S "$PEANO_DIR/llvm" -B "$PEANO_BUILD_DIR" \
    -DLLVM_ENABLE_PROJECTS="clang;lld" \
    -DLLVM_TARGETS_TO_BUILD="X86;AIE" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON

echo -e "\033[93m[SYSTEM] Forging Peano C++ Backend (Thermodynamically Capped at $CORES Threads)...\033[0m"
ninja -j"$CORES"

echo -e "\033[1;32m[SUCCESS] Sovereign Peano Substrate Forged.\033[0m"

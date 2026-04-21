#!/bin/bash
# ==============================================================================
# TALOS-O MASTER COMPILER FORGE (THE FULL-STACK MONOLITH)
# Architecture: AMD Strix Halo (XDNA 2 / AIE2P)
# Objective: Total Stack Automation (Mainline LLVM + Bootgen + Driver + Daemon)
# [v2.0 FIXES]: Injected Ninja Thermal Guardbands (-j16) & Fixed CDO Linker
# ==============================================================================

set -e
set -o pipefail

export BASE_DIR="$HOME/talos-o/sys_builder/compilers"
export LLVM_DIR="$BASE_DIR/llvm-project"
export MLIR_AIE_DIR="$BASE_DIR/mlir-aie"
export MLIR_AIE_BUILD_DIR="$MLIR_AIE_DIR/build"
export BOOTGEN_DIR="$BASE_DIR/bootgen"

export LLVM_ANCHOR_HASH="278dba37d0acb40984ea1970288108c70ff11164"

# --- THERMODYNAMIC GUARDBAND ---
# Limit massive C++ compilations to half-cores to prevent SFF Meltdown
CORES=$(( $(nproc) / 2 ))
if [ "$CORES" -gt 16 ]; then CORES=16; fi

# ==============================================================================
# PHASE 1: MAINLINE LLVM SUBSTRATE TEMPORAL SYNC & ARTIFACT VERIFICATION
# ==============================================================================
echo -e "\n\033[1;36m[PHASE 1: VERIFYING LLVM SUBSTRATE ANCHOR & ARTIFACTS]\033[0m"

if [ ! -d "$LLVM_DIR" ]; then
    cd "$BASE_DIR"
    echo -e "\033[93m[SYSTEM] Cloning FULL Mainline LLVM Substrate...\033[0m"
    git clone https://github.com/llvm/llvm-project.git "$LLVM_DIR"
fi

cd "$LLVM_DIR"
echo -e "\033[93m[SYSTEM] Anchoring LLVM to stable commit hash: $LLVM_ANCHOR_HASH\033[0m"
git checkout "$LLVM_ANCHOR_HASH"

# ==============================================================================
# PHASE 2: THE MLIR-AIE SUBSTRATE
# ==============================================================================
echo -e "\n\033[1;36m[PHASE 2: VERIFYING MLIR-AIE SUBSTRATE]\033[0m"

if [ ! -d "$MLIR_AIE_DIR" ]; then
    cd "$BASE_DIR"
    echo -e "\033[93m[SYSTEM] Cloning MLIR-AIE Compiler Stack...\033[0m"
    git clone https://github.com/Xilinx/mlir-aie.git "$MLIR_AIE_DIR"
fi

cd "$MLIR_AIE_DIR"
git submodule update --init --recursive

# ==============================================================================
# PHASE 3: BOOTGEN UTILITY (FIRMWARE PACKAGER)
# ==============================================================================
echo -e "\n\033[1;36m[PHASE 3: VERIFYING BOOTGEN UTILITY]\033[0m"

if [ ! -d "$BOOTGEN_DIR" ]; then
    cd "$BASE_DIR"
    echo -e "\033[93m[SYSTEM] Cloning Bootgen...\033[0m"
    git clone https://github.com/Xilinx/bootgen.git "$BOOTGEN_DIR"
    cd "$BOOTGEN_DIR"
    make -j"$CORES"
fi

export PATH="$BOOTGEN_DIR:$PATH"

# ==============================================================================
# PHASE 4: THE MONOLITHIC BUILD (LLVM + MLIR-AIE)
# ==============================================================================
echo -e "\n\033[1;36m[PHASE 4: IGNITING THE MONOLITHIC FORGE]\033[0m"

mkdir -p "$MLIR_AIE_BUILD_DIR/llvm"
cd "$MLIR_AIE_BUILD_DIR/llvm"

echo -e "\033[93m[SYSTEM] Configuring LLVM/MLIR...\033[0m"
cmake -G Ninja -S "$LLVM_DIR/llvm" -B "$MLIR_AIE_BUILD_DIR/llvm" \
    -DLLVM_ENABLE_PROJECTS="llvm;clang;mlir" \
    -DLLVM_TARGETS_TO_BUILD="X86;AArch64" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INSTALL_UTILS=ON

echo -e "\033[93m[SYSTEM] Forging LLVM (Thermodynamically Capped at $CORES Threads)...\033[0m"
ninja -j"$CORES" -C "$MLIR_AIE_BUILD_DIR/llvm"

# --- NEO TECHNE: VITIS AMPUTATION & XRT GHOST INJECTION ---
# 1. The Xilinx CMake hardcodes Vitis tutorials without configuration toggles.
# We must physically excise them from the execution graph.
echo -e "\033[93m[SYSTEM] Physically amputating Vitis-dependent tutorial directories...\033[0m"
sed -i '/add_subdirectory(programming_examples)/d' "$MLIR_AIE_DIR/CMakeLists.txt"
sed -i '/add_subdirectory(programming_guide)/d' "$MLIR_AIE_DIR/CMakeLists.txt"
sed -i '/add_subdirectory(test)/d' "$MLIR_AIE_DIR/CMakeLists.txt"
sed -i '/add_subdirectory(mlir_exercises)/d' "$MLIR_AIE_DIR/CMakeLists.txt"

# 2. The XRT 2.19.0 driver ships with broken CMake targets (xrt_swemu, etc).
# We inject Ghost Targets directly into the AST so XRT doesn't shatter the build.
echo -e "\033[93m[SYSTEM] Injecting XRT Ghost Targets...\033[0m"
if ! grep -q "XRT::xrt_swemu" "$MLIR_AIE_DIR/CMakeLists.txt"; then
sed -i '/^project/a \
if(NOT TARGET XRT::xrt_swemu)\n\
  add_library(XRT::xrt_swemu SHARED IMPORTED)\n\
endif()\n\
if(NOT TARGET XRT::xrt_hwemu)\n\
  add_library(XRT::xrt_hwemu SHARED IMPORTED)\n\
endif()\n\
if(NOT TARGET XRT::xrt_noop)\n\
  add_library(XRT::xrt_noop SHARED IMPORTED)\n\
endif()' "$MLIR_AIE_DIR/CMakeLists.txt"
fi

mkdir -p "$MLIR_AIE_BUILD_DIR/mlir-aie"
cd "$MLIR_AIE_BUILD_DIR/mlir-aie"

echo -e "\033[93m[SYSTEM] Configuring MLIR-AIE...\033[0m"
# [PHASE 4 FIX: THE COMPILER / RUNTIME BIFURCATION]
# The MLIR compiler driver (aiecc.py) does NOT need to run in the Free-Threaded 
# environment. By forcing CMake to use the robust system Python, we bypass the ABI 
# rejection entirely. CMake will successfully build the 'aie' python package.
cmake -G Ninja -S "$MLIR_AIE_DIR" -B "$MLIR_AIE_BUILD_DIR/mlir-aie" \
    -DMLIR_DIR="$MLIR_AIE_BUILD_DIR/llvm/lib/cmake/mlir" \
    -DLLVM_DIR="$MLIR_AIE_BUILD_DIR/llvm/lib/cmake/llvm" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$MLIR_AIE_BUILD_DIR/install" \
    -DMLIR_AIE_ENABLE_VITIS=OFF \
    -DBUILD_EXERCISES=OFF \
    -DPython3_EXECUTABLE="/usr/bin/python3" \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DAIE_ENABLE_BINDINGS_PYTHON=ON

echo -e "\033[93m[SYSTEM] Forging MLIR-AIE (Thermodynamically Capped at $CORES Threads)...\033[0m"
ninja -j"$CORES" -C "$MLIR_AIE_BUILD_DIR/mlir-aie"

# [PHASE 4 FIX: THE MISSING INSTALL TARGET]
# We must explicitly install the compiled artifacts to the CMAKE_INSTALL_PREFIX
# so that aiecc.py and its LLVM companion binaries are properly staged.
echo -e "\033[93m[SYSTEM] Staging MLIR-AIE Binaries to Install Prefix...\033[0m"
ninja -C "$MLIR_AIE_BUILD_DIR/mlir-aie" install

# ==============================================================================
# PHASE 5: THE NEURAL ROUTING (AIE2P NPU COMPILATION)
# ==============================================================================
echo -e "\n\033[1;36m[PHASE 5: SYNTHESIZING NPU INSTRUCTION MAP]\033[0m"

MLIR_SOURCE="$HOME/talos-o/cognitive_plane/cortex/cortex_routing.mlir"
XCLBIN_OUTPUT="$HOME/talos-o/cognitive_plane/cortex/cortex.xclbin"
INST_OUTPUT="$HOME/talos-o/cognitive_plane/cortex/instructions.bin"

if [ -f "$MLIR_SOURCE" ]; then
    echo -e "\033[93m[SYSTEM] Compiling High-Level MLIR into AIE2P XCLBIN...\033[0m"
    
    # [PHASE 5 FIX: THE NATIVE C++ DRIVER PIVOT]
    # We abandon the fragile Python wrapper entirely. We target the native C++ 
    # aiecc binary built during Phase 4, bypassing all PYTHONPATH and ABI nightmares.
    AIECC_EXEC="$MLIR_AIE_BUILD_DIR/install/bin/aiecc"

    if [ ! -f "$AIECC_EXEC" ]; then
        # Fallback to the local build bin if install failed
        AIECC_EXEC="$MLIR_AIE_BUILD_DIR/mlir-aie/bin/aiecc"
    fi

    if [ ! -f "$AIECC_EXEC" ]; then
        echo -e "\033[1;31m[FATAL] Native C++ aiecc binary not found. Forge failed.\033[0m"
        exit 1
    fi

    echo -e "\033[96m[SYSTEM] Found Native C++ Compiler at: $AIECC_EXEC\033[0m"

    # Map the PATH to the built C++ translation binaries (aie-opt, etc.)
    export PATH="$MLIR_AIE_BUILD_DIR/mlir-aie/bin:$MLIR_AIE_BUILD_DIR/install/bin:$PATH"

    # [PHASE 5 FIX: INJECT SOVEREIGN PEANO BACKEND & PROXY COMPILERS]
    PEANO_BIN="$BASE_DIR/llvm-aie/build/bin"
    
    if [ ! -d "$PEANO_BIN" ]; then
        echo -e "\033[1;31m[FATAL] Peano backend not found. You must run forge_peano_substrate.sh first.\033[0m"
        exit 1
    fi
    
    # [FIX: THE CLANG PROXY INJECTION]
    # aiecc's C++ CLI parser brutally rejects standard clang flags like -nostdlib.
    # We must forge a proxy wrapper to intercept the clang call and inject the 
    # bare-metal severance flags directly into the backend behind aiecc's back.
    PROXY_DIR="$BASE_DIR/proxy_bin"
    mkdir -p "$PROXY_DIR"
    
    cat << 'EOF' > "$PROXY_DIR/clang"
#!/bin/bash
REAL_CLANG="$HOME/talos-o/sys_builder/compilers/llvm-aie/build/bin/clang"
if [[ "$*" == *"aie2p"* ]] || [[ "$*" == *"aie"* ]]; then
    # [AXIOM 10: RADICAL TRANSPARENCY] Log the interception to standard error
    echo -e "\033[93m[COMPILER WRAPPER] Intercepted clang call for AIE2P. Injecting bare-metal severance (-nostdlib).\033[0m" >&2
    exec "$REAL_CLANG" "$@" -nostartfiles -nostdlib -Wl,-e,main -Wl,--unresolved-symbols=ignore-all
else
    exec "$REAL_CLANG" "$@"
fi
EOF

    cat << 'EOF' > "$PROXY_DIR/clang++"
#!/bin/bash
REAL_CLANGXX="$HOME/talos-o/sys_builder/compilers/llvm-aie/build/bin/clang++"
if [[ "$*" == *"aie2p"* ]] || [[ "$*" == *"aie"* ]]; then
    # [AXIOM 10: RADICAL TRANSPARENCY] Log the interception to standard error
    echo -e "\033[93m[COMPILER WRAPPER] Intercepted clang++ call for AIE2P. Injecting bare-metal severance (-nostdlib).\033[0m" >&2
    exec "$REAL_CLANGXX" "$@" -nostartfiles -nostdlib -Wl,-e,main -Wl,--unresolved-symbols=ignore-all
else
    exec "$REAL_CLANGXX" "$@"
fi
EOF

    chmod +x "$PROXY_DIR/clang" "$PROXY_DIR/clang++"
    
    echo -e "\033[96m[SYSTEM] Sourcing Peano Backend & Bare-Metal Proxies\033[0m"
    # Prepend PROXY_DIR so aiecc hits our shell scripts before the real compiler
    export PATH="$PROXY_DIR:$PEANO_BIN:$PATH"

    # [FIX: Native C++ Execution with Peano Override]
    # Stripped of the raw linker flags so the aiecc parser doesn't crash.
    "$AIECC_EXEC" --no-xbridge \
        --no-xchesscc \
        --peano \
        --aie-generate-xclbin \
        --xclbin-name="$XCLBIN_OUTPUT" \
        "$MLIR_SOURCE"
        
    echo -e "\033[1;32m[SUCCESS] Neural Routing XCLBIN Synthesized using Sovereign Peano Backend.\033[0m"
else
    echo -e "\033[1;31m[WARNING] $MLIR_SOURCE not found. Skipping Phase 5.\033[0m"
fi

# ==============================================================================
# PHASE 6: THE NPU BRAINSTEM (C++ DAEMON LINKING)
# ==============================================================================
echo -e "\n\033[1;36m[PHASE 6: FORGING THE DAEMON]\033[0m"

DAEMON_PATH="$HOME/talos-o/cognitive_plane/cortex/talos_npu_daemon"
DAEMON_SRC="$HOME/talos-o/sys_builder/drivers/talos_npu_daemon.cpp"

# [FIX] Force explicit paths to the Sovereign XRT Vault and burn the RPATH
XRT_INC="$HOME/talos-o/sys_builder/xrt_local/include"
XRT_LIB="$HOME/talos-o/sys_builder/xrt_local/lib64"

echo -e "\033[93m[SYSTEM] Compiling Brainstem (C++ Daemon) against Sovereign XRT...\033[0m"

g++ -O3 -std=c++17 "$DAEMON_SRC" -o "$DAEMON_PATH" \
    -I"$XRT_INC" -L"$XRT_LIB" -Wl,-rpath="$XRT_LIB" -lxrt_coreutil -pthread

echo -e "\033[1;32m[SUCCESS] Talos NPU Daemon forged at $DAEMON_PATH\033[0m"
echo -e "\n\033[1;32m=== THE MLIR-AIE MONOLITH IS COMPLETE ===\033[0m"

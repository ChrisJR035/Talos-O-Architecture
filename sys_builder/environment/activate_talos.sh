#!/bin/bash
# ==============================================================================
# TALOS-O: NEURAL LINK (AUTO-DETECT MODE)
# Target: AMD Strix Halo (gfx1151) on Fedora Rawhide
# ==============================================================================

# 1. Detect ROCm Path (Priority: Fedora System -> Upstream -> Custom)
if [ -d "/usr/lib64/rocm" ] || [ -d "/usr/include/rocm" ]; then
    export ROCM_PATH="/usr"
    export LIB_PATH="/usr/lib64"
    echo "[*] Detected System ROCm (Fedora): $ROCM_PATH"
elif [ -d "/opt/rocm" ]; then
    export ROCM_PATH="/opt/rocm"
    export LIB_PATH="/opt/rocm/lib"
    echo "[*] Detected Upstream ROCm: $ROCM_PATH"
else
    export ROCM_PATH="$HOME/rocm-native"
    export LIB_PATH="$ROCM_PATH/lib"
    echo "[!] WARNING: ROCm not found in system paths. Using fallback: $ROCM_PATH"
fi

export HIP_PATH="$ROCM_PATH"

# 2. Inject Binary Paths
# We prepend to PATH to ensure we find 'hipcc' and 'rocminfo'
export PATH="$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH"

# 3. Inject Library Paths (Critical for PyTorch Runtime)
export LD_LIBRARY_PATH="$LIB_PATH:$ROCM_PATH/llvm/lib:$LD_LIBRARY_PATH"

# 4. CMake Prefix (For building extensions)
export CMAKE_PREFIX_PATH="$ROCM_PATH"

# 5. Hardware Masquerade (Strix Halo Identity - CRITICAL)
# Forces the RDNA 3 driver to treat this chip as a supported 11.5.1 target
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export PYTORCH_ROCM_ARCH=gfx1151
export HIP_VISIBLE_DEVICES=0
export HIPBLASLT_DISABLE=1 # [PHASE 2 FIX: Suppress datacenter library warnings]
# 6. Memory Optimization (Unified Memory / APU)
export HSA_ENABLE_SDMA=0 # Disable System DMA (CPU/GPU share RAM)
export HIP_HOST_COHERENT=1 # Enable Coherent Access

# [FIX: SVA / XNACK FRACTURE]
# Required for PyTorch and ROCm to share page tables with CPU without hanging the GPU
export HSA_XNACK=1

# [FIX: HARDWARE QUEUE BACKPRESSURE]
# Prevent PyTorch from starving llama.cpp of VMIDs on the Strix Halo
export GPU_MAX_HW_QUEUES=4
export HSA_ENABLE_INTERRUPT=0 # Replace PCIe interrupts with memory polling

# 7. Wavefront Control (Native RDNA 3.5)
# 7. Wavefront Control (Native RDNA 3.5)
export AMDR_WAVEFRONT_SIZE=32

echo "[TALOS] Neural Link Active. Connected to Substrate at $ROCM_PATH"

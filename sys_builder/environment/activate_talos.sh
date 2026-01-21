#!/bin/bash
# ==============================================================================
# TALOS-O: NEURAL LINK (SOURCE MODE)
# Connects the shell to the custom Strix Halo Substrate in $HOME/rocm-native
# ==============================================================================

# 1. Target the Custom Build (NOT /usr)
export ROCM_PATH="$HOME/rocm-native"
export HIP_PATH="$ROCM_PATH"

# 2. Inject Binary Paths (Priority High)
export PATH="$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH"

# 3. Inject Library Paths (Critical for Runtime)
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/llvm/lib:$LD_LIBRARY_PATH"

# 4. CMake Prefix (For building PyTorch later)
export CMAKE_PREFIX_PATH="$ROCM_PATH"

# 5. Hardware Masquerade (Strix Halo Identity)
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export PYTORCH_ROCM_ARCH=gfx1151
export HIP_VISIBLE_DEVICES=0

# 6. Memory Optimization (Unified Memory / APU)
export HSA_ENABLE_SDMA=0       # Disable DMA (CPU/GPU share RAM, DMA is slower)
export HIP_HOST_COHERENT=1     # Enable coherency

# 7. Wavefront Control (Native RDNA 3.5)
export AMDR_WAVEFRONT_SIZE=32

echo "[TALOS] Neural Link Active. Connected to Substrate at $ROCM_PATH"

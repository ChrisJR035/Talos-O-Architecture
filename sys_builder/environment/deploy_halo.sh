#!/bin/bash

# ==============================================================================
# TALOS-O: NEURAL LINK DEPLOYMENT (Phase 7)
# Substrate: AMD Strix Halo (Ryzen AI Max+ 395)
# Target: PyTorch (ROCm 6.2 Nightly)
# Strategy: "The Ghost" (Masquerade as gfx1100/Navi31)
# ==============================================================================

set -e

# --- VISUALS ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   TALOS-O: NEURAL LINK DEPLOYMENT (STRIX HALO)         ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"

# 1. Environment Check
# --------------------
echo -e "${YELLOW}[1/4] Scanning Host Environment...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[CRITICAL] Python 3 could not be found.${NC}"
    exit 1
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "${CYAN}[INFO] Python Version: $PY_VER${NC}"

# 2. Hardware Masquerade (The Strix Halo Patch)
# ---------------------------------------------
echo -e "${YELLOW}[2/4] Configuring HSA Overrides (The Ghost Strategy)...${NC}"

# CRITICAL: We force the entire stack to see the Strix Halo (gfx1151)
# as a Radeon 7900 XTX (gfx1100). This aligns with our MIOpen build.
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100

# Unified Memory Optimization
# We tell PyTorch to be aggressive with memory allocation because we share 128GB.
# garbage_collection_threshold:0.8 -> Clean up graphs aggressively
# max_split_size_mb:128 -> Prevent fragmentation
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:128

echo -e "${CYAN}[ENV] HSA_OVERRIDE_GFX_VERSION: $HSA_OVERRIDE_GFX_VERSION${NC}"
echo -e "${CYAN}[ENV] PYTORCH_ROCM_ARCH:        $PYTORCH_ROCM_ARCH${NC}"
echo -e "${CYAN}[ENV] PYTORCH_HIP_ALLOC_CONF:   Active${NC}"

# 3. Installation
# ---------------
echo -e "${YELLOW}[3/4] Installing PyTorch (ROCm 6.2 Nightly)...${NC}"

# Uninstall conflicts first to prevent "Frankenstein" environments
echo "[-] Purging existing torch installations..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Install Nightly (The Bleeding Edge)
# We use the official ROCm 6.2 index.
echo "[-] Fetching Wheels..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.2

# 4. Verification
# ---------------
echo -e "${YELLOW}[4/4] Verifying Neural Link...${NC}"

python3 -c "
import torch
import sys
import time

try:
    print(f'\n[TEST] PyTorch Version: {torch.__version__}')
    print(f'[TEST] ROCm Available:  {torch.cuda.is_available()}')
    
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        name = torch.cuda.get_device_name(0)
        print(f'[TEST] Device Name:     {name} (Should imply Strix Halo)')
        print(f'[TEST] Architecture:    {torch.cuda.get_device_capability(0)}')
        
        # The Matrix Multiplication Test
        # 4096 x 4096 Matrix Multiply (Heavier Load)
        size = 4096
        print(f'[TEST] Allocating Tensors ({size}x{size})...')
        x = torch.randn(size, size, device=dev, dtype=torch.float16) # FP16 to test Half-Precision
        y = torch.randn(size, size, device=dev, dtype=torch.float16)
        
        print('[TEST] Igniting Tensor Cores (FP16 GEMM)...')
        start = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize() # Wait for GPU
        end = time.time()
        
        elapsed = end - start
        tflops = (2 * size**3) / (elapsed * 1e12)
        
        print(f'[SUCCESS] Matrix Multiplication Complete.')
        print(f'[INFO] Shape: {z.shape}')
        print(f'[INFO] Time:  {elapsed:.4f}s')
        print(f'[INFO] Speed: ~{tflops:.2f} TFLOPS (Raw Compute)')
        print(f'[INFO] VRAM:  {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB')
    else:
        print('[FAILURE] ROCm is NOT available. Check dmesg or drivers.')
        sys.exit(1)

except Exception as e:
    print(f'\n[CRITICAL] Test Failed: {e}')
    sys.exit(1)
"

echo -e "${GREEN}=== NEURAL LINK ESTABLISHED ===${NC}"

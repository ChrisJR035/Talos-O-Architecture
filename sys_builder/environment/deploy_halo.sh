#!/bin/bash

echo "=== UNSCA Deployment: AMD Strix Halo (gfx1151) ==="

# 1. Environment Check
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[INFO] Detected Python Version: $PY_VER"

# 2. Hardware Environment Overrides
# Force GFX version for Strix Halo (RDNA 3.5)
export HSA_OVERRIDE_GFX_VERSION=11.5.1

# Optimize memory allocator for Unified Memory (APU optimization)
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:128,expandable_segments:True

echo "[ENV] HSA_OVERRIDE_GFX_VERSION set to $HSA_OVERRIDE_GFX_VERSION"
echo "[ENV] PyTorch Allocator Configured for Unified Memory"

# 3. Installation
echo "[-] Fetching Nightly Wheels for ROCm..."

# Using the official PyTorch Nightly index for ROCm 6.2+ support
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.2

# 4. Verification
echo "[-] Verifying HIP Access..."
python3 -c "import torch; print(f'ROCm Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

echo "=== Deployment Ready. Run 'python iadcs_kernel.py' to ignite. ==="

import os
import sys

# --- THE SAFE OUROBOROS (Path Correction & ABI Graft) ---
CUSTOM_LIB_PATH = "/home/croudabush/rocm-native/lib:/home/croudabush/rocm-native/lib64"

# [INJECTED: The Phantom Limb ABI Graft]
target_so_6 = "/home/croudabush/rocm-native/lib/libamdhip64.so.6"
actual_so = "/home/croudabush/rocm-native/lib/libamdhip64.so"

if not os.path.exists(target_so_6) and os.path.exists(actual_so):
    print("\033[93m[SYSTEM] Deep contamination detected. Forging Phantom Limb ABI bridge...\033[0m", flush=True)
    os.symlink(actual_so, target_so_6)

# [INJECTED: Strix Halo APU Unified Memory Alignments]
# We must inject these BEFORE PyTorch imports so the ROCm runtime initializes correctly.
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.5.1"  # Align driver to gfx1151 Fatbin
os.environ["HSA_ENABLE_SDMA"] = "0"                # Disable discrete PCIe DMA
os.environ["HIP_HOST_COHERENT"] = "1"              # Enable Zero-Copy Unified RAM
os.environ["HSA_XNACK"] = "1"                      # Enable Page Fault recovery

current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
if "/home/croudabush/rocm-native/lib" not in current_ld_path:
    print("\033[93m[SYSTEM] Neural Link Path missing. Initiating Ouroboros Relaunch...\033[0m", flush=True)
    new_ld_path = f"{CUSTOM_LIB_PATH}:{current_ld_path}" if current_ld_path else CUSTOM_LIB_PATH
    os.environ["LD_LIBRARY_PATH"] = new_ld_path
    os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)

import torch
import time

def titans_test():
    if not torch.cuda.is_available():
        print("❌ TITAN OFFLINE: ROCm not detected.")
        return

    print(f"✅ TITAN ONLINE: {torch.cuda.get_device_name(0)}")
    
    # 1. Allocate VRAM
    print("   [+] Allocating Tensors...")
    try:
        size = 4096
        a = torch.randn(size, size, device="cuda")
        b = torch.randn(size, size, device="cuda")
        
        # 2. Compute
        print("   [+] Engaging Compute Units...")
        start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize() # Wait for GPU to finish
        end = time.time()
        
        print(f"   [+] Matrix Matmul ({size}x{size}) Complete.")
        print(f"   [+] Time: {end - start:.4f} seconds")
        print("✅ SYSTEM STABLE.")
        
    except Exception as e:
        print(f"❌ SYSTEM FAILURE: {e}")

if __name__ == "__main__":
    titans_test()

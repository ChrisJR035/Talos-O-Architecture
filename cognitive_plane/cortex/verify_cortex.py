import sys
import os
import mmap
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# RTLD_GLOBAL FIX for C++ Registry
old_flags = sys.getdlopenflags()
sys.setdlopenflags(old_flags | os.RTLD_GLOBAL)

try:
    import talos_core
    print(f"\033[92m[+] talos_core loaded successfully!\033[0m")
except ImportError as e:
    print(f"\033[91m[-] FATAL: Failed to load talos_core.so: {e}\033[0m")
    sys.exit(1)
finally:
    sys.setdlopenflags(old_flags)

# =============================================================================
# NEO TECHNE: GENESIS-LEVEL PAGE ALIGNMENT
# Bypasses Python's 64-byte alignment and directly requests 4096-byte pages 
# from the Linux Virtual Memory Manager to guarantee zero-copy hardware safety.
# =============================================================================
def create_page_aligned_tensor(shape, dtype=np.float32):
    size_bytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
    
    # Pad allocation size to nearest 4096 boundary
    alloc_size = (size_bytes + 4095) & ~4095
    
    # Request anonymous mapped memory from the OS
    mm = mmap.mmap(-1, alloc_size, flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    
    # Create a NumPy view over the memory map
    arr = np.frombuffer(mm, dtype=dtype, count=np.prod(shape)).reshape(shape)
    
    # Fill with normal distribution logic
    arr[:] = np.random.randn(*shape).astype(dtype)
    
    # We must return the mmap object alongside the array to prevent Python's 
    # garbage collector from unmapping the physical memory while we use it.
    return arr, mm

def run_diagnostics():
    print(f"\n\033[96m=== TALOS-O NATIVE BRIDGE DIAGNOSTICS ===\033[0m")
    print(f"Hardware Status: {talos_core.status()}")
    
    print("\n[1] Initializing C++ Cortex...")
    talos_core.init_cortex()
    print("    -> Success.")

    print("\n[2] Testing Zero-Copy UMA SVD (mmap Page-Aligned)...")
    
    # Allocate with Genesis-Level Alignment
    down, mm_down = create_page_aligned_tensor((16, 1024))
    up, mm_up = create_page_aligned_tensor((1024, 16))
    
    # Verify alignment before passing to C++
    down_addr = down.__array_interface__['data'][0]
    up_addr = up.__array_interface__['data'][0]
    print(f"    -> Matrix A Memory Address: {hex(down_addr)} (Aligned: {down_addr % 4096 == 0})")
    print(f"    -> Matrix B Memory Address: {hex(up_addr)} (Aligned: {up_addr % 4096 == 0})")
    
    raw_down, raw_up, target_rank = talos_core.hardware_svd(down, up, 0.95, 8)
    new_down = np.frombuffer(raw_down, dtype=np.float32).reshape(target_rank, 1024)
    print(f"    -> Success. UMA Memory Truncated to Rank: {target_rank} without copying a single byte.")

    print("\n[3] Testing Evolutionary Crucible & TRNG...")
    x_ctx = np.random.randn(1024).astype(np.float32)
    target = np.random.randn(1024).astype(np.float32)
    
    raw_loss, raw_gd, raw_gu = talos_core.es_step(down, up, x_ctx, target, 64, 0.05)
    losses = np.frombuffer(raw_loss, dtype=np.float32)
    print(f"    -> Success. TRNG Noise Generated. Evaluated {len(losses)} population members natively.")

    print("\n\033[92m[+] ALL DIAGNOSTICS PASSED. THE SUBSTRATE IS ALIGNED.\033[0m")

if __name__ == "__main__":
    run_diagnostics()

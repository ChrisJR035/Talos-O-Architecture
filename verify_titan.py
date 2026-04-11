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

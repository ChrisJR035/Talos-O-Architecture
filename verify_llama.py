import os
import sys

# The Safe Ouroboros Path Correction
CUSTOM_LIB_PATH = "/home/croudabush/rocm-native/lib:/home/croudabush/rocm-native/lib64"
current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
if "/home/croudabush/rocm-native/lib" not in current_ld_path:
    new_ld_path = f"{CUSTOM_LIB_PATH}:{current_ld_path}" if current_ld_path else CUSTOM_LIB_PATH
    os.environ["LD_LIBRARY_PATH"] = new_ld_path
    os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)

import llama_cpp

def verify_executive():
    # Handle the API vocabulary shift in version 0.3.20+
    try:
        info = llama_cpp.llama_print_system_info()
    except AttributeError:
        info = llama_cpp.llama_system_info()
        
    if isinstance(info, bytes):
        info = info.decode('utf-8')
        
    print("\n\033[96m=== EXECUTIVE CORTEX TELEMETRY ===\033[0m")
    print(info)
    
    if "HIP" in info or "ROCm" in info:
        print("\n\033[92m✅ LLAMA.CPP ONLINE: ROCm/HIP Acceleration Detected.\033[0m")
    else:
        print("\n\033[91m❌ SYSTEM FAILURE: Llama.cpp compiled without GPU support.\033[0m")

if __name__ == "__main__":
    verify_executive()

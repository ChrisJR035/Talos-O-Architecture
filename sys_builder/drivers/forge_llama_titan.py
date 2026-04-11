#!/usr/bin/env python3
import os
import subprocess
import sys
import multiprocessing

TALOS_HOME = os.path.expanduser("~/talos-o")
DRIVERS_DIR = os.path.join(TALOS_HOME, "sys_builder/drivers")
LLAMA_DIR = os.path.join(DRIVERS_DIR, "llama_cpp")
ROCM_PATH = os.path.expanduser("~/rocm-native")

CORES = min(multiprocessing.cpu_count(), 12)
TARGET = "gfx1151"

CYAN = "\033[96m"
GREEN = "\033[92m"
RESET = "\033[0m"

def run_cmd(cmd, cwd=None, env=None):
    print(f"{CYAN}EXEC: {cmd}{RESET}")
    subprocess.run(cmd, shell=True, cwd=cwd, env=env, check=True)

def forge_llama():
    print(f"{GREEN}=== FORGING LLAMA.CPP (NO OPENMP / PURE GPU) ==={RESET}")
    
    if not os.path.exists(LLAMA_DIR):
        run_cmd(f"git clone https://github.com/ggerganov/llama.cpp.git {LLAMA_DIR}")
    else:
        run_cmd("git pull", cwd=LLAMA_DIR)

    build_dir = os.path.join(LLAMA_DIR, "build")
    run_cmd(f"rm -rf {build_dir}")
    
    env = os.environ.copy()
    env["ROCM_PATH"] = ROCM_PATH
    env["HIP_PATH"] = ROCM_PATH
    env["CXX"] = os.path.join(ROCM_PATH, "bin/hipcc")
    env["CC"] = os.path.join(ROCM_PATH, "bin/hipcc")
    
    # 1. Compile Bare-Metal Engine (OpenMP Excised)
    print(f"\n{GREEN}[1/2] Synthesizing Core C++ Engine...{RESET}")
    cmake_cmd = (
        f"cmake -B {build_dir} -DGGML_HIP=ON -DAMDGPU_TARGETS={TARGET} "
        f"-DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH={ROCM_PATH} "
        f"-DCMAKE_HIP_FLAGS=\"-D__AMDGCN_WAVEFRONT_SIZE=32\" "
        f"-DCMAKE_CXX_FLAGS=\"-D__AMDGCN_WAVEFRONT_SIZE=32\" "
        f"-DCMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES=/usr/include "
        f"-DCMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES=/usr/include "
        f"-DLLAMA_BUILD_TESTS=OFF "
        f"-DLLAMA_BUILD_EXAMPLES=OFF "
        f"-DLLAMA_BUILD_SERVER=OFF "
        f"-DGGML_OPENMP=OFF "
        f"-DLLAMA_OPENMP=OFF"
    )
    run_cmd(cmake_cmd, cwd=LLAMA_DIR, env=env)
    run_cmd(f"cmake --build {build_dir} --config Release -j {CORES}", cwd=LLAMA_DIR, env=env)
    
    # 2. Compile Python Binding (Redirected to Bleeding-Edge GitHub Repo)
    print(f"\n{GREEN}[2/2] Synthesizing Python Neural Link (llama-cpp-python)...{RESET}")
    py_env = env.copy()
    py_env["CMAKE_ARGS"] = (
        f"-DGGML_HIP=ON -DAMDGPU_TARGETS={TARGET} -DCMAKE_PREFIX_PATH={ROCM_PATH} "
        f"-DCMAKE_HIP_FLAGS=\"-D__AMDGCN_WAVEFRONT_SIZE=32\" "
        f"-DCMAKE_CXX_FLAGS=\"-D__AMDGCN_WAVEFRONT_SIZE=32\" "
        f"-DCMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES=/usr/include "
        f"-DCMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES=/usr/include "
        f"-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF "
        f"-DGGML_OPENMP=OFF -DLLAMA_OPENMP=OFF"
    )
    py_env["FORCE_CMAKE"] = "1"
    
    # SURGICAL FIX: Pull from git directly to secure Qwen3.5 MoE tensor architecture support
    pip_cmd = f"{sys.executable} -m pip install --upgrade --force-reinstall --no-cache-dir git+https://github.com/abetlen/llama-cpp-python.git"
    run_cmd(pip_cmd, cwd=LLAMA_DIR, env=py_env)
    
    print(f"\n{GREEN}[+] Llama.cpp Substrate Forged. The Sovereign Node is fully armed.{RESET}")

if __name__ == "__main__":
    forge_llama()

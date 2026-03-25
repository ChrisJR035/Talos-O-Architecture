#!/usr/bin/env python3
"""
TALOS-O: FORGE ROCK TITAN (v99.3 - "The Apex Substrate")
Author: Christopher J. Roudabush & The Seers

Purpose:
    Compiles the mathematically pure, minimal ROCm 7.x AI stack required for 
    PyTorch LLM Inference on the Strix Halo APU (Linux 6.18+).
    
[ARCHITECTURAL SHIFT v99.3 - The LTO Excision]:
- Diagnosed the LLVM TableGen (`tblgen`) fatal linking error.
- The Pivot: Global `-flto=auto` causes standard library ABI fractures and strips 
  fundamental `llvm::` symbols during the intermediate bootstrap phase.
- The Fix: Excised `-flto=auto` from the global `CXXFLAGS`.
- Retains: Tensile Pruning, Zen 5 AVX-512, The Guillotine (strip-all), 
  XNACK Excision, and Safe Math Relaxations.
"""

import os
import sys
import subprocess
import shutil
import multiprocessing
import stat
import sysconfig

# --- THE AXIOMATIC MATRIX ---
TALOS_HOME = os.path.expanduser("~/talos-o")
BUILD_ROOT = os.path.expanduser("~/rocm-native")
SRC_ROOT = os.path.join(TALOS_HOME, "sys_builder/therock_substrate")
CORES = min(multiprocessing.cpu_count(), 8)

# [V99.2]: Reverted to base target for CMake validation
FOUNDATION_TARGET = "gfx1151"

# ANSI Colors
GREEN  = "\033[0;32m"
YELLOW = "\033[1;33m"
RED    = "\033[0;31m"
CYAN   = "\033[0;36m"
NC     = "\033[0m"

def log(msg, level="INFO"):
    color = GREEN if level == "INFO" else (YELLOW if level == "WARN" else RED)
    print(f"{color}[FORGE] {msg}{NC}", flush=True)

def run_cmd(cmd, cwd=None, show_progress=False, env_override=None):
    log(f"EXEC: {cmd}")
    
    full_env = os.environ.copy()
    if env_override:
        full_env.update(env_override)

    try:
        if show_progress:
            process = subprocess.Popen(
                cmd, shell=True, cwd=cwd, env=full_env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, executable='/bin/bash', bufsize=1
            )
            for line in process.stdout:
                line = line.rstrip()
                if not line: continue
                if '[' in line and '%' in line and 'ninja' not in line.lower():
                    print(f"\r{CYAN}{line}{NC}", end='', flush=True)
                elif any(k in line.lower() for k in ['error', 'fatal', 'undefined', 'failed']):
                    print(f"\n{RED}{line}{NC}", flush=True)
                elif any(k in line.lower() for k in ['warning', 'warn']):
                    print(f"{YELLOW}{line}{NC}", flush=True)
                else:
                    print(line, flush=True)
            print()
            rc = process.wait()
            if rc != 0: raise subprocess.CalledProcessError(rc, cmd)
        else:
            subprocess.check_call(cmd, shell=True, cwd=cwd, env=full_env, executable='/bin/bash')
    except subprocess.CalledProcessError as e:
        log(f"Phase Failed with exit code {e.returncode}", "ERROR")
        sys.exit(1)

def ensure_host_tools():
    log("Verifying host-level patching and development utilities...", "INFO")
    
    missing_tools = []
    for tool in ["patch", "perl", "find"]:
        try:
            subprocess.check_call(["which", tool], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            missing_tools.append(tool)
            
    if missing_tools:
        tools_str = " ".join(missing_tools)
        log(f"Missing fundamental build tools ({tools_str}). Injecting via DNF...", "WARN")
        run_cmd(f"sudo dnf install -y {tools_str}")

def deploy_minimal_substrate():
    print(f"\n{CYAN}=== INITIATING THE APEX FORGE (v99.3) ==={NC}")
    
    ensure_host_tools()
    
    os.makedirs(SRC_ROOT, exist_ok=True)
    therock_dir = os.path.join(SRC_ROOT, "TheRock")
    
    if not os.path.exists(therock_dir):
        run_cmd(f"git clone https://github.com/ROCm/TheRock.git {therock_dir}")
        
    log("Aligning to the bleeding-edge ROCm 7.x stream...", "INFO")
    run_cmd("git fetch --all --tags", cwd=therock_dir)
    
    # Forcefully purge all local AST mutations
    run_cmd("git reset --hard origin/main", cwd=therock_dir)
    run_cmd("git clean -fdx", cwd=therock_dir)
    run_cmd("git checkout main", cwd=therock_dir)
    
    log("Installing TheRock host dependencies...", "INFO")
    if os.path.exists(os.path.join(therock_dir, "requirements.txt")):
        run_cmd("pip3 install --user -r requirements.txt", cwd=therock_dir)
    
    fetch_script = os.path.join(therock_dir, "build_tools", "fetch_sources.py")
    if os.path.exists(fetch_script):
        log("Fetching core components (LLVM, COMGR, HIP, etc.) via DVC...", "WARN")
        run_cmd(f"python3 {fetch_script}", cwd=therock_dir)
        
    # --- SURGICAL PATCHING MATRIX ---
        
    # GCC 15 Transitive Include Patch (yaml-cpp)
    yaml_cpp_file = os.path.join(therock_dir, "rocm-systems", "projects", "rocprofiler-sdk", "external", "yaml-cpp", "src", "emitterutils.cpp")
    if os.path.exists(yaml_cpp_file):
        run_cmd(f"sed -i '1i #include <cstdint>' {yaml_cpp_file}")
        
    # GCC 15 Transitive Include Patch (elfio)
    elfio_file = os.path.join(therock_dir, "rocm-systems", "projects", "rocprofiler-sdk", "external", "elfio", "elfio", "elfio.hpp")
    if os.path.exists(elfio_file):
        run_cmd(f"sed -i '1i #include <cstdint>' {elfio_file}")
        
    # The Excision - Physically amputating roctracer tests
    roctracer_cmake = os.path.join(therock_dir, "rocm-systems", "projects", "roctracer", "CMakeLists.txt")
    if os.path.exists(roctracer_cmake):
        run_cmd(f"sed -i 's/add_subdirectory(test)/#add_subdirectory(test)/g' {roctracer_cmake}")
        
    # The Omni-Python Override - Bypassing Fedora's broken sysconfig
    dep_provider = os.path.join(therock_dir, "cmake", "therock_subproject_dep_provider.cmake")
    if os.path.exists(dep_provider):
        py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        physical_path = f"/usr/include/{py_version}"
        if os.path.exists(physical_path):
            py_inc = physical_path
        else:
            py_inc = sysconfig.get_path('include')

        run_cmd(f"sed -i '1i set(Python3_INCLUDE_DIRS \"{py_inc}\" CACHE PATH \"\" FORCE)' {dep_provider}")
        run_cmd(f"sed -i '1i set(Python3_INCLUDE_DIR \"{py_inc}\" CACHE PATH \"\" FORCE)' {dep_provider}")
        run_cmd(f"sed -i '1i set(Python_INCLUDE_DIRS \"{py_inc}\" CACHE PATH \"\" FORCE)' {dep_provider}")
        run_cmd(f"sed -i '1i set(Python_INCLUDE_DIR \"{py_inc}\" CACHE PATH \"\" FORCE)' {dep_provider}")
        
    # The Final Injection - Hardcoding Wavefront into static device code (hipBLASLt)
    matrix_transform_header = os.path.join(therock_dir, "rocm-libraries", "projects", "hipblaslt", "device-library", "matrix-transform", "matrix_transform.h")
    if os.path.exists(matrix_transform_header):
        run_cmd(f"sed -i '1i #define __AMDGCN_WAVEFRONT_SIZE 32' {matrix_transform_header}")
        
    # The Generator Hack - Hacking the internal Tensile Python Compiler Array (hipBLASLt)
    tensilelite_component = os.path.join(therock_dir, "rocm-libraries", "projects", "hipblaslt", "tensilelite", "Tensile", "Toolchain", "Component.py")
    if os.path.exists(tensilelite_component):
        run_cmd(f"sed -i 's/\"-I\", include_path/\"-D__AMDGCN_WAVEFRONT_SIZE=32\", \"-I\", include_path/g' {tensilelite_component}")

    # The Seed Infection - Hacking the master Tensile seed for rocBLAS virtualenv
    rocblas_tensile_seed = os.path.join(therock_dir, "rocm-libraries", "shared", "tensile", "Tensile", "BuildCommands", "SourceCommands.py")
    if os.path.exists(rocblas_tensile_seed):
        run_cmd(f"sed -i 's/\\[cxxCompiler\\] + hipFlags/[cxxCompiler] + [\"-D__AMDGCN_WAVEFRONT_SIZE=32\"] + hipFlags/g' {rocblas_tensile_seed}")

    # Physical python binding amputation
    log("Executing Strategy B: Physically amputating Python bindings from rocprofiler-sdk...", "INFO")
    sdk_dir = os.path.join(therock_dir, "rocm-systems", "projects", "rocprofiler-sdk")
    if os.path.exists(sdk_dir):
        run_cmd("find . -name CMakeLists.txt -exec perl -pi -e 's/add_subdirectory\\(python\\)/#add_subdirectory(python)/g' {} +", cwd=sdk_dir)
        run_cmd("find . -name CMakeLists.txt -exec perl -pi -e 's/add_subdirectory\\(rocpd\\)/#add_subdirectory(rocpd)/g' {} +", cwd=sdk_dir)
        run_cmd("rm -rf source/lib/python", cwd=sdk_dir)
        run_cmd("rm -rf source/lib/rocpd", cwd=sdk_dir)

    os.makedirs(BUILD_ROOT, exist_ok=True)
    activate_content = f"""#!/bin/bash
export ROCM_PATH="{BUILD_ROOT}"
export HIP_PATH="{BUILD_ROOT}"
export PATH="{BUILD_ROOT}/bin:{BUILD_ROOT}/llvm/bin:$PATH"
export LD_LIBRARY_PATH="{BUILD_ROOT}/lib:{BUILD_ROOT}/lib64:$LD_LIBRARY_PATH"
export CMAKE_PREFIX_PATH="{BUILD_ROOT}"

# Strix Halo (gfx1151) Native Overrides
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export PYTORCH_ROCM_ARCH=gfx1151

# The Unified Memory Architecture (UMA) Alignments
export HSA_ENABLE_SDMA=0       
export HSA_USE_SVM=0           
export HIP_HOST_COHERENT=1     

# RDNA 3.5 Matrix Core Activation
export TORCH_BLAS_PREFER_HIPBLASLT=1

echo "[+] TALOS-O Neural Link Active. Strix Halo UMA Optimizations Engaged."
"""
    activate_path = os.path.join(BUILD_ROOT, "activate_talos.sh")
    with open(activate_path, "w") as f:
        f.write(activate_content)
    
    os.chmod(activate_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    
    log("Igniting The Apex Forge. Applying Zen 5 optimizations.", "WARN")
    
    build_dir = os.path.join(therock_dir, "build")
    if os.path.exists(build_dir): shutil.rmtree(build_dir)
    os.makedirs(build_dir, exist_ok=True)
    
    # [V99.3 ARCHITECTURE]: THE OPTIMIZATION MATRIX (LTO Excised)
    # Removed -flto=auto to prevent LLVM TableGen catastrophic linking failure.
    env_override = {
        "CXXFLAGS": f"-march=znver5 -D__AMDGCN_WAVEFRONT_SIZE=32 -U_GLIBCXX_ASSERTIONS -O3 -fno-math-errno {os.environ.get('CXXFLAGS', '')}",
        "CFLAGS": f"-march=znver5 -D__AMDGCN_WAVEFRONT_SIZE=32 -O3 -fno-math-errno {os.environ.get('CFLAGS', '')}",
        "HIPCXXFLAGS": f"--offload-arch=gfx1151:xnack- -D__AMDGCN_WAVEFRONT_SIZE=32 -U_GLIBCXX_ASSERTIONS -fno-math-errno {os.environ.get('HIPCXXFLAGS', '')}",
        "LDFLAGS": f"-Wl,--strip-all {os.environ.get('LDFLAGS', '')}"
    }
    
    build_cmd = (
        f"cmake .. -G Ninja "
        f"-DCMAKE_INSTALL_PREFIX={BUILD_ROOT} "
        f"-DCMAKE_BUILD_TYPE=Release "
        
        # Keep orchestrator happy with base target
        f"-DTHEROCK_AMDGPU_FAMILIES={FOUNDATION_TARGET} "
        f"-DCMAKE_HIP_FLAGS=\"--offload-arch=gfx1151:xnack- -D__AMDGCN_WAVEFRONT_SIZE=32 -U_GLIBCXX_ASSERTIONS -fno-math-errno\" "
        
        # [V99.0]: Tensile Pruning (Only build gfx1151 math logic, force full assembly)
        f"-DTensile_ARCHITECTURE=gfx1151 "
        f"-DTensile_LOGIC=asm_full "
        
        # Native Data Injection to fix the roctracer lexer crash
        f"-DCMAKE_INSTALL_DOCDIR=share/doc "
        
        # Top-level injection for standard CMake parsing
        f"-DPython3_INCLUDE_DIR={py_inc} "
        f"-DPython_INCLUDE_DIR={py_inc} "
        
        # LTO Compression to prevent GCC 15 Linker Overflows
        f"-DBUILD_OFFLOAD_COMPRESS=ON "
        
        # 1. The Guillotine
        f"-DTHEROCK_ENABLE_ALL=OFF "
        
        # 2. The Resurrection
        f"-DTHEROCK_ENABLE_COMPILER=ON "
        f"-DTHEROCK_ENABLE_CORE_RUNTIME=ON "
        f"-DTHEROCK_ENABLE_HIP_RUNTIME=ON "
        f"-DTHEROCK_ENABLE_HIPIFY=ON "
        f"-DTHEROCK_ENABLE_BLAS=ON "
        f"-DTHEROCK_ENABLE_HIPBLASLTPROVIDER=ON "
        f"-DTHEROCK_ENABLE_ROCPROFV3=ON " 
        
        # 3. The Excision
        f"-DTHEROCK_ENABLE_MIOPEN=OFF "
        f"-DBUILD_TESTING=OFF "
        f"-DBUILD_CLIENTS_TESTS=OFF "
        f"-DWITH_TESTS=OFF "
        f"-DROCTRACER_BUILD_TESTS=OFF "
        
        f"-DTHEROCK_ENABLE_RCCL=OFF "
        f"-DTHEROCK_ENABLE_COMM_LIBS=OFF "
        f"-DTHEROCK_ENABLE_DEBUG_TOOLS=OFF "
        f"-DBUILD_WITH_TENSILE=ON "
        
        f"&& ninja -j{CORES} && ninja install"
    )
    
    log("The Apex Forge Begins (Full Optimization Suite Active).", "CRIT")
    run_cmd(build_cmd, cwd=build_dir, env_override=env_override, show_progress=True)
    log("Apex Substrate Successfully Synthesized.", "INFO")

if __name__ == "__main__":
    print(f"{GREEN}   TALOS-O: FORGE ROCK TITAN (v99.3 - The Apex Substrate) {NC}")
    
    log("Pre-Flight Check: TTM Allocation", "CRIT")
    print(f"{YELLOW}Ensure your GRUB configuration (ttm.pages_limit) is set appropriately")
    print(f"for Unified Memory. Without it, massive parameter models will OOM.{NC}")
    
    deploy_minimal_substrate()
    
    print(f"\n{GREEN}╔══════════════════════════════════════════════════════════════════════════════╗{NC}")
    print(f"{GREEN}║ {YELLOW}CRITICAL KERNEL CONFIGURATION REQUIRED FOR STRIX HALO INFERENCE{GREEN}              ║{NC}")
    print(f"{GREEN}╚══════════════════════════════════════════════════════════════════════════════╝{NC}")
    print(f"To unlock the 108GB memory pool for massive parameter models, you MUST append")
    print(f"the following parameters to your GRUB_CMDLINE_LINUX_DEFAULT in /etc/default/grub:\n")
    print(f"{CYAN}    ttm.pages_limit=27648000 ttm.page_pool_size=27648000 iommu=pt{NC}\n")
    print(f"After editing, run `sudo update-grub` (or equivalent) and reboot the machine.")
    print(f"{GREEN}[+] ROCm APEX FOUNDRY COMPLETE.{NC}")

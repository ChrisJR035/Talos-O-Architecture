#!/usr/bin/env python3
"""
TALOS-O: FORGE ROCK TITAN (v99.4 - "The Architectural Truth")
Author: Christopher J. Roudabush & The Seers

Purpose:
    Compiles the mathematically pure, minimal ROCm 7.x AI stack required for 
    PyTorch LLM Inference on the Strix Halo APU (Linux 6.18+).
    
[ARCHITECTURAL SHIFT v99.4]:
- Integrates verbatim string matches for ROCm 7.x (LLVM 18/19) runtimes.
- Excises the `offload` target to permanently cure the Fortran Emissary Phantom Limb.
"""

import os
import sys
import subprocess
import shutil
import stat
import sysconfig

# --- THE AXIOMATIC MATRIX ---
TALOS_HOME = os.path.expanduser("~/talos-o")
BUILD_ROOT = os.path.expanduser("~/rocm-native")
SRC_ROOT = os.path.join(TALOS_HOME, "sys_builder/therock_substrate")
CORES = 4
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
    if env_override: full_env.update(env_override)

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
        run_cmd(f"sudo dnf install -y {tools_str}")

def synthesize_gcc15_shim():
    shim_path = os.path.join(SRC_ROOT, "TheRock", "rocm_gcc15_compat.h")
    log(f"Synthesizing Shim Header: {shim_path}", "INFO")
    content = """#ifndef ROCM_GCC15_COMPAT_H
#define ROCM_GCC15_COMPAT_H
/* Activate for C++ compilers on x86_64 to protect against libstdc++ fractures */
#if defined(__cplusplus) && defined(__x86_64__)
  #include <emmintrin.h>
  #include <pmmintrin.h>
  #include <tmmintrin.h>
  #include <immintrin.h>
  #include <cstdint>
  #include <stdint.h> /* Ensures global uint16_t without std:: */
#endif
#endif
"""
    with open(shim_path, "w") as f: f.write(content)

def amputate_flang_and_cauterize():
    log("Executing Auditable AST Mutation: Verbatim ROCm 7.x Excision...", "WARN")
    hook_path = os.path.join(SRC_ROOT, "TheRock", "compiler", "pre_hook_amd-llvm.cmake")
    if os.path.exists(hook_path):
        with open(hook_path, "r") as f: c = f.read()
            
        # 1. Excise Flang (Host compiler Phase)
        old_proj = '"clang;lld;clang-tools-extra;flang"'
        new_proj = '"clang;lld;clang-tools-extra"'
        c = c.replace(old_proj, new_proj)
        
        # 2. Excise Offload ONLY (Restore OpenMP for hipBLASLt host threading)
        old_runt = '"compiler-rt;libunwind;libcxx;libcxxabi;openmp;offload"'
        new_runt = '"compiler-rt;libunwind;libcxx;libcxxabi;openmp"' # <-- RESTORED openmp
        c = c.replace(old_runt, new_runt)
        
        # 3. Safety Fallback: Neutralize the Master Override directly if it exists
        c = c.replace('set(LIBOMPTARGET_BUILD_DEVICE_FORTRT ON)', 'set(LIBOMPTARGET_BUILD_DEVICE_FORTRT OFF)')

        with open(hook_path, "w") as f: f.write(c)
        log("Mutation Successful: Flang and Offload excised. OpenMP restored.", "INFO")
    else:
        log(f"CRITICAL: {hook_path} not found.", "RED")

def deploy_minimal_substrate():
    print(f"\n{CYAN}=== INITIATING THE APEX FORGE (v99.4) ==={NC}")
    ensure_host_tools()
    
    os.makedirs(SRC_ROOT, exist_ok=True)
    therock_dir = os.path.join(SRC_ROOT, "TheRock")
    
    if not os.path.exists(therock_dir):
        run_cmd(f"git clone https://github.com/ROCm/TheRock.git {therock_dir}")
        
    run_cmd("git fetch --all --tags --force", cwd=therock_dir)
    run_cmd("git reset --hard origin/main", cwd=therock_dir)
    run_cmd("git clean -fdx", cwd=therock_dir)
    run_cmd("git checkout main", cwd=therock_dir)
    
    if os.path.exists(os.path.join(therock_dir, "requirements.txt")):
        run_cmd("pip3 install -r requirements.txt", cwd=therock_dir)
        
    amputate_flang_and_cauterize()
    synthesize_gcc15_shim()
    
    fetch_script = os.path.join(therock_dir, "build_tools", "fetch_sources.py")
    if os.path.exists(fetch_script):
        run_cmd(f"python3 {fetch_script}", cwd=therock_dir)
        
    # --- SURGICAL PATCHING MATRIX ---
    
    # Define the shim path ONCE for all injections
    shim_path = os.path.join(therock_dir, "rocm_gcc15_compat.h")

    # GCC 15 Shim Injection (rocprofiler-sdk)
    # This prevents FetchContent/Submodules from overwriting our physical file patches
    sdk_dir = os.path.join(therock_dir, "rocm-systems", "projects", "rocprofiler-sdk")
    if os.path.exists(os.path.join(sdk_dir, "CMakeLists.txt")):
        run_cmd(f"sed -i '1i add_compile_options(\"-include\" \"{shim_path}\")' CMakeLists.txt", cwd=sdk_dir)
        
    # GCC 15 Shim Injection (rocr-runtime)
    hsa_dir = os.path.join(therock_dir, "rocm-systems", "projects", "rocr-runtime")
    if os.path.exists(os.path.join(hsa_dir, "CMakeLists.txt")):
        run_cmd(f"sed -i '1i add_compile_options(\"-include\" \"{shim_path}\")' CMakeLists.txt", cwd=hsa_dir)
        run_cmd(f"sed -i '1i set(CMAKE_CXX_STANDARD 20)' CMakeLists.txt", cwd=hsa_dir)
        
    roctracer_cmake = os.path.join(therock_dir, "rocm-systems", "projects", "roctracer", "CMakeLists.txt")
    if os.path.exists(roctracer_cmake): run_cmd(f"sed -i 's/add_subdirectory(test)/#add_subdirectory(test)/g' {roctracer_cmake}")
        
    dep_provider = os.path.join(therock_dir, "cmake", "therock_subproject_dep_provider.cmake")
    if os.path.exists(dep_provider):
        py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        py_inc = f"/usr/include/{py_version}" if os.path.exists(f"/usr/include/{py_version}") else sysconfig.get_path('include')
        for var in ["Python3_INCLUDE_DIRS", "Python3_INCLUDE_DIR", "Python_INCLUDE_DIRS", "Python_INCLUDE_DIR"]:
            run_cmd(f"sed -i '1i set({var} \"{py_inc}\" CACHE PATH \"\" FORCE)' {dep_provider}")
        
    matrix_transform_header = os.path.join(therock_dir, "rocm-libraries", "projects", "hipblaslt", "device-library", "matrix-transform", "matrix_transform.h")
    if os.path.exists(matrix_transform_header): run_cmd(f"sed -i '1i #define __AMDGCN_WAVEFRONT_SIZE 32' {matrix_transform_header}")
        
    tensilelite_component = os.path.join(therock_dir, "rocm-libraries", "projects", "hipblaslt", "tensilelite", "Tensile", "Toolchain", "Component.py")
    if os.path.exists(tensilelite_component): run_cmd(f"sed -i 's/\"-I\", include_path/\"-D__AMDGCN_WAVEFRONT_SIZE=32\", \"-I\", include_path/g' {tensilelite_component}")

    rocblas_tensile_seed = os.path.join(therock_dir, "rocm-libraries", "shared", "tensile", "Tensile", "BuildCommands", "SourceCommands.py")
    if os.path.exists(rocblas_tensile_seed): run_cmd(f"sed -i 's/\\[cxxCompiler\\] + hipFlags/[cxxCompiler] + [\"-D__AMDGCN_WAVEFRONT_SIZE=32\"] + hipFlags/g' {rocblas_tensile_seed}")

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

# Strix Halo Native Overrides
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export PYTORCH_ROCM_ARCH=gfx1151

# UMA Alignments
export HSA_ENABLE_SDMA=0       
export HSA_USE_SVM=0           
export HIP_HOST_COHERENT=1     
export TORCH_BLAS_PREFER_HIPBLASLT=1

echo "[+] TALOS-O Neural Link Active. Strix Halo UMA Optimizations Engaged."
"""
    activate_path = os.path.join(BUILD_ROOT, "activate_talos.sh")
    with open(activate_path, "w") as f: f.write(activate_content)
    os.chmod(activate_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    
    build_dir = os.path.join(therock_dir, "build")
    if os.path.exists(build_dir): shutil.rmtree(build_dir)
    os.makedirs(build_dir, exist_ok=True)
    
    env_override = {
        "CXXFLAGS": f"-march=znver5 -D__AMDGCN_WAVEFRONT_SIZE=32 -U_GLIBCXX_ASSERTIONS -O3 -fno-math-errno {os.environ.get('CXXFLAGS', '')}",
        "CFLAGS": f"-march=znver5 -D__AMDGCN_WAVEFRONT_SIZE=32 -O3 -fno-math-errno {os.environ.get('CFLAGS', '')}",
        "HIPCXXFLAGS": f"--offload-arch=gfx1151:xnack+ -D__AMDGCN_WAVEFRONT_SIZE=32 -U_GLIBCXX_ASSERTIONS -fno-math-errno {os.environ.get('HIPCXXFLAGS', '')}",
        "LDFLAGS": f"-Wl,--strip-all {os.environ.get('LDFLAGS', '')}"
    }
    
    build_cmd = (
        f"cmake .. -G Ninja "
        f"-DCMAKE_INSTALL_PREFIX={BUILD_ROOT} "
        f"-DCMAKE_BUILD_TYPE=Release "
        f"-DTHEROCK_AMDGPU_FAMILIES={FOUNDATION_TARGET} "
        f"-DCMAKE_HIP_FLAGS=\"--offload-arch=gfx1151:xnack+ -D__AMDGCN_WAVEFRONT_SIZE=32 -U_GLIBCXX_ASSERTIONS -fno-math-errno\" "
        f"-DTensile_ARCHITECTURE=gfx1151 -DTensile_LOGIC=asm_full "
        f"-DCMAKE_INSTALL_DOCDIR=share/doc "
        f"-DPython3_INCLUDE_DIR={py_inc} -DPython_INCLUDE_DIR={py_inc} "
        f"-DBUILD_OFFLOAD_COMPRESS=ON "
        
        f"-DTHEROCK_ENABLE_ALL=OFF "
        f"-DTHEROCK_ENABLE_COMPILER=ON "
        f"-DTHEROCK_ENABLE_CORE_RUNTIME=ON "
        f"-DTHEROCK_ENABLE_HIP_RUNTIME=ON "
        f"-DTHEROCK_ENABLE_HIPIFY=ON "
        f"-DTHEROCK_ENABLE_BLAS=ON "
        f"-DTHEROCK_ENABLE_SPARSE=ON " 
        f"-DTHEROCK_ENABLE_SOLVER=ON " 
        f"-DTHEROCK_ENABLE_HIPBLASLTPROVIDER=ON "
        f"-DTHEROCK_ENABLE_ROCPROFV3=ON " 
        
        f"-DTHEROCK_ENABLE_MIOPEN=OFF "
        f"-DBUILD_TESTING=OFF -DBUILD_CLIENTS_TESTS=OFF -DWITH_TESTS=OFF -DROCTRACER_BUILD_TESTS=OFF "
        f"-DTHEROCK_ENABLE_RCCL=OFF -DTHEROCK_ENABLE_COMM_LIBS=OFF -DTHEROCK_ENABLE_DEBUG_TOOLS=OFF "
        f"-DBUILD_WITH_TENSILE=ON -DLLVM_ENABLE_PROJECTS=\"clang;lld;compiler-rt\" -DBUILD_FLANG=OFF "
        
        f"&& ninja -j{CORES} && ninja install"
    )
    
    log("The Apex Forge Begins (Full Optimization Suite Active).", "CRIT")
    run_cmd(build_cmd, cwd=build_dir, env_override=env_override, show_progress=True)

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

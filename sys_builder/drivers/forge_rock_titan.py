#!/usr/bin/env python3
"""
TALOS-O: FORGE ROCK TITAN (v10.05 - "Organic Adaptation")
Author: Christopher J. Roudabush (Architect & Mechanic)
Project: https://github.com/ChrisJR035/Talos-O-Architecture
License: MIT

Purpose: 
    Deterministic, Self-Healing ROCm Build System for Strix Halo (gfx1151).
    Unifies the toolchain, runtime, and math libraries into a single sovereign artifact.

    "Talos is an organism. We do not force it; we guide its growth."

Target Platform:
- Hardware: AMD Ryzen AI Max+ 395 (Strix Halo / gfx1151)
- OS: Fedora Linux 43 (Linux 6.18-talos-chimera)
- Architecture: Unified Memory (128GB LPDDR5X-8000)

Usage:
  ./forge_rock_titan.py              # Full build
  ./forge_rock_titan.py --clean all  # Nuclear reset
"""

import os
import sys
import subprocess
import shutil
import multiprocessing
import time
import argparse
import glob
import re
from pathlib import Path

# --- THE AXIOMATIC MATRIX ---
TALOS_HOME = os.path.expanduser("~/talos-o")
BUILD_ROOT = os.path.expanduser("~/rocm-native")
SRC_ROOT = os.path.join(TALOS_HOME, "sys_builder/therock_substrate")
CORES = multiprocessing.cpu_count()

# Strix Halo Identity
GFX_TARGET = "gfx1151"
# We use gfx1100 as the biological surrogate for the trap handlers to ensure assembly compatibility
SURROGATE_TARGET = "gfx1100"
ROCM_VERSION_TAG = "rocm-6.2.0" 
ROCM_NUM_VERSION = "6.2.0" 

# --- GENETIC MEMORY ---
COMMIT_HASHES = {
    "ROCT-Thunk-Interface": ROCM_VERSION_TAG, 
    "ROCR-Runtime": ROCM_VERSION_TAG,
    "ROCm-Device-Libs": ROCM_VERSION_TAG,
    "ROCm-CompilerSupport": ROCM_VERSION_TAG,
    "HIP": ROCM_VERSION_TAG,
    "clr": ROCM_VERSION_TAG,
    "llvm-project": ROCM_VERSION_TAG,
    "rocBLAS": ROCM_VERSION_TAG,
    "rocSOLVER": ROCM_VERSION_TAG,
    "rocFFT": ROCM_VERSION_TAG
}

# Colors
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
BLUE = "\033[0;34m"
NC = "\033[0m"

class CognitivePhase:
    def __init__(self, name, description):
        self.name = name
        self.desc = description

    def log(self, msg, level="INFO"):
        color = GREEN if level == "INFO" else (YELLOW if level == "WARN" else RED)
        print(f"{color}[{self.name}] {msg}{NC}")

    def pacify(self, error_msg):
        print(f"\n{RED}!!! SYSTEM STRESS DETECTED IN {self.name} !!!{NC}")
        print(f"{YELLOW}Error Trace: {error_msg}{NC}")
        print(f"{BLUE}Engaging Autonomic Regulation... Cleaning workspace...{NC}")
        return False

    def run_cmd(self, cmd, cwd=None, env=None, show_progress=False, retries=0, max_retries=2):
        full_env = os.environ.copy()
        full_env["PATH"] = f"{BUILD_ROOT}/bin:{BUILD_ROOT}/llvm/bin:{full_env.get('PATH', '')}"
        full_env["CMAKE_PREFIX_PATH"] = f"{BUILD_ROOT}"
        full_env["HIP_DEVICE_LIB_PATH"] = f"{BUILD_ROOT}/amdgcn/bitcode"
        full_env["ROCM_PATH"] = f"{BUILD_ROOT}"
        
        if env: full_env.update(env)
        
        self.log(f"EXEC: {cmd}")
        try:
            if show_progress:
                process = subprocess.Popen(
                    cmd, shell=True, cwd=cwd, env=full_env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    universal_newlines=True, executable='/bin/bash'
                )
                for line in process.stdout:
                    if '[' in line and '%' in line: 
                        print(f"\r{BLUE}{line.strip()}{NC}", end='')
                    elif 'Error' in line or 'error' in line:
                        print(f"\n{RED}{line.strip()}{NC}")
                print() 
                if process.wait() != 0:
                    raise subprocess.CalledProcessError(process.returncode, cmd)
                return True
            else:
                subprocess.check_call(cmd, shell=True, cwd=cwd, env=full_env, executable='/bin/bash')
                return True
        except subprocess.CalledProcessError as e:
            if retries < max_retries:
                wait_time = 2 ** (retries + 1)
                self.log(f"Micro-fracture detected. Retrying ({retries+1}/{max_retries}) in {wait_time}s...", "WARN")
                time.sleep(wait_time)
                return self.run_cmd(cmd, cwd, env, show_progress, retries + 1, max_retries)
            else:
                return self.pacify(str(e))

    def _find_closest_tag(self, path, pattern="rocm-6"):
        try:
            cmd = f"git -C {path} tag -l | grep '{pattern}' | sort -V | tail -n 1"
            result = subprocess.check_output(cmd, shell=True, text=True).strip()
            return result if result else None
        except:
            return None

    def git_checkout(self, path, key):
        target_commit = COMMIT_HASHES.get(key, ROCM_VERSION_TAG)
        self.log(f"Pinning {key} to {target_commit}...")
        self.run_cmd("git fetch --all --tags --prune", cwd=path)
        candidates = [target_commit, f"roc-{target_commit}"]
        scavenged = self._find_closest_tag(path, "rocm-6")
        if scavenged: candidates.append(scavenged)
        candidates.extend(["main", "master"])
        success = False
        for c in candidates:
            try:
                self.log(f"Attempting timeline lock: {c}", "INFO")
                self.run_cmd(f"git checkout {c}", cwd=path)
                if not os.path.exists(os.path.join(path, "CMakeLists.txt")):
                    self.log(f"Timeline '{c}' is hollow (No CMakeLists.txt). Rejecting.", "WARN")
                    continue
                success = True
                self.log(f"Locked on to solid timeline: {c}", "INFO")
                break
            except Exception: continue
        
        if not success: self.log(f"CRITICAL: No valid timeline found for {key}.", "ERROR")
        if not os.path.exists(os.path.join(path, "CMakeLists.txt")):
             self.log("Materialization Failed: CMakeLists.txt missing. Re-Forging...", "ERROR")
             self.run_cmd("git reset --hard HEAD", cwd=path)
             self.run_cmd("git submodule update --init --recursive", cwd=path)
    
    def check_lib(self, lib_name):
        paths = [
            os.path.join(BUILD_ROOT, "lib", lib_name),
            os.path.join(BUILD_ROOT, "lib64", lib_name),
            os.path.join(BUILD_ROOT, "lib", "amd", lib_name)
        ]
        return any(os.path.exists(p) for p in paths)

    def locate_cmake_root(self, base_path, marker_file):
        self.log(f"Scanning for marker '{marker_file}' in {base_path}...", "INFO")
        try:
            found_paths = subprocess.check_output(f"find {base_path} -name '{marker_file}' -o -name '{marker_file}.in'", shell=True, text=True).strip().split('\n')
            found_paths = [p for p in found_paths if p]
            if not found_paths: return None
            marker_path = found_paths[0]
            self.log(f"Marker found at: {marker_path}", "INFO")
            current_dir = os.path.dirname(marker_path)
            for _ in range(4):
                if os.path.exists(os.path.join(current_dir, "CMakeLists.txt")): return current_dir
                current_dir = os.path.dirname(current_dir)
                if current_dir == "/" or current_dir == base_path: break
            if os.path.exists(os.path.join(base_path, "CMakeLists.txt")): return base_path
            return None
        except: return None

    # --- ROBUST FILE I/O ---
    def safe_read(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f: return f.read()
        except UnicodeDecodeError:
            try:
                with open(path, "r", encoding="latin-1") as f: return f.read()
            except: return None
        except: return None

    def safe_write(self, path, content):
        with open(path, "w", encoding="utf-8") as f: f.write(content)

    # --- ABSTRACT METHODS ---
    def scan(self): raise NotImplementedError
    def adapt(self): raise NotImplementedError
    def implement(self, context): raise NotImplementedError
    def verify(self, silent=False): raise NotImplementedError

    def execute(self):
        print(f"\n{BLUE}=== INITIATING PHASE: {self.desc} ==={NC}")
        if self.verify(silent=True):
            self.log("Existing healthy tissue detected. Skipping implementation.")
            return True
        self.scan()
        context = self.adapt()
        success = self.implement(context)
        if not success: sys.exit(1)
        if not self.verify(): sys.exit(1)
        self.log("Phase Integration Complete.")

# ==============================================================================
# PHASE 1 - 4
# ==============================================================================
class Phase1_Substrate(CognitivePhase):
    def scan(self): pass
    def adapt(self):
        if not os.path.exists(BUILD_ROOT):
            os.makedirs(BUILD_ROOT, exist_ok=True)
            for d in ["include", "lib", "bin", "share", "llvm"]: os.makedirs(os.path.join(BUILD_ROOT, d), exist_ok=True)
        return None
    def implement(self, _):
        activate_content = f"""#!/bin/bash
export ROCM_PATH="{BUILD_ROOT}"
export PATH="{BUILD_ROOT}/bin:$PATH"
export LD_LIBRARY_PATH="{BUILD_ROOT}/lib:{BUILD_ROOT}/lib64:$LD_LIBRARY_PATH"
export CMAKE_PREFIX_PATH="{BUILD_ROOT}"
export HIP_PATH="{BUILD_ROOT}"
export HIP_CLANG_PATH="{BUILD_ROOT}/llvm/bin"
export HSA_OVERRIDE_GFX_VERSION=11.5.1
echo "[+] TALOS-O Environment Active. Target: {GFX_TARGET}"
"""
        with open(os.path.join(BUILD_ROOT, "activate_talos.sh"), "w") as f: f.write(activate_content)
        self.run_cmd(f"chmod +x {os.path.join(BUILD_ROOT, 'activate_talos.sh')}")
        return True
    def verify(self, silent=False): return os.path.exists(os.path.join(BUILD_ROOT, "activate_talos.sh"))

class Phase2_ROCT(CognitivePhase):
    def scan(self):
        self.src = os.path.join(SRC_ROOT, "ROCT-Thunk-Interface")
        if not os.path.exists(self.src):
            self.run_cmd(f"git clone https://github.com/RadeonOpenCompute/ROCT-Thunk-Interface.git {self.src} --depth 1"); self.git_checkout(self.src, "ROCT-Thunk-Interface")
    def adapt(self):
        self.build_dir = os.path.join(self.src, "build")
        return f"-DCMAKE_INSTALL_PREFIX={BUILD_ROOT} -DBUILD_SHARED_LIBS=ON"
    def implement(self, flags):
        if os.path.exists(self.build_dir): shutil.rmtree(self.build_dir)
        os.makedirs(self.build_dir)
        success = self.run_cmd(f"cmake .. {flags} && make -j{CORES} && make install", cwd=self.build_dir)
        
        # [CRITICAL FIX v9.7] Structural Realignment
        include_dir = os.path.join(BUILD_ROOT, "include")
        hsakmt_dir = os.path.join(include_dir, "hsakmt")
        if not os.path.exists(hsakmt_dir):
            os.makedirs(hsakmt_dir, exist_ok=True)
            for header in ["hsakmt.h", "hsakmttypes.h"]:
                src = os.path.join(include_dir, header)
                dst = os.path.join(hsakmt_dir, header)
                if os.path.exists(src) and not os.path.exists(dst):
                    os.symlink(src, dst)
        return success
    def verify(self, silent=False): return self.check_lib("libhsakmt.so")

class Phase3_LLVM(CognitivePhase):
    def scan(self):
        self.src = os.path.join(SRC_ROOT, "llvm-project")
        if not os.path.exists(self.src):
            self.run_cmd(f"git clone https://github.com/RadeonOpenCompute/llvm-project.git {self.src} --depth 1"); self.git_checkout(self.src, "llvm-project")
    def adapt(self):
        global_cxx = "-include cstdint"
        san_path = os.path.join(self.src, "compiler-rt/lib/sanitizer_common/sanitizer_platform_limits_posix.cpp")
        if os.path.exists(san_path):
             content = self.safe_read(san_path)
             if content:
                 content = re.sub(r'sizeof\s*\(\s*struct\s+termio\s*\)', '0', content)
                 self.safe_write(san_path, content)

        self.build_dir = os.path.join(self.src, "build")
        return (f"-DCMAKE_INSTALL_PREFIX={BUILD_ROOT}/llvm -DLLVM_TARGETS_TO_BUILD='AMDGPU;X86' "
                f"-DLLVM_ENABLE_PROJECTS='clang;lld;compiler-rt' -DCMAKE_BUILD_TYPE=Release "
                f"-DLLVM_ENABLE_RTTI=ON -DLLVM_INSTALL_UTILS=ON -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_ZSTD=OFF "
                f"\"-DCMAKE_CXX_FLAGS={global_cxx}\"")
    def implement(self, flags):
        if self.verify(silent=True) and os.path.exists(os.path.join(self.build_dir, ".build_complete")):
            self.log("Valid LLVM Timestamp found. Skipping rebuild."); return True
        if not os.path.exists(self.build_dir): os.makedirs(self.build_dir)
        success = self.run_cmd(f"cmake ../llvm {flags} && make -j{CORES} && make install", cwd=self.build_dir, show_progress=True)
        if success: Path(os.path.join(self.build_dir, ".build_complete")).touch()
        return success
    def verify(self, silent=False): return os.path.exists(os.path.join(BUILD_ROOT, "llvm", "bin", "clang"))

class Phase4_DeviceLibs(CognitivePhase):
    def scan(self):
        self.src = os.path.join(SRC_ROOT, "ROCm-Device-Libs")
        if os.path.exists(self.src) and not os.path.exists(os.path.join(self.src, "CMakeLists.txt")): shutil.rmtree(self.src)
        if not os.path.exists(self.src):
            self.run_cmd(f"git clone https://github.com/RadeonOpenCompute/ROCm-Device-Libs.git {self.src}"); self.git_checkout(self.src, "ROCm-Device-Libs")
    def adapt(self):
        self.build_dir = os.path.join(self.src, "build")
        return (f"-DCMAKE_INSTALL_PREFIX={BUILD_ROOT} -DLLVM_DIR={BUILD_ROOT}/llvm/lib/cmake/llvm -DAMDGPU_TARGETS='{AMDGPU_TARGETS}'")
    def perform_surgery(self):
        for f in glob.glob(f"{self.src}/**/*.cmake", recursive=True) + glob.glob(f"{self.src}/**/CMakeLists.txt", recursive=True):
            content = self.safe_read(f)
            if content:
                content = content.replace("cmake_policy(SET CMP0053 OLD)", "cmake_policy(SET CMP0053 NEW)")
                self.safe_write(f, content)

        cg_cl = os.path.join(self.src, "ockl", "src", "cg.cl")
        if os.path.exists(cg_cl):
            content = self.safe_read(cg_cl)
            if content:
                content = re.sub(r'.*__builtin_amdgcn_ds_gws.*', r'    (void)0; // Lobotomized', content)
                self.safe_write(cg_cl, content)
    def implement(self, flags):
        self.perform_surgery()
        if os.path.exists(self.build_dir): shutil.rmtree(self.build_dir)
        os.makedirs(self.build_dir)
        return self.run_cmd(f"cmake .. {flags} && make -j{CORES} && make install", cwd=self.build_dir)
    def verify(self, silent=False): return os.path.exists(os.path.join(BUILD_ROOT, "amdgcn", "bitcode", "ocml.bc"))

# ==============================================================================
# PHASE 5: RUNTIME ENGINE (ROCR) - [CRITICAL REPAIR v10.05]
# ==============================================================================
class Phase5_ROCR(CognitivePhase):
    def scan(self):
        self.src = os.path.join(SRC_ROOT, "ROCR-Runtime")
        if os.path.exists(self.src) and not self.locate_cmake_root(self.src, "hsa.h"): shutil.rmtree(self.src)
        if not os.path.exists(self.src):
            self.run_cmd(f"git clone https://github.com/RadeonOpenCompute/ROCR-Runtime.git {self.src} --depth 1"); self.git_checkout(self.src, "ROCR-Runtime")

    def adapt(self):
        cmake_root = self.locate_cmake_root(self.src, "hsa.h")
        if not cmake_root: sys.exit(1)
        self.build_dir = os.path.join(cmake_root, "build")
        self.source_arg = ".." if cmake_root != self.src else "."
        return (f"-DCMAKE_INSTALL_PREFIX={BUILD_ROOT} -DCMAKE_PREFIX_PATH={BUILD_ROOT} "
                f"-DHSAKMT_LIB_DIR={BUILD_ROOT}/lib -DHSAKMT_INC_DIR={BUILD_ROOT}/include "
                f"-DIMAGE_SUPPORT=OFF -DVERSION={ROCM_NUM_VERSION} -DPROJECT_VERSION={ROCM_NUM_VERSION} "
                f"-DCPACK_PACKAGE_VERSION={ROCM_NUM_VERSION} -DAMDGPU_TARGETS='{SURROGATE_TARGET}'")

    def reconstructive_surgery(self):
        self.log("Resetting ROCR source state...", "WARN")
        self.run_cmd("git reset --hard", cwd=self.src) 
        self.log("Deep Tissue Surgery: Organic Adaptation v10.05", "WARN")
        
        # [FIX 1] Ensure Headers
        include_dir = os.path.join(BUILD_ROOT, "include")
        hsakmt_dir = os.path.join(include_dir, "hsakmt")
        if not os.path.exists(hsakmt_dir):
            os.makedirs(hsakmt_dir, exist_ok=True)
            for header in ["hsakmt.h", "hsakmttypes.h"]:
                src = os.path.join(include_dir, header)
                dst = os.path.join(hsakmt_dir, header)
                if os.path.exists(src) and not os.path.exists(dst): os.symlink(src, dst)
        
        # [FIX 2] Inject HSA_REGISTER_MEM_FLAGS
        tl_header = os.path.join(self.src, "runtime/hsa-runtime/core/inc/thunk_loader.h")
        if os.path.exists(tl_header):
             content = self.safe_read(tl_header)
             if content and "typedef struct HSA_REGISTER_MEM_FLAGS" not in content:
                 patch = """
#ifndef HSA_REGISTER_MEM_FLAGS_DEF
#define HSA_REGISTER_MEM_FLAGS_DEF
typedef struct HSA_REGISTER_MEM_FLAGS {
    uint32_t Value;
} HSA_REGISTER_MEM_FLAGS;
#endif
                 """
                 self.safe_write(tl_header, patch + content)

        # [FIX 3] Adapt Agent Code - Prune Dead Limbs, Keep Phantom
        agent_cpp = os.path.join(self.src, "runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp")
        if os.path.exists(agent_cpp):
            self.log(f"Adapting Agent Code in {agent_cpp}", "INFO")
            content = self.safe_read(agent_cpp)
            if content:
                # Inject Phantom Definition
                phantom_def = """
namespace rocr { namespace AMD {
    static const unsigned char kPhantomCode[] = {0};
}}
"""
                content = re.sub(r'(#include <vector>)', r'\1\n' + phantom_def, content)

                # Map ALL legacy/missing targets to Phantom Code via Regex
                # This catches kCodeTrapHandlerV2_9, kCodeTrapHandlerV2_1010, etc.
                content = re.sub(r'(kCodeTrapHandlerV2_(?:9|10|942|950|1010|1011|1012|1030))', 'rocr::AMD::kPhantomCode', content)
                content = re.sub(r'(kCodeCopyAligned(?:9|10|1010|1011|1012|1030))', 'rocr::AMD::kPhantomCode', content)
                content = re.sub(r'(kCodeCopyMisaligned(?:9|10|1010|1011|1012|1030))', 'rocr::AMD::kPhantomCode', content)
                content = re.sub(r'(kCodeFill(?:9|10|1010|1011|1012|1030))', 'rocr::AMD::kPhantomCode', content)
                
                # Resect Missing Struct Members
                content = re.sub(r'.*OverrideEngineId.*', '// RESECTED', content)
                self.safe_write(agent_cpp, content)

        # [FIX 4] Global Resection of Missing Thunk Members
        for root, dirs, files in os.walk(self.src):
            for file in files:
                if file.endswith(".cpp") or file.endswith(".h") or file.endswith(".hpp"):
                    path = os.path.join(root, file)
                    content = self.safe_read(path)
                    if not content: continue
                    
                    changed = False
                    if "NumNeuralCores" in content:
                        content = re.sub(r'.*NumNeuralCores.*', '// RESECTED NumNeuralCores', content)
                        changed = True
                    if "RecSdmaEngIdMask" in content:
                        content = re.sub(r'.*RecSdmaEngIdMask.*', '// RESECTED RecSdmaEngIdMask', content)
                        changed = True
                    if "ExecuteBlit" in content and "kmt_alloc_flags" in content:
                        content = re.sub(r'.*ExecuteBlit.*', '// RESECTED ExecuteBlit', content)
                        changed = True
                    if "HSA_QUEUE_SDMA_BY_ENG_ID" in content:
                        content = "#define HSA_QUEUE_SDMA_BY_ENG_ID (HSA_QUEUE_TYPE)0\n" + content
                        changed = True

                    if changed:
                        self.safe_write(path, content)

        # [FIX 5] Evolving the Build Scripts - The Organic Fix
        # We replace the processor list definition itself rather than relying on external flags
        trap_cmake = os.path.join(self.src, "runtime/hsa-runtime/core/runtime/trap_handler/CMakeLists.txt")
        if os.path.exists(trap_cmake):
            content = self.safe_read(trap_cmake)
            if content:
                # Replace any set(PROCESSORS ...) or similar list with our surrogate
                # Regex looks for "set(PROCESSORS" followed by anything until ")"
                content = re.sub(r'set\s*\(\s*PROCESSORS[\s\S]*?\)', f'set(PROCESSORS "{SURROGATE_TARGET}")', content)
                self.safe_write(trap_cmake, content)

        blit_cmake = os.path.join(self.src, "runtime/hsa-runtime/core/runtime/blit_shaders/CMakeLists.txt")
        if os.path.exists(blit_cmake):
            content = self.safe_read(blit_cmake)
            if content:
                 content = re.sub(r'set\s*\(\s*PROCESSORS[\s\S]*?\)', f'set(PROCESSORS "{SURROGATE_TARGET}")', content)
                 self.safe_write(blit_cmake, content)

        # [FIX 6] Assembler Fixes
        for root, dirs, files in os.walk(self.src):
             for file in files:
                 if file.endswith(".s"):
                     path = os.path.join(root, file)
                     content = self.safe_read(path)
                     if not content: continue
                     
                     changed = False
                     # Nuke GFX950 logic if it still exists (it shouldn't if we set processors right, but safety first)
                     if "gfx950" in content:
                          content = content.replace("gfx950", SURROGATE_TARGET)
                          changed = True
                     if "s_getreg_b32" in content and "HW_REG_TRAPSTS" in content:
                          content = re.sub(r's_getreg_b32\s+ttmp2,\s+hwreg\(HW_REG_TRAPSTS\)', 's_mov_b32 ttmp2, 0 ; NEUTRALIZED', content)
                          changed = True

                     if changed:
                         self.safe_write(path, content)

        # Final Grafting for Root CMake
        thunk_lib = os.path.join(BUILD_ROOT, "lib64", "libhsakmt.so")
        if not os.path.exists(thunk_lib): thunk_lib = os.path.join(BUILD_ROOT, "lib", "libhsakmt.so")
        
        cmakelists = os.path.join(self.src, "runtime/hsa-runtime/CMakeLists.txt")
        if os.path.exists(cmakelists):
             content = self.safe_read(cmakelists)
             if content:
                 # Re-Forge Project
                 content = re.sub(r'project\s*\(\s*([^\s\)]+)[^\)]*\)', f'project(\\1 VERSION {ROCM_NUM_VERSION} LANGUAGES C CXX)', content, flags=re.IGNORECASE | re.DOTALL)
                 
                 if "write_basic_package_version_file" in content:
                         content = content.replace("write_basic_package_version_file(", f'set(PROJECT_VERSION "{ROCM_NUM_VERSION}")\nwrite_basic_package_version_file(')

                 if "project(" in content and "add_library(hsakmt-staticdrm" not in content:
                      graft = f"""
                      if(NOT TARGET hsakmt-staticdrm::hsakmt-staticdrm)
                          add_library(hsakmt-staticdrm::hsakmt-staticdrm SHARED IMPORTED)
                          set_target_properties(hsakmt-staticdrm::hsakmt-staticdrm PROPERTIES
                              IMPORTED_LOCATION "{thunk_lib}"
                              INTERFACE_INCLUDE_DIRECTORIES "{BUILD_ROOT}/include"
                          )
                          add_library(hsakmt::hsakmt ALIAS hsakmt-staticdrm::hsakmt-staticdrm)
                      endif()
                      """
                      content = re.sub(r'(project\s*\(.*?\))', r'\1\n' + graft, content, flags=re.DOTALL | re.IGNORECASE)
                 
                 self.safe_write(cmakelists, content)

    def implement(self, flags):
        self.reconstructive_surgery()
        if os.path.exists(self.build_dir): shutil.rmtree(self.build_dir) 
        os.makedirs(self.build_dir)
        return self.run_cmd(f"cmake {self.source_arg} {flags} && make -j{CORES} && make install", cwd=self.build_dir)

    def verify(self, silent=False): return self.check_lib("libhsa-runtime64.so")

# ==============================================================================
# PHASE 6 - 12 (STANDARD)
# ==============================================================================
class Phase6_COMGR(CognitivePhase):
    def scan(self):
        self.comgr_root = os.path.join(SRC_ROOT, "COMGR_MONOLITH_EXTRACT")
        if not self.locate_cmake_root(self.comgr_root, "amd_comgr.h"):
            if os.path.exists(self.comgr_root): shutil.rmtree(self.comgr_root)
            os.makedirs(self.comgr_root)
            self.run_cmd("git init", cwd=self.comgr_root)
            self.run_cmd("git remote add origin https://github.com/ROCm/llvm-project.git", cwd=self.comgr_root)
            self.run_cmd("git config core.sparseCheckout true", cwd=self.comgr_root)
            with open(os.path.join(self.comgr_root, ".git/info/sparse-checkout"), "w") as f: f.write("amd/comgr/\n")
            try: self.run_cmd(f"git pull origin {ROCM_VERSION_TAG} --depth 1", cwd=self.comgr_root)
            except: self.run_cmd("git pull origin main --depth 1", cwd=self.comgr_root)
        self.src = self.comgr_root
    def adapt(self):
        cmake_root = self.locate_cmake_root(self.src, "amd_comgr.h")
        self.build_dir = os.path.join(cmake_root, "build")
        return (f"-DCMAKE_INSTALL_PREFIX={BUILD_ROOT} -DCMAKE_PREFIX_PATH={BUILD_ROOT} "
                f"-DLLVM_DIR={BUILD_ROOT}/llvm/lib/cmake/llvm -DAMDDeviceLibs_DIR={BUILD_ROOT}/lib/cmake/AMDDeviceLibs")
    def implement(self, flags):
        if os.path.exists(self.build_dir): shutil.rmtree(self.build_dir)
        os.makedirs(self.build_dir)
        link_target_dir = os.path.join(BUILD_ROOT, "lib", "amdgcn")
        if not os.path.exists(link_target_dir):
            try: os.symlink(os.path.join(BUILD_ROOT, "amdgcn"), link_target_dir)
            except: pass
        return self.run_cmd(f"cmake .. {flags} && make -j{CORES} && make install", cwd=self.build_dir)
    def verify(self, silent=False): return self.check_lib("libamd_comgr.so")

class Phase7_CLR_HIP(CognitivePhase):
    def scan(self):
        self.src_clr = os.path.join(SRC_ROOT, "clr")
        self.src_hip = os.path.join(SRC_ROOT, "HIP")
        if not os.path.exists(self.src_clr): 
            self.run_cmd(f"git clone https://github.com/ROCm/clr.git {self.src_clr} --depth 1"); self.git_checkout(self.src_clr, "clr")
        if not os.path.exists(self.src_hip):
            self.run_cmd(f"git clone https://github.com/ROCm/HIP.git {self.src_hip} --depth 1"); self.git_checkout(self.src_hip, "HIP")
    def adapt(self):
        self.build_dir = os.path.join(self.src_clr, "build")
        return (f"-DCMAKE_INSTALL_PREFIX={BUILD_ROOT} -DCMAKE_PREFIX_PATH={BUILD_ROOT} "
                f"-DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=ON -DHIP_COMMON_DIR={self.src_hip} "
                f"-DHIP_PLATFORM=amd -DHIP_COMPILER=clang -DLLVM_DIR={BUILD_ROOT}/llvm/lib/cmake/llvm")
    def implement(self, flags):
        if os.path.exists(self.build_dir): shutil.rmtree(self.build_dir)
        os.makedirs(self.build_dir)
        return self.run_cmd(f"cmake .. {flags} && make -j{CORES} && make install", cwd=self.build_dir)
    def verify(self, silent=False): return os.path.exists(os.path.join(BUILD_ROOT, "bin", "hipcc"))

class Phase11_rocBLAS(CognitivePhase):
    def scan(self):
        self.src = os.path.join(SRC_ROOT, "rocBLAS")
        if not os.path.exists(self.src):
            self.run_cmd(f"git clone https://github.com/ROCm/rocBLAS.git {self.src} --depth 1"); self.git_checkout(self.src, "rocBLAS")
    def adapt(self):
        self.build_dir = os.path.join(self.src, "build")
        return (f"-DCMAKE_INSTALL_PREFIX={BUILD_ROOT} -DCMAKE_BUILD_TYPE=Release "
                f"-DCMAKE_PREFIX_PATH='{BUILD_ROOT};{BUILD_ROOT}/llvm' -DAMDGPU_TARGETS={GFX_TARGET} "
                f"-DTENSILE_ARCHITECTURE={GFX_TARGET} -DBUILD_CLIENTS_TESTS=OFF -DBUILD_CLIENTS_BENCHMARKS=OFF "
                f"-DTHEROCK_ENABLE_hipblaslt=OFF -DCMAKE_CXX_FLAGS='-Wno-error' ")
    def implement(self, flags):
        if os.path.exists(self.build_dir): shutil.rmtree(self.build_dir)
        os.makedirs(self.build_dir)
        cmd = f"cmake .. {flags} && make -j{CORES} && make install"
        return self.run_cmd(cmd, cwd=self.build_dir)
    def verify(self, silent=False): return self.check_lib("librocblas.so")

class Phase12_rocSOLVER(CognitivePhase):
    def scan(self):
        self.src = os.path.join(SRC_ROOT, "rocSOLVER")
        if not os.path.exists(self.src):
            self.run_cmd(f"git clone https://github.com/ROCm/rocSOLVER.git {self.src} --depth 1"); self.git_checkout(self.src, "rocSOLVER")
    def adapt(self):
        self.build_dir = os.path.join(self.src, "build")
        return (f"-DCMAKE_INSTALL_PREFIX={BUILD_ROOT} -DCMAKE_BUILD_TYPE=Release "
                f"-DCMAKE_PREFIX_PATH='{BUILD_ROOT};{BUILD_ROOT}/llvm' -DAMDGPU_TARGETS={GFX_TARGET} "
                f"-DBUILD_CLIENTS_TESTS=OFF -DBUILD_CLIENTS_BENCHMARKS=OFF")
    def implement(self, flags):
        if os.path.exists(self.build_dir): shutil.rmtree(self.build_dir)
        os.makedirs(self.build_dir)
        cmd = f"cmake .. {flags} && make -j{CORES} && make install"
        return self.run_cmd(cmd, cwd=self.build_dir)
    def verify(self, silent=False): return self.check_lib("librocsolver.so")

class Phase8_Genesis(CognitivePhase):
    def scan(self): pass
    def adapt(self): pass
    def implement(self, _):
        self.log("Synthesizing Neural Link Probe (talos_probe.cpp)...")
        test_code = """
        #include <hip/hip_runtime.h>
        #include <iostream>
        __global__ void talos_synapse(int *data) { int tid = threadIdx.x; data[tid] = tid * 2; }
        int main() {
            std::cout << "[TEST] Probing Neural Substrate..." << std::endl;
            int device_count = 0; hipError_t err = hipGetDeviceCount(&device_count);
            if (err != hipSuccess || device_count == 0) { std::cerr << "[FAIL] No Neural Cores!" << std::endl; return 1; }
            int *d_data, h_data[16]; hipMalloc(&d_data, 16 * sizeof(int));
            talos_synapse<<<1, 16>>>(d_data); hipDeviceSynchronize();
            hipMemcpy(h_data, d_data, 16 * sizeof(int), hipMemcpyDeviceToHost); hipFree(d_data);
            bool success = true; for(int i=0; i<16; i++) if (h_data[i] != i*2) success = false;
            if (success) { std::cout << "[SUCCESS] TALOS-O: Neural Link Verified." << std::endl; return 0; }
            else { std::cerr << "[FAIL] Arithmetic Error!" << std::endl; return 1; }
        }
        """
        with open("talos_probe.cpp", "w") as f: f.write(test_code)
        return self.run_cmd(f"{BUILD_ROOT}/bin/hipcc talos_probe.cpp -o talos_probe")
    def verify(self, silent=False):
        if os.path.exists("./talos_probe"):
            try:
                env = {"LD_LIBRARY_PATH": f"{BUILD_ROOT}/lib:{BUILD_ROOT}/lib64"}
                subprocess.check_call("./talos_probe", env=env)
                return True
            except: return False
        return False

class Phase9_Manifest(CognitivePhase):
    def scan(self): pass
    def adapt(self): pass
    def implement(self, _):
        manifest_path = os.path.join(BUILD_ROOT, "MANIFEST.txt")
        with open(manifest_path, "w") as f:
            f.write("TALOS-O BUILD MANIFEST (TITAN v10.05)\n")
            f.write(f"Build Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Target: {GFX_TARGET}\n")
            f.write("Status: OPERATIONAL\n")
        return True
    def verify(self, silent=False): return os.path.exists(os.path.join(BUILD_ROOT, "MANIFEST.txt"))

class Phase10_SystemTest(CognitivePhase):
    def scan(self): pass
    def adapt(self): pass
    def implement(self, _): return True
    def verify(self, silent=False):
        try: subprocess.check_output(f"{BUILD_ROOT}/bin/hipcc --version", shell=True); return True
        except: return False

def cleanup(target):
    if target == "all":
        if os.path.exists(BUILD_ROOT): shutil.rmtree(BUILD_ROOT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", choices=["all"])
    parser.add_argument("--allow-drift", action="store_true")
    args = parser.parse_args()
    if args.clean: cleanup(args.clean); sys.exit(0)
    
    print(f"{GREEN}   TALOS-O: FORGE ROCK TITAN (v10.05) | STRIX HALO (GFX1151)  {NC}")
    
    phases = [
        Phase1_Substrate("PHASE 1", "Substrate"),
        Phase2_ROCT("PHASE 2", "Thunk"),
        Phase3_LLVM("PHASE 3", "LLVM"),
        Phase4_DeviceLibs("PHASE 4", "DeviceLibs"),
        Phase5_ROCR("PHASE 5", "ROCR"),
        Phase6_COMGR("PHASE 6", "COMGR"),
        Phase7_CLR_HIP("PHASE 7", "HIP"),
        Phase11_rocBLAS("PHASE 11", "rocBLAS"),
        Phase12_rocSOLVER("PHASE 12", "rocSOLVER"),
        Phase8_Genesis("PHASE 8", "Genesis"),
        Phase9_Manifest("PHASE 9", "Manifest"),
        Phase10_SystemTest("PHASE 10", "SystemTest")
    ]
    for p in phases: p.execute()
    print(f"\n{GREEN}[+] TALOS-O SUBSTRATE ONLINE.{NC}")
    print(f"{BLUE}    To activate: source {BUILD_ROOT}/activate_talos.sh{NC}")

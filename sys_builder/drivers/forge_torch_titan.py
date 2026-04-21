#!/usr/bin/env python3
"""
================================================================================
TALOS-O: FORGE TORCH TITAN (v105.0 - "The Genetic Splice")
Target: AMD Strix Halo (gfx1151) | Kernel: 6.19+ 
Philosophy: "The Artifact is the History. Mutate the DNA natively."

[DIAGNOSTIC FIXES]:
1. The Temporal Paradox Cure: 
   Manual patches to `ir.cpp` were being overwritten by `HIPIFY` every time 
   the forge ignited. We have embedded `splice_dna()` to automatically correct 
   the `c10::hip` namespace mangling post-HIPIFY and pre-CMake.
================================================================================
"""

import os
import sys
import subprocess
import asyncio
import time
import signal
import shutil

TALOS_HOME = os.path.expanduser("~/talos-o")
BUILD_ROOT = os.path.join(TALOS_HOME, "sys_builder/drivers")
SOURCE_DIR = os.path.join(BUILD_ROOT, "pytorch_src")

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

def log(organelle: str, msg: str, color=RESET):
    timestamp = time.strftime("%H:%M:%S")
    print(f"{color}[{timestamp}] [{organelle}] {msg}{RESET}", flush=True)

class Mitochondria:
    def __init__(self):
        self.max_jobs = str(min((os.cpu_count() or 16) // 2, 16))

class Nucleus:
    def __init__(self):
        self.target_arch = "gfx1151"

    def transcribe_dna(self):
        log("NUCLEUS", "Pre-running HIPIFY Translation Matrix...", YELLOW)
        hipify_cmd = [
            sys.executable,
            os.path.join(SOURCE_DIR, "tools", "amd_build", "build_amd.py")
        ]
        try:
            subprocess.run(hipify_cmd, cwd=SOURCE_DIR, check=True)
            log("NUCLEUS", "HIPIFY translation complete.", GREEN)
        except subprocess.CalledProcessError:
            log("NUCLEUS", "HIPIFY failed.", RED)
            sys.exit(1)

    def splice_dna(self):
        """Surgically corrects HIPIFY mangling before compilation begins."""
        log("NUCLEUS", "Splicing DNA to cure HIPIFY temporal paradoxes...", YELLOW)
        ir_cpp_path = os.path.join(SOURCE_DIR, "torch/csrc/jit/ir/ir.cpp")
        
        try:
            with open(ir_cpp_path, "r") as f:
                content = f.read()

            # The 4 Namespace Corrections
            replacements = [
                ("case c10::hip::set_stream:", "case c10::cuda::set_stream:"),
                ("case cuda::set_stream:", "case c10::cuda::set_stream:"),
                ("case c10::hip::_set_device:", "case c10::cuda::_set_device:"),
                ("case cuda::_set_device:", "case c10::cuda::_set_device:"),
                ("case c10::hip::_current_device:", "case c10::cuda::_current_device:"),
                ("case cuda::_current_device:", "case c10::cuda::_current_device:"),
                ("case c10::hip::synchronize:", "case c10::cuda::synchronize:"),
                ("case cuda::synchronize:", "case c10::cuda::synchronize:")
            ]

            for bad, good in replacements:
                content = content.replace(bad, good)

            with open(ir_cpp_path, "w") as f:
                f.write(content)
                
            # [INJECTED: The Vestigial Type Polyfill for ROCm 7.1.3+]
            gemm_header = os.path.join(SOURCE_DIR, "aten/src/ATen/hip/tunable/GemmHipblaslt.h")
            if os.path.exists(gemm_header):
                with open(gemm_header, "r") as f:
                    gemm_content = f.read()
                
                # If PyTorch hasn't caught up, force the translation of the amputated type
                if "typedef hipDataType hipblasDatatype_t;" not in gemm_content:
                    gemm_content = gemm_content.replace(
                        "#pragma once",
                        "#pragma once\n#include <hip/library_types.h>\ntypedef hipDataType hipblasDatatype_t;\n"
                    )
                    # --- AFTER ---
                    with open(gemm_header, "w") as f:
                        f.write(gemm_content)
                    log("NUCLEUS", "Vestigial Type Polyfill applied to GemmHipblaslt.h.", YELLOW)

            log("NUCLEUS", "Genetic Splice complete. Substrate stabilized.", GREEN)
        except Exception as e:
            log("NUCLEUS", f"Genetic Splice failed: {e}", RED)
            sys.exit(1)

class Ribosome:
    def __init__(self, nucleus, mitochondria):
        self.nucleus = nucleus
        self.mitochondria = mitochondria

    async def synthesize(self):
        build_dir = os.path.join(SOURCE_DIR, "build")
        if os.path.exists(build_dir):
            log("LYSOSOME", "Purging poisoned CMake cache...", YELLOW)
            shutil.rmtree(build_dir, ignore_errors=True)

        # 1. Translate
        self.nucleus.transcribe_dna()
        # 2. Fix the Translation
        self.nucleus.splice_dna()

        env = os.environ.copy()
        
        env["USE_ROCM"] = "1"
        env["PYTORCH_ROCM_ARCH"] = self.nucleus.target_arch
        env["USE_CUDA"] = "0"
        
        env["USE_MPI"] = "0"
        env["USE_NCCL"] = "0"
        env["USE_RCCL"] = "0"
        env["USE_GLOO"] = "0"
        
        # [RESTORED: Embrace the Native MIOpen Organ]
        env["USE_MIOPEN"] = "1"
        
        # [INJECTED: ROCm 7.x & GCC 15 Survival Flags]
        env["USE_KINETO"] = "0"
        env["BUILD_LAZY_TS_BACKEND"] = "1"
        env["USE_EXPERIMENTAL_CUDNN_V8_API"] = "0"
        
        env["BUILD_PYTHON"] = "1" 
        env["USE_NUMPY"] = "1"
        env["MAX_JOBS"] = self.mitochondria.max_jobs
        
        # [INJECTED: Force ROCm Build Info to bypass Tunable.cpp fatal error]
        env["CXXFLAGS"] = '-Wno-error -DROCM_BUILD_INFO=\\"7.1.3-Talos\\"'
        
        # [INJECTED: The Linker Bridge - Forcing 'ld' to see the Apex Substrate]
        rocm_lib_path = "/home/croudabush/rocm-native/lib"
        env["LDFLAGS"] = f"-L{rocm_lib_path} {os.environ.get('LDFLAGS', '')}"
        
        env["BUILD_TEST"] = "0"
        # [RESTORED: Clean CMake Args without MIOpen suppressions]
        env["CMAKE_ARGS"] = env.get("CMAKE_ARGS", "") + " -DCMAKE_POLICY_VERSION_MINIMUM=3.5"

        if "ROCM_PATH" not in env:
            env["ROCM_PATH"] = "/usr"

        env["EXTRA_CMAKE_ARGS"] = (
            f"-DPython_EXECUTABLE={sys.executable} "
            f"-DPython_FIND_VIRTUALENV=ONLY "
            f"-DPython_FIND_STRATEGY=LOCATION"
        )
        
        env["PYTORCH_PYTHON"] = sys.executable
        
        existing_cmake_path = env.get("CMAKE_PREFIX_PATH", "/usr")
        env["CMAKE_PREFIX_PATH"] = f"{sys.prefix};{existing_cmake_path}"

        log("RIBOSOME", f"Igniting the Forge via setup.py (Max Threads: {self.mitochondria.max_jobs})...", BOLD + YELLOW)
        
        python_exe = os.path.join(TALOS_HOME, "cognitive_plane/venv/bin/python3")
        process = await asyncio.create_subprocess_exec(
            python_exe, "setup.py", "bdist_wheel",
            cwd=SOURCE_DIR,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )

        while True:
            await asyncio.sleep(0.001)
            line = await process.stdout.readline()
            if not line: break
            decoded = line.decode().strip()
            if not decoded: continue
            print(f" | {decoded}")

        await process.wait()
        
        if process.returncode != 0:
            log("LYSOSOME", "Synthesis Failed. Review terminal blood.", RED)
            sys.exit(1)
        else:
            log("GOLGI", "Synthesis Complete. The Python Wheel has been forged.", BOLD + GREEN)
            log("GOLGI", f"To install: pip3 install {SOURCE_DIR}/dist/*.whl", CYAN)

class SiliconCell:
    def __init__(self):
        self.mitochondria = Mitochondria()
        self.nucleus = Nucleus()

    async def live(self):
        log("MEMBRANE", "Cellular Awakening...", BOLD + CYAN)
        ribosome = Ribosome(self.nucleus, self.mitochondria)
        await ribosome.synthesize()

if __name__ == "__main__":
    def signal_handler(sig, frame):
        log("MEMBRANE", "\nApoptosis Initiated (User Interrupt).", RED)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    cell = SiliconCell()
    asyncio.run(cell.live())

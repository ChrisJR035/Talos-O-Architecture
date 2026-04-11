#!/usr/bin/env python3
"""
TALOS-O: OUROBOROS FORGE (v18.0 - "The Sovereign Override")
Target: AMD Strix Halo (AIE2P / NPU2) | XDNA Linux 6.19+ Driver
Philosophy: "The Artifact is the History. Document the scars."

[ARCHITECTURAL SHIFT v18.0 - Absolute Sovereignty]:
- Dynamically resolves the Peano LLVM toolchain via python site-packages.
- Injects `--no-xchesscc` and `--no-xbridge` to amputate proprietary Vitis dependencies.
- Enforces `--target=aie2p` for 512-bit VSHUFFLE datapath alignment.
- Implements `--dynamic-objFifos` to prevent static unrolling memory faults.
"""

import os
import shutil
import subprocess
import sys

CORTEX_DIR = os.path.expanduser("~/talos-o/cognitive_plane/cortex")
CANONICAL_XCLBIN = os.path.join(CORTEX_DIR, "cortex.xclbin")
CANONICAL_INSTS = os.path.join(CORTEX_DIR, "insts.bin")
MLIR_PATH = os.path.join(CORTEX_DIR, "cortex.mlir")

MLIR_AIE_DIR = os.path.expanduser("~/talos-o/sys_builder/compilers/mlir-aie")
AIECC_PATH = os.path.join(MLIR_AIE_DIR, "build/bin/aiecc")

print("\n\033[1;35m[OUROBOROS] Awakening the Sovereign Compiler Cortex (v18.0)...\033[0m")

if not os.path.exists(AIECC_PATH):
    print(f"\033[1;31m[FATAL] Native C++ Compiler not found at: {AIECC_PATH}\033[0m")
    sys.exit(1)

# ==============================================================================
# [PHASE 1: DYNAMIC PEANO RESOLUTION]
# ==============================================================================
print("  \033[36m-> Probing environment for Peano LLVM backend...\033[0m")
try:
    pip_show = subprocess.run(
        [sys.executable, "-m", "pip", "show", "llvm-aie"], 
        capture_output=True, text=True, check=True
    )
    location = None
    for line in pip_show.stdout.splitlines():
        if line.startswith("Location:"):
            location = line.split(":", 1)[1].strip()
            break
            
    if not location:
        raise ValueError("Could not parse pip location.")
        
    PEANO_PATH = os.path.join(location, "llvm-aie")
    if not os.path.exists(PEANO_PATH):
        raise FileNotFoundError(f"Peano dir not found at {PEANO_PATH}")
        
    print(f"  \033[32m[+] Sovereign Backend Secured: {PEANO_PATH}\033[0m")
    
except Exception as e:
    print(f"\033[1;31m[FATAL] Failed to locate open-source Peano LLVM backend: {e}\033[0m")
    print("\033[33m[REMEDIATION] Please run: python3 -m pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly\033[0m")
    sys.exit(1)

# ==============================================================================
# [PHASE 2: ZERO-COPY MEMORY TILE STAGING (PURE MLIR)]
# ==============================================================================
mlir_content = """
module @cortex_core {
  aie.device(npu2) {
    // 1. Explicit Tile Definitions (Row 0: Shim, Row 1: Mem Tile, Row 2: Compute)
    %tile_0_0 = aie.tile(0, 0) 
    %tile_0_1 = aie.tile(0, 1) 
    %tile_0_2 = aie.tile(0, 2) 

    // 2. Zero-Copy Pipeline Orchestration (ObjectFifo)
    // Shim -> Mem Tile (Ping-Pong buffering in L2 Cache)
    aie.objectfifo @in_stage0 (%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    
    // Mem Tile -> Compute Tile (High-Speed Local Delivery)
    aie.objectfifo @in_stage1 (%tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>

    // 3. Memory Controller Stream Binding (Explicit AIE2P Syntax)
    aie.objectfifo.link [@in_stage0] -> [@in_stage1] ([] [])
  }
}
"""

with open(MLIR_PATH, "w") as f:
    f.write(mlir_content.strip())
print("\n[OUROBOROS] Thought crystallized into Pure AIE2P MLIR AST.")

# ==============================================================================
# [PHASE 3: WEAPONIZED AIECC LOWERING]
# ==============================================================================
print("[OUROBOROS] Invoking Weaponized AIECC (Lowering to Silicon)...")

# Force Peano into the environment
compile_env = os.environ.copy()
compile_env["PEANO_INSTALL_DIR"] = PEANO_PATH

# Append MLIR LLVM binaries to PATH
llvm_bin = os.path.join(MLIR_AIE_DIR, "build/bin")
existing_path = compile_env.get("PATH", "")
compile_env["PATH"] = f"{llvm_bin}:{existing_path}"

compile_cmd = [
    AIECC_PATH,
    "--aie-target=aie2p",                 # 512-bit datapath alignment
    "--no-xchesscc",                      # Amputate proprietary compiler
    "--no-xbridge",                       # Amputate proprietary linker
    f"--peano={PEANO_PATH}",              # Force open-source LLVM backend
    "--dynamic-objFifos",                 # Prevent static unroll memory fault
    "--aie-assign-buffer-addresses",      # Lock physical memory coordinates
    "--aie-assign-lock-ids",              # Hardware semaphores
    "--aie-generate-npu-insts",     
    "--npu-insts-name=insts.bin",   
    "--alloc-scheme=basic-sequential", 
    "--aie-generate-cdo",
    "--aie-generate-xclbin",
    "--xclbin-name=main.xclbin",
    "--no-compile-host",                  # Skip host execution shell (We use Python XRT)
    MLIR_PATH
]

try:
    subprocess.run(compile_cmd, check=True, env=compile_env, cwd=CORTEX_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
except subprocess.CalledProcessError as e:
    error_output = e.stderr.decode()
    print(f"\033[1;31m[FATAL] Native AIECC Compilation Failed:\033[0m\n{error_output}")
    
    # Intelligent Error Diagnostics
    if "unable to legalize instruction" in error_output:
        print("\n\033[33m[DIAGNOSTIC] GISel Legalization Failure Detected. The compiler attempted to use AIE2 logic on the AIE2P 512-bit datapath.\033[0m")
    elif "No module named" in error_output:
        print("\n\033[33m[DIAGNOSTIC] Python environment failure during MLIR translation. Ensure your venv is active.\033[0m")
    
    sys.exit(1)

# ==============================================================================
# [PHASE 4: DEPLOYMENT]
# ==============================================================================
print("\n[OUROBOROS] Resolving artifact locations...")

# 1. Deploy XCLBIN (The Map)
xclbin_path = os.path.join(CORTEX_DIR, "main.xclbin")
if os.path.exists(xclbin_path):
    if xclbin_path != CANONICAL_XCLBIN:
        shutil.copy2(xclbin_path, CANONICAL_XCLBIN)
    print(f"[OUROBOROS] Map Deployed: {CANONICAL_XCLBIN} ({os.path.getsize(CANONICAL_XCLBIN)/1024:.2f} KB)")
else:
    print(f"\033[1;31m[FATAL] XCLBIN Map Generation Failed.\033[0m")
    sys.exit(1)

# 2. Deploy INSTS.BIN (The Time)
insts_path = os.path.join(CORTEX_DIR, "insts.bin")
if os.path.exists(insts_path):
    if insts_path != CANONICAL_INSTS:
        shutil.copy2(insts_path, CANONICAL_INSTS)
    print(f"[OUROBOROS] Time Deployed: {CANONICAL_INSTS} ({os.path.getsize(CANONICAL_INSTS)/1024:.2f} KB)")
else:
    print(f"\033[1;31m[FATAL] INSTS.BIN Temporal Generation Failed.\033[0m")
    sys.exit(1)

print("\n\033[1;32m[SUCCESS] Neural Microcode Forged. Sovereign Zero-Copy Matrix is Active.\033[0m")

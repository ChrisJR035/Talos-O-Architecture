#!/usr/bin/env python3
"""
TALOS-O: FORGE MATHEMATIKOS (v1.0 - The ABI Purge)
Purpose: Eradicates GIL-dependent pre-compiled wheels for numpy/scipy 
and forces a native GCC compilation with Free-Threading (-DPy_GIL_DISABLED=1) macros.
"""
import os
import subprocess
import sys

TALOS_HOME = os.path.expanduser("~/talos-o")
VENV_PIP = os.path.join(TALOS_HOME, "cognitive_plane/venv/bin/pip")

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def run_cmd(cmd, env=None):
    print(f"{CYAN}EXEC: {cmd}{RESET}")
    subprocess.run(cmd, shell=True, env=env, check=True)

def forge_mathematikos():
    print(f"{GREEN}=== FORGING MATHEMATIKOS (NO-GIL PURE C-EXTENSIONS) ==={RESET}")
    
    if not os.path.exists(VENV_PIP):
        print(f"\n{YELLOW}[FATAL] Virtual environment not found at {VENV_PIP}.{RESET}")
        sys.exit(1)

    # Inject the Free-Threading macros directly into the C/C++ compiler pipeline
    env = os.environ.copy()
    env["CFLAGS"] = "-DPy_GIL_DISABLED=1"
    env["CXXFLAGS"] = "-DPy_GIL_DISABLED=1"
    
    # [FIX: PIERCING THE ISOLATION VEIL]
    # Force Meson to see the host OS's native pkg-config paths for OpenBLAS
    env["PKG_CONFIG_PATH"] = "/usr/lib64/pkgconfig:/usr/lib/pkgconfig:/usr/share/pkgconfig"
    
    # 1. Eradicate the poisoned wheels
    print(f"\n{GREEN}[1/2] Purging Genetic Contamination (Pre-compiled Wheels)...{RESET}")
    run_cmd(f"{VENV_PIP} uninstall -y numpy scipy", env=env)
    
    # 2. Forge from source (Warning: Scipy requires OpenBLAS and gfortran)
    print(f"\n{GREEN}[2/2] Synthesizing No-GIL Math Libraries from Source...{RESET}")
    print(f"{YELLOW}(Note: Scipy C++ / Fortran compilation may take several minutes. Let the silicon burn.){RESET}")
    
    # --no-binary numpy,scipy strictly forbids downloading pre-compiled wheels
    # [FIX: FLEXIBLAS ROUTING] Tell Meson to use Fedora's FlexiBLAS wrapper instead of OpenBLAS
    build_cmd = f"{VENV_PIP} install --upgrade --no-cache-dir --no-binary numpy,scipy numpy scipy -C setup-args=\"-Dblas=flexiblas\" -C setup-args=\"-Dlapack=flexiblas\""
    run_cmd(build_cmd, env=env)
    
    print(f"\n{GREEN}[+] Mathematikos Substrate Forged. The C-ABI is secure and Free-Threaded.{RESET}")

if __name__ == "__main__":
    forge_mathematikos()

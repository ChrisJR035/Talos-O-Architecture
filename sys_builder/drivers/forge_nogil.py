#!/usr/bin/env python3
"""
TALOS-O: FORGE NOGIL v3.0 (The Biosphere Synthesis)
Purpose: 
    1. Compiles CPython 3.13 (Free-Threading) from source.
    2. Bootstraps a "Pure-Blood" Virtual Environment.
    3. Forces native ROCm/No-GIL compilation for core cognitive libraries.
"""

import os
import sys
import subprocess
import multiprocessing
import shutil
import time

# --- THE IMMUTABLE PATHS ---
TALOS_HOME = os.path.expanduser("~/talos-o")
BUILD_ROOT = os.path.join(TALOS_HOME, "sys_builder/nogil_src")
INSTALL_PREFIX = os.path.join(TALOS_HOME, "talos-nogil")
VENV_PATH = os.path.join(TALOS_HOME, "cognitive_plane/venv")

PYTHON_REPO = "https://github.com/python/cpython.git"
PYTHON_BRANCH = "3.13"

# --- COLORS ---
BOLD, RED, GREEN, YELLOW, CYAN, RESET = "\033[1m", "\033[31m", "\033[32m", "\033[33m", "\033[36m", "\033[0m"

class TheBlacksmith:
    def __init__(self):
        # Axiom 2: Thermodynamic Cost - Limit parallel burn to keep Cerberus alive
        self.cpu_count = min(multiprocessing.cpu_count() // 2, 16)
        os.makedirs(BUILD_ROOT, exist_ok=True)
        self.bin_path = os.path.join(INSTALL_PREFIX, "bin/python3.13t")

    def log(self, msg, level="INFO"):
        color = {"INFO": GREEN, "WARN": YELLOW, "CRIT": RED, "SUCCESS": CYAN}[level]
        print(f"{color}[FORGE] {msg}{RESET}", flush=True)

    def forge_substrate(self):
        """Compiles the No-GIL CPython binary."""
        target_dir = os.path.join(BUILD_ROOT, "cpython")
        if not os.path.exists(target_dir):
            self.log(f"Cloning CPython {PYTHON_BRANCH}...", "INFO")
            subprocess.run(["git", "clone", "--depth", "1", "--branch", PYTHON_BRANCH, PYTHON_REPO, target_dir], check=True)
        
        self.log("Configuring No-GIL Substrate...", "INFO")
        # Axiom 1: Material Reality - Optimized for Strix Halo (znver5)
        subprocess.run([
            "./configure", "--disable-gil", "--enable-optimizations", 
            f"--prefix={INSTALL_PREFIX}", "CFLAGS=-march=znver5"
        ], cwd=target_dir, check=True)

        self.log(f"Striking the Anvil ({self.cpu_count} threads)...", "INFO")
        subprocess.run(["make", f"-j{self.cpu_count}"], cwd=target_dir, check=True)
        subprocess.run(["make", "install"], cwd=target_dir, check=True)

    def bootstrap_biosphere(self):
        """Creates the Venv and injects No-GIL compatible sensors."""
        self.log("Initializing Pure-Blood Virtual Environment...", "INFO")
        if os.path.exists(VENV_PATH):
            self.log("Old environment detected. Purging to prevent genetic drift.", "WARN")
            shutil.rmtree(VENV_PATH)
        
        subprocess.run([self.bin_path, "-m", "venv", VENV_PATH], check=True)
        
        venv_python = os.path.join(VENV_PATH, "bin/python3")
        
        # Axiom 5: Embodied Grounding - We must force-reinstall libraries from source 
        # where wheels are not No-GIL ready.
        core_deps = ["pip", "setuptools", "wheel", "numpy>=2.1.0"]
        
        self.log("Injecting Base Metabolic Libraries...", "INFO")
        subprocess.run([venv_python, "-m", "pip", "install", "--upgrade"] + core_deps, check=True)

    def inject_cognitive_drivers(self):
        """Forces the ROCm-enabled llama-cpp-python build within the No-GIL venv."""
        self.log("Injecting ROCm-Aware Cognitive Drivers (No-GIL Edition)...", "INFO")
        venv_python = os.path.join(VENV_PATH, "bin/python3")
        
        # Enforce ROCm alignment for Strix Halo (gfx1151)
        env = os.environ.copy()
        env["CMAKE_ARGS"] = "-DGGML_HIPBLAS=on -DAMDGPU_TARGETS=gfx1151"
        env["FORCE_CMAKE"] = "1"
        
        subprocess.run([
            venv_python, "-m", "pip", "install", 
            "--force-reinstall", "--no-cache-dir", "llama-cpp-python"
        ], env=env, check=True)

    def verify_integrity(self):
        """Final Phenotype check."""
        venv_python = os.path.join(VENV_PATH, "bin/python3")
        check_cmd = "import sys; print(f'GIL Disabled: {not sys._is_gil_enabled()}'); import llama_cpp; print('Llama-Link: ACTIVE')"
        try:
            subprocess.run([venv_python, "-c", check_cmd], check=True)
            self.log("Integrity Verified. The Substrate Breathes.", "SUCCESS")
        except:
            self.log("Integrity Breach. Check compilation logs.", "CRIT")

if __name__ == "__main__":
    smith = TheBlacksmith()
    smith.forge_substrate()
    smith.bootstrap_biosphere()
    smith.inject_cognitive_drivers()
    smith.verify_integrity()

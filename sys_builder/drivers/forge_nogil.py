#!/usr/bin/env python3
"""
forge_nogil.py - The Python Forge (v2.0)
"Forging the Free-Threaded Runtime"

Purpose:
Compiles CPython 3.13 (Free-Threading / No-GIL) from source.
This provides the multi-core capability for the Cortex.

Architecture:
- Inherits 'TheBlacksmith' pattern from forge_grammatikos.py
- Enforces --disable-gil and --enable-optimizations
- Installs to ~/talos-nogil (Isolated from System Python)

Target: AMD Ryzen AI Max+ 395 (Strix Halo)
"""

import os
import sys
import subprocess
import multiprocessing
import shutil
import time

# --- CONFIGURATION ---
TALOS_HOME = os.path.expanduser("~/talos-o")
BUILD_ROOT = os.path.join(TALOS_HOME, "sys_builder/nogil_src")
INSTALL_PREFIX = os.path.join(TALOS_HOME, "talos-nogil")

PYTHON_REPO = "https://github.com/python/cpython.git"
PYTHON_BRANCH = "3.13" # The Free-Threaded Branch

# --- COLORS ---
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"

class TheBlacksmith:
    def __init__(self):
        # [FIX: THERMAL IMBALANCE]
        # Prevents SFF chassis meltdown by capping compilation threads to 16 maximum
        raw_cores = multiprocessing.cpu_count()
        self.cpu_count = min(raw_cores // 2, 16) 
        
        if not os.path.exists(BUILD_ROOT):
            os.makedirs(BUILD_ROOT)

    def log(self, msg, level="INFO"):
        color = GREEN if level == "INFO" else (YELLOW if level == "WARN" else RED)
        print(f"{color}[FORGE] {msg}{RESET}", flush=True)

    def fetch_source(self):
        target_dir = os.path.join(BUILD_ROOT, "cpython")
        if os.path.exists(target_dir):
            self.log("Source found. Purging old artifacts...")
            subprocess.run(["git", "clean", "-fdx"], cwd=target_dir, check=True)
            return target_dir

        self.log(f"Cloning CPython {PYTHON_BRANCH}...", "INFO")
        subprocess.run([
            "git", "clone", "--depth", "1",
            "--branch", PYTHON_BRANCH,
            PYTHON_REPO, target_dir
        ], check=True)
        return target_dir

    def forge(self, source_dir):
        self.log("Configuring CPython 3.13 (No-GIL Mode)...")
        
        # 1. CONFIGURE
        cmd_conf = [
            "./configure",
            "--disable-gil",
            "--enable-optimizations",
            f"--prefix={INSTALL_PREFIX}"
        ]
        
        try:
            subprocess.run(cmd_conf, cwd=source_dir, check=True)
        except subprocess.CalledProcessError:
            self.log("Configuration Failed. Missing deps?", "CRIT")
            sys.exit(1)

        # 2. BUILD
        self.log(f"Igniting the Forge ({self.cpu_count} threads)...")
        try:
            subprocess.run(["make", f"-j{self.cpu_count}"], cwd=source_dir, check=True)
        except subprocess.CalledProcessError:
            self.log("Compilation Failed.", "CRIT")
            sys.exit(1)

        # 3. INSTALL
        self.log(f"Installing to {INSTALL_PREFIX}...")
        subprocess.run(["make", "install"], cwd=source_dir, check=True)
        self.log(f"Python 3.13t forged successfully at {INSTALL_PREFIX}/bin/python3.13t", "SUCCESS")

if __name__ == "__main__":
    print(f"{BOLD}=== THE PYTHON FORGE v2.0 (NO-GIL) ==={RESET}")
    smith = TheBlacksmith()
    src = smith.fetch_source()
    smith.forge(src)

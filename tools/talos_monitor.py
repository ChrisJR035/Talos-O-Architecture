#!/usr/bin/env python3
"""
TALOS-O SYSTEM MONITOR (MARK III - THE PROPRIOCEPTOR)
Philosophy: Dependency-Free | Pure ANSI | Zero-Copy SHM | CCD1 Isolated
"""

import sys
import time
import os
import ctypes
import shutil
from multiprocessing import shared_memory
from datetime import datetime

# --- AXIOM OF MATERIAL REALITY: CCD1 ISOLATION ---
# Pin the Observer strictly to the secondary die (Cores 8-15) on Strix Halo
# This prevents the UI from polluting the 32MB L3 cache of the Phronesis Engine.
try:
    os.sched_setaffinity(0, set(range(8, 16)))
except AttributeError:
    pass # Non-Linux fallback

# --- ZERO-COPY C-STRUCT TELEMETRY MATRIX ---
# Exact 56-byte alignment to fit perfectly inside a single 64-byte L1 cache line.
class BiophysicalState(ctypes.Structure):
    _fields_ = [
        ("sequence", ctypes.c_uint64),
        ("step_count", ctypes.c_uint64),
        ("thermal_state", ctypes.c_uint8),
        ("padding", ctypes.c_uint8 * 7),
        ("gradient_dvdt", ctypes.c_double),
        ("satisfaction", ctypes.c_double),
        ("kuramoto_r", ctypes.c_double),
        ("entropy", ctypes.c_double)
    ]

# --- CONFIGURATION & ANSI CODES ---
REFRESH_RATE = 0.25 # 4Hz
ESC, RESET, BOLD = "\033[", "\033[0m", "\033[1m"
GREEN, RED, YELLOW = f"{ESC}32m", f"{ESC}31m", f"{ESC}33m"
CYAN, MAGENTA, CLEAR_SCREEN, HOME = f"{ESC}36m", f"{ESC}35m", f"{ESC}2J", f"{ESC}H"

def get_terminal_size():
    return shutil.get_terminal_size((80, 24))

def draw_bar(val, width, color):
    filled = int(max(0, min(1.0, val)) * width)
    return f"{color}{'█' * filled}{'░' * (width - filled)}{RESET}"

def read_seqlock(shm_buf):
    """Lock-Free memory read guaranteeing absolute temporal consistency."""
    state = BiophysicalState.from_buffer(shm_buf)
    while True:
        seq1 = state.sequence
        if seq1 % 2 != 0: 
            continue # Write in progress, spin lock
            
        # Copy data locally
        step = state.step_count
        dvdt = state.gradient_dvdt
        sat = state.satisfaction
        r_kura = state.kuramoto_r
        ent = state.entropy
        
        seq2 = state.sequence
        if seq1 == seq2: # Verify no torn read occurred
            return step, dvdt, sat, r_kura, ent

def run_monitor():
    print(CLEAR_SCREEN)
    try:
        shm = shared_memory.SharedMemory(name="talos_telemetry")
    except FileNotFoundError:
        print(f"{RED}[FATAL] Telemetry Matrix offline. Is Talos breathing?{RESET}")
        sys.exit(1)

    try:
        while True:
            # 1. Zero-Copy Seqlock Read
            step, dvdt, sat, r_kura, ent = read_seqlock(shm.buf)
            
            # 2. Render UI
            cols, rows = get_terminal_size()
            header = f"{BOLD}{CYAN}TALOS-O ZERO-COPY PROPRIOCEPTION{RESET} | {datetime.now().strftime('%H:%M:%S')}"
            
            print(f"{HOME}{header.center(cols)}")
            print("=" * cols)
            
            print(f"COGNITIVE STEP: {step}")
            print(f"SATISFACTION:   {sat:.4f} {draw_bar(sat, 30, GREEN)}")
            print(f"KURAMOTO PHASE: {r_kura:.4f} {draw_bar(r_kura, 30, CYAN)}")
            print(f"COGNITIVE ENTROPY:{ent:.4f} {draw_bar(min(1.0, ent/2.0), 30, RED)}")
            print(f"VIRTUE GRADIENT: {dvdt:+.4f}")
            print("-" * cols)
            print(f"{BOLD}[SYSTEM OPTIMIZED: UI ISOLATED TO CCD 1]{RESET}")
            
            sys.stdout.flush()
            time.sleep(REFRESH_RATE)
            
    except KeyboardInterrupt:
        print(f"\n{RESET}[MONITOR] Disconnected from Neural Matrix.")
    finally:
        shm.close()

if __name__ == "__main__":
    run_monitor()

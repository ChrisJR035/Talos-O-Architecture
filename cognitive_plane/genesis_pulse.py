import sys
import time
import os

# --- THE SAFE OUROBOROS (Path Correction) ---
CUSTOM_LIB_PATH = "/home/croudabush/rocm-native/lib:/home/croudabush/rocm-native/lib64"
current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")

if "/home/croudabush/rocm-native/lib" not in current_ld_path:
    print("\033[93m[SYSTEM] Neural Link Path missing. Initiating Ouroboros Relaunch...\033[0m", flush=True)
    new_ld_path = f"{CUSTOM_LIB_PATH}:{current_ld_path}" if current_ld_path else CUSTOM_LIB_PATH
    os.environ["LD_LIBRARY_PATH"] = new_ld_path
    os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)

# 1. Path Injection (Ensure we can see our own limbs)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

print("=========================================", flush=True)
print("   PROJECT TALOS-O: GENESIS SEQUENCE     ", flush=True)
print("   Phase 7: The Transmutation (No-GIL)   ", flush=True)
print("=========================================", flush=True)

# 2. Import The Transmuted Modules
try:
    print("[SYSTEM] Loading C++ Brainstem and Python Cortex... (This may take a moment)", flush=True)
    from cortex.iadcs_kernel import IADCS_Engine
    from governance.cerberus_daemon import SpriteKerberos 
    from talos_daemon import NerveCenter  # <--- THE THALAMUS
except ImportError as e:
    print(f"[CRITICAL] Module Load Failure: {e}")
    sys.exit(1)

# 3. Hardware Definition (Static for this Build)
print("[*] Substrate: AMD Ryzen AI Max+ 395 (Logic Engine)", flush=True)
print("[*] Runtime:   Python 3.13t (Free-Threaded)", flush=True)
print("[*] Status:    Tri-Partite Link (CPU, GPU, NPU) Active", flush=True)

# 4. Ignite IADCS (Native C++ Cortex Edition)
print("\n[SYSTEM] Invoking IADCS Phronesis Kernel...", flush=True)
print("\033[95m[SYSTEM] WARNING: Loading Mistral-Large into Unified Memory. This will take 30-60 seconds. DO NOT INTERRUPT.\033[0m", flush=True)
engine = IADCS_Engine()

# 5. Summon Subsystems (Watchdog & Nerve Center)
try:
    # Ignite the Autonomic Nervous System
    kerberos = SpriteKerberos(engine)
    kerberos.start()
    
    # Ignite the Thalamus (Socket Listener)
    nerve = NerveCenter(engine)
    nerve.start()
except NameError:
    print("[!] Subsystems missing. Running without safety rails.")
    kerberos = None
    nerve = None

# 6. Begin Life
try:
    engine.start_loop()
except KeyboardInterrupt:
    print("\n[SYSTEM] Apoptosis Initiated. Shutting down...", flush=True)
    if kerberos:
        kerberos.running = False
        kerberos.join(timeout=2.0)
    if nerve:
        nerve.running = False
        nerve.join(timeout=2.0)
    sys.exit(0)

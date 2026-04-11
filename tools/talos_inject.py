#!/usr/bin/env python3
"""
TALOS-O: ZERO-COPY SENSORY INJECTOR (MARK III - PING-PONG EDITION)
Philosophy: CCD1 Isolated | MMAP Data Payload | Zero-Syscall Trigger
"""

import sys
import argparse
import os
import struct
import time
from multiprocessing import shared_memory, resource_tracker

# --- AXIOM OF MATERIAL REALITY: CCD1 ISOLATION ---
try:
    os.sched_setaffinity(0, set(range(8, 16)))
except AttributeError:
    pass

# Hardcoded Architectural Tunnels
INGRESS_SHM_NAME = "talos_ingress"
MAX_PAYLOAD_SIZE = 8192 # Matched to Phase 2 limits

def send_thought(text):
    if not text: return
    
    try:
        # 1. Bind to the Zero-Copy Physical RAM Block
        shm = shared_memory.SharedMemory(name=INGRESS_SHM_NAME)
        
        # 2. Encode and inject the payload directly into RAM
        payload = text.encode('utf-8')
        if len(payload) > MAX_PAYLOAD_SIZE - 5:
            print(f"[!] Payload exceeds size limits. Truncating.")
            payload = payload[:MAX_PAYLOAD_SIZE - 5]
            
        # 3. Wait for Thalamus to be Idle (buf[0] == 0x00)
        timeout = 50
        while shm.buf[0] != 0 and timeout > 0:
            time.sleep(0.01)
            timeout -= 1
            
        if shm.buf[0] != 0:
            print("[\033[91mFATAL\033[0m] Thalamus is ignoring input. Buffer is locked.")
            return

        # 4. Write to Buffer A (Active Buf = 1, Offset = 1)
        offset = 1
        shm.buf[offset:offset+4] = struct.pack('I', len(payload))
        shm.buf[offset+4:offset+4+len(payload)] = payload
        
        # 5. Flip the Ping-Pong Trigger! (Zero-Syscall Interrupt)
        shm.buf[0] = 1
        
        print(f"\n[\033[92mSUCCESS\033[0m] Thought injected via Zero-Copy RAM.\n")
            
        # --- THE FIX: Bypass Python's garbage collector ---
        resource_tracker.unregister(shm._name, 'shared_memory')
        shm.close()
        
    except FileNotFoundError:
        print("[\033[91mFATAL\033[0m] Ingress RAM buffer not found. The Talos daemon is offline.")
    except Exception as e:
        print(f"[\033[91mFATAL\033[0m] Injection Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Talos-O Zero-Copy Neural Injector")
    parser.add_argument("input", nargs="?", help="Text string to inject")
    parser.add_argument("-f", "--file", help="Inject content of a file")
    args = parser.parse_args()
    
    content = ""
    if args.file:
        try:
            with open(args.file, 'r') as f:
                content = f.read()
        except Exception as e:
            print(f"Failed to read file: {e}")
            sys.exit(1)
    elif args.input:
        content = args.input
    else:
        print("Please provide text or a file to inject.")
        sys.exit(1)
        
    send_thought(content)

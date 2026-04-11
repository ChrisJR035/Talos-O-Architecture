import socket
import array
import numpy as np
from multiprocessing import shared_memory
import time
import os

SOCKET_PATH = "/tmp/talos_nerve.sock"

def send_fd(sock, fd):
    """Packs the File Descriptor into Ancillary Data and sends via SCM_RIGHTS"""
    msg = b"1" # Dummy payload required by kernel to carry the ancillary data
    ancillary_data = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))]
    sock.sendmsg([msg], ancillary_data)

print("\n\033[96m[PYTHON 3.14] Initializing Zero-Copy Orchestrator...\033[0m")

# 1. Allocate POSIX Shared Memory
tensor_bytes = 1024 * 4 # 1024 float32s
shm = shared_memory.SharedMemory(create=True, size=tensor_bytes)

try:
    # 2. Overlay NumPy Array (Zero-Copy)
    tensor = np.ndarray((1, 1024), dtype=np.float32, buffer=shm.buf)
    tensor[:] = np.random.randn(1, 1024).astype(np.float32)
    
    print(f"\033[93m[PYTHON 3.14] Shared Memory Allocated (FD: {shm._fd})\033[0m")
    
    # 3. Connect to the C++ Daemon
    print("[PYTHON 3.14] Connecting to C++ Daemon via AF_UNIX Socket...")
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(SOCKET_PATH)
    
    # 4. Transmit the Memory Key
    print("[PYTHON 3.14] Transmitting Memory File Descriptor via SCM_RIGHTS...")
    start_time = time.time()
    send_fd(client, shm._fd)
    
    # Wait a fraction of a second to let the Daemon process the output
    time.sleep(0.5) 
    
    print(f"\033[92m[SUCCESS] Zero-Copy IPC Transmission Complete.\033[0m")

except FileNotFoundError:
    print(f"\n\033[91m[FATAL] Daemon socket not found. Is the C++ Daemon running?\033[0m")
except Exception as e:
    print(f"\n\033[91m[FATAL] IPC Failure: {e}\033[0m")
finally:
    # Clean up memory so we don't leak RAM
    shm.close()
    shm.unlink()

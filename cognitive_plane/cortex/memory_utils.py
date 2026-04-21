import mmap
import numpy as np
import ctypes
import os
import errno
import time

# ==============================================================================
# NEO-TECHNE: PIERCING THE ABSTRACTION
# Loading the core Linux C standard library to execute bare-metal system calls.
# ==============================================================================
libc = ctypes.CDLL("libc.so.6", use_errno=True)

# [CRITICAL SECURITY SPLICE] 
# Explicitly define argtypes and restype. Without this, Python's ctypes defaults 
# to 32-bit integers, which will brutally truncate 64-bit LPDDR5X memory pointers 
# and cause immediate, fatal Segmentation Faults.
libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mlock.restype = ctypes.c_int

libc.munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.munlock.restype = ctypes.c_int

class NPUContextManifold:
    """
    A strictly controlled Context Manager that manages the lifecycle of NPU memory.
    It connects to the POSIX shared memory forged by the C++ daemon, maps it, and
    violently pins it to physical DRAM via mlock() to prevent kswapd eviction.
    """
    def __init__(self, shm_name, shape, dtype=np.float32):
        self.shm_name = shm_name
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.size_bytes = int(np.prod(self.shape) * self.dtype.itemsize)
        # Mathematical spatial alignment to 4096-byte boundaries
        self.alloc_size = (self.size_bytes + 4095) & ~4095 
        
        self.shm_path = f"/dev/shm{self.shm_name}"
        self.fd = None
        self.mm = None
        self.arr = None
        self.buffer_address = None

    def __enter__(self):
        # 1. THE BRIDGE: Connect to the exact physical IPC file created by the Daemon
        if not os.path.exists(self.shm_path):
            raise FileNotFoundError(f"[FATAL] Brainstem inactive. Cannot find {self.shm_path}")
        
        self.fd = os.open(self.shm_path, os.O_RDWR)
        
        # 2. THE MAP: Map the shared memory into Python's virtual address space
        self.mm = mmap.mmap(self.fd, self.alloc_size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        os.close(self.fd) # Safe to close the file descriptor after mmap is established
        
        # Structure the raw buffer into an explicit Numpy tensor
        self.arr = np.frombuffer(self.mm, dtype=self.dtype, count=np.prod(self.shape)).reshape(self.shape)
        
        # 3. THE EXTRACTION: Strip Python's abstraction to get the raw 64-bit physical pointer
        self.buffer_address = self.arr.ctypes.data
        
        # 4. ENFORCING MATERIAL REALITY: Staged Ingestion (mlock Throttling)
        # [PHASE 4 FIX] Fracture the instantaneous mlock into 256MB chunks with micro-sleeps
        # to prevent 120W PCIe/UMA thermal shock during massive tensor allocations.
        chunk_size = 256 * 1024 * 1024  # 256 MB chunks
        offset = 0
        
        while offset < self.alloc_size:
            current_chunk = min(chunk_size, self.alloc_size - offset)
            
            # Lock the specific memory segment using pointer arithmetic
            target_ptr = ctypes.c_void_p(self.buffer_address + offset)
            res = libc.mlock(target_ptr, ctypes.c_size_t(current_chunk))
            
            if res != 0:
                err = ctypes.get_errno()
                raise OSError(err, f"[FATAL] mlock() failed at offset {offset}! The Linux Kernel refused to pin the memory. "
                                   f"Did you set memlock to unlimited in limits.conf? (errno={err})")
            
            offset += current_chunk
            
            # Introduce artificial temporal friction (50ms) to allow Cerberus to calculate dE/dt
            if offset < self.alloc_size:
                time.sleep(0.05)
        
        return self.arr

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 5. ENFORCING HOMEOSTASIS: Safe lifecycle termination
        # mlock tracks per-page. We must precisely munlock and close the map upon exit
        # to prevent overlapping lock leaks and memory exhaustion.
        if self.buffer_address and self.mm:
            libc.munlock(ctypes.c_void_p(self.buffer_address), ctypes.c_size_t(self.alloc_size))
            self.mm.close()
            self.arr = None

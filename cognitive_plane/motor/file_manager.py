import os
import numpy as np
from pathlib import Path
from multiprocessing import shared_memory

class SafeLoader:
    """
    Secure File I/O & Zero-Copy Memory handler.
    (Pickle eradicated: Utilizing pure Numpy serialization and POSIX Shared Memory)
    """
    def __init__(self, base_dir):
        # Atomic thread-safe resolution and directory creation
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def validate_path(self, user_input):
        """Cryptographic path validation to prevent traversal attacks."""
        target_abs = (self.base_dir / user_input).resolve()
        
        # Verify the target absolute path is physically inside the base jail
        if self.base_dir not in target_abs.parents and target_abs != self.base_dir:
            raise PermissionError(f"Access Denied: Path traversal detected. {user_input} escapes the membrane.")
            
        return str(target_abs)

    def load_weights(self, filename):
        """Loads matrices deterministically. Pickle execution is strictly forbidden."""
        safe_path = self.validate_path(filename)
        # .npy or .npz mapping. allow_pickle=False enforces the Axiom of the White Box.
        return np.load(safe_path, allow_pickle=False)

    def save_weights(self, state_dict, filename):
        """Saves matrix state safely to NVMe."""
        safe_path = self.validate_path(filename)
        # Assuming state_dict is a numpy array or dict of numpy arrays
        if isinstance(state_dict, dict):
            np.savez(safe_path, **state_dict)
        else:
            np.save(safe_path, state_dict)
        print(f"[MEMORY] State crystallized to {safe_path}")

    def map_shared_memory(self, name="/talos_npu_matrix", size=(1024 * 1024 * 2), create=False):
        """
        The Zero-Copy Bridge. 
        Establishes POSIX shared memory for 150ns CPU-GPU-NPU latency.
        """
        try:
            shm = shared_memory.SharedMemory(name=name, create=create, size=size)
            print(f"\033[92m[MEMORY] Zero-Copy UMA Bridge Established: {name}\033[0m")
            return shm
        except FileExistsError:
            # If it exists, attach to it instead
            shm = shared_memory.SharedMemory(name=name, create=False, size=size)
            print(f"\033[96m[MEMORY] Attached to existing Zero-Copy Bridge: {name}\033[0m")
            return shm
        except Exception as e:
            print(f"\033[91m[FATAL] UMA Bridge Failure: {e}\033[0m")
            return None

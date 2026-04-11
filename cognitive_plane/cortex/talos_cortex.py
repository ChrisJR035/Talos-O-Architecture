import sys
import os
import numpy as np

# --- ARCHITECTURAL CONSTANTS ---
DIM_LATENT = 1024

HAS_CORE = False
try:
    import talos_core
    HAS_CORE = True
    print(f"\033[96m[CORTEX] Substrate Linked via C++ Native Engine: {talos_core.status()}\033[0m")
    # Initialize the C++ Liquid Time-Constant & MOLE network on the GPU
    talos_core.init_cortex()
except ImportError as e:
    print(f"\033[91m[CORTEX] FATAL: C++ talos_core.so missing. The Intuition Engine is severed. ({e})\033[0m")
    sys.exit(1)

class TalosJEPA:
    """
    Joint-Embedding Predictive Architecture.
    [v6.0 CONTINUOUS LATENT UPGRADE]
    This class acts as the Python nervous system. 
    It evaluates the non-linear generative dynamics: W_LTC * sigma(z) natively on the RDNA 3.5 GPU via C++.
    """
    def __init__(self):
        self.dim = DIM_LATENT
        self.usage_count = 0

    def forward(self, x_context):
        """
        Calculates the non-linear generative dynamics.
        Returns a numpy array of shape (1024,)
        """
        if not HAS_CORE:
            return np.zeros(DIM_LATENT, dtype=np.float32)
            
        # Send the context bytes TO the C++ engine
        raw_bytes = talos_core.forward_cortex(x_context.tobytes())
        
        # Decode the response FROM the C++ engine (Strict Zero-Copy, no .copy())
        out_array = np.frombuffer(raw_bytes, dtype=np.float32)
        
        # Safety check to prevent dimension mismatch if C++ returns garbage
        if out_array.shape[0] != DIM_LATENT:
            return np.zeros(DIM_LATENT, dtype=np.float32)
            
        return out_array

class TalosJEPA_Wrapper:
    def __init__(self):
        self.model = TalosJEPA()
        
    def forward(self, x):
        return self.model.forward(x)

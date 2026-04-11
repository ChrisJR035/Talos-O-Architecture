import numpy as np
import time
import sys
import os
import threading

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import talos_core
except ImportError:
    print("[DREAM] CRITICAL: talos_core C++ implant missing. Subconscious is paralyzed.")
    sys.exit(1)

LORA_ENERGY_KEEP = 0.95
MIN_LORA_RANK = 8

dream_event = threading.Event()
dream_mutex = threading.Lock()

class DaydreamDaemon(threading.Thread):
    def __init__(self, ham_instance, cortex_instance, cleanup_instance, ltn_ref=None):
        super().__init__(name="Talos_Daydream", daemon=True)
        self.ham_instance = ham_instance
        self.cortex = cortex_instance
        self.cleanup = cleanup_instance
        self.ltn_ref = ltn_ref
        self.t_cpu = 45.0

    def autonomic_trigger(self, t_cpu):
        self.t_cpu = t_cpu
        dream_event.set()
        return "[DREAM] Background consolidation signaled."

    def run(self):
        print("[DREAM] Daemon initialized. Entering zero-cost hibernation...")
        while True:
            dream_event.wait()
            dream_event.clear()
            _perform_iterative_resonance(self.ham_instance, self.cleanup, self.t_cpu)
            _perform_lora_consolidation(self.cortex)

def _perform_iterative_resonance(ham_instance, cleanup_instance, t_cpu):
    print(f"[DREAM] Iterative Resonance (T_die: {t_cpu:.1f}C)...")
    iterations = 3
    # [FIX H-1]: Changed from 10240 to 1024
    dummy_codebook = [np.random.randn(1024).astype(np.float32) for _ in range(5)]
    if hasattr(ham_instance, 'iterative_resonance'):
        ham_instance.iterative_resonance(dummy_codebook, iterations=iterations)
    
    if cleanup_instance and hasattr(cleanup_instance, 'anneal_memory'):
        cleanup_instance.anneal_memory(ham_instance.trace, t_cpu)

def _perform_lora_consolidation(cortex):
    """
    Skill Sharpening via Native C++ Hardware-Accelerated SVD (hipSOLVER).
    """
    # [FIX H-2]: Execute blindly and let the C++ module catch errors.
    if cortex is None: 
        return 0.0
        
    print("[DREAM] Initiating Hardware-Accelerated LoRA Consolidation...")
    
    # Defaults to 1 if the wrapper hasn't mapped it yet
    num_experts = getattr(cortex.predictor, 'num_experts', 1) if hasattr(cortex, 'predictor') else 1
    
    # RESTORED LOGIC: Python iterates over each expert natively
    for i in range(num_experts):
        try:
            expert_down = cortex.predictor.experts_down[i]
            expert_up = cortex.predictor.experts_up[i]
            
            # THE HANDOFF: We delegate the heavy math to C++ to avoid melting the CPU.
            raw_down, raw_up, target_rank = talos_core.hardware_svd(
                expert_down, 
                expert_up, 
                LORA_ENERGY_KEEP, 
                MIN_LORA_RANK
            )
            
            dim_in = expert_down.shape[1]
            dim_out = expert_up.shape[0]
            
            # Decode ABI bytes back into Numpy Arrays
            new_down = np.frombuffer(raw_down, dtype=np.float32).copy().reshape(target_rank, dim_in)
            new_up = np.frombuffer(raw_up, dtype=np.float32).copy().reshape(dim_out, target_rank)
            
            # Sync truncated matrices back to Python storage
            cortex.predictor.experts_down[i] = new_down
            cortex.predictor.experts_up[i] = new_up
            print(f"[DREAM] Expert {i} Consolidation Complete. Rank optimized to {new_down.shape[0]}.")
            
        except Exception as e:
            # GPU SVD failed to converge. Abandon and preserve current matrices.
            print(f"[DREAM] Expert {i} SVD failed: {e}. Preserving current state.")
            continue

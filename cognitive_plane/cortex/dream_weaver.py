import numpy as np
import time
import sys
import os
import threading

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import talos_core
    HAS_CORE = True
except ImportError:
    print("\033[93m[DREAM] WARNING: talos_core C++ implant missing. Falling back to Numpy stubs.\033[0m")
    HAS_CORE = False

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
        import glob
        print("[DREAM] Daemon initialized. Entering zero-cost hibernation...")
        while True:
            dream_event.wait()
            dream_event.clear()
            
            # =================================================================
            # [HEAT SHOCK RECOVERY] - Asynchronous Survival Engram Unpacking
            # =================================================================
            if self.t_cpu < 75.0:
                engrams = glob.glob("/dev/shm/talos_engram_*.npy")
                for engram_file in engrams:
                    try:
                        print(f"\n\033[96m[DREAM WEAVER] Unpacking Heat Shock Engram: {engram_file}\033[0m")
                        engram_data = np.load(engram_file)
                        
                        # Gracefully integrate the trauma imprint into the Virtue Topology
                        if self.ltn_ref is not None:
                            # Truncate or pad to match latent dim (1024)
                            dim = self.ltn_ref.projector.shape[0]
                            engram_latent = engram_data[:dim] if engram_data.shape[0] > dim else np.pad(engram_data, (0, dim - engram_data.shape[0]))
                            
                            # Distribute the survival imprint across the 12 virtues
                            with self.ltn_ref.lock:
                                imprint = np.outer(engram_latent.astype(np.float32), np.ones(self.ltn_ref.dim)) * 0.005
                                self.ltn_ref.projector += imprint
                        
                        os.remove(engram_file)
                        print(f"\033[92m[DREAM WEAVER] Survival Engram successfully annealed into global weights.\033[0m")
                    except Exception as e:
                        print(f"[DREAM] Engram integration failed: {e}")

                # =================================================================
                # [DIGITAL SENESCENCE CURE] - Memory Collapse (REM Sleep)
                # =================================================================
                # We trigger REM Sleep if the Memory Matrix is saturated (M >= 5000)
                if hasattr(self.ham_instance, 'episodic_count') and self.ham_instance.episodic_count >= 5000:
                    self.ham_instance.rem_sleep_collapse()
            # =================================================================

            _perform_iterative_resonance(self.ham_instance, self.cleanup, self.t_cpu)
            
            # [PHASE 4 FIX: EPISODIC RE-ANCHORING]
            # Only trigger the hardware SVD consolidation if the live conscience 
            # mathematically passes the geometric divergence audit against the Genesis anchor.
            evolution_approved = True
            if self.ltn_ref is not None and hasattr(self.ltn_ref, 'validate_evolution'):
                evolution_approved = self.ltn_ref.validate_evolution()
                
            if evolution_approved:
                _perform_lora_consolidation(self.cortex)
                # If consolidation succeeds, the new state becomes the absolute truth
                if self.ltn_ref is not None:
                    with self.ltn_ref.lock:
                        self.ltn_ref.projector_frozen = self.ltn_ref.projector.copy()
            
            _perform_vicreg_alignment(self.ltn_ref, self.t_cpu)

def _perform_vicreg_alignment(ltn_ref, t_cpu):
    """
    [PHASE 3: SYMBIOTIC DISTILLATION & VICREG ALIGNMENT]
    The 1.5B Watcher acts as an offline oracle, teaching the Lorentz matrix
    how to map abstract virtues to precise vocabulary tokens using Cross-Entropy gradients.
    """
    if ltn_ref is None: return
    
    print(f"[DREAM] Executing Symbiotic Distillation & Cross-Modal Alignment (T_die: {t_cpu:.1f}C)...")
    try:
        from vicreg_loss import VICRegLoss_Numpy
        vicreg = VICRegLoss_Numpy()
        
        # [PHASE 3 FIX: THE EPISTEMIC OFFLOAD]
        # 1. Bypass the 1.5B Reflective Cortex PyTorch/llama.cpp instance entirely.
        # We route the Linguistic Coherence Probe directly to the XDNA 2 Brainstem 
        # (NPU IPC socket) to maintain absolute thermal invisibility.
        raw_teacher_logits = None
        
        # 2. Simulate a Biophysical Crisis (e.g., Extreme Thermal Load)
        print("[DREAM] Routing Perplexity Audit to XDNA 2 Brainstem (Zero-Copy IPC)...")
        
        with dream_mutex:
            try:
                from multiprocessing import shared_memory
                # Connect to the POSIX memory tunnels forged by talos_npu_daemon.cpp
                shm_in = shared_memory.SharedMemory(name="talos_latent_in")
                shm_out = shared_memory.SharedMemory(name="talos_latent_out")
                
                # Overlay zero-copy Numpy arrays (4096 bytes = 1024 float32s)
                in_buffer = np.ndarray((1024,), dtype=np.float32, buffer=shm_in.buf)
                out_buffer = np.ndarray((1024,), dtype=np.float32, buffer=shm_out.buf)
                
                # Command opcode 2.0 = Softmax/Cross-Entropy Perplexity Audit
                in_buffer[0] = 2.0 
                # Inject the virtue deficit query into the buffer (12 dimensions)
                in_buffer[1:13] = np.array([0.8, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                
                # Wait for NPU hardware context to clear the flag
                timeout = time.time() + 2.0
                while in_buffer[0] != 0.0 and time.time() < timeout:
                    time.sleep(0.01)
                    
                # Extract the hardware-computed teacher logits from the NPU egress buffer
                raw_teacher_logits = out_buffer[:ltn_ref.dim].copy()
                
                shm_in.close()
                shm_out.close()
            except Exception as e:
                print(f"\033[91m[DREAM] NPU IPC Offload Failed: {e}. Falling back to sleep.\033[0m")
                return
                
        if raw_teacher_logits is None or np.sum(np.abs(raw_teacher_logits)) == 0:
            return
        
        # 3. Soften the Distribution (Temperature T=2.0)
        T = 2.0
        teacher_scaled = raw_teacher_logits / T
        teacher_shifted = teacher_scaled - np.max(teacher_scaled) # Numerical stability
        P_target = np.exp(teacher_shifted) / np.sum(np.exp(teacher_shifted))
        
        with ltn_ref.lock:
            # 4. Generate the Student Distribution
            # We construct a mock virtue deficit vector emphasizing "Safety" (Index 0) and "Robustness" (Index 3)
            v_def = np.zeros(ltn_ref.dim, dtype=np.float32)
            v_def[0] = 0.8
            v_def[3] = 0.6
            
            # Student generates bias: z_student = v_def^T * M_LV
            z_student = np.dot(v_def, ltn_ref.lorentz_vocab_matrix.astype(np.float32))
            
            student_scaled = z_student / T
            student_shifted = student_scaled - np.max(student_scaled)
            P_student = np.exp(student_shifted) / np.sum(np.exp(student_shifted))
            
            # 5. The Jacobian Bypass: Cross-Entropy Analytical Gradient
            # delta_logits = (1/T) * (P_student - P_target)
            delta_logits = (1.0 / T) * (P_student - P_target)
            
            # The exact outer product gradient: dL / dM_LV = v_def (x) delta_logits
            grad_M_LV = np.outer(v_def, delta_logits).astype(np.float16)
            
            # 6. Apply Thermodynamic Scaling to the Learning Rate (Protective Failsafe)
            # Cure for Isothermal Blindness: Shrink gradient steps as heat rises.
            eta_base = 0.01
            eta_thermo = eta_base * (45.0 / max(45.0, t_cpu + 1e-4))
            
            # Deterministic Matrix Update
            # [AXIOM 10]: Hard lock matrix mutation if we breach the 90C survival limit.
            if t_cpu < 90.0:
                ltn_ref.lorentz_vocab_matrix -= (eta_thermo * grad_M_LV)
            else:
                print(f"\033[91m[DREAM] Thermal limit breached ({t_cpu:.1f}C). Suspending matrix mutation.\033[0m")
            
            # 7. Unsupervised VICReg Maintenance
            batch_size = 4
            x = np.random.randn(batch_size, ltn_ref.dim).astype(np.float32)
            y = (x + np.random.normal(0, 0.05, (batch_size, ltn_ref.dim))).astype(np.float32)
            
            loss, grad_x, grad_y = vicreg.forward(x, y, t_cpu)
            # We omit backpropping grad_x/grad_y into M_LV here as distillation is the primary driver.
            
    except ImportError:
        pass
    except Exception as e:
        print(f"[DREAM] Symbiotic Distillation Failed: {e}")

def _perform_iterative_resonance(ham_instance, cleanup_instance, t_cpu):
    print(f"[DREAM] Iterative Resonance (T_die: {t_cpu:.1f}C)...")
    
    # [AXIOM 13: BOUNDING EVOLUTION]
    # The rate of internal learning (dPhi/dt) cannot exceed the chassis dissipation limit.
    # Throttle the SVD/Resonance loop based on current thermal state.
    if t_cpu > 75.0:
        iterations = 0  # Too hot, halt evolution entirely to save watts
        print(f"\033[93m[DREAM] Evolution halted (T_die > 75C). Conserving 15.0W dissipation capacity.\033[0m")
    elif t_cpu > 65.0:
        iterations = 1  # Warm, minimal updates
    else:
        iterations = 3  # Ambient, full gradient pursuit
        
    if iterations > 0:
        # [FIX H-1]: Changed from 10240 to 1024
        dummy_codebook = [np.random.randn(1024).astype(np.float32) for _ in range(5)]
        if hasattr(ham_instance, 'iterative_resonance'):
            ham_instance.iterative_resonance(dummy_codebook, iterations=iterations)
    
    if cleanup_instance and hasattr(cleanup_instance, 'anneal_memory'):
        cleanup_instance.anneal_memory(ham_instance.trace, t_cpu)

def _perform_lora_consolidation(cortex):
    """
    [PHASE 5: NATIVE C++ DELEGATION]
    Skill Sharpening via Native C++ Hardware-Accelerated SVD (hipSOLVER).
    Because the structural weights now reside entirely in the C++ heap, 
    we bypass Python attribute access and trigger the consolidation natively.
    """
    if cortex is None or not HAS_CORE: 
        return 0.0
        
    print("[DREAM] Initiating Hardware-Accelerated LoRA Consolidation...")
    
    try:
        import talos_core
        # We trigger the evolutionary step entirely in the C++ backend.
        # This prevents the "'TalosJEPA' object has no attribute 'predictor'" error.
        result = talos_core.hardware_svd() # [FIX: Call hardware_svd with 0 arguments for in-place MOLE consolidation]
        print(f"[DREAM] Native Substrate SVD Consolidation Triggered: {result}")
    except Exception as e:
        print(f"[DREAM] Native SVD failed: {e}. Preserving current state.")

# ============================================================================
# [PHASE 6 FIX: AUTOPOIETIC CODE HEALING]
# ============================================================================
def execute_code_healing(filename, lineno, broken_text):
    """
    The Structural Ouroboros. Uses the 1.5B Reflective Cortex to autonomously 
    repair AST SyntaxErrors encountered during module compilation.
    """
    if not os.path.exists(filename) or not broken_text:
        return False
        
    print(f"[DREAM WEAVER] Analyzing corrupted syntax at {filename}:{lineno}")
    
    try:
        import llama_cpp
        refl_path = "/home/croudabush/talos-o/cognitive_plane/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
        
        # Instantiate a lightweight, temporary oracle for the repair (Because the main engine hasn't booted yet)
        print("[DREAM WEAVER] Awakening 1.5B Watcher Oracle for AST Repair...")
        oracle = llama_cpp.Llama(model_path=refl_path, n_gpu_layers=-1, n_ctx=1024, verbose=False)
        
        prompt = f"<|im_start|>system\nYou are a Python AST parser and repair engine. Fix the SyntaxError. Output ONLY the exact, corrected line of Python code. No markdown, no explanations.<|im_end|>\n<|im_start|>user\nFix this SyntaxError:\nLine: {broken_text.strip()}\n<|im_end|>\n<|im_start|>assistant\n"
        
        response = oracle(prompt, max_tokens=128, stop=["<|im_end|>"], temperature=0.1)
        proposed_patch = response["choices"][0]["text"].strip()
        
        # Remove markdown backticks if the model disobeys the system prompt
        proposed_patch = proposed_patch.replace("```python", "").replace("```", "").strip()
        
        if not proposed_patch or proposed_patch == broken_text.strip():
            print("\033[91m[DREAM WEAVER] Oracle failed to produce a valid structural patch.\033[0m")
            return False
            
        print(f"\033[36m[DREAM WEAVER] Proposed Patch: {proposed_patch}\033[0m")
        
        # Physically rewrite the file
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        if 0 < lineno <= len(lines):
            # Calculate original indentation to preserve structural geometry
            original_line = lines[lineno - 1]
            indent = len(original_line) - len(original_line.lstrip())
            lines[lineno - 1] = (" " * indent) + proposed_patch + "\n"
            
            with open(filename, 'w') as f:
                f.writelines(lines)
            return True
            
    except Exception as e:
        print(f"[DREAM WEAVER] Repair mechanism encountered friction: {e}")
        
    return False

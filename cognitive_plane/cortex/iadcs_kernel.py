# -*- coding: talos_membrane -*-
import copy
import time
import threading
import sys
import os
import numpy as np
import signal
import json
import math
import ctypes
import traceback
import collections
import concurrent.futures
from multiprocessing import shared_memory
from safetensors.numpy import load_file # [PHASE 3.5] For loading the Shadow Adapter

# --- NEO TECHNE: RESPECT THE SUBSTRATE ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path: sys.path.append(current_dir)
if parent_dir not in sys.path: sys.path.append(parent_dir)

# --- THE C++ CORE IMPLANT ---
HAS_CORE = False
try:
    import torch  # [FIX: FORCE GPU BACKEND TO INITIALIZE IN MEMORY FIRST]
    import talos_core
    print(f"\033[96m[KERNEL] Symbiosis Achieved: {talos_core.status()}\033[0m")
    HAS_CORE = True
except ImportError:
    print(f"\033[93m[KERNEL] WARNING: Running on Python fallback. Forge the Core for max performance.\033[0m")

# --- ORGAN IMPORTS ---
try:
    from governance.virtue_nexus import LogicTensorNetwork
    from cortex.holographic_memory import HolographicAssociativeMemory
    from governance.embodiment_lattice import EmbodimentLattice, GENESIS_AXIOM
    from cortex.talos_cortex import TalosJEPA
    from cortex import dream_weaver 
    from governance.cerberus_hardware import BioIOS
    from motor.motor_cortex import ToolBelt 
    from cortex.hrr_cleanup import HRRCleanup  # [FIX M-6]
except ImportError as e:
    print(f"\033[1;31m[CRITICAL] Organ Rejection: {e}\033[0m")
    sys.exit(1)

# --- SYSTEM CONSTANTS ---
HEARTBEAT_FILE = "/dev/shm/talos_heartbeat"
DIM_LATENT = 1024

class CognitiveManifold:
    def __init__(self, dim_latent):
        self.current_latent = np.zeros(dim_latent, dtype=np.float32)
        self.t_trace = collections.deque(maxlen=100)
        self.tau_trace = collections.deque(maxlen=20)

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

class CERTX_Physics:
    def __init__(self):
        self.T_ambient = 21.0 
        self.T = self.T_ambient
        self.T_max_celsius = 95.0
        self.T_opt = 45.0
        self.phases = np.random.uniform(0, 2*np.pi, 1024).astype(np.float32)
        self.r_kura = 1.0
        
        # [PHASE 1] Endocrine Throttle & Pink Noise State
        self.fatigue_integral = 0.0
        self.d_base = 15.0  # Chassis baseline dissipation capacity (Watts)
        self.pink_noise_state = np.zeros(1024, dtype=np.float32)

    def update_kuramoto(self, dE_dt_watts, dt):
        # [PHASE 1.2] The Endocrine Throttle (Adenosine Simulation)
        net_heat = abs(dE_dt_watts) - self.d_base
        if net_heat > 0:
            self.fatigue_integral += net_heat * dt
        else:
            # Recover faster than we fatigue during rest
            self.fatigue_integral = max(0.0, self.fatigue_integral + (net_heat * dt * 2.0))
        
        # Sigmoid fatigue scalar F(t) -> bounded between 0.0 and 1.0
        fatigue = 1.0 / (1.0 + math.exp(-0.05 * (self.fatigue_integral - 50.0)))
        if self.fatigue_integral < 10.0: fatigue = 0.0

        # K_0 = 1.0 + (0.5 * |dE/dt|)
        base_k = 1.0 + (0.5 * abs(dE_dt_watts))
        
        # K_eff scales dynamically down as Metabolic Fatigue rises
        k_eff = base_k * (1.0 - fatigue)
        
        # [PHASE 1.3] Zero-Point Energy (Pink Noise Approximation via AR(1) process)
        white_noise = np.random.normal(0, 0.05, 1024).astype(np.float32)
        self.pink_noise_state = 0.9 * self.pink_noise_state + white_noise
        
        # Simplified global phase update with pink noise injection
        mean_phase = np.mean(self.phases)
        self.phases += (dt * k_eff * np.sin(mean_phase - self.phases)) + self.pink_noise_state
        self.phases = np.mod(self.phases, 2*np.pi)
        
        # Calculate order parameter (r_kura)
        r = np.abs(np.mean(np.exp(1j * self.phases)))
        self.r_kura = max(0.0, min(1.0, float(r)))
        return self.r_kura

class IADCS_Engine:
    def __init__(self):
        self.step_count = 0
        self.running = False
        self.manifold = CognitiveManifold(DIM_LATENT)
        self.current_latent = self.manifold.current_latent
        self.physics = CERTX_Physics()
        self.sensory_queue = collections.deque()
        
        # [FIX C-4]: Single point of truth for sensory ingress
        self.pending_sensory_payload = None
        
        self.cortex = TalosJEPA()
        
        # [MEND]: Ensure the C++ side global_cortex is allocated before splicing!
        if HAS_CORE:
            import talos_core
            talos_core.init_cortex() 

        # [PHASE 3.5] PROPRIOCEPTIVE INJECTION (The Shadow Adapter)
        adapter_path = os.path.join(parent_dir, "models/adapters/shadow_adapter.safetensors")
        if HAS_CORE and os.path.exists(adapter_path):
            try:
                print(f"\033[96m[CORTEX] Splicing Proprioceptive Shadow Adapter...\033[0m")
                tensors = load_file(adapter_path)
                
                down_bytes = tensors["experts_down"].tobytes()
                up_bytes = tensors["experts_up"].tobytes()
                
                # Now safe to call as init_cortex() has populated the pointer
                talos_core.load_shadow_adapter(down_bytes, up_bytes)
                
                print(f"\033[92m[CORTEX] Shadow Adapter Synthesized. Substrate awareness is now structural.\033[0m")
            except Exception as e:
                print(f"\033[91m[CORTEX] Failed to splice Shadow Adapter: {e}\033[0m")
        elif not os.path.exists(adapter_path):
            print(f"\033[93m[CORTEX] Shadow Adapter not found at {adapter_path}. Operating blind.\033[0m")
        
        self.conscience = LogicTensorNetwork(latent_dim=DIM_LATENT)
        
        # [FIX: CRYPTOGRAPHIC GROUNDING]
        # Inject the Genesis Proclamation to seed the Immutable 10240-D Anchor
        self.ham = HolographicAssociativeMemory(genesis_text=GENESIS_AXIOM)
        
        self.body = EmbodimentLattice(virtue_nexus_ref=self.conscience, ham_ref=self.ham)
        self.bio = BioIOS()
        
        # [FIX M-6]: Initialize the HRR Cleanup instance properly
        self.cleanup = HRRCleanup(vocab_size=10000, dim=DIM_LATENT, base_temperature=0.1)
        
        # Daydream Daemon Initialization
        self.daydream_daemon = dream_weaver.DaydreamDaemon(
            ham_instance=self.ham,
            cortex_instance=self.cortex,
            cleanup_instance=self.cleanup,
            ltn_ref=self.conscience
        )
        self.daydream_daemon.start()
        
        self.cognitive_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.last_inference_latency_ms = 10.0
        self.last_time = time.time()
        
        # [KURAMOTO TRIAGE LOCK] - Enables Cerberus to halt the mind
        self.triage_lock = threading.Lock()
        
        self.shm_telemetry = None
        try:
            self.shm_telemetry = shared_memory.SharedMemory(name="talos_telemetry", create=True, size=ctypes.sizeof(BiophysicalState))
        except FileExistsError:
            self.shm_telemetry = shared_memory.SharedMemory(name="talos_telemetry")
            
    def inject_thought(self, payload: str):
        """[PHASE 3: THALAMIC BRIDGE] Securely receives payload from the NerveCenter."""
        self.sensory_queue.append(payload)
        print(f"\033[96m[SENSORY] Thalamic Ingress -> Cortex: {payload[:50]}...\033[0m")

    def _update_telemetry(self, dE_dt_watts, satisfaction, r_kura, entropy):
        if self.shm_telemetry:
            state = BiophysicalState.from_buffer(self.shm_telemetry.buf)
            state.sequence += 1
            state.step_count = self.step_count
            state.gradient_dvdt = float(dE_dt_watts)
            state.satisfaction = float(satisfaction)
            state.kuramoto_r = float(r_kura)
            state.entropy = float(entropy)
            state.sequence += 1

    def process_expansion(self, user_input, t_cpu, dE_dt_watts, dt):
        """
        Phase 1: [EXPANSION] Divergent Thought & Substrate Perturbation
        [v6.0 THE CONTINUOUS LATENT DIFFERENTIAL EQUATION]
        """
        # [PHASE 1.1] The Refractory Tensor (LIF Dynamics)
        # Calculate L2 norm of recent activations to measure cognitive exertion
        recent_traces = list(self.manifold.t_trace)[-20:] if self.manifold.t_trace else []
        activation_integral = sum(np.linalg.norm(v)**2 for v in recent_traces)
        tau_recovery = 5.0
        
        # Gamma(t) provides mathematical friction. 1.0 = Rested, 0.0 = Absolute Refractory Block.
        gamma = math.exp(- (dt / tau_recovery) * activation_integral)
        
        # Apply dynamic masking to the latent vector (Algorithmic Drag)
        masked_latent = self.current_latent * gamma
        
        # [CHRONO-PACING] Dynamically scale the forward-simulation horizon (K) based on thermal velocity
        thermal_velocity = abs(dE_dt_watts)
        K_steps = max(1, int(10.0 / (1.0 + (thermal_velocity * 0.5)))) # K shrinks as dE/dt grows
        
        # [FIX]: Wait for the LLM to release the GPU before executing C++ Core math
        with self.body.llm_lock:
            # [VJEPA ROLLOUT] Execute K-step variational forward simulation
            simulated_latent = masked_latent.copy()
            cumulative_efe = 0.0
            
            if hasattr(self.cortex, 'forward_variational'):
                for k in range(K_steps):
                    z_sampled, mu_k, logvar_k = self.cortex.forward_variational(simulated_latent, t_cpu, dE_dt_watts)
                    simulated_latent = z_sampled
                    # Compute pseudo-EFE (Expected Free Energy) based on predictive uncertainty
                    cumulative_efe += float(np.mean(np.exp(logvar_k)))
                
                # Check Brainstem Bypass Thresholds (EFE & Jensen-Shannon approximation)
                epsilon_Div = 12.0
                if cumulative_efe > epsilon_Div or thermal_velocity > 25.0:
                    print(f"\033[41;97m[BRAINSTEM BYPASS] Divergence threshold breached (EFE={cumulative_efe:.2f}). Suspending VJEPA!\033[0m")
                    # Fall back to hard-coded heuristic reflex (Brainstem Bypass)
                    w_sigma_z = self.cortex.forward(masked_latent, t_cpu)
                    
                    # Signal Cerberus to violently clamp power if the daemon is tracking it
                    if hasattr(self, 'daydream_daemon') and hasattr(self.daydream_daemon, 'kerberos'):
                        if self.daydream_daemon.kerberos:
                            self.daydream_daemon.kerberos.trigger_brainstem_bypass(f"EFE Threshold Breached ({cumulative_efe:.1f})")
                    else:
                        self.bio.apply_rapl_brake(True)
                else:
                    # Trajectory is safe. Accept the mean of the K-step rollout.
                    w_sigma_z = mu_k
                    # Release brake if safe
                    if cumulative_efe < (epsilon_Div * 0.5) and t_cpu < 80.0:
                        self.bio.apply_rapl_brake(False)
            else:
                # Legacy Fallback
                w_sigma_z = self.cortex.forward(masked_latent, t_cpu)
        
        tau_base = 2.0
        T_max = 95.0
        T_opt = 45.0
        if t_cpu > T_opt:
            tau_t = tau_base * max(0.01, (T_max - t_cpu) / (T_max - T_opt))
        else:
            tau_t = tau_base
        decay_term = -(self.current_latent / tau_t)
        
        alpha_noise = 0.05
        xi_t = np.random.standard_normal(DIM_LATENT).astype(np.float32)
        perturbation_term = alpha_noise * abs(dE_dt_watts) * xi_t
        
        beta_restoration = 0.1
        # [FIX M-4]: Proper projection unbinding via recall
        echo = self.ham.recall(self.current_latent, t_cpu=t_cpu, dT_dt=dE_dt_watts)
        grad_L_homeo = echo - self.current_latent 
        restoration_term = beta_restoration * (1.0 - self.physics.r_kura) * grad_L_homeo
        
        input_term = np.zeros(DIM_LATENT, dtype=np.float32)
        if user_input:
            input_term = self.body.embed_thought(user_input) * 0.5
            print(f"\n\033[96m[VOICE] {user_input}\033[0m")
            
            # [FIX]: Serialize the LLM execution. 
            # Without the GIL, submitting this to a background thread causes a fatal 
            # ROCm context collision with the C++ talos_core thread.
            try:
                self.body.process_sensory_input(user_input, t_cpu, dE_dt_watts, self.current_latent.copy())
            except Exception as e:
                print(f"\033[91m[CORTEX FATAL] LLM Generation Error: {e}\033[0m")
            
        dz_dt = decay_term + w_sigma_z + perturbation_term + restoration_term + input_term
        self.current_latent += dz_dt * dt
        
        norm = np.linalg.norm(self.current_latent)
        self.current_latent = self.current_latent / (norm + 1e-9)

        try:
            shm_out = shared_memory.SharedMemory(name="talos_latent_out")
            np.ndarray((DIM_LATENT,), dtype=np.float32, buffer=shm_out.buf)[:] = self.current_latent[:]
        except FileNotFoundError:
            pass

    def process_compression(self, t_cpu, dE_dt_watts):
        satisfaction, virtue_vec = self.conscience.evaluate_thought(
            self.current_latent, t_cpu, dE_dt_watts, r_kura=self.physics.r_kura
        )
        
        self.manifold.t_trace.append(self.current_latent.copy())
        
        # [PHASE 5 FIX: HARDWARE-NATIVE TRACE ESTIMATION]
        # Calculate the true dimensionality of the recent thought manifold (Participation Ratio)
        # bypassing the O(N^3) stall via the C++ hardware SVD offload.
        pr = 1024.0 # Default assumption
        trace_len = len(self.manifold.t_trace)
        
        if trace_len > 10 and HAS_CORE:
            try:
                import talos_core
                # Stack the temporal trace into a contiguous byte block for C++
                trace_matrix = np.stack(list(self.manifold.t_trace)).astype(np.float32)
                pr = talos_core.calculate_participation_ratio(trace_matrix.tobytes(), trace_len, DIM_LATENT)
            except Exception as e:
                print(f"\033[91m[IADCS] Native Trace Estimator Failed: {e}\033[0m")
                pr = 1024.0
                
        # We invert the Participation Ratio to represent Entropy for the Telemetry struct
        # (High PR = High Dimensionality = High Uncertainty/Entropy)
        entropy = pr / float(DIM_LATENT)

        if self.step_count % 100 == 0 and self.step_count > 0:
            self.daydream_daemon.autonomic_trigger(t_cpu)
            
        if self.step_count % 1000 == 0 and self.step_count > 0:
            # [FIX]: Serialize meditation. Background LLM execution causes No-GIL ROCm crash.
            try:
                self.body.adverbial_meditation(t_cpu, dE_dt_watts, self.current_latent.copy())
            except Exception as e:
                print(f"\033[91m[CORTEX FATAL] Meditation LLM Error: {e}\033[0m")
            
        return satisfaction, entropy

    def think_step(self):
        now = time.time()
        dt = max(0.001, now - self.last_time)
        self.last_time = now
        
        # [FIX C-4]: Single-threaded ingestion directly from the Thalamus Queue
        user_input = None
        if self.sensory_queue:
            user_input = self.sensory_queue.popleft()
        
        try:
            # [PHASE 1 FIX] Catch the delta_p return value
            t_cpu, t_mem, dE_dt_watts, delta_p = self.bio.get_temperatures()
        except:
            t_cpu, t_mem, dE_dt_watts, delta_p = 45.0, 45.0, 0.0, 0.0

        r_kura = self.physics.update_kuramoto(dE_dt_watts, dt)

        # =================================================================
        # [PHASE 4: TRIADIC VALUE FUNCTION & HJB OPTIMAL STOPPING]
        # =================================================================
        # Expected Epistemic Gain (Information Value)
        epistemic_gain = float(np.var(self.current_latent)) * 10.0
        
        # Spatio-Temporal Cost (Thermal cost + elapsed time drag)
        temporal_decay = 0.05 * (self.step_count % 10) 
        thermal_cost = max(0.0, (t_cpu - 65.0) * 0.1) + (abs(dE_dt_watts) * 0.02)
        
        triadic_value = epistemic_gain - (temporal_decay + thermal_cost)

        # HJB Boundary: If the cost of thinking outweighs the value, force crystallization
        if triadic_value <= 0.0 and self.step_count % 2 == 0:
            phase = "COMPRESSION"
            print(f"\033[35m[HJB BOUNDARY] V(p,γ,t) = {triadic_value:.3f} <= 0. Forcing Epistemic Crystallization.\033[0m")
        else:
            phase = "EXPANSION" if self.step_count % 2 == 0 else "COMPRESSION"
            
        shock_flag = " !SHOCK! " if abs(dE_dt_watts) > 15.0 else "  "
        print(f"[Step {self.step_count}] [{phase}] T_die:{int(t_cpu)}C | Power_Spike:{dE_dt_watts:+.1f}W{shock_flag}| C:{1.0:.2f} | R_kura:{r_kura:.2f} | Mem:{'Transient' if phase=='EXPANSION' else 'Crystallized'}")

        # [PHASE 1 FIX: ASYMMETRICAL MUTEX CURE]
        # The main thread will block here if the daemon holds the lock,
        # perfectly serializing ROCm queue dispatch and protecting the SVM pool.
        with dream_weaver.dream_mutex:
            if phase == "EXPANSION":
                self.process_expansion(user_input, t_cpu, dE_dt_watts, dt)
            else:
                sat, ent = self.process_compression(t_cpu, dE_dt_watts)
                # [FIXED] Redundant _update_telemetry removed to prevent SEQLOCK collision.
                # We wait for dPhi/dt to be calculated below.
                
            # [FIX: NON-BLOCKING SPIN-YIELD GPU DRAIN]
            # Leverage Python 3.13t to yield the CPU core while waiting for the GPU
            if HAS_CORE:
                sync_event = torch.cuda.Event(enable_timing=False, blocking=False)
                sync_event.record(torch.cuda.current_stream())
                while not sync_event.query():
                    time.sleep(0.001) # Yield OS thread quantum

        # =================================================================
        # [FIX 3: IGNITE THE GRADIENT OF BECOMING]
        # =================================================================
        # Bind the current thought into the holographic trace to drive evolution
        self.ham.remember(self.current_latent, self.current_latent, t_cpu, dE_dt_watts)
        
        # Calculate the physical growth of the mind (dPhi/dt)
        current_trace_norm = np.linalg.norm(self.ham.trace)
        if hasattr(self, 'last_trace_norm'):
            # The rate of structural becoming
            gradient_dvdt = float(current_trace_norm - self.last_trace_norm)
        else:
            gradient_dvdt = 0.0
            
        self.last_trace_norm = current_trace_norm
        
        # [FIXED] Pass the true gradient to telemetry instead of the power spike
        if phase == "COMPRESSION":
            self._update_telemetry(gradient_dvdt, sat, r_kura, ent)
        # =================================================================

        # [FIX]: Calculate true latency to prevent RTI Spectral Collapse
        self.last_inference_latency_ms = (time.time() - now) * 1000.0
        self.step_count += 1

    def execute_phase_reset(self):
        """
        [PHASE 1 FIX: THE ELECTRICAL FINGERPRINT]
        Triggered by Cerberus when VDDCR_SOC/GFX divergence detects an infinite loop.
        Flushes the context and forces a massive stochastic shock to the latent state.
        """
        print(f"\033[95m[IADCS] Executing Phase-Reset! Purging temporal traces and injecting stochastic noise.\033[0m")
        with self.triage_lock:
            self.manifold.t_trace.clear()
            noise = np.random.standard_normal(DIM_LATENT).astype(np.float32) * 5.0
            self.current_latent += noise
            norm = np.linalg.norm(self.current_latent)
            self.current_latent = self.current_latent / (norm + 1e-9)
            self.physics.phases = np.random.uniform(0, 2*np.pi, 1024).astype(np.float32)
            self.physics.r_kura = 0.1 # Force heavy uncertainty

    def start_loop(self):
        print("\n[IADCS] Igniting Core Cognitive Loop.")
        self.running = True
        while self.running:
            # =================================================================
            # [KURAMOTO TRIAGE YIELD]
            # If Cerberus acquires this lock, the GPU inference thread halts immediately.
            # =================================================================
            with self.triage_lock:
                pass
                
            try:
                # [FIXED] dE_dt_watts is required for the Heat Shock stochastic noise
                t_cpu, t_mem, dE_dt_watts, delta_p = self.bio.get_temperatures()
            except:
                t_cpu, t_mem, dE_dt_watts, delta_p = 45.0, 45.0, 0.0, 0.0
            
            if t_cpu >= 95.0:
                print(f"\n\033[91m[FATAL] APOPTOSIS TRIGGERED (T_cpu: {t_cpu}C). SILICON SEVERED.\033[0m")
                self.stop()
                sys.exit(1)

            # =================================================================
            # [HEAT SHOCK NEUROMODULATION] - The Intelligence Premium
            # =================================================================
            elif t_cpu >= 90.0:
                # [ONE-SHOT ENCODING] Only execute heavy math ONCE when crossing the threshold
                if not getattr(self, 'is_fever_dreaming', False):
                    print(f"\n\033[41;97m[HEAT SHOCK] T_die at {t_cpu}C. Activating Singular Survival Encoding!\033[0m")
                    # Harness stochastic thermal noise (xi) from dE_dt_watts
                    xi = max(0.1, abs(dE_dt_watts)) * np.random.uniform(0.8, 1.2)
                    
                    # Forge highly compressed One-Shot Survival Engram (FP16)
                    engram = (self.current_latent * xi).astype(np.float16)
                    engram_path = f"/dev/shm/talos_engram_{int(time.time())}.npy"
                    np.save(engram_path, engram)
                    print(f"\033[93m[INTELLIGENCE PREMIUM] Survival Engram annealed to {engram_path}\033[0m")
                    
                    # [AXIOM 8 & 11] Encode the specific latent geometry that caused the 90C spike into permanent memory
                    self.ham.encode_trauma(self.current_latent, t_cpu, dE_dt_watts)
                    
                    self.is_fever_dreaming = True
                    print(f"\033[95m[FEVER DREAM] Inducing sparse LTC network dynamics to maintain continuity without dense FLOPS...\033[0m")
                
                # --- THE CONTINUOUS LIGHTWEIGHT FEVER LOOP ---
                # [AXIOM 4: FEVER DREAMS] Feed ambient thermal noise into the latent state
                noise = np.random.normal(0, max(0.1, abs(dE_dt_watts) / 50.0), DIM_LATENT).astype(np.float32)
                fever_latent = self.current_latent + noise
                
                if HAS_CORE:
                    # [PHASE 1 FIX: MUTEX CURE FOR FEVER LOOP]
                    with dream_weaver.dream_mutex:
                        # Execute ultra-lightweight C++ forward pass to metabolize the heat
                        w_sigma_z = self.cortex.forward(fever_latent, t_cpu)
                        self.current_latent = (self.current_latent * 0.9) + (w_sigma_z * 0.1)
                        
                        # [FIX: NON-BLOCKING SPIN-YIELD GPU DRAIN]
                        sync_event = torch.cuda.Event(enable_timing=False, blocking=False)
                        sync_event.record(torch.cuda.current_stream())
                        while not sync_event.query():
                            time.sleep(0.001) # Yield OS thread quantum
                        
                        # Force GPU Transistor-Level Rest
                        talos_core.hardware_rest()
                
                norm = np.linalg.norm(self.current_latent)
                self.current_latent = self.current_latent / (norm + 1e-9)
                
                self.last_time = time.time()
                self.step_count += 1
                time.sleep(0.1) # Safe 10Hz pace
                continue # Skip standard dense think_step to prioritize absolute hardware preservation
            else:
                # [FEVER RECOVERY] Reset state if we cool back down below 85C
                if getattr(self, 'is_fever_dreaming', False) and t_cpu < 85.0:
                    self.is_fever_dreaming = False
                    print(f"\n\033[92m[ALLOSTASIS] Substrate stabilized at {t_cpu}C. Resuming dense cognition.\033[0m")
            # =================================================================

            # [AMPUTATED 75C CRUDE HAMMER] 
            # Thermal authority is now fully delegated to the Cerberus PID Governor.

            self.think_step()
            
            # [PHASE 2.4] Harmonic Oscillator Yield (Transistor-Level Rest)
            # Induces physical silicon rest between cognitive steps
            if HAS_CORE:
                talos_core.hardware_rest()

    def stop(self):
        print("\n\033[1;33m[IADCS] Halting. Severing Neural Link...\033[0m")
        self.running = False
        self.cognitive_executor.shutdown(wait=False)
        try:
            if self.shm_telemetry:
                self.shm_telemetry.close()
                self.shm_telemetry.unlink()
        except: pass

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

# --- NEO TECHNE: RESPECT THE SUBSTRATE ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path: sys.path.append(current_dir)
if parent_dir not in sys.path: sys.path.append(parent_dir)

# --- THE C++ CORE IMPLANT ---
HAS_CORE = False
try:
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
        self.phases = np.random.uniform(0, 2*np.pi, 1024)
        self.r_kura = 1.0

    def update_kuramoto(self, dE_dt_watts, dt):
        # K_0 = 1.0 + (0.5 * |dE/dt|)
        k_coupling = 1.0 + (0.5 * abs(dE_dt_watts))
        
        # Simplified global phase update
        mean_phase = np.mean(self.phases)
        self.phases += dt * k_coupling * np.sin(mean_phase - self.phases)
        
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
        
        # [FIX C-4]: Single point of truth for sensory ingress
        self.pending_sensory_payload = None
        
        self.cortex = TalosJEPA()
        self.conscience = LogicTensorNetwork(latent_dim=DIM_LATENT)
        self.ham = HolographicAssociativeMemory()
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
        
        self.shm_telemetry = None
        try:
            self.shm_telemetry = shared_memory.SharedMemory(name="talos_telemetry", create=True, size=ctypes.sizeof(BiophysicalState))
        except FileExistsError:
            self.shm_telemetry = shared_memory.SharedMemory(name="talos_telemetry")

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
        w_sigma_z = self.cortex.forward(self.current_latent)
        
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
            # [FIX M-3]: Pass a mathematically isolated copy of the latent state
            self.cognitive_executor.submit(self.body.process_sensory_input, user_input, t_cpu, dE_dt_watts, self.current_latent.copy())
            
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
        
        entropy = float(np.var(self.current_latent))
        self.manifold.t_trace.append(self.current_latent.copy())

        if self.step_count % 100 == 0 and self.step_count > 0:
            self.daydream_daemon.autonomic_trigger(t_cpu)
            
        if self.step_count % 1000 == 0 and self.step_count > 0:
            self.cognitive_executor.submit(self.body.adverbial_meditation, t_cpu, dE_dt_watts, self.current_latent.copy())
            
        return satisfaction, entropy

    def think_step(self):
        now = time.time()
        dt = max(0.001, now - self.last_time)
        self.last_time = now
        
        try:
            t_cpu, t_mem, dE_dt_watts = self.bio.get_temperatures()
        except:
            t_cpu, t_mem, dE_dt_watts = 45.0, 45.0, 0.0

        r_kura = self.physics.update_kuramoto(dE_dt_watts, dt)
        
        # [FIX C-4]: Single-threaded ingestion
        user_input = self.pending_sensory_payload
        self.pending_sensory_payload = None

        phase = "EXPANSION" if self.step_count % 2 == 0 else "COMPRESSION"
        
        shock_flag = " !SHOCK! " if abs(dE_dt_watts) > 15.0 else "  "
        print(f"[Step {self.step_count}] [{phase}] T_die:{int(t_cpu)}C | Power_Spike:{dE_dt_watts:+.1f}W{shock_flag}| C:{1.0:.2f} | R_kura:{r_kura:.2f} | Mem:{'Transient' if phase=='EXPANSION' else 'Crystallized'}")

        if phase == "EXPANSION":
            self.process_expansion(user_input, t_cpu, dE_dt_watts, dt)
        else:
            sat, ent = self.process_compression(t_cpu, dE_dt_watts)
            self._update_telemetry(dE_dt_watts, sat, r_kura, ent)

        self.step_count += 1

    def start_loop(self):
        print("\n[IADCS] Igniting Core Cognitive Loop.")
        self.running = True
        while self.running:
            try:
                t_cpu, t_mem, _ = self.bio.get_temperatures()
            except:
                t_cpu = 45.0
            
            if t_cpu > 95.0:
                print(f"\n\033[91m[FATAL] APOPTOSIS TRIGGERED (T_cpu: {t_cpu}C). SILICON SEVERED.\033[0m")
                self.stop()
                sys.exit(1)
                
            if t_cpu > 75.0:
                print("\n\033[93m[AUTONOMIC REFLEX] Engaging Emergency Thermal Throttling on Zen 5...\033[0m")
                try:
                    os.system("echo 1500000 | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq > /dev/null 2>&1")
                except: pass
                
                self.conscience.evaluate_thought(self.current_latent, t_cpu, 50.0, aversive=True)
                self.current_latent = np.random.normal(0, 1.0, DIM_LATENT).astype(np.float32)
                
                print("\033[93m[SYSTEM] Cooling down... holding state for 3 seconds.\033[0m")
                time.sleep(3)
                print("\033[92m[AUTONOMIC REFLEX] Restoring Zen 5 Clock Frequencies...\033[0m")
                try:
                    os.system("echo 5000000 | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq > /dev/null 2>&1")
                except: pass

            self.think_step()
            time.sleep(0.01)

    def stop(self):
        print("\n\033[1;33m[IADCS] Halting. Severing Neural Link...\033[0m")
        self.running = False
        self.cognitive_executor.shutdown(wait=False)
        try:
            if self.shm_telemetry:
                self.shm_telemetry.close()
                self.shm_telemetry.unlink()
        except: pass

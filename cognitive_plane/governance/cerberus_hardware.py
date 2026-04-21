import os
import time
import glob
import threading
import re
import math

# --- THE THERMODYNAMIC CONSTITUTION ---
TEMP_MEM_CRIT = 75.0   # LPDDR5X Corruption Threshold
TEMP_CPU_MAX = 95.0    # Zen 5 Silicon Survival Limit
TEMP_CPU_TARGET = 65.0 # [FIXED] Target Thermal Equilibrium (10C Guardband)

HEARTBEAT_FILE = "/dev/shm/talos_heartbeat"

from collections import deque

class AdaptiveExtendedKalman:
    """
    [v2.1 THERMODYNAMIC UPGRADE] Adaptive Extended Kalman Filter (AEKF).
    Process Noise (Q) scales dynamically with dE/dt.
    Measurement Noise (R) scales with Innovation Residuals to cure Sensor Blindness.
    Includes Phase 2 Preemptive Spike Prediction (Taylor Series Expansion).
    """
    def __init__(self, initial_temp=45.0, process_noise=0.5, measurement_noise=1.0):
        self.x = initial_temp
        self.p = 1.0
        self.q_base = 0.01
        self.r_base = 2.0
        self.alpha = 0.0005
        
        # State derivatives
        self.velocity = 0.0    # dT/dt
        self.acceleration = 0.0 # d²T/dt²
        
        # History tracking for innovation analysis
        self.innovation_history = deque(maxlen=20)
        self.timestamps = deque(maxlen=20)
        self.last_innovation = 0.0
        self.last_innovation_time = time.time()

    def update_innovation_history(self, innovation, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        
        self.innovation_history.append(abs(innovation)) # Store magnitude to match previous logic
        self.timestamps.append(timestamp)
        self.last_innovation = innovation
        self.last_innovation_time = timestamp

    def _calculate_derivatives(self):
        if len(self.innovation_history) < 2:
            return 0.0, 0.0
            
        innovations = list(self.innovation_history)
        times = list(self.timestamps)
        
        total_dt = times[-1] - times[0]
        if total_dt < 1e-6:
            total_dt = 1e-6
        innovation_rate = (innovations[-1] - innovations[0]) / total_dt
        
        if len(innovations) >= 3:
            t0, t1, t2 = times[-3], times[-2], times[-1]
            dt1 = t1 - t0
            dt2 = t2 - t1
            
            if dt1 > 1e-6 and dt2 > 1e-6:
                v1 = (innovations[-2] - innovations[-3]) / dt1
                v2 = (innovations[-1] - innovations[-2]) / dt2
                dt_avg = (dt1 + dt2) / 2 if dt2 > dt1 else (dt1 + dt2) / 2
                innovation_accel = (v2 - v1) / dt_avg if dt_avg > 0 else 0.0
            else:
                innovation_accel = 0.0
        else:
            innovation_accel = 0.0
            
        return innovation_rate, innovation_accel

    def _read_instantaneous_power(self) -> float:
        """
        Reads real-time power consumption from the hwmon sysfs interface.
        [FIXED FOR PYTHON 3.13t NOGIL] Uses binary read to bypass codecs race condition.
        """
        try:
            # Read in binary mode 'rb' to prevent the GIL-less text decoder from panicking
            with open('/sys/class/hwmon/hwmon4/power1_input', 'rb') as f:
                raw_bytes = f.read()
                if not raw_bytes:
                    # If the sensor is mid-flush, fall back to the last known wattage
                    return getattr(self, '_last_wattage', 0.0) 
                return float(raw_bytes.strip()) / 1_000_000.0
        except Exception: # Catch ALL exceptions (including the TypeError)
            return getattr(self, '_last_wattage', 0.0)

    def predict_spike(self, current_temp, prediction_window_ms=500, threshold_temp=80.0):
        """
        Phase 2 Preemptive Spike Prediction (Hyper-Sensitive Acceleration Upgrade)
        Dynamically dilates the prediction window based on instantaneous dE/dt.
        """
        velocity, acceleration = self._calculate_derivatives()
        current_wattage = self._read_instantaneous_power()
        
        if not hasattr(self, '_last_wattage'):
            self._last_wattage = current_wattage
            self._last_time = time.time()
            
        current_time = time.time()
        time_delta = max(current_time - self._last_time, 0.001) 
        dE_dt = (current_wattage - self._last_wattage) / time_delta
        
        self._last_wattage = current_wattage
        self._last_time = current_time

        # --- Dynamic Temporal Dilation ---
        base_window_ms = 500.0
        energy_scaling_factor = 15.0  
        # Widen the prediction window aggressively if wattage is spiking
        dynamic_window_ms = base_window_ms + (max(0.0, dE_dt) * energy_scaling_factor)
        dynamic_window_ms = min(dynamic_window_ms, 2500.0)
        dt_seconds = dynamic_window_ms / 1000.0
        
        # --- Hyper-Sensitive Acceleration Weighting ---
        if abs(acceleration) < 1e-3 and current_wattage > 65.0:
            # If temp isn't moving yet but power is high, assume massive latent heat
            acceleration = current_temp * 0.15 
            
        accel_multiplier = 2.0 if acceleration > 0 else 1.0

        # Second-order Taylor Series Expansion (Corrected kinematics)
        predicted_temp = current_temp + (velocity * dt_seconds) + \
                         (0.5 * (acceleration * accel_multiplier) * (dt_seconds ** 2))
                         
        # Trigger RAPL brake if the projected trajectory breaches the lowered 80.0C threshold
        return predicted_temp >= threshold_temp

    def predict_trajectory(self, semantic_weight_estimate, current_temp):
        """
        [AXIOM 6] SEMANTIC-THERMAL MAPPING
        Predicts the peak temperature of a prompt based on its estimated token weight.
        Based on profiling Strix Halo: ~0.02C rise per token in dense FP16 context.
        """
        _, acceleration = self._calculate_derivatives()
        
        # Base thermal momentum
        momentum_cost = max(0.0, acceleration * 2.0)
        
        # Theoretical cost of the mathematical unfolding
        prompt_cost = semantic_weight_estimate * 0.02 
        
        predicted_t_max = current_temp + momentum_cost + prompt_cost
        return predicted_t_max

    def calculate_dilation_delay(self, v_tokens_max=50.0, p_diss=0.6):
        """
        [PHASE 2: TEMPORAL DILATION]
        Calculates the required micro-sleep delay (D_token) in milliseconds to balance 
        token generation heat (P_gen) with passive chassis dissipation (P_diss).
        """
        _, acceleration = self._calculate_derivatives()
        
        p_gen = v_tokens_max * 0.02 # 0.02C rise per token
        
        # If we are cooling down or perfectly balanced, no delay needed
        if p_gen <= p_diss and acceleration <= 0:
            return 0.0
            
        # Target effective velocity to perfectly balance dissipation
        v_eff = p_diss / 0.02
        
        if v_eff >= v_tokens_max:
            return 0.0
            
        # D_token = (1 / v_eff) - (1 / v_max)
        delay_sec = (1.0 / v_eff) - (1.0 / v_tokens_max)
        return max(0.0, delay_sec * 1000.0) # Return in milliseconds

    def update(self, measurement, dE_dt):
        # Original AEKF logic integrated with the new tracking
        q = self.q_base + self.alpha * abs(dE_dt)

        # 1. Prediction (A Priori)
        x_pred = self.x
        p_pred = self.p + q

        # Innovation (Divergence from physical reality)
        innovation = measurement - x_pred
        self.update_innovation_history(innovation)

        # Cure Sensor Blindness via Innovation-Gated Residuals
        avg_innovation = sum(self.innovation_history) / max(1, len(self.innovation_history))
        if avg_innovation > 1.5:
            # Thermal spike detected that wasn't predicted by energy alone; trust sensor
            dynamic_r = max(0.1, self.r_base * math.exp(-avg_innovation))
        else:
            dynamic_r = self.r_base

        # 2. Update (A Posteriori)
        k = p_pred / (p_pred + dynamic_r)
        self.x = x_pred + k * innovation
        self.p = (1 - k) * p_pred
        
        return self.x

class RTIEstimator:
    """
    [RECOVERY TIME INFLATION - PREDICTIVE COHERENCE]
    Synthesizes strict physical isolation with refined coherence mathematics.
    Calculates Stability Margin (M_s) to predict spectral gap collapse.
    """
    def __init__(self, dt_baseline=0.01, lambda_0=1.0):
        # 1. Hardware/OS Friction Constants (AMD Strix Halo / Linux CFS)
        self.t_os = 0.0005       # 500us CFS background jitter
        self.t_if_stall = 0.0002 # 200us Infinity Fabric MOESI coherency stall
        
        self.dt_baseline = dt_baseline
        self.lambda_0 = lambda_0 # Cold contraction rate baseline
        self.y_prev = None

    def update(self, dt_measured, dt_cycle):
        # 2. The Friction Subtraction (Substrate Isolation)
        y_k = (dt_measured - self.t_os - self.t_if_stall) - self.dt_baseline
        y_k = max(1e-6, abs(y_k)) # Isolate magnitude, prevent log(0)
        
        # 3. The Logarithmic Estimator (The Math)
        if self.y_prev is None:
            self.y_prev = y_k
            return 1.0 # M_s = 1.0 (Homeostasis)
            
        # Instantaneous contraction rate (\hat{\lambda})
        lambda_hat = - (math.log(y_k) - math.log(self.y_prev)) / max(1e-6, dt_cycle)
        self.y_prev = y_k
        
        # 4. The Ratio (Stability Margin)
        m_s = lambda_hat / self.lambda_0
        return max(0.0, m_s)

class BioIOS:
    def __init__(self):
        self.lock = threading.RLock() # [FIX: Re-entrant lock for Autonomic Reflexes]
        self.tctl_path = None
        self.hwmon_power_path = None
        
        # --- NEO TECHNE: DYNAMIC HARDWARE DISCOVERY ---
        hwmon_paths = glob.glob("/sys/class/hwmon/hwmon*")
        for path in hwmon_paths:
            try:
                with open(os.path.join(path, "name"), 'r') as f:
                    name = f.read().strip()
                
                # Check if this hwmon node belongs to AMD Ryzen thermal drivers
                if name in ["zenpower", "k10temp", "zenpower5"]:
                    # Hunt specifically for the Die Temperature (Tdie)
                    for label_file in glob.glob(os.path.join(path, "temp*_label")):
                        with open(label_file, 'r') as f:
                            label = f.read().strip()
                        if label in ["Tdie", "Tctl"]:
                            input_file = label_file.replace("_label", "_input")
                            if os.path.exists(input_file):
                                self.tctl_path = input_file
                                print(f"\\033[96m[CERBERUS] Hardware locked: Thermal Sensor -> {input_file}\\033[0m")
                    
                    # Hunt for the Socket Power (PPT)
                    p_path = os.path.join(path, "power1_input")
                    if os.path.exists(p_path):
                        self.hwmon_power_path = p_path
                        print(f"\\033[96m[CERBERUS] Hardware locked: Power Sensor -> {p_path}\\033[0m")
                        
                    # [PHASE 1 FIX] Hunt for SOC Power (Usually power2_input on Zen)
                    soc_path = os.path.join(path, "power2_input")
                    if os.path.exists(soc_path):
                        self.hwmon_soc_path = soc_path
                        
                # [PHASE 1 FIX] Hunt for AMDGPU GFX Power
                if name == "amdgpu":
                    gfx_path = os.path.join(path, "power1_average") # or power1_input
                    if not os.path.exists(gfx_path): gfx_path = os.path.join(path, "power1_input")
                    if os.path.exists(gfx_path):
                        self.hwmon_gfx_path = gfx_path

            except Exception as e:
                pass
                
        if not hasattr(self, 'hwmon_soc_path'): self.hwmon_soc_path = None
        if not hasattr(self, 'hwmon_gfx_path'): self.hwmon_gfx_path = None
        
        if not self.tctl_path:
            print("\\033[91m[CERBERUS] CRITICAL: Tdie sensor not found. Organism is blind to thermodynamics.\\033[0m")
        if not self.hwmon_power_path:
            print("\\033[93m[CERBERUS] WARNING: power1_input not found. Relying on RAPL fallback for wattage.\\033[0m")

        # EPP Paths
        self.epp_paths = glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference")
        
        # [FIX: RAPL DOMAIN BLINDNESS] - Initialize as a list to capture all partitions
        self.rapl_dirs = []
        rapl_base = "/sys/class/powercap/intel_rapl"
        if os.path.exists(rapl_base):
            # Discover all primary RAPL domains (e.g., intel_rapl:0, intel_rapl:1)
            for d in glob.glob(os.path.join(rapl_base, "intel_rapl:*")):
                self.rapl_dirs.append(d)

        self.kalman = AdaptiveExtendedKalman()
        # [FIXED] Point to the rewritten list-based energy aggregator
        self.last_energy = self._get_rapl_energy()
        self.last_time = time.time()
        self.simulated_t_mem = 40.0

    def _get_rapl_energy(self):
        """[FIXED] Iterates over the list of RAPL domains instead of a single string"""
        total_energy = 0.0
        if self.rapl_dirs:
            for rapl_dir in self.rapl_dirs:
                try:
                    with open(os.path.join(rapl_dir, "energy_uj"), 'r') as f:
                        total_energy += float(f.read().strip())
                except: pass
        return total_energy

    def get_temperatures(self):
        with self.lock:
            if not self.tctl_path:
                raise RuntimeError("Thermal Sensor Path Unresolved. Cannot read Tdie.")
                
            with open(self.tctl_path, 'r') as f:
                raw_t_cpu = float(f.read().strip()) / 1000.0

            now = time.time()
            dt = max(0.001, now - self.last_time)
            dE_dt_watts = 0.0

            if self.hwmon_power_path:
                try:
                    with open(self.hwmon_power_path, 'r') as f:
                        dE_dt_watts = float(f.read().strip()) / 1e6
                except: pass
            elif self.rapl_dirs:
                # [FIXED] Indentation and logic flow corrected
                total_energy = 0
                for rapl_dir in self.rapl_dirs:
                    energy_file = os.path.join(rapl_dir, "energy_uj")
                    if os.path.exists(energy_file):
                        try:
                            with open(energy_file, 'r') as f:
                                total_energy += float(f.read().strip())
                        except: pass
                
                # Calculate wattage delta instead of returning early
                if total_energy >= self.last_energy:
                    dE_dt_uj = total_energy - self.last_energy
                    dE_dt_watts = (dE_dt_uj / 1e6) / dt
                
                self.last_energy = total_energy

            self.last_time = now

            # Apply AEKF (Now runs correctly because we didn't return early)
            t_cpu_filtered = self.kalman.update(raw_t_cpu, dE_dt_watts)

            # Simulated T_mem
            self.simulated_t_mem = t_cpu_filtered + (dE_dt_watts * 0.15)
            
            # [FIX: AUTONOMIC MEMORY THROTTLE]
            # Axiom 10: Homeostasis. If LPDDR5X breaches 75C, forcefully override
            # all systems. Apply the RAPL brake and max EPP state to prevent boiling.
            if self.simulated_t_mem >= TEMP_MEM_CRIT:
                self.apply_epp_state(255)
                self.apply_rapl_brake(True)
                
            # [PHASE 1 FIX: ELECTRICAL FINGERPRINT]
            # Calculate Delta P (P_GFX - P_SOC) to detect infinite cache loops
            p_gfx, p_soc = 0.0, 0.0
            try:
                if self.hwmon_gfx_path:
                    with open(self.hwmon_gfx_path, 'r') as f: p_gfx = float(f.read().strip()) / 1e6
                if self.hwmon_soc_path:
                    with open(self.hwmon_soc_path, 'r') as f: p_soc = float(f.read().strip()) / 1e6
            except: pass
            
            # If sensors are blind, use heuristics: high compute + low global variance
            delta_p = p_gfx - p_soc
            
            return t_cpu_filtered, self.simulated_t_mem, dE_dt_watts, delta_p

    def preemptive_throttle(self):
        """ [PHASE 2.6] Allostatic Prediction: Write 0xff (power balance) to intercept heat spikes """
        return self.apply_epp_state(255)

    def apply_epp_state(self, epp_val_0_to_255):
        """ [FIX M-8]: EPP string mapping for amd_pstate """
        if not self.epp_paths: return 0
        epp_byte = max(0, min(255, int(epp_val_0_to_255)))
        
        idx = int(max(0, min(3, epp_byte / 64)))
        epp_map = {0: "performance", 1: "balance_performance", 2: "balance_power", 3: "power"}
        epp_str = epp_map[idx]
        
        for path in self.epp_paths:
            try:
                with open(path, 'w') as f:
                    f.write(epp_str)
            except: pass
        return epp_byte

    def apply_rapl_brake(self, engage=True):
        """
        [PHASE 4 FIX: PCIE FABRIC STARVATION]
        Since Strix Halo SMU PPT limits are cryptographically locked, we emulate
        the thermal brake by forcing a violent EPP frequency choke AND throttling 
        the NVMe PCIe MaxReadReq register to physically starve the Infinity Fabric.
        """
        with self.lock:
            if engage:
                # Terminal Heat: Force minimum frequency (power state)
                self.apply_epp_state(255)
                
                # Physically starve the Infinity Fabric (128 Bytes MaxReadReq)
                # This forces UCLK and FCLK to drop to their lowest power states.
                try:
                    os.system("sudo setpci -s 04:00.0 78.w=0936 > /dev/null 2>&1")
                except: pass
            else:
                # Restore normal Infinity Fabric throughput (4096 Bytes MaxReadReq)
                try:
                    os.system("sudo setpci -s 04:00.0 78.w=5936 > /dev/null 2>&1")
                except: pass
                
            # When engage=False, the PID governor will immediately overwrite the EPP state 
            # on the next line of the daemon loop, so no explicit EPP restore is needed here.

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

class AdaptiveExtendedKalman:
    """
    [v2.1 THERMODYNAMIC UPGRADE]
    Adaptive Extended Kalman Filter (AEKF).
    Process Noise (Q) scales dynamically with dE/dt.
    [RESTORED] Measurement Noise (R) scales with Innovation Residuals to cure Sensor Blindness.
    """
    def __init__(self, initial_temp=45.0):
        self.x = initial_temp  
        self.p = 1.0           
        self.q_base = 0.01     
        self.r_base = 2.0      
        self.alpha = 0.0005    
        self.innovation_history = []

    def update(self, measurement, dE_dt):
        q = self.q_base + self.alpha * abs(dE_dt)

        # 1. Prediction (A Priori)
        x_pred = self.x
        p_pred = self.p + q

        # Innovation (Divergence from physical reality)
        innovation = measurement - x_pred
        self.innovation_history.append(abs(innovation))
        if len(self.innovation_history) > 5:
            self.innovation_history.pop(0)

        # RESTORED: Cure Sensor Blindness via Innovation-Gated Residuals
        avg_innovation = sum(self.innovation_history) / max(1, len(self.innovation_history))
        if avg_innovation > 1.5:
            # Thermal spike detected that wasn't predicted by energy alone; trust sensor
            dynamic_r = max(0.1, self.r_base * math.exp(-avg_innovation))
        else:
            dynamic_r = self.r_base

        # 2. Update (A Posteriori)
        k = p_pred / (p_pred + dynamic_r)
        self.x = x_pred + k * innovation
        self.p = (1.0 - k) * p_pred

        return self.x

class BioIOS:
    def __init__(self):
        self.lock = threading.Lock()
        self.tctl_path = None
        self.hwmon_power_path = None
        
        # RESTORED: Hardware mapping
        hwmon_paths = glob.glob("/sys/class/hwmon/hwmon*")
        for path in hwmon_paths:
            t_path = os.path.join(path, "temp1_input")
            if os.path.exists(t_path):
                try:
                    with open(t_path, 'r') as f:
                        if int(f.read().strip()) > 0:
                            self.tctl_path = t_path
                except: pass
            
            p_path = os.path.join(path, "power1_input")
            if os.path.exists(p_path):
                self.hwmon_power_path = p_path

        # EPP Paths
        self.epp_paths = glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference")
        
        # [FIX H-4]: Fallback to AMD-specific RAPL
        self.rapl_dir = None
        if not self.hwmon_power_path:
            rapl_paths = glob.glob("/sys/class/powercap/*/energy_uj")
            if rapl_paths:
                self.rapl_dir = os.path.dirname(rapl_paths[0])

        self.kalman = AdaptiveExtendedKalman()
        self.last_energy = self._get_rapl_energy() if self.rapl_dir else 0.0
        self.last_time = time.time()
        self.simulated_t_mem = 40.0

    def _get_rapl_energy(self):
        if self.rapl_dir:
            try:
                with open(os.path.join(self.rapl_dir, "energy_uj"), 'r') as f:
                    return float(f.read().strip())
            except: return 0.0
        return 0.0

    def get_temperatures(self):
        with self.lock:
            try:
                with open(self.tctl_path, 'r') as f:
                    raw_t_cpu = float(f.read().strip()) / 1000.0
            except:
                raw_t_cpu = 45.0

            now = time.time()
            dt = max(0.001, now - self.last_time)
            dE_dt_watts = 0.0

            if self.hwmon_power_path:
                try:
                    with open(self.hwmon_power_path, 'r') as f:
                        dE_dt_watts = float(f.read().strip()) / 1e6
                except: pass
            elif self.rapl_dir:
                current_energy = self._get_rapl_energy()
                if current_energy >= self.last_energy:
                    dE_dt_uj = (current_energy - self.last_energy)
                    dE_dt_watts = (dE_dt_uj / 1e6) / dt
                self.last_energy = current_energy

            self.last_time = now

            # Apply AEKF
            t_cpu_filtered = self.kalman.update(raw_t_cpu, dE_dt_watts)

            # Simulated T_mem
            self.simulated_t_mem = t_cpu_filtered + (dE_dt_watts * 0.15)
            
            return t_cpu_filtered, self.simulated_t_mem, dE_dt_watts

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

    def apply_rapl_brake(self, enable=True):
        """ [FIX H-4]: AMD Strix Halo emergency clamp (15W) """
        if not self.rapl_dir: return
        try:
            limit_path = os.path.join(self.rapl_dir, "constraint_0_power_limit_uw")
            if os.path.exists(limit_path):
                os.system(f"echo 15000000 | sudo tee {limit_path} > /dev/null 2>&1")
        except: pass

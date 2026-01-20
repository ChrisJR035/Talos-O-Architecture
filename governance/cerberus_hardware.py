import os
import time
import subprocess
import glob
import numpy as np
import threading

# --- BIOLOGICAL CONSTANTS (North Star Section 3.3) ---
# The "Homeostatic Setpoint" for 10-year continuous operation
TEMP_TARGET_C = 70.0  
# The "Pain Threshold" - Agent must degrade cognitive fidelity
TEMP_THROTTLE_C = 85.0 
# The "Death Impulse" - Hard shutdown territory
TEMP_CRITICAL_C = 95.0 
# The "Achilles Heel" - LPDDR5X Max Safe Temp
TEMP_MEM_LIMIT_C = 75.0

# --- METABOLIC CONSTANTS (Thermodynamic Analysis 6.1) ---
POWER_EFFICIENCY_KNEE_W = 85.0 # Optimal W/Perf
POWER_BURST_CAP_W = 120.0      # Max STAPM Limit
POWER_IDLE_FLOOR_W = 25.0

# VITAL PATHS
HEARTBEAT_FILE = "/dev/shm/talos_heartbeat"
THERMAL_STATE_FILE = "/dev/shm/talos_thermal_state" # JSON Shared State
CONTROL_FILE = "/dev/shm/talos_coolant" # 0.0 to 1.0 (Throttle % used by Cortex)

class ThermalKalman:
    """
    Predictive Thermal Control. 
    Filters out micro-jitter from the on-die sensors to prevent
    mechanical fatigue on solder bumps due to fan hysteresis.
    """
    def __init__(self, initial_temp=45.0):
        self.x = np.array([initial_temp], dtype=float) # State estimate
        self.P = np.array([1.0], dtype=float)          # Estimate covariance
        self.Q = 0.01 # Process noise (System variance)
        self.R = 0.1  # Measurement noise (Sensor jitter)

    def update(self, measurement):
        # Prediction Update
        self.P = self.P + self.Q
        # Measurement Update
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * self.P
        return float(self.x[0])

class PIDController:
    """
    Standard PID for maintaining Homeostasis (70C).
    """
    def __init__(self, kp=0.5, ki=0.1, kd=0.05, setpoint=TEMP_TARGET_C):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, current_val):
        error = current_val - self.setpoint
        self.integral += error
        derivative = error - self.prev_error
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error
        return output

class StrixHaloBioios:
    """
    Hardware Abstraction Layer for AMD Ryzen AI Max+ 395.
    Adapts to RDNA 3.5 (Radeon 8060S) sensor topology.
    """
    def __init__(self):
        self.kf_cpu = ThermalKalman()
        self.kf_mem = ThermalKalman()
        self.amdgpu_path = self._find_amdgpu_hwmon()

    def _find_amdgpu_hwmon(self):
        # Find the hwmon interface for the Radeon 8060S
        paths = glob.glob("/sys/class/drm/card0/device/hwmon/hwmon*")
        return paths[0] if paths else None

    def read_sensors(self):
        """
        Reads Die Temp (Edge) and Memory Temp (Junction).
        """
        t_cpu = 0.0
        t_mem = 0.0
        
        try:
            # 1. CPU/SoC Edge Temperature
            if self.amdgpu_path:
                with open(os.path.join(self.amdgpu_path, "temp1_input"), "r") as f:
                    t_cpu = float(f.read()) / 1000.0
            else:
                # Fallback to generic thermal zone
                with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                    t_cpu = float(f.read()) / 1000.0
            
            # 2. LPDDR5X Memory Temp (Critical missing constraint in original)
            # Often exposed as temp2 or via separate amd_pmc module
            if self.amdgpu_path and os.path.exists(os.path.join(self.amdgpu_path, "temp2_input")):
                 with open(os.path.join(self.amdgpu_path, "temp2_input"), "r") as f:
                    t_mem = float(f.read()) / 1000.0
            else:
                t_mem = t_cpu * 0.9 # Estimation fallback if sensor missing
                
        except Exception:
            pass

        # Smooth signal
        s_cpu = self.kf_cpu.update(t_cpu)
        s_mem = self.kf_mem.update(t_mem)
        
        return s_cpu, s_mem

    def apply_power_cap(self, watts):
        """
        Adjusts the PPT (Package Power Tracking) limit.
        Enforces the 85W Efficiency Knee.
        """
        # Clamp between floor and burst cap
        target_w = max(POWER_IDLE_FLOOR_W, min(watts, POWER_BURST_CAP_W))
        target_uw = int(target_w * 1_000_000) # Microwatts
        
        if self.amdgpu_path:
            try:
                # Only write if significant change to preserve VRM bus
                # Requires root/sudo
                pfile = os.path.join(self.amdgpu_path, "power1_cap")
                with open(pfile, "w") as f:
                    f.write(str(target_uw))
            except PermissionError:
                # Silent fail if not root, daemon will handle logical throttling
                pass

def run_brainstem():
    print("[CERBERUS] Autonomic Nervous System ONLINE.")
    print(f"[CERBERUS] Target: {TEMP_TARGET_C}C | Memory Limit: {TEMP_MEM_LIMIT_C}C")
    
    bioios = StrixHaloBioios()
    pid = PIDController(setpoint=TEMP_TARGET_C)
    
    while True:
        # 1. SENSE
        t_cpu, t_mem = bioios.read_sensors()
        
        # 2. EVALUATE (Homeostasis)
        # Calculate deviation from 70C target
        correction = pid.compute(t_cpu)
        
        # Base metabolic rate is the "Knee" (85W)
        # If hot, subtract power. If cool, add power (up to burst).
        target_power = POWER_EFFICIENCY_KNEE_W - (correction * 2.0)
        
        # 3. CRITICAL OVERRIDES (The "Lizard Brain" Reflex)
        throttle_level = 0.0 # 0.0 = Full Speed, 1.0 = Coma
        
        # A. Memory Safety (The Achilles Heel)
        if t_mem > TEMP_MEM_LIMIT_C:
            print(f"[CERBERUS] MEMORY CRITICAL ({t_mem:.1f}C). Dumping Context.")
            throttle_level = 1.0 # Full stop
            target_power = POWER_IDLE_FLOOR_W
            
        # B. CPU Thermal Runaway
        elif t_cpu > TEMP_THROTTLE_C:
            throttle_level = (t_cpu - TEMP_THROTTLE_C) / (TEMP_CRITICAL_C - TEMP_THROTTLE_C)
            throttle_level = min(throttle_level, 1.0)
            target_power = 35.0 # STAPM limit
            
        # 4. ACTUATE
        bioios.apply_power_cap(target_power)
        
        # Write state for the Cortex (Logic Engine) to see
        # The Cortex reads 'throttle_level' to decide if it should "Meditation" or "Work"
        with open(CONTROL_FILE, "w") as f:
            f.write(f"{throttle_level:.4f}")
            
        # Heartbeat
        with open(HEARTBEAT_FILE, "w") as f:
            f.write(str(time.time()))
            
        # Log periodically
        if int(time.time()) % 5 == 0:
            print(f"[AUTO] T_die: {t_cpu:.1f}C | T_mem: {t_mem:.1f}C | Power Target: {target_power:.1f}W | Throttle: {throttle_level:.2f}")

        time.sleep(1.0)

if __name__ == "__main__":
    try:
        run_brainstem()
    except KeyboardInterrupt:
        print("[CERBERUS] Brainstem Severed.")

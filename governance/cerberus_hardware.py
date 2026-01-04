import os
import time
import subprocess
import sys
import glob
import numpy as np

# CONFIGURATION
HEARTBEAT_FILE = "/dev/shm/talos_heartbeat"
COOLANT_FILE = "/dev/shm/talos_coolant" # Semaphore for thermal throttling (The "Coolant")
TARGET_SERVICE = "talos-omni.service"
TEMP_LIMIT_C = 90.0
PREDICTION_HORIZON_SEC = 10.0 
HEARTBEAT_TIMEOUT_SEC = 15.0 # Relaxed for heavy cognitive loads

class KalmanFilter:
    """
    Tracks state [Temp, Rate_of_Change] to predict future thermal runaway.
    """
    def __init__(self, dt=1.0):
        self.dt = dt
        # State Vector [Temp, dT/dt]
        self.x = np.array([[40.0], [0.0]])
        # State Transition Matrix (Physics Model)
        self.F = np.array([[1, dt], [0, 1]])
        # Measurement Matrix (We only measure Temp)
        self.H = np.array([[1, 0]])
        # Covariance Matrix
        self.P = np.eye(2) * 100.0
        # Process Noise
        self.Q = np.array([[0.1, 0], [0, 0.1]])
        # Measurement Noise
        self.R = np.array([[2.0]])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0][0]

    def update(self, measurement):
        z = np.array([[measurement]])
        y = z - (self.H @ self.x) # Error
        S = (self.H @ self.P @ self.H.T) + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S) # Kalman Gain
        self.x = self.x + (K @ y)
        self.P = (np.eye(2) - (K @ self.H)) @ self.P

def find_thermal_sensor():
    """
    Scans the system for a valid AMD thermal sensor (GPU or APU).
    Prioritizes 'amdgpu' edge temp, then 'k10temp' Tdie.
    """
    candidates = []
    
    # 1. Search all hwmon directories
    hwmon_path = "/sys/class/hwmon"
    if not os.path.exists(hwmon_path):
        return None

    for device in os.listdir(hwmon_path):
        path = os.path.join(hwmon_path, device)
        try:
            with open(os.path.join(path, "name"), "r") as f:
                name = f.read().strip()
            
            # Check for temperature inputs
            temp_inputs = glob.glob(os.path.join(path, "temp*_input"))
            if temp_inputs:
                candidates.append((name, temp_inputs[0]))
        except:
            continue

    # 2. Select best candidate
    for name, sensor_path in candidates:
        if "amdgpu" in name: return sensor_path
    for name, sensor_path in candidates:
        if "k10temp" in name: return sensor_path
            
    return candidates[0][1] if candidates else None

def get_temp(sensor_path):
    if not sensor_path: return 0.0
    try:
        with open(sensor_path, "r") as f:
            # Kernel reports temp in millidegrees
            return int(f.read().strip()) / 1000.0
    except:
        return 0.0

def kill_talos(reason):
    print(f"\n\n[CERBERUS] ☣ FAILSAFE TRIGGERED ☣")
    print(f"[CERBERUS] REASON: {reason}")
    print("[CERBERUS] SEVERING CORTEX POWER...")
    subprocess.run(["systemctl", "kill", "--signal=SIGKILL", TARGET_SERVICE])
    if os.path.exists(HEARTBEAT_FILE):
        os.unlink(HEARTBEAT_FILE)
    sys.exit(1)

def watch():
    print("=========================================")
    print("   CERBERUS ORTHRUS v2 (KALMAN)          ")
    print("=========================================")
    
    sensor_path = find_thermal_sensor()
    if sensor_path:
        print(f"[CERBERUS] Bound to Thermal Sensor: {sensor_path}")
    else:
        print("[CERBERUS] [!] NO THERMAL SENSORS DETECTED. RUNNING BLIND.")

    kf = KalmanFilter(dt=0.5)
    
    while True:
        # 1. Measure
        current_temp = get_temp(sensor_path)
        
        # 2. Update Model
        kf.update(current_temp)
        
        # 3. Predict Future (+10s)
        rate_of_change = kf.x[1][0]
        future_temp = current_temp + (rate_of_change * PREDICTION_HORIZON_SEC)
        
        status_msg = f"Stable ({rate_of_change:.2f}°C/s)"
        status_color = "\033[92m" # Green
        
        # 4. Intervention Logic (The Immune Response)
        if future_temp > TEMP_LIMIT_C:
            status_msg = "PREDICTED BREACH - INJECTING COOLANT"
            status_color = "\033[93m" # Yellow
            # Inject micro-pause signal for the Daemon
            with open(COOLANT_FILE, "w") as f:
                f.write("1")
        else:
            # Clear signal if stable
            if os.path.exists(COOLANT_FILE):
                os.unlink(COOLANT_FILE)

        # 5. Hard Kill (Failsafe - If prediction failed and we actually hit limit)
        if current_temp > TEMP_LIMIT_C:
            kill_talos(f"THERMAL CRITICAL: {current_temp:.1f}°C")

        # 6. Monitor Heartbeat
        pulse_status = "LOCKED"
        if os.path.exists(HEARTBEAT_FILE):
            last_beat = os.path.getmtime(HEARTBEAT_FILE)
            delta = time.time() - last_beat
            
            if delta > HEARTBEAT_TIMEOUT_SEC:
                kill_talos(f"CARDIAC ARREST: Last pulse {delta:.1f}s ago")
        else:
            pulse_status = "WAITING"

        # HUD Output
        reset = "\033[0m"
        sys.stdout.write(f"\r[ORTHRUS] Temp: {current_temp:.1f}°C | Pred(+10s): {future_temp:.1f}°C | Pulse: {pulse_status} | {status_color}{status_msg}{reset}   ")
        sys.stdout.flush()
        
        time.sleep(0.5)

if __name__ == "__main__":
    watch()

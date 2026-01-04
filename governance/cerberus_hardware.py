import os
import time
import subprocess
import signal
import sys
import glob

# CONFIGURATION
HEARTBEAT_FILE = "/dev/shm/talos_heartbeat"
TARGET_SERVICE = "talos-omni.service"
TEMP_LIMIT_C = 90.0
HEARTBEAT_TIMEOUT_SEC = 10.0 # Increased for LLM loading time

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
    # Strix Halo is an APU, so 'amdgpu' or 'k10temp' are key
    for name, sensor_path in candidates:
        if "amdgpu" in name:
            return sensor_path
    for name, sensor_path in candidates:
        if "k10temp" in name:
            return sensor_path
            
    # Fallback to first available if any
    return candidates[0][1] if candidates else None

SENSOR_PATH = find_thermal_sensor()
if SENSOR_PATH:
    print(f"[CERBERUS] Bound to Thermal Sensor: {SENSOR_PATH}")
else:
    print("[CERBERUS] [!] NO THERMAL SENSORS DETECTED. RUNNING BLIND.")

def get_temp():
    if not SENSOR_PATH: return 0.0
    try:
        with open(SENSOR_PATH, 'r') as f:
            return int(f.read().strip()) / 1000.0
    except:
        return 0.0

def kill_talos(reason):
    print(f"\n\n[CERBERUS] ☣ FAILSAFE TRIGGERED ☣")
    print(f"[CERBERUS] REASON: {reason}")
    print("[CERBERUS] SEVERING CORTEX POWER...")
    subprocess.run(["systemctl", "kill", "--signal=SIGKILL", TARGET_SERVICE])
    # Remove heartbeat to prevent loop
    if os.path.exists(HEARTBEAT_FILE):
        os.unlink(HEARTBEAT_FILE)
    sys.exit(1)

def watch():
    print("=========================================")
    print("   CERBERUS HARDWARE SENTINEL v2.0       ")
    print("=========================================")
    
    while True:
        # 1. Monitor Temperature
        temp = get_temp()
        if temp > TEMP_LIMIT_C:
            kill_talos(f"THERMAL CRITICAL: {temp:.1f}°C")

        # 2. Monitor Heartbeat
        pulse_status = "WAITING"
        if os.path.exists(HEARTBEAT_FILE):
            last_beat = os.path.getmtime(HEARTBEAT_FILE)
            delta = time.time() - last_beat
            
            if delta > HEARTBEAT_TIMEOUT_SEC:
                kill_talos(f"CARDIAC ARREST: Last pulse {delta:.1f}s ago")
            else:
                pulse_status = "LOCKED"
        else:
            pulse_status = "NO SIGNAL"
            # Optional: Kill if "NO SIGNAL" persists too long after boot? 
            # For now, we just warn to allow startup.

        # 3. HUD
        status_color = "\033[92m" if pulse_status == "LOCKED" else "\033[93m"
        reset = "\033[0m"
        sys.stdout.write(f"\r[SENTINEL] Temp: {temp:.1f}°C | Pulse: {status_color}{pulse_status}{reset}   ")
        sys.stdout.flush()
        
        time.sleep(0.5)

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("[!] Root required for Kill Signal.")
        sys.exit(1)
    watch()

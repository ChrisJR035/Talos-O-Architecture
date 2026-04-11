import threading
import time
import sys
import os
import signal
import glob

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from cerberus_hardware import BioIOS, TEMP_CPU_TARGET, TEMP_MEM_CRIT, TEMP_CPU_MAX
except ImportError:
    from governance.cerberus_hardware import BioIOS, TEMP_CPU_TARGET, TEMP_MEM_CRIT, TEMP_CPU_MAX

class PIDGovernor:
    """
    [v2.0 THERMODYNAMIC UPGRADE]
    Proportional-Integral-Derivative (PID) Controller with Back-Calculation.
    Prevents Integral Windup when the 4.4L chassis reaches sustained thermal saturation.
    """
    def __init__(self, target_temp):
        self.target = target_temp
        
        # Aggressive tuning for 120W TDP in SFF Chassis
        self.kp = 12.0   # Immediate, aggressive scaling on error
        self.ki = 0.5    # Gentle pull for steady-state fever
        self.kd = 3.0    # Predictive braking against rapid heating
        
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()
        
        # Anti-Windup Tracking Constant
        self.tracking_time_constant = 1.5 

    def update(self, current_temp):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0: dt = 0.01
        self.last_time = now

        error = current_temp - self.target
        derivative = (error - self.last_error) / dt
        self.last_error = error

        # 1. Theoretical PID Output
        p_out = self.kp * error
        i_out = self.ki * self.integral
        d_out = self.kd * derivative
        
        theoretical_out = p_out + i_out + d_out
        
        # 2. Physical Clamping (CPPC EPP bounds)
        actual_out = max(0.0, min(255.0, theoretical_out))

        # 3. Integral Back-Calculation (Anti-Windup)
        # If the output hits the 255 ceiling, we actively unwind the integral 
        # so the system recovers instantly when the inference spike ends.
        if theoretical_out != actual_out:
            excess = theoretical_out - actual_out
            self.integral -= (excess / self.tracking_time_constant) * dt
        else:
            self.integral += error * dt

        return actual_out, derivative


class SpriteKerberos(threading.Thread):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.running = True
        self.daemon = True
        self.bio = BioIOS()
        self.pid = PIDGovernor(TEMP_CPU_TARGET)
        
        self._setup_apoptosis_handlers()

    def _setup_apoptosis_handlers(self):
        """Ensures POSIX shared memory is unlinked on fatal exit."""
        def graceful_exit(signum, frame):
            print("\n\033[93m[CERBERUS] Apoptosis Signal Received. Purging Memory Locks...\033[0m")
            self.running = False
            self.bio.apply_rapl_brake(False) # Release brake on exit
            
            # Eradicate POSIX ghosts
            for shm_file in glob.glob("/dev/shm/talos_*"):
                try: os.unlink(shm_file)
                except: pass
                
            os._exit(0) # We use os._exit here because the main thread handles sys.exit

        signal.signal(signal.SIGINT, graceful_exit)
        signal.signal(signal.SIGTERM, graceful_exit)

    def run(self):
        print("\033[96m[CERBERUS] Autonomic Nervous System Active. AEKF & PID Governance Online.\033[0m")
        
        brake_engaged = False
        
        while self.running:
            t_cpu, t_mem, wattage = self.bio.get_temperatures()
            
            # =================================================================
            # 1. THE ABSOLUTE OVERRIDE (Hardware Preservation)
            # =================================================================
            if t_mem > TEMP_MEM_CRIT or t_cpu > TEMP_CPU_MAX:
                self.bio.apply_epp_state(255)
                self.bio.apply_rapl_brake(True)
                
                print(f"\033[1;41m[FATAL] APOPTOSIS TRIGGERED (T_cpu: {t_cpu:.1f}C, T_mem: {t_mem:.1f}C). SILICON SEVERED.\033[0m", flush=True)
                
                # Eradicate POSIX ghosts before dying
                for shm_file in glob.glob("/dev/shm/talos_*"):
                    try: os.unlink(shm_file)
                    except: pass
                os._exit(1)

            # =================================================================
            # 2. DUAL-STAGE ACTUATION (PID EPP + Parasympathetic RAPL)
            # =================================================================
            epp_val, temp_momentum = self.pid.update(t_cpu)
            
            # If temperature is high AND rising fast, engage the 15W hardware brake
            if t_cpu > 85.0 and temp_momentum > 2.0:
                if not brake_engaged:
                    self.bio.apply_rapl_brake(True)
                    brake_engaged = True
                    print(f"\033[41m[CERBERUS] PARASYMPATHETIC BRAKE ENGAGED! (15W Clamp)\033[0m", flush=True)
                actual_epp = self.bio.apply_epp_state(255)
            else:
                # Normal Operation
                if brake_engaged:
                    self.bio.apply_rapl_brake(False)
                    brake_engaged = False
                    print(f"\033[92m[CERBERUS] Brake Released. Restoring 120W limit.\033[0m", flush=True)
                    
                actual_epp = self.bio.apply_epp_state(epp_val)
            
            # Link to the main engine for HUD telemetry
            if hasattr(self.engine, 'thermal_state'):
                self.engine.thermal_state = f"EPP:{actual_epp}"
            
            # Terminal Logging
            if int(time.time() * 10) % 10 == 0:
                brake_warn = " [BRAKE ACTIVE]" if brake_engaged else ""
                print(f"[CERBERUS] T_die: {t_cpu:.1f}C | T_mem: {t_mem:.1f}C | Power: {wattage:.1f}W | EPP: {int(actual_epp)}/255{brake_warn}")
            
            time.sleep(0.1) # 10Hz Polling Rate

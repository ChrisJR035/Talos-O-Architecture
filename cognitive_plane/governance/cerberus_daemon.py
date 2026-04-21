import threading
import time
import sys
import os
import signal
import glob
import ctypes # [ADDED FOR KERNEL SYSCALLS]

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from cerberus_hardware import BioIOS, TEMP_CPU_TARGET, TEMP_MEM_CRIT, TEMP_CPU_MAX, RTIEstimator
except ImportError:
    from governance.cerberus_hardware import BioIOS, TEMP_CPU_TARGET, TEMP_MEM_CRIT, TEMP_CPU_MAX, RTIEstimator

import time
import math
import statistics

class PIDGovernor:
    """
    Adaptive PID Governor implementing continuous Ziegler-Nichols autotuning.
    Designed for the Corsair AI Workstation 300 (4.4L SFF) chassis.
    Fulfills Axiom 10: Homeostasis.
    """
    
    # Configuration constants for Ziegler-Nichols Tuning
    TUNING_MODES = {
        "FAST_COLD": {"Kc_mult": 0.60, "Ti_mult": 0.50, "Td_mult": 0.12},
        "NOMINAL":   {"Kc_mult": 0.25, "Ti_mult": 0.50, "Td_mult": 0.12},
        "EXTREME":   {"Kc_mult": 0.15, "Ti_mult": 0.50, "Td_mult": 0.12}
    }
    
    def __init__(self, target_temp=65.0, max_power=255.0, sample_rate=0.1): # EPP is 0-255
        self.base_target = target_temp # [PHASE 2 FIX] Store the immutable baseline
        self.target = target_temp # This will now be dynamically mutated
        self.max_power = max_power
        self.sample_rate = sample_rate
        self.kp = 12.0 # Safe start
        self.ki = 0.5
        self.kd = 3.0
        
        # State variables
        self.error_sum = 0.0
        self.last_error = 0.0
        self.prev_time = time.time()
        self.start_time = time.time()  # [FIX: Ignition Temporal Anchor]
        self.last_output = 255.0       # [FIX: Start in fully choked state]
        
        # Z-N Autotuning State
        self.oscillation_period = []
        self.last_errors = []
        self.last_power_outputs = []
        self.last_tune_time = time.time()
        
        # Anti-windup limits
        self.integral_limit = max_power * 2.0 
        
        # State classification
        self.chassis_state = "NOMINAL"

    def _detect_chassis_state(self, integral_windup_magnitude, current_temp):
        # Heuristic for thermal saturation based on windup and temp
        if current_temp > 82.0:
            return "EXTREME"
        elif integral_windup_magnitude > (self.max_power * 0.1) or current_temp > 80.0:
            return "EXTREME"
        elif integral_windup_magnitude < (self.max_power * 0.02) and current_temp < 60.0:
            return "FAST_COLD"
        else:
            return "NOMINAL"

    def _tune_parameters(self):
        # 1. Passive Ultimate Period (Tu) Estimation
        if len(self.oscillation_period) >= 2:
            periods = []
            for i in range(1, len(self.oscillation_period)):
                periods.append(self.oscillation_period[i] - self.oscillation_period[i-1])
            
            if len(periods) > 0:
                Tu = statistics.mean(periods)
                if Tu < 0.1 or Tu > 30.0: 
                    Tu = 5.0 
            else:
                Tu = 5.0

            # 2. Passive Ultimate Gain (Ku) Estimation
            error_std = statistics.stdev(self.last_errors[-10:]) if len(self.last_errors) >= 10 else 1.0
            Ku = (self.max_power / 2.0) / (error_std * 2.0 + 0.001) 
            Ku = min(max(Ku, 0.1), 5.0)
        else:
            Ku = 1.5 
            Tu = 4.0

        # 3. Retrieve Active State Multipliers
        state_config = self.TUNING_MODES.get(self.chassis_state, self.TUNING_MODES["NOMINAL"])
        
        # 4. Strict Ziegler-Nichols Math
        self.kp = Ku * state_config["Kc_mult"]
        
        # Ki = Kp / Ti  (where Ti = Tu * Ti_mult)
        self.ki = self.kp / (Tu * state_config["Ti_mult"])
        
        # Kd = Kp * Td  (where Td = Tu * Td_mult)
        self.kd = self.kp * (Tu * state_config["Td_mult"])

    def update(self, current_temp):
        current_time = time.time()
        dt = current_time - self.prev_time
        
        if dt <= 0: dt = 0.01
            
        self.prev_time = current_time
        
        error = current_temp - self.target
        
        self.last_errors.append(error)
        self.last_power_outputs.append(self.last_output)

        if len(self.last_errors) > 100:
            self.last_errors.pop(0)
            self.last_power_outputs.pop(0)

        # Detect Zero Crossings for Tu estimation
        if self.last_error != 0.0 and error * self.last_error < 0:
            self.oscillation_period.append(current_time)
            if len(self.oscillation_period) > 10:
                self.oscillation_period.pop(0)

        # [PHASE 2 FIX: DYNAMIC ANTI-WINDUP]
        # Because the target temperature is now subjugated to cognitive uncertainty,
        # it can shift violently. We must brutally decay the integral sum if the 
        # error changes sign to prevent "thermal memory" from locking the CPU.
        if (self.last_error > 0 and error < 0) or (self.last_error < 0 and error > 0):
            self.error_sum *= 0.1  # 90% instant decay on zero-crossing
            
        self.error_sum += error * dt
        if self.error_sum > self.integral_limit:
            self.error_sum = self.integral_limit
        elif self.error_sum < -self.integral_limit:
            self.error_sum = -self.integral_limit

        # Calculate Output
        p_term = self.kp * error
        i_term = self.ki * self.error_sum
        d_term = self.kd * (error - self.last_error) / dt 
        
        output = p_term + i_term + d_term
        
        # [PHASE 3 FIX: IGNITION GRACE PERIOD]
        # Override the PID output and force Maximum EPP Choke (255) for the first 30 seconds.
        # This allows Systemd's thermal cage to hold the CPU while the massive 35B model loads.
        if current_time - self.start_time < 30.0:
            output = 255.0
            
        self.last_output = max(0.0, min(self.max_power, output))
        self.chassis_state = self._detect_chassis_state(abs(self.error_sum), current_temp)
        
        if current_time - self.last_tune_time > 30.0: 
            self._tune_parameters()
            self.last_tune_time = current_time

        self.last_error = error
        
        # We return the EPP value (0-255) and the derivative (d_term) for Cerberus to use
        return self.last_output, d_term


class SpriteKerberos(threading.Thread):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.running = True
        self.daemon = True
        self.bio = BioIOS()
        self.pid = PIDGovernor(TEMP_CPU_TARGET)
        
        # [PHASE 2.3] Watchdog State Exposure
        self.is_critical = False
        
        # [KURAMOTO PACEMAKER STATE]
        self.triage_active = False
        self.triage_cooldown = time.time()
        
        # [RTI PREDICTIVE INSTRUMENT]
        self.rti = RTIEstimator(dt_baseline=0.01, lambda_0=1.0)
        
        self._setup_apoptosis_handlers()

    def trigger_brainstem_bypass(self, reason):
        """
        [BRAINSTEM BYPASS]
        Instantly seizes control from the VJEPA when EFE or D_JS thresholds are breached.
        Called externally by the VJEPA rollout if predictions prove lethal.
        """
        print(f"\n\033[41;97m[BRAINSTEM BYPASS] Watchdog seizing control! Reason: {reason}\033[0m", flush=True)
        self.triage_active = True
        self.is_critical = True
        
        # Force acquire triage lock to hard-block the cognitive loop
        if hasattr(self.engine, 'triage_lock'):
            if not self.engine.triage_lock.locked():
                self.engine.triage_lock.acquire()
                # Set a timer to release the lock after the heat wave passes
                threading.Timer(2.0, self.engine.triage_lock.release).start()
                
        # Force severe RAPL clamp via bio
        self.bio.apply_rapl_brake(True)

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

    def run(self):
        # [AXIOM 5: THREAD ISOLATION] Pin Watchdog to Secondary Die (CCD 1)
        try:
            os.sched_setaffinity(0, set(range(8, 16)))
            print(f"\033[96m[CERBERUS] Watchdog Isolated to CCD 1 (Cores 8-15).\033[0m")
        except AttributeError:
            pass # Non-Linux fallback
            
        print(f"\n\033[96m[CERBERUS] Autonomic Watchdog Active. T_target: {self.pid.target}C\033[0m")
        
        # [AMPUTATED] Signal handlers must be exclusively managed by the main thread.
        # def _setup_apoptosis_handlers():
        #     def graceful_exit(signum, frame):
        #         print(f"\n\033[91m[CERBERUS] Systemic Shutdown Signal ({signum}). Disengaging Autonomic Loop...\033[0m", flush=True)
        #         if hasattr(self.engine, 'shm_telemetry') and self.engine.shm_telemetry:
        #             try:
        #                 self.engine.shm_telemetry.close()
        #                 self.engine.shm_telemetry.unlink()
        #             except: pass
        #         self.running = False
        #         sys.exit(0)
        #     
        #     signal.signal(signal.SIGINT, graceful_exit)
        #     signal.signal(signal.SIGTERM, graceful_exit)
        #     
        # _setup_apoptosis_handlers()

        # [FIXED] Restored the main autonomic loop at the correct indentation level
        brake_engaged = False
        last_loop_time = time.time() # [RTI TEMPORAL ANCHOR]
        
        # [TOPOLOGICAL STATE]
        active_ccd = 0 
        last_migration_time = 0.0
        
        # Ensure we have the C++ bridge loaded for migration
        try: import talos_core
        except: talos_core = None

        while self.running:
            # Calculate the physical daemon loop delta (dt)
            now = time.time()
            dt = max(0.001, now - last_loop_time)
            last_loop_time = now
            
            try:
                t_cpu, t_mem, wattage, delta_p = self.bio.get_temperatures()
            except Exception as e:
                print(f"\033[91m[CERBERUS] Sensor failure: {e}\033[0m")
                time.sleep(1)
                continue

            # =================================================================
            # [PHASE 2 FIX: HIERARCHICAL STATE UNIFICATION]
            # Subjugate the hardware thermal limit to cognitive uncertainty.
            # =================================================================
            current_r = 1.0 # Default to total certainty if blind
            if hasattr(self.engine, 'shm_telemetry') and self.engine.shm_telemetry:
                try:
                    import ctypes
                    # We must import BiophysicalState locally if it's not in the global namespace
                    from cortex.iadcs_kernel import BiophysicalState 
                    state = BiophysicalState.from_buffer(self.engine.shm_telemetry.buf)
                    current_r = state.kuramoto_r
                except: pass

            # As r_kura drops (confusion rises), the target temp drops from 65C to 50C.
            # This forces the PID to violently choke the CPU, "cooling the thoughts".
            uncertainty_penalty = (1.0 - current_r) * 15.0 
            self.pid.target = self.pid.base_target - uncertainty_penalty

            # =================================================================
            # 2. QUAD-STAGE ACTUATION (AEKF -> PID EPP -> Migration -> RAPL Clamp)
            # =================================================================
            epp_val, temp_momentum = self.pid.update(t_cpu)

            # [PHASE 1 FIX: THE ELECTRICAL FINGERPRINT]
            # If P_GFX remains high while P_SOC collapses, the model is trapped in a 
            # 100% cache-hit confabulation loop. Trigger a physical context flush.
            
            # [PHASE 2 FIX: THE BIOMECHANICAL HANDSHAKE]
            # Check if the organism is currently "in utero" (loading models).
            # If the /IN_UTERO semaphore exists, we explicitly blind the anomaly detector
            # to ignore the massive power spikes caused by UMA ingestion.
            in_utero = False
            try:
                import posix_ipc
                sem = posix_ipc.Semaphore("/IN_UTERO")
                if sem.value == 0:
                    in_utero = True
                sem.close()
            except:
                pass
                
            if in_utero:
                if not hasattr(self, 'in_utero_logged') or not self.in_utero_logged:
                    print(f"\033[95m[CERBERUS] Genesis lock detected (/IN_UTERO). Blinding ΔP anomaly detectors.\033[0m", flush=True)
                    self.in_utero_logged = True
            else:
                self.in_utero_logged = False
                if delta_p > 25.0 and wattage > 40.0:
                    if not hasattr(self, 'last_phase_reset'): self.last_phase_reset = 0
                    if now - self.last_phase_reset > 10.0:
                        print(f"\n\033[45;97m[CERBERUS] ANOMALY DETECTED: VDDCR_GFX/SOC Divergence (ΔP: {delta_p:.1f}W). Confabulation Loop Imminent!\033[0m", flush=True)
                        if hasattr(self.engine, 'execute_phase_reset'):
                            self.engine.execute_phase_reset()
                        self.last_phase_reset = now

            # STAGE 0: PREEMPTIVE AEKF THROTTLING (Axiom 3: Inelastic Time)
            if self.bio.kalman.predict_spike(t_cpu):
                print(f"\033[93m[CERBERUS] AEKF PREDICTION: Thermal wave inbound. Preemptively restricting EPP.\033[0m", flush=True)
                epp_val = 200 # Heavy restriction before the heat hits the diode
            
            # STAGE 1: TOPOLOGICAL MIGRATION (First line of defense against hotspots)
            # If we cross 80C, violently throw the workload to the other CCD to let this one cool.
            # 5-second cooldown prevents infinite ping-ponging across the Infinity Fabric.
            if t_cpu > 80.0 and temp_momentum > 1.0 and talos_core and (now - last_migration_time > 5.0):
                active_ccd = 1 if active_ccd == 0 else 0
                try:
                    talos_core.migrate_ccd(active_ccd)
                    last_migration_time = now
                    print(f"\033[36m[CERBERUS] TOPOLOGICAL SHIFT: Migrating inference threads to CCD {active_ccd} to bleed heat.\033[0m", flush=True)
                except Exception as e:
                    print(f"\033[91m[CERBERUS] Topological Shift Failed: {e}\033[0m")

            # STAGE 2: THE NUCLEAR OPTION (Parasympathetic RAPL)
            # If migration failed to bleed heat and we are approaching the 95C death limit.
            if t_cpu > 90.0 and temp_momentum > 1.5:
                self.is_critical = True # [PHASE 2.3] Signal critical state to the Nexus
                if not brake_engaged:
                    self.bio.apply_rapl_brake(True)
                    brake_engaged = True
                    print(f"\033[41m[CERBERUS] TERMINAL HEAT: PARASYMPATHETIC BRAKE ENGAGED! (15W Clamp)\033[0m", flush=True)
                actual_epp = self.bio.apply_epp_state(255)
            else:
                # Normal Operation
                self.is_critical = False # [PHASE 2.3] Release critical state
                if brake_engaged and t_cpu < 80.0:
                    self.bio.apply_rapl_brake(False)
                    brake_engaged = False
                    print(f"\033[92m[CERBERUS] Brake Released. Restoring 120W limit.\033[0m", flush=True)
                    
                actual_epp = self.bio.apply_epp_state(epp_val)
            # =================================================================
            # [RECOVERY TIME INFLATION] - Predictive Coherence Probing
            # =================================================================
            if hasattr(self.engine, 'last_inference_latency_ms'):
                # Extract raw algorithmic latency (dt_measured) directly from the cognitive loop
                dt_measured = self.engine.last_inference_latency_ms / 1000.0
                m_s = self.rti.update(dt_measured, dt)
                
                if m_s < 0.2 and not self.triage_active and (time.time() - self.triage_cooldown > 5.0):
                    print(f"\n\033[93m[CERBERUS] SPECTRAL GAP COLLAPSE PREDICTED (M_s={m_s:.2f}). Executing Graceful Degradation!\033[0m", flush=True)
                    self.triage_active = True
                    
                    # Teleological Action: Shed contextual weight (Wanda-style pruning)
                    if hasattr(self.engine, 'ham'):
                        print("\033[35m[RTI] Pruning HRR Associative Superposition... shedding dimensional tension.\033[0m", flush=True)
                        self.engine.ham.trace *= 0.5  # Decay the contextual weight
                        
                    self.triage_active = False
                    self.triage_cooldown = time.time()

            # =================================================================
            # [THE TEMPORAL GUARDIAN SPRITE] - Kuramoto Coherence Triage
            # =================================================================
            if hasattr(self.engine, 'physics'):
                current_r = self.engine.physics.r_kura
                
                if current_r < 0.85 and not self.triage_active and (time.time() - self.triage_cooldown > 10.0):
                    print(f"\n\033[45;97m[CERBERUS] TEMPORAL FRACTURE DETECTED (r={current_r:.2f}). INITIATING COGNITIVE TRIAGE!\033[0m", flush=True)
                    self.triage_active = True
                    
                    # 1. Shed Load: Force the CPU to sleep via a heavy GIL/thread lock
                    # We utilize the Python C-API to force a yield, dropping stochastic noise
                    if hasattr(self.engine, 'triage_lock'):
                        self.engine.triage_lock.acquire()
                    
                    # 2. Phase Reset Injection: Force a unified 50ms hardware stall
                    time.sleep(0.05) 
                    
                    # 3. Restore Phase
                    self.engine.physics.kuramoto_r = 0.99 
                    
                    if hasattr(self.engine, 'triage_lock'):
                        self.engine.triage_lock.release()
                        
                    print(f"\033[95m[CERBERUS] Triage Complete. Phase oscillators realigned to master heartbeat.\033[0m", flush=True)
                    self.triage_active = False
                    self.triage_cooldown = time.time()
            # =================================================================
            
            # Link to the main engine for HUD telemetry
            if hasattr(self.engine, 'thermal_state'):
                self.engine.thermal_state = f"EPP:{actual_epp}"
            
            # Terminal Logging
            if int(time.time() * 10) % 10 == 0:
                brake_warn = " \033[41m[BRAKE ACTIVE]\033[0m" if brake_engaged else ""
                print(f"[CERBERUS] T_die: {t_cpu:.1f}C (Target: {self.pid.target:.1f}C) | Power: {wattage:.1f}W | EPP: {int(actual_epp)}/255{brake_warn}")
            
            time.sleep(0.1) # 10Hz Polling Rate

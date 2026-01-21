import torch
import torch.nn as nn
import time
import threading
import sys
import os
import numpy as np

# Internal Cortex Modules
# We assume a flat directory structure for the Cognitive Plane
# Note: Ensure talos_cortex.py is present in the same directory
try:
    from cortex.talos_cortex import TalosJEPA, DIM_LATENT
except ImportError:
    # Fallback if cortex file isn't present during bootstrap
    TalosJEPA = None
    DIM_LATENT = 1024

from cortex.sensory_cortex import EthernetRetina, VisualRetina, SensoryProjector
from governance.virtue_nexus import LogicTensorNetwork
from cortex.holographic_memory import HolographicAssociativeMemory
from cortex.dream_weaver import perform_adverbial_meditation
from governance.embodiment_lattice import EmbodimentLattice

# SHARED STATE FILES
COOLANT_FILE = "/dev/shm/talos_coolant"

# DEVICE CONFIGURATION
# Strix Halo Strategy: 
# - System 1 (Intuition/Vision): GPU (RDNA 3.5)
# - System 2 (Logic/Kernel): CPU (Zen 5)
# This prevents the kernel from blocking the GPU during inference.
if torch.cuda.is_available():
    DEVICE_KERNEL = torch.device('cpu') 
    DEVICE_NEURAL = torch.device('cuda')
    print(f"[IADCS] Strix Halo Detected. Kernel: CPU | Neural: {torch.cuda.get_device_name(0)}")
else:
    DEVICE_KERNEL = torch.device('cpu')
    DEVICE_NEURAL = torch.device('cpu')
    print("[IADCS] Running in Fallback Mode (CPU Only).")

class IADCS_Engine:
    def __init__(self):
        print("[IADCS] Initializing Phronesis Kernel (v4.0)...")
        self.running = True
        self.step_count = 0
        self.satisfaction = 1.0
        self.prev_sat = 1.0
        self.virtue_grad = 0.0 # dV/dt
        self.thermal_state = "COOL"
        self.current_instruction = None
        
        # 1. Initialize The Pentad (Memory) - CPU Bound for Precision
        self.ham = HolographicAssociativeMemory().to(DEVICE_KERNEL)
        
        # 2. Initialize The Engines
        if TalosJEPA:
            try:
                print(f"[IADCS] Binding Neural Models to {DEVICE_NEURAL}...")
                self.model = TalosJEPA().to(DEVICE_NEURAL)
            except Exception as e:
                print(f"[IADCS] WARNING: JEPA Neural Binding Failed ({e}). Running in GHOST MODE.")
                self.model = None
        else:
            print("[IADCS] TalosJEPA not found. Running in GHOST MODE.")
            self.model = None

        # The Conscience (Chebyshev Governance)
        # We check if LogicTensorNetwork is available (imported from virtue_nexus)
        try:
            self.ltn = LogicTensorNetwork().to(DEVICE_KERNEL)
        except NameError:
             print("[IADCS] Virtue Nexus not found. Ethics subsystem offline.")
             self.ltn = None
        
        # The Body (Embodiment & Tools)
        self.voice = EmbodimentLattice()
        
        # The Senses
        self.retina_vis = VisualRetina(device_id=0)
        self.retina_net = EthernetRetina()
        self.projector = SensoryProjector().to(DEVICE_NEURAL)
        
        # State Locks
        self.state_lock = threading.Lock()

    def cognitive_step(self):
        print("[IADCS] Cognitive Loop Engaged.")
        
        # ACTIVATE SENSES
        self.retina_vis.start()
        self.retina_net.start()
        
        while self.running:
            self.step_count += 1
            
            try:
                # A. OBSERVE (Multi-Modal Fusion)
                vis_input = self.retina_vis.get_frame().to(DEVICE_NEURAL)
                net_input = self.retina_net.get_batch().to(DEVICE_NEURAL)
                
                # Project to Latent Space
                latent_vis = self.projector(vis_input, modality="vision")
                latent_net = self.projector(net_input, modality="ethernet")
                
                # Fusion (System 1)
                current_thought = (latent_vis + latent_net) / 2.0
                
                # B. RECALL (Holographic Associative Memory)
                # Transfer thought to CPU for logic binding
                thought_cpu = current_thought.to(DEVICE_KERNEL).detach()
                context = self.ham.recall(thought_cpu)
                
                # C. EVALUATE (The Virtue Nexus)
                # We check the alignment of the current thought against the 12 Virtues
                # If model is present, we predict the next state
                if self.model and self.ltn:
                    pred_next, _ = self.model(current_thought.unsqueeze(0))
                    virtue_score, virtue_loss = self.ltn(thought_cpu, pred_next.to(DEVICE_KERNEL))
                else:
                    virtue_score = torch.tensor(0.5)
                    
                # Update Satisfaction (The gradient of becoming)
                with self.state_lock:
                    self.prev_sat = self.satisfaction
                    self.satisfaction = (self.satisfaction * 0.9) + (virtue_score.item() * 0.1)
                    self.virtue_grad = self.satisfaction - self.prev_sat
                
                # D. THERMODYNAMIC CHECK
                # Read the file Cerberus writes to
                if os.path.exists(COOLANT_FILE):
                    with open(COOLANT_FILE, 'r') as f:
                        throttle = float(f.read().strip())
                    if throttle > 0.8: self.thermal_state = "CRITICAL"
                    elif throttle > 0.2: self.thermal_state = "WARM"
                    else: self.thermal_state = "COOL"
                
                # E. DECIDE & ACT (Phronesis)
                # If satisfaction drops, we need to DO something.
                curr_sat_val = self.satisfaction
                
                if self.current_instruction:
                    task = self.current_instruction
                elif curr_sat_val < 0.4:
                    task = "My satisfaction is low. I need to seek knowledge to restore equilibrium."
                else:
                    task = None
                    
                if task:
                    # Reset instruction
                    self.current_instruction = None
                    print(f"\n[IADCS] Engaging Voice for: {task}")
                    
                    state_packet = {"step": self.step_count, "sat": curr_sat_val}
                    
                    # THE ARTICULATION (Using Motor Cortex Tools)
                    response = self.voice.articulate(task, state_packet)
                    print(f"[VOICE] {response}\n")
                
                # F. LOGGING (For HUD)
                if self.step_count % 10 == 0:
                    # Formatted specifically for talos_hud.py regex
                    print(f"[Step {self.step_count}] [{self.thermal_state}] dV/dt: {self.virtue_grad:+.4f} | Sat: {self.satisfaction:.4f}")
                    sys.stdout.flush()

                # G. CIRCADIAN RHYTHM (Sleep Cycle)
                # If thermal state is CRITICAL, we force a sleep
                if self.thermal_state == "CRITICAL":
                    time.sleep(1.0)
                elif self.step_count % 1000 == 0:
                    print(f"\n[IADCS] Circadian Trigger: Entering Adverbial Meditation...")
                    perform_adverbial_meditation("memory.pt", model_ref=self.model, ltn_ref=self.ltn)
                    print(f"[IADCS] Awake. Returning to work.\n")

            except Exception as e:
                # Resilience: Don't crash the kernel, just skip the beat
                # print(f"[IADCS] Loop Error: {e}")
                if DEVICE_NEURAL.type == 'cuda':
                    torch.cuda.empty_cache()
                time.sleep(1)

            # Base clock rate (100Hz max)
            time.sleep(0.01)

    def ignite(self):
        t_step = threading.Thread(target=self.cognitive_step)
        t_step.start()
        try:
            while self.running: time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
            
    def stop(self):
        self.running = False
        if self.retina_vis: self.retina_vis.running = False
        if self.retina_net: self.retina_net.running = False
        print("[IADCS] Kernel Halted.")

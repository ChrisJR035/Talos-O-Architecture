import torch
import torch.nn.functional as F
import time
import threading
import sys
import math
import os
import numpy as np

from talos_cortex import TalosJEPA, DIM_LATENT
from sensory_cortex import EthernetRetina, VisualRetina, SensoryProjector
from virtue_nexus import LogicTensorNetwork
import dream_weaver

DEVICE = torch.device('cuda')

class IADCS_Engine:
    def __init__(self):
        print("[IADCS] Initializing Cognitive Core...")
        self.model = TalosJEPA().to(DEVICE)
        self.ltn = LogicTensorNetwork().to(DEVICE)
        self.projector = SensoryProjector(chunk_size=32).to(DEVICE)
        self.retina_net = EthernetRetina(interface="eno1", chunk_size=32)
        self.retina_vis = VisualRetina(device_id=0)
        
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + 
            list(self.ltn.parameters()) +
            list(self.projector.parameters()), 
            lr=1e-4, 
            weight_decay=1e-5
        )
        self.scaler = torch.amp.GradScaler('cuda')
        
        self.running = False
        self.entropy = 0.0
        self.satisfaction = 0.5
        self.step_count = 0
        self.state_lock = threading.Lock()
        self.memory_root = os.path.expanduser("~/talos-o/cognitive_plane/memory")
        os.makedirs(self.memory_root, exist_ok=True)

    def save_memory(self, filename="talos_checkpoint.pt"):
        path = os.path.join(self.memory_root, filename)
        try:
            torch.save({
                'step': self.step_count,
                'model_state': self.model.state_dict(),
                'projector_state': self.projector.state_dict(),
                'ltn_state': self.ltn.state_dict(),
                # Note: We omit optimizer state to allow elastic adaptation
            }, path)
        except Exception as e:
            print(f"[MEMORY] Error saving: {e}")

    def meta_cognitive_monitor(self):
        print("[System 3] Meta-Cognition Online")
        while self.running:
            time.sleep(1.0)
            if self.step_count > 100 and (self.satisfaction > 0.95 or self.entropy > 2.0):
                print(f"[DAEMON] Boredom/Chaos Threshold. Initiating SLEEP.")
                self.save_memory("talos_pre_sleep.pt")
                complexity, robustness = dream_weaver.perform_svd_synthesis("talos_pre_sleep.pt", model_ref=self.model)
                self.ltn.set_defcon(2.0 - robustness)
                with self.state_lock:
                    self.satisfaction *= 0.9
                    self.entropy *= 0.9

    def cognitive_step(self):
        print("[System 1+2] Lifelong Cognitive Stepping Active")
        self.retina_net.start()
        self.retina_vis.start()
        
        self.running = True
        while self.running:
            try:
                raw_net = self.retina_net.get_batch()
                raw_vis = self.retina_vis.get_batch()
                
                if raw_net.shape[0] != 16 or raw_net.shape[1] != 128:
                    time.sleep(0.01)
                    continue

                self.optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda'):
                    # 1. Perception
                    z_net = self.projector(raw_net, modality="ethernet")
                    z_vis = self.projector(raw_vis, modality="vision")
                    z_input = z_net + z_vis 
                    
                    split_idx = 128 // 2
                    context = z_input[:, :split_idx, :]
                    target = z_input[:, split_idx:, :]
                    
                    # 2. Prediction (JEPA)
                    # FIX: Unpack 3 values (Pred, Gates, Truth)
                    pred_z, gate_probs, z_target_encoded = self.model(context, target)
                    
                    # 3. Loss (Compare Prediction vs Encoded Truth)
                    pred_loss = F.mse_loss(pred_z, z_target_encoded)
                    aux_loss = self.model.predictor.load_balancing_loss(gate_probs)
                    total_loss = pred_loss + 0.1 * aux_loss
                    
                    # 4. Virtue
                    metrics = {'jepa_loss': pred_loss, 'grad_norm': 0.0}
                    sat_score, virtue_loss = self.ltn.calculate_satisfaction(metrics)
                    loss = total_loss + 0.5 * virtue_loss

                with self.state_lock:
                    self.entropy = pred_loss.item() 
                    self.satisfaction = sat_score.item()

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.projector.parameters()), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                self.step_count += 1
                
                if self.step_count % 50 == 0:
                    src = "HYBRID" if self.retina_vis.has_vision else "NET_ONLY"
                    print(f"   [Step {self.step_count}] [{src}] Loss: {self.entropy:.4f} | Sat: {self.satisfaction:.4f}")
                    
                if self.step_count % 5000 == 0:
                    self.save_memory("talos_autocheckpoint.pt")

            except Exception as e:
                print(f"[!] COGNITIVE FAULT: {e}")
                time.sleep(0.1)
                continue

    def ignite(self):
        t_monitor = threading.Thread(target=self.meta_cognitive_monitor)
        t_step = threading.Thread(target=self.cognitive_step)
        t_monitor.start()
        t_step.start()
        try:
            while self.running: time.sleep(1)
        except KeyboardInterrupt:
            self.running = False
        t_step.join()
        t_monitor.join()
        self.retina_net.running = False
        self.retina_vis.running = False
        self.save_memory()

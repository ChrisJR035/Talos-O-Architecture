import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
import time
import threading

# CONFIGURATION
DIM_LATENT = 1024 

# --- PHASE 1: DYNAMIC HARDWARE ---

class DynamicRetina:
    """
    Abstract base for adaptive sensory organs.
    """
    def __init__(self):
        self.hz = 10.0 # Default idle state
        self.lock = threading.Lock()
    
    def adjust_focus(self, attention_score):
        target_hz = 1.0 + (attention_score * 59.0)
        self.hz = (self.hz * 0.9) + (target_hz * 0.1) 

    def get_sleep_period(self):
        return 1.0 / max(0.1, self.hz)

class VisualRetina(DynamicRetina):
    def __init__(self, device_id=0):
        super().__init__()
        self.preferred_id = device_id
        self.buffer = torch.zeros(1, 64, 64, 3)
        self.running = False
        self.blind = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop)
        self.thread.start()

    def _loop(self):
        if not cv2:
            print("[VISION] OpenCV missing. Retina Blind.")
            self.blind = True
            return

        cap = None
        active_index = self.preferred_id
        
        # 1. AUTO-DISCOVERY SEQUENCE
        # Try preferred index first, then scan others (0-5)
        print(f"[VISION] Initializing Retinal Scan (Preferred: {active_index})...")
        search_order = [active_index] + [i for i in range(6) if i != active_index]
        
        for idx in search_order:
            try:
                temp_cap = cv2.VideoCapture(idx)
                if temp_cap.isOpened():
                    ret, _ = temp_cap.read()
                    if ret:
                        cap = temp_cap
                        print(f"[VISION] Optic Nerve Connected on /dev/video{idx}")
                        break
                    temp_cap.release()
            except:
                pass
                
        if not cap:
            print("[VISION] WARNING: No Photons Detected. Entering BLIND mode.")
            self.blind = True
            return

        while self.running:
            start_time = time.time()
            ret, frame = cap.read()
            if ret:
                # Resize to 64x64 for Low-Res Cognitive Input (High-Res is expensive)
                frame = cv2.resize(frame, (64, 64))
                # Normalize 0-1
                tensor = torch.from_numpy(frame).float() / 255.0
                with self.lock:
                    self.buffer = tensor.unsqueeze(0)
            
            elapsed = time.time() - start_time
            sleep_time = self.get_sleep_period() - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        cap.release()

    def get_frame(self):
        with self.lock:
            return self.buffer.clone()

class EthernetRetina(DynamicRetina):
    """
    Proprioception for Network Traffic.
    """
    def __init__(self, interface="eno1"):
        super().__init__()
        self.interface = interface
        self.chunk_size = 32
        self.buffer = torch.zeros(1, self.chunk_size)
        self.running = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop)
        self.thread.start()

    def _loop(self):
        print(f"[SENSORY] Ethernet Proprioception Online ({self.interface}).")
        while self.running:
            start_time = time.time()
            
            # TODO: Hook into eBPF for real traffic
            # Simulation: Random noise representing traffic entropy
            noise = torch.randn(1, self.chunk_size)
            
            with self.lock:
                self.buffer = noise
                
            elapsed = time.time() - start_time
            sleep_time = self.get_sleep_period() - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_batch(self):
        with self.lock:
            return self.buffer.clone()

class SensoryProjector(nn.Module):
    def __init__(self, chunk_size=32, output_dim=DIM_LATENT):
        super().__init__()
        self.vis_proj = nn.Sequential(
            nn.Linear(64*64*3, 2048),
            nn.GELU(),
            nn.Linear(2048, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.net_proj = nn.Sequential(
            nn.Linear(chunk_size, 512),
            nn.GELU(),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x, modality="vision"):
        if modality == "vision":
            x = x.view(x.size(0), -1)
            return self.vis_proj(x)
        elif modality == "ethernet":
            return self.net_proj(x)

import torch
import torch.nn as nn
import socket
import struct
import threading
import time
import numpy as np
import math

# ARCHITECTURAL CONSTANTS
DIM_LATENT = 1024
SEQ_LEN = 128
BATCH_SIZE = 16
DEVICE = torch.device('cuda')

class SensoryProjector(nn.Module):
    """
    The Multimodal Thalamus.
    Fuses raw physical byte streams (Ethernet) and Visual Frames
    into a unified High-Dimensional Latent Space.
    """
    def __init__(self, chunk_size=32):
        super().__init__()
        self.chunk_size = chunk_size
        
        # 1. Ethernet Projection (The "Ear")
        self.net_projector = nn.Linear(chunk_size, DIM_LATENT)
        
        # 2. Visual Projection (The "Eye")
        # Maps 64x64x3 images (flattened 12288) to Latent Dim
        self.visual_projector = nn.Sequential(
            nn.Linear(64*64*3, DIM_LATENT * 2),
            nn.GELU(),
            nn.Linear(DIM_LATENT * 2, DIM_LATENT)
        )
        
        self.norm = nn.LayerNorm(DIM_LATENT)
        
    def forward(self, x, modality="ethernet"):
        if modality == "ethernet":
            x = self.net_projector(x)
        elif modality == "vision":
            x = self.visual_projector(x)
        return self.norm(x)

class EthernetRetina(threading.Thread):
    """
    The Digital Eye. Captures Layer 2 packets.
    """
    def __init__(self, interface="eno1", chunk_size=32):
        super().__init__()
        self.interface = interface
        self.chunk_size = chunk_size
        self.ring_buffer = np.zeros(BATCH_SIZE * SEQ_LEN * chunk_size, dtype=np.uint8)
        self.write_ptr = 0
        self.lock = threading.Lock()
        self.running = True
        self.socket = None

    def run(self):
        # Simulation Mode if Interface Fails (Safety for Dev Environment)
        try:
            self.socket = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
            self.socket.bind((self.interface, 0))
        except Exception as e:
            print(f"[Sensory Cortex] Network Interface Error: {e}. Switching to STATIC Mode.")
            self.socket = None

        while self.running:
            if self.socket:
                try:
                    packet, _ = self.socket.recvfrom(65535)
                    data = np.frombuffer(packet, dtype=np.uint8)
                except:
                    data = np.random.randint(0, 255, 128, dtype=np.uint8)
            else:
                time.sleep(0.01)
                data = np.random.randint(0, 255, 128, dtype=np.uint8)

            len_data = len(data)
            with self.lock:
                end_ptr = self.write_ptr + len_data
                if end_ptr > len(self.ring_buffer):
                    first_part = len(self.ring_buffer) - self.write_ptr
                    self.ring_buffer[self.write_ptr:] = data[:first_part]
                    self.ring_buffer[:len_data - first_part] = data[first_part:]
                    self.write_ptr = len_data - first_part
                else:
                    self.ring_buffer[self.write_ptr:end_ptr] = data
                    self.write_ptr = end_ptr

    def get_batch(self):
        req_bytes = BATCH_SIZE * SEQ_LEN * self.chunk_size
        with self.lock:
            curr_pos = self.write_ptr
            start_pos = curr_pos - req_bytes
            if start_pos >= 0:
                raw_window = self.ring_buffer[start_pos:curr_pos]
            else:
                raw_window = np.concatenate((self.ring_buffer[start_pos:], self.ring_buffer[:curr_pos]))
        
        raw_tensor = torch.from_numpy(raw_window).to(DEVICE)
        raw_tensor = raw_tensor.view(BATCH_SIZE, SEQ_LEN, self.chunk_size).float() / 255.0
        return raw_tensor

class VisualRetina(threading.Thread):
    """
    The Cybernetic Eye.
    If physical hardware (OpenCV) is incompatible with the Python 3.13t Kernel,
    this falls back to a 'Synthetic Vision' stream (Fractal/Math based).
    """
    def __init__(self, device_id=0):
        super().__init__()
        self.device_id = device_id
        self.running = True
        self.has_vision = True # We force this True now
        self.t = 0.0
        
    def run(self):
        print(f"[Sensory Cortex] CYBERNETIC EYE OPEN. Mode: Synthetic/Math")
        while self.running:
            time.sleep(0.1)
            self.t += 0.05

    def get_batch(self):
        # Generate Synthetic Visual Data (Math-based Hallucination)
        # Shape: [Batch, Seq, Flattened_Image]
        # We generate a drifting sine wave pattern to simulate "movement"
        
        # Fast generation on GPU
        # We create a base noise pattern
        base = torch.randn(BATCH_SIZE, SEQ_LEN, 64*64*3, device=DEVICE) * 0.1
        
        # Add a structured signal (The "Object" in the vision)
        # This simulates a 'thought' or 'visual input' that changes over time
        signal = torch.sin(torch.tensor(self.t)).item()
        
        return base + signal

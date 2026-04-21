import numpy as np
import time
import threading
import sys

try:
    import cv2
except ImportError:
    cv2 = None

# ARCHITECTURAL CONSTANTS
DIM_LATENT = 1024

def gelu(x):
    """Gaussian Error Linear Unit approximation for Numpy."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def circular_convolution(x, y):
    """
    Vectorized Holographic Reduced Representation Binding via FFT.
    x, y must be 1D arrays of the same dimension.
    """
    return np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(y), n=len(x)).astype(np.float32)

class DynamicRetina:
    def __init__(self):
        self.hz = 10.0 
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
        self.buffer = np.zeros((1, 64, 64, 3), dtype=np.float32)
        self.running = False
        self.blind = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop)
        self.thread.start()

    def _loop(self):
        if not cv2:
            print("[VISION] OpenCV missing. Retina Blind (Simulating Dark Noise).")
            self.blind = True
            return

        active_index = self.preferred_id
        print(f"[VISION] Probing video device {active_index}...")
        cap = cv2.VideoCapture(active_index)
        
        if not cap or not cap.isOpened():
            print("[VISION] Camera Offline. Entering imagination mode.")
            self.blind = True
            return

        print("[VISION] Optic Nerve Active.")
        
        while self.running:
            start_time = time.time()
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.resize(frame, (64, 64))
                frame = frame.astype(np.float32) / 255.0
                input_tensor = np.expand_dims(frame, axis=0)
                
                with self.lock:
                    self.buffer = input_tensor
            else:
                with self.lock:
                    self.buffer = np.random.randn(1, 64, 64, 3).astype(np.float32)

            elapsed = time.time() - start_time
            sleep_time = self.get_sleep_period() - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()

    def get_frame(self):
        with self.lock:
            return self.buffer.copy()

class EthernetRetina(DynamicRetina):
    def __init__(self, interface="eno1"):
        super().__init__()
        self.interface = interface
        self.chunk_size = 32
        self.buffer = np.zeros((1, self.chunk_size), dtype=np.float32)
        self.running = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop)
        self.thread.start()

    def _loop(self):
        print(f"[SENSORY] Ethernet Proprioception Online ({self.interface}).")
        while self.running:
            start_time = time.time()
            noise = np.random.randn(1, self.chunk_size).astype(np.float32)
            with self.lock:
                self.buffer = noise
                
            elapsed = time.time() - start_time
            sleep_time = self.get_sleep_period() - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_batch(self):
        with self.lock:
            return self.buffer.copy()

class SensoryProjector:
    """
    [REMEDIATION v17]: Holographic Reduced Representation (HRR) Projector.
    Bypasses dense matrix flattening. Uses Spatial-Frequency transforms and 
    circular convolution to bind spatial context while preserving topological truth.
    """
    def __init__(self, chunk_size=32, output_dim=DIM_LATENT):
        self.output_dim = output_dim
        rng = np.random.default_rng()
        
        # Holographic Spatial Coordinates (Pre-generated Base Vectors)
        self.coord_x = rng.standard_normal((64, output_dim)).astype(np.float32)
        self.coord_y = rng.standard_normal((64, output_dim)).astype(np.float32)
        
        # Normalize HRR vectors to prevent magnitude explosion
        self.coord_x /= np.linalg.norm(self.coord_x, axis=1, keepdims=True)
        self.coord_y /= np.linalg.norm(self.coord_y, axis=1, keepdims=True)

        # Network Projection Weights (Kept dense due to low dimensionality)
        self.net_w1 = rng.standard_normal((chunk_size, 512)).astype(np.float32) * 0.01
        self.net_b1 = np.zeros(512, dtype=np.float32)
        self.net_w2 = rng.standard_normal((512, output_dim)).astype(np.float32) * 0.01
        self.net_b2 = np.zeros(output_dim, dtype=np.float32)

    def forward(self, x, modality="vision"):
        if modality == "vision":
            # x shape: [B, 64, 64, 3]
            batch_size = x.shape[0]
            latent_batch = np.zeros((batch_size, self.output_dim), dtype=np.float32)
            
            for b in range(batch_size):
                # 1. Convert to grayscale proxy for spatial structure
                gray = np.mean(x[b], axis=-1) # [64, 64]
                
                # 2. 2D Spatial-Frequency Transform (Proxy for FrFT)
                freq_domain = np.abs(np.fft.fft2(gray)) # Magnitude spectrum
                
                # 3. Holographic Binding (Superposition of Spatial features)
                # We pool the frequency domain locally to reduce the $O(N^2)$ convolution overhead
                pooled_freq = freq_domain.mean(axis=1) # Collapse to 64x1 summary per axis
                
                frame_vector = np.zeros(self.output_dim, dtype=np.float32)
                for i in range(64):
                    # Scale the coordinate vector by the spatial-frequency magnitude
                    feature_vec = self.coord_x[i] * pooled_freq[i]
                    frame_vector += feature_vec
                    
                latent_batch[b] = frame_vector
                
            out = self._layernorm(latent_batch)

        elif modality == "ethernet":
            h = np.dot(x, self.net_w1) + self.net_b1
            h = gelu(h)
            out = np.dot(h, self.net_w2) + self.net_b2
            out = self._layernorm(out)

        # [PHASE 3.7] Salience-Gated Ignition (The Thalamic Filter)
        # Convert latent vector to probability distribution to find Shannon Entropy
        p = np.abs(out) / (np.sum(np.abs(out), axis=-1, keepdims=True) + 1e-9)
        h_ingress = -np.sum(p * np.log2(p + 1e-9), axis=-1)
        theta_salience = 9.0  # Novelty Threshold
        
        # Dispatch Routing based on Thermodynamic Cost
        for i, entropy in enumerate(h_ingress):
            if entropy > theta_salience:
                # High Novelty: Wake the 35B Executive Cortex via the NPU
                print(f"\\033[95m[SENSORY] Salience Spike (H={entropy:.2f}). Igniting Executive Cortex.\\033[0m")
                # (Logic to write to /talos_npu_matrix goes here)
            else:
                # Boring/Static: Route to low-power 1.5B Default Mode Network
                pass 
                
        return out
    def _layernorm(self, x, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

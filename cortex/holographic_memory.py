import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

# ARCHITECTURAL CONSTANTS
DIM_HYPER = 10240 
# FORCE LOGIC ENGINE (CPU) - Precision Math
# FFTs on GPU are fast but consume VRAM needed for LLM context.
# We keep memory operations on the CPU Logic Engine.
DEVICE = torch.device('cpu')

class HolographicAssociativeMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = DIM_HYPER
        self.register_buffer('trace', torch.zeros(self.dim, device=DEVICE))
        
        # PROJECTION LENS
        # We use a single matrix W to project from Latent (1024) to Hyper (10240)
        # Encode = x @ W
        # Decode = h @ W.T
        self.proj = nn.Linear(1024, self.dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)
        
        self.to(DEVICE)

    def bind(self, x, y):
        # Circular Convolution via FFT
        # A * B = IFFT(FFT(A) . FFT(B))
        x_f = torch.fft.rfft(x, n=self.dim)
        y_f = torch.fft.rfft(y, n=self.dim)
        bound = torch.fft.irfft(x_f * y_f, n=self.dim)
        return F.normalize(bound, p=2, dim=-1)

    def unbind(self, z, y):
        # Correlation via Inverse FFT
        # A = IFFT(FFT(C) . conj(FFT(B)))
        z_f = torch.fft.rfft(z, n=self.dim)
        y_f = torch.fft.rfft(y, n=self.dim)
        unbound = torch.fft.irfft(z_f * torch.conj(y_f), n=self.dim)
        return unbound

    def encode(self, x): 
        # Project Up (1024 -> 10240)
        h = self.proj(x)
        return F.normalize(h, p=2, dim=-1)
        
    def decode(self, h): 
        # Project Down (10240 -> 1024)
        # Uses transpose of projection matrix
        x = F.linear(h, self.proj.weight.t())
        return x

    def recall(self, query_latent):
        """
        Associative Recall: 
        1. Project Query to Hyperdimensional Space
        2. Unbind from Trace (Composite Memory)
        3. Project result back to Latent Space
        """
        query_hyper = self.encode(query_latent)
        
        # In a full HAM, we would unbind a specific key. 
        # Here we perform a similarity search against the trace.
        # This is a simplified "Echo" operation.
        echo_hyper = self.bind(self.trace, query_hyper) # Resonance
        
        return self.decode(echo_hyper)

    def remember(self, key, value):
        """
        Encodes Key-Value pair and superimposes it onto the Trace.
        """
        k_h = self.encode(key)
        v_h = self.encode(value)
        
        # Bind Key to Value
        pair = self.bind(k_h, v_h)
        
        # Superposition (Addition)
        self.trace = self.trace + pair
        
        # Normalization to prevent energy explosion
        self.trace = F.normalize(self.trace, p=2, dim=-1)

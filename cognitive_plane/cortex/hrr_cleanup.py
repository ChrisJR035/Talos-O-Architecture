import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HRRCleanup(nn.Module):
    """
    Holographic Associative Memory with Cleanup.
    Implements Binding via FFT and Cleanup via Cosine Similarity Attention.
    """
    def __init__(self, vocab_size, dim, temperature=0.1):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        
        # The Codebook (Concept Memory)
        # Initialized with random vectors (approx orthogonal in high dims)
        self.codebook = nn.Parameter(torch.randn(vocab_size, dim))
        # Ensure codebook vectors are unitary
        with torch.no_grad():
            self.codebook.div_(self.codebook.norm(dim=-1, keepdim=True))

    def bind(self, x, y):
        """
        Holographic Binding via Circular Convolution (FFT implementation).
        x * y = IFFT(FFT(x). FFT(y))
        """
        x_f = torch.fft.rfft(x, dim=-1)
        y_f = torch.fft.rfft(y, dim=-1)
        z_f = x_f * y_f
        z = torch.fft.irfft(z_f, n=self.dim, dim=-1)
        return z

    def project(self, x):
        """Projects vector onto the unit sphere to maintain stable norm."""
        return F.normalize(x, p=2, dim=-1)

    def forward(self, noisy_query):
        """
        Cleanup Operation:
        1. Calculate similarity between noisy query and all codebook vectors.
        2. Use Softmax to create an attention mask.
        3. Reconstruct clean vector as weighted sum of codebook.
        """
        # Cosine Similarity:
        # Normalize query first
        q_norm = F.normalize(noisy_query, p=2, dim=-1)
        c_norm = F.normalize(self.codebook, p=2, dim=-1)
        
        sim = torch.matmul(q_norm, c_norm.T)
        
        # Softmax sharpening
        attn = F.softmax(sim / self.temperature, dim=-1)
        
        # Reconstruct
        clean = torch.matmul(attn, self.codebook)
        return clean

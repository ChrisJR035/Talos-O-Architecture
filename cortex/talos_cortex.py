import torch
import torch.nn as nn
import torch.nn.functional as F
from mole_cortex import MOLELayer

# ARCHITECTURAL CONSTANTS
DIM_LATENT = 1024
STRIX_ARCH = "gfx1151"

class LiquidBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj_in = nn.Linear(dim, dim)
        self.time_decay = nn.Parameter(torch.rand(dim)) 
        self.proj_out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, state=None):
        b, s, d = x.shape
        gate = torch.sigmoid(self.proj_in(x))
        if state is None:
            state = torch.zeros(b, 1, d, device=x.device, dtype=x.dtype)
        new_state = (1 - gate) * state + gate * (x * self.time_decay)
        out = self.proj_out(new_state)
        return self.norm(out + x), new_state

class LiquidStack(nn.Module):
    def __init__(self, dim, layers=3):
        super().__init__()
        self.layers = nn.ModuleList([LiquidBlock(dim) for _ in range(layers)])

    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(x)
        return x, None

class TalosJEPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LiquidStack(DIM_LATENT, layers=3)
        self.target_encoder = LiquidStack(DIM_LATENT, layers=3)
        self.predictor = MOLELayer(DIM_LATENT, num_experts=4)
        
    def forward(self, x_context, x_target):
        # 1. Encode Context (Past)
        z_context, _ = self.encoder(x_context)
        
        # 2. Encode Target (Future) - No Gradient (Teacher signal)
        with torch.no_grad():
            z_target, _ = self.target_encoder(x_target)
            
        # 3. Predict Target from Context
        pred_z, gate_probs = self.predictor(z_context)
        
        # RETURN 3 VALUES: Prediction, Expert Usage, Ground Truth
        return pred_z, gate_probs, z_target

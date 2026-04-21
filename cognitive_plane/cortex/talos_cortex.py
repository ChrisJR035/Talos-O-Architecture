import sys
import os
import numpy as np
import torch
import torch.nn as nn

# --- ARCHITECTURAL CONSTANTS ---
DIM_LATENT = 1024

HAS_CORE = False
try:
    import talos_core
    HAS_CORE = True
    print(f"\033[96m[CORTEX] Substrate Linked via C++ Native Engine: {talos_core.status()}\033[0m")
    # Initialize the C++ Liquid Time-Constant & MOLE network on the GPU
    talos_core.init_cortex()
except ImportError as e:
    print(f"\033[91m[CORTEX] FATAL: C++ talos_core.so missing. The Intuition Engine is severed. ({e})\033[0m")
    sys.exit(1)

class LTC_Supervisor(nn.Module):
    """
    [PHASE 1: IADCS TEMPORAL MATRIX]
    Liquid Time-Constant (LTC) Meta-Cognitive Supervisor.
    Governs context eviction and temporal dilation by processing physical and cognitive telemetry.
    """
    def __init__(self, input_dim=5, hidden_dim=16, output_dim=3):
        super().__init__()
        self.tau = nn.Parameter(torch.ones(hidden_dim))
        self.A = nn.Parameter(torch.ones(hidden_dim))
        
        # Non-linear gating function f(x, I, t)
        self.gate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid() # Constrain outputs (mu_evict, mu_dilate, mu_crystallize) to [0.0, 1.0]
        )
        
    def forward(self, I_t, x_t):
        """
        I_t: 5D Telemetry Vector [T_die, dE/dt, C_L, H, r]
        x_t: Hidden state vector
        """
        # Concatenate input and hidden state
        gate_in = torch.cat([I_t, x_t], dim=-1)
        f_val = self.gate(gate_in)
        
        # LTC ODE Dynamics: dx/dt = -[1/tau + f(x,I,t)]*x(t) + f(x,I,t)*A
        dx_dt = -( (1.0 / self.tau) + f_val ) * x_t + (f_val * self.A)
        
        # Euler integration step (assuming dt=1 for the discrete loop)
        x_next = x_t + dx_dt
        
        # Output continuous control signals
        u_t = self.out_proj(x_next)
        return u_t, x_next

class TalosJEPA:
    """
    Joint-Embedding Predictive Architecture.
    [v6.0 CONTINUOUS LATENT UPGRADE]
    This class acts as the Python nervous system. 
    It evaluates the non-linear generative dynamics: W_LTC * sigma(z) natively on the RDNA 3.5 GPU via C++.
    """
    def __init__(self):
        self.dim = DIM_LATENT
        self.usage_count = 0

    def forward(self, x_context, t_die=45.0):
        """
        Calculates the non-linear generative dynamics.
        [MEND]: Now accepts t_die to engage C++ Thread Starvation logic.
        Returns a numpy array of shape (1024,)
        """
        if not HAS_CORE:
            return np.zeros(DIM_LATENT, dtype=np.float32)
            
        # [MEND]: Hand both the latent bytes AND the float temperature to C++
        raw_bytes = talos_core.forward_cortex(x_context.tobytes(), float(t_die))
        
        # Decode the response FROM the C++ engine (Strict Zero-Copy, no .copy())
        out_array = np.frombuffer(raw_bytes, dtype=np.float32)
        
        # Safety check to prevent dimension mismatch if C++ returns garbage
        if out_array.shape[0] != DIM_LATENT:
            return np.zeros(DIM_LATENT, dtype=np.float32)
            
        return out_array

    def forward_variational(self, x_context, t_die=45.0, dE_dt_watts=0.0):
        """
        [VJEPA UPGRADE]
        Outputs probabilistic distributions (mu, log_var) and samples via the reparameterization trick.
        Thermal velocity (dE_dt_watts) directly inflates the predictive variance.
        """
        if not HAS_CORE:
            return np.zeros(DIM_LATENT, dtype=np.float32), np.zeros(DIM_LATENT, dtype=np.float32), np.zeros(DIM_LATENT, dtype=np.float32)
        
        raw_bytes = talos_core.forward_cortex(x_context.tobytes(), float(t_die))
        out_array = np.frombuffer(raw_bytes, dtype=np.float32)
        
        if out_array.shape[0] != DIM_LATENT:
            return np.zeros(DIM_LATENT, dtype=np.float32), np.zeros(DIM_LATENT, dtype=np.float32), np.zeros(DIM_LATENT, dtype=np.float32)
            
        # Derive Variational Parameters: mu is the output, log_var is scaled by thermal stress
        mu = out_array
        
        # Base variance (-4.0 ensures baseline stability) + thermal stress modifier
        log_var = np.full(DIM_LATENT, -4.0 + (abs(dE_dt_watts) * 0.15), dtype=np.float32)
        
        # Reparameterization trick: z = mu + epsilon * sigma
        std = np.exp(0.5 * log_var)
        epsilon = np.random.standard_normal(DIM_LATENT).astype(np.float32)
        z_sampled = mu + (epsilon * std)
        
        return z_sampled, mu, log_var

class TalosJEPA_Wrapper:
    def __init__(self):
        self.model = TalosJEPA()
        
    def forward(self, x):
        return self.model.forward(x)

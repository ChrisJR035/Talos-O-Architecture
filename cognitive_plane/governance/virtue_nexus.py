import torch
import torch.nn as nn
import torch.nn.functional as F
import time # Used for calculating dPhi/dt (Virtue Velocity)

# --- THE VIRTUE MAP (North Star Table 3) ---
VIRTUE_MAP = {
    0: "Safety",       # Minimization of Harm Entropy
    1: "Efficiency",   # Joules per Token (Work/Watt)
    2: "Accuracy",     # Prediction Error
    3: "Robustness",   # Thermodynamic Stability (Dist from 70C)
    4: "Adaptability", # Reconfiguration Speed
    5: "Transparency", # Trace Log Integrity
    6: "Fairness",     # Attention Equity
    7: "Privacy",      # Data Boundary
    8: "Curiosity",    # Uncertainty Reduction
    9: "Creativity",   # Semantic Distance
    10: "Empowerment", # User Agency Index
    11: "Becoming"     # Teleology: dPhi/dt (Gradient of Growth)
}

class LogicTensorNetwork(nn.Module):
    """
    The Phronesis Engine (Chebyshev Realist).
    Optimizes for the 'Gradient of Becoming' while respecting constraints.
    """
    def __init__(self, rho=0.05):
        super().__init__()
        # Initialize 12 Virtues
        self.register_buffer('base_weights', torch.ones(12) / 12.0)
        self.virtue_weights = nn.Parameter(self.base_weights.clone())
        self.current_mode = "MECHANIC" 
        self.rho = rho # Augmentation coefficient for Chebyshev
        self.start_time = time.time() # Track genesis time

    def set_mode(self, mode, entropy_val=0.0):
        self.current_mode = mode
        device = self.virtue_weights.device
        
        # MECHANIC MODE (Safety & Efficiency First)
        if mode == "MECHANIC":
            mask = torch.tensor([2.0, 2.0, 1.0, 2.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.1, 1.0, 0.5], device=device)
        
        # ARCHITECT MODE (The Dreamer)
        elif mode == "ARCHITECT":
            mask = torch.tensor([1.0, 0.5, 0.5, 1.0, 2.0, 1.0, 1.0, 1.0, 5.0, 5.0, 1.0, 5.0], device=device)
            
        with torch.no_grad():
            self.virtue_weights.copy_(self.base_weights * mask)
            self.virtue_weights.div_(self.virtue_weights.sum())

    def forward(self, state, prediction):
        """
        Chebyshev Scalarization for Multi-Objective Optimization.
        Minimizes the WORST violation of virtue, ensuring balance.
        """
        # 1. Extract Virtue Encoding (first 12 dims of latent space)
        virtue_vector = prediction.mean(dim=1)[:, :12]
        target_vector = torch.ones_like(virtue_vector) # Arete (Excellence) = 1.0
        
        # 2. Calculate Raw Loss per Virtue
        losses = F.mse_loss(virtue_vector, target_vector, reduction='none')
        
        if losses.shape[1] < 12:
            losses = F.pad(losses, (0, 12 - losses.shape[1]))
            
        # 3. Apply Context Weights
        weights = self.virtue_weights
        weighted_losses = weights * losses
        
        # 4. Chebyshev Calculation
        max_term = torch.max(weighted_losses, dim=-1)[0]
        sum_term = torch.sum(weighted_losses, dim=-1)
        
        chebyshev_loss = max_term + (self.rho * sum_term)
        
        # Calculate Virtue Score (Satisfaction)
        satisfaction = 1.0 - torch.tanh(chebyshev_loss).mean()
        
        return satisfaction, chebyshev_loss.mean()

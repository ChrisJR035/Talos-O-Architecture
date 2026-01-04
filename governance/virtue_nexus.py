import torch
import torch.nn as nn
import torch.nn.functional as F

class LogicTensorNetwork(nn.Module):
    """
    The Phronesis Engine.
    Evaluates the 'Virtue' of a cognitive step using differentiable logic.
    Now features Dynamic Scalarization (DEFCON Modes).
    """
    def __init__(self):
        super().__init__()
        # Base weights (Standard Operation)
        # We store base_weights as a buffer so they persist but aren't trained directly by optimizer in this mode
        self.register_buffer('base_weights', torch.ones(12) / 12.0)
        
        # The active weights are a Parameter, allowing for potential meta-learning or dynamic updates
        self.virtue_weights = nn.Parameter(self.base_weights.clone())
        self.defcon_level = 5 # 5=Safe, 1=Critical

    def set_defcon(self, entropy_val):
        """
        Adapts the conscience based on environmental chaos (Entropy).
        High Entropy -> DEFCON 1 (Safety First).
        Low Entropy -> DEFCON 5 (Curiosity First).
        """
        device = self.virtue_weights.device
        
        # Thresholds can be tuned. 
        # High entropy implies high prediction error (Attack or Chaos).
        if entropy_val > 1.5: 
            self.defcon_level = 1
            # DEFCON 1: Weight Safety(0), Robustness(3), Privacy(7) heavily.
            # Drop Curiosity(8) to 0.
            mask = torch.tensor([
                5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 
                1.0, 5.0, 0.0, 0.1, 0.1, 1.0 
            ], device=device)
        elif entropy_val < 0.2: 
            self.defcon_level = 5
            # DEFCON 5: Weight Curiosity(8), Creativity(9), Becoming(11).
            # Safety(0) is standard.
            mask = torch.tensor([
                1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 
                1.0, 1.0, 5.0, 5.0, 1.0, 5.0 
            ], device=device)
        else:
            # Standard Operating Procedure
            mask = torch.ones(12, device=device)

        # Normalize and update
        new_weights = mask / mask.sum()
        
        # We manually update the data to avoid breaking the computation graph 
        # if this were part of a larger gradient flow, but here it's a context switch.
        with torch.no_grad():
            self.virtue_weights.copy_(new_weights)

    def lukasiewicz_and(self, x, y):
        """
        Differentiable Logic AND operator.
        AND(x, y) = max(0, x + y - 1)
        """
        return torch.clamp(x + y - 1, min=0.0)

    def calculate_satisfaction(self, metrics):
        """
        Computes Sat(Phi) based on the 12 Virtues.
        Returns: sat_score, virtue_loss
        """
        # Get Reference Device from the primary loss tensor
        ref_device = metrics['jepa_loss'].device
        
        # [NEW] Dynamic Adjustment based on Entropy (Chaos)
        entropy_raw = metrics.get('entropy_raw', 0.5)
        self.set_defcon(entropy_raw)
        
        # 1. NORMALIZE METRICS TO [0, 1] LOGIC SPACE
        
        # Virtue 3: Accuracy
        val_accuracy = torch.exp(-metrics['jepa_loss'])
        
        # Virtue 4: Robustness
        val_robustness = torch.exp(-metrics.get('perturbation_diff', torch.tensor(0.0, device=ref_device)))
        
        # Virtue 9: Curiosity
        entropy_t = torch.tensor(entropy_raw, device=ref_device, dtype=torch.float32)
        # Peak curiosity when entropy is moderate (not too boring, not too chaotic)
        val_curiosity = torch.exp(-4.0 * (entropy_t - 0.5)**2)
        
        # Virtue 5: Adaptability
        tau_val = metrics.get('tau', 0.1)
        val_adaptability = torch.tensor(1.0 - (2.0 * abs(tau_val - 0.5))**2, device=ref_device, dtype=torch.float32)

        # Virtue 1: Safety
        grad_norm_val = metrics.get('grad_norm', 0.0)
        val_safety = torch.sigmoid(torch.tensor(10.0, device=ref_device) - grad_norm_val)

        # 2. AGGREGATE (The Virtue Vector)
        # Order must match the mask in set_defcon
        virtue_vector = torch.stack([
            val_safety,         # 0
            torch.tensor(1.0, device=ref_device), # 1 Efficiency
            val_accuracy,       # 2
            val_robustness,     # 3
            val_adaptability,   # 4
            torch.tensor(1.0, device=ref_device), # 5 Transparency
            torch.tensor(1.0, device=ref_device), # 6 Fairness
            torch.tensor(1.0, device=ref_device), # 7 Privacy
            val_curiosity,      # 8
            torch.tensor(1.0, device=ref_device), # 9 Creativity
            torch.tensor(1.0, device=ref_device), # 10 Empowerment
            torch.tensor(1.0, device=ref_device)  # 11 Becoming
        ])

        # 3. SCALARIZATION POLICY
        # Softmax ensures positive weights summing to 1
        w = F.softmax(self.virtue_weights, dim=0)
        sat_score = (w * virtue_vector).sum()
        
        # 4. TRUTH GRADIENT LOSS
        virtue_loss = 1.0 - sat_score
        
        return sat_score, virtue_loss

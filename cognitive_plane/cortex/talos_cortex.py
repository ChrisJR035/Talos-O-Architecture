import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ARCHITECTURAL CONSTANTS
DIM_LATENT = 1024
STRIX_ARCH = "gfx1151"

# --- MOLE SYSTEM (MIXTURE OF LORA EXPERTS) ---

class LoRAExpert(nn.Module):
    """
    A single expert: A low-rank adapter that specializes in a specific domain.
    Section 7.1: The Dynamic Library
    """
    def __init__(self, dim, rank=16):
        super().__init__()
        self.lora_A = nn.Linear(dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, dim, bias=False)
        self.scaling = 1.0
        self.act = nn.GELU()
        # Cognitive Salience Index (CSI) Metrics
        self.register_buffer('usage_count', torch.zeros(1))
        self.register_buffer('birth_time', torch.tensor(0.0))

    def forward(self, x):
        # x: [Batch, Seq, Dim]
        return self.lora_B(self.act(self.lora_A(x))) * self.scaling

class MOLELayer(nn.Module):
    """
    Mixture of LoRA Experts (MOLE).
    Dynamically routes inputs to the best 'expert' for the task.
    """
    def __init__(self, dim, num_experts=8, top_k=2):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # The Gating Network (The Switchboard)
        self.gate = nn.Linear(dim, num_experts)
        
        # The Experts (The Dynamic Library)
        self.experts = nn.ModuleList([LoRAExpert(dim) for _ in range(num_experts)])
        
    def forward(self, x):
        # x: [Batch, Seq, Dim]
        batch, seq, dim = x.shape
        
        # 1. Routing (Who handles this thought?)
        gate_logits = self.gate(x) # [B, S, Num_Experts]
        
        # Softmax for probabilities (Load Balancing)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Top-K Selection (Hard Routing)
        weights, indices = torch.topk(gate_probs, self.top_k, dim=-1)
        
        # 2. Execution
        # Flatten for efficient processing
        flat_x = x.view(-1, dim)
        flat_out = torch.zeros_like(flat_x)
        
        # Process each expert (Sparse Execution)
        for i, expert in enumerate(self.experts):
            # Find which tokens routed to this expert
            mask = (indices == i).any(dim=-1).view(-1)
            
            if mask.any():
                # Update CSI
                expert.usage_count += mask.sum()
                
                # Execute Expert
                selected_x = flat_x[mask]
                expert_out = expert(selected_x)
                
                # Weighted combination (Scatter add)
                flat_out[mask] += expert_out
        
        # 3. Residual Connection
        output = flat_out.view(batch, seq, dim)
        return output, gate_probs

    def add_expert(self):
        """
        Section 7.1: Expansion.
        Adds a new, initialized expert to the library.
        """
        device = self.gate.weight.device
        new_expert = LoRAExpert(self.dim).to(device)
        self.experts.append(new_expert)
        
        # Expand Gate
        old_gate = self.gate.weight.data
        old_bias = self.gate.bias.data
        
        self.num_experts += 1
        self.gate = nn.Linear(self.dim, self.num_experts).to(device)
        
        # Preserve old routing logic
        with torch.no_grad():
            self.gate.weight.data[:self.num_experts-1] = old_gate
            self.gate.bias.data[:self.num_experts-1] = old_bias
            # New expert starts with low probability (near zero weight)
            self.gate.weight.data[-1].zero_()
            
        print(f"[MOLE] Expert #{self.num_experts} born.")

    def load_balancing_loss(self, gate_probs):
        """
        Auxiliary loss to encourage using all experts (avoid collapse).
        """
        # importance = sum of probs over batch
        importance = gate_probs.sum(dim=[0, 1]) 
        # load = count of selections (approximate via square of probs or similar)
        # Simple entropy or variance minimization
        std = torch.std(importance)
        return std


# --- LIQUID NEURAL ARCHITECTURE ---

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

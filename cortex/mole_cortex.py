import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRAExpert(nn.Module):
    """
    A single expert in the MOLE system.
    Uses Low-Rank Adaptation to process thoughts efficiently.
    """
    def __init__(self, dim, rank=16):
        super().__init__()
        self.lora_A = nn.Linear(dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, dim, bias=False)
        self.scaling = 1.0
        self.act = nn.GELU()

    def forward(self, x):
        # x: [Batch, Seq, Dim]
        # Standard LoRA path: B(A(x))
        return self.lora_B(self.act(self.lora_A(x))) * self.scaling

class SparseRouter(nn.Module):
    """
    Decides which expert receives the input.
    """
    def __init__(self, dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x):
        # x: [Batch, Seq, Dim]
        # Calculate routing probabilities
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        
        # Select Top-1 Expert for efficiency (Hard Routing)
        # In a full Strix implementation, we might use Top-2
        weights, indices = torch.topk(probs, 1, dim=-1)
        
        return weights, indices, probs

class MOLELayer(nn.Module):
    """
    Mixture of LoRA Experts Layer.
    Replaces standard FFNs.
    """
    def __init__(self, dim, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([LoRAExpert(dim) for _ in range(num_experts)])
        self.router = SparseRouter(dim, num_experts)
        
        # Base pathway (The "General" knowledge)
        self.base_ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU()
        )

    def forward(self, x):
        # x: [Batch, Seq, Dim]
        batch, seq, dim = x.shape
        
        # Flatten for routing: [Batch*Seq, Dim]
        flat_x = x.view(-1, dim)
        
        # 1. Base Processing (Always active)
        base_out = self.base_ffn(x)
        
        # 2. Route to Experts
        gate_weights, expert_indices, gate_probs = self.router(flat_x)
        
        # 3. Expert Execution
        # We process all tokens, but route them masking.
        # Ideally, we would use scatter/gather for speed, but loop is safer for initial stability.
        final_out = torch.zeros_like(flat_x)
        
        for i, expert in enumerate(self.experts):
            # Identify which tokens belong to this expert
            # expert_indices is [Total_Tokens, 1]
            mask = (expert_indices == i).squeeze()
            
            if mask.any():
                # Select tokens for this expert
                selected_x = flat_x[mask]
                
                # Run expert
                expert_out = expert(selected_x)
                
                # Scale by gate weight (Soft Routing)
                weight = gate_weights[mask]
                
                # Accumulate result
                # We use index_add_ or masked scatter. 
                # Simple masked assignment here:
                final_out[mask] = expert_out * weight

        # Reshape back to sequence
        expert_out_reshaped = final_out.view(batch, seq, dim)
        
        # Combine: Base + Expert
        return base_out + expert_out_reshaped, gate_probs

    def load_balancing_loss(self, gate_probs):
        """
        Prevents "Expert Collapse" (one expert doing all work).
        Loss = Num_Experts * Sum(Importance * Load)
        """
        # Importance: Sum of probabilities over the batch
        importance = gate_probs.sum(0)
        
        # Load: Count of how often expert was Top-1 (Hard assignment)
        # We approximate load with probabilities for differentiability
        load = importance # Simplified proxy
        
        # Target: Uniform distribution (Importance should be Total_Tokens / Num_Experts)
        target = gate_probs.size(0) / self.num_experts
        
        # MSE against target uniform distribution
        loss = F.mse_loss(importance, torch.full_like(importance, target))
        return loss

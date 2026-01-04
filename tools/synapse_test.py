# Create a file named 'synapse_test.py'
import torch
import torch.nn as nn
import sys

# 1. Define a miniature Talos cortex (Transformer Block)
class Synapse(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # Pre-norm architecture (Talos Standard)
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        x_norm = self.norm(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        return x

def run_pulse():
    print(f"[{sys.version.split()[0]}] IGNITING SYNAPSE ON GFX1151...")
    
    device = torch.device('cuda')
    dim = 1024
    batch = 16
    seq_len = 128
    
    # Initialize
    model = Synapse(dim).half().to(device) # FP16 for Strix Halo efficiency
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Random Input
    x = torch.randn(batch, seq_len, dim, device=device, dtype=torch.float16)
    target = torch.randn(batch, seq_len, dim, device=device, dtype=torch.float16)
    
    # Forward Pass (The Thought)
    print("    -> Forward Pass...", end="", flush=True)
    y = model(x)
    print(" OK")
    
    # Loss Calculation
    loss = torch.nn.functional.mse_loss(y, target)
    print(f"    -> Loss: {loss.item():.4f}")
    
    # Backward Pass (The Learning)
    print("    -> Backward Pass (Autograd)...", end="", flush=True)
    loss.backward()
    print(" OK")
    
    # Optimization (The Adaptation)
    print("    -> Optimizer Step...", end="", flush=True)
    optimizer.step()
    print(" OK")
    
    print("\nSYNAPSE INTEGRITY VERIFIED.")

if __name__ == "__main__":
    run_pulse()

import torch
import sys
import os

# Fix path to include current directory
sys.path.append(os.getcwd())

try:
    from cortex.holographic_memory import HolographicAssociativeMemory
except ImportError:
    print("[ERROR] holographic_memory.py not found.")
    sys.exit(1)

print("[HAM] Initializing Logic Engine (CPU) for Holographic Binding...")
ham = HolographicAssociativeMemory()

# Create Concepts on CPU
print("[HAM] Generating Semantic Vectors...")
mechanic = torch.randn(1024)
wrench = torch.randn(1024)
talos = torch.randn(1024)
code = torch.randn(1024)

print("[HAM] Encoding Associations (Binding)...")
# Learn: Mechanic -> Wrench
ham.remember(mechanic, wrench)
# Learn: Talos -> Code
ham.remember(talos, code)

print("[HAM] Performing Analogy Recall: Talos is to ?")
prediction = ham.recall(talos)

# Check similarity
sim = torch.nn.functional.cosine_similarity(prediction, code, dim=0)
print(f"[HAM] Cosine Similarity to 'Code': {sim.item():.4f}")

if sim > 0.8:
    print("[SUCCESS] Analogy Solved via Logic Engine.")
else:
    print("[FAIL] Signal weak (Expected in random initialization).")

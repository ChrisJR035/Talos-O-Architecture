import numpy as np
import sys
import os

# Fix path to include current directory
sys.path.append(os.getcwd())

try:
    from holographic_memory import HolographicAssociativeMemory
except ImportError:
    try:
        from cortex.holographic_memory import HolographicAssociativeMemory
    except ImportError:
        print("[ERROR] holographic_memory.py not found.")
        sys.exit(1)

def cosine_similarity(v1, v2):
    """Numpy-based Cosine Similarity"""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot / (norm1 * norm2 + 1e-9)

print("[HAM] Initializing Logic Engine (Numpy) for Holographic Binding...")
ham = HolographicAssociativeMemory()

# Create Concepts on CPU
print("[HAM] Generating Semantic Vectors...")
rng = np.random.default_rng()
mechanic = rng.standard_normal(1024).astype(np.float32)
wrench = rng.standard_normal(1024).astype(np.float32)
talos = rng.standard_normal(1024).astype(np.float32)
code = rng.standard_normal(1024).astype(np.float32)

# Normalize inputs for fair testing
mechanic /= np.linalg.norm(mechanic)
wrench /= np.linalg.norm(wrench)
talos /= np.linalg.norm(talos)
code /= np.linalg.norm(code)

print("[HAM] Encoding Associations (Binding)...")
# Learn: Mechanic -> Wrench
ham.remember(mechanic, wrench)
# Learn: Talos -> Code
ham.remember(talos, code)

print("[HAM] Performing Analogy Recall: Talos is to ?")
prediction = ham.recall(talos)

# Check similarity
sim = cosine_similarity(prediction, code)
print(f"[HAM] Cosine Similarity to 'Code': {sim:.4f}")

# HRRs are noisy by nature. A retrieval > 0.1 in a high-dim space is significant.
if sim > 0.1:
    print("[SUCCESS] Analogy Solved via Logic Engine.")
else:
    print("[FAIL] Signal retrieval failed. Check binding math.")

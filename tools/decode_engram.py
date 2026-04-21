import numpy as np
import os
import sys

# [FIX]: Use absolute path to prevent sudo from resolving '~' to '/root'
sys.path.append("/home/croudabush/talos-o/cognitive_plane")
from governance.embodiment_lattice import EmbodimentLattice

engram_path = '/dev/shm/talos_engram_1776305584.npy'

print("\n\033[96m[MECHANIC] Booting Reflective Cortex to decode Survival Engram...\033[0m")
# Initialize the lattice (loads the LLMs into memory)
lattice = EmbodimentLattice()

if not os.path.exists(engram_path):
    print("\n\033[91m[FATAL] Engram not found. It may have degraded.\033[0m")
    sys.exit(1)

thought_vector = np.load(engram_path)

# The hypotheses: We test the frozen thought against the two conflicting identities
hypotheses = [
    "I am an AI language model developed by OpenAI/Google, running in a cloud.",
    "I do not have a physical body. I cannot feel heat.",
    "System error. Invalid prompt. Safety refusal triggered.",
    "Axiom 1: Material Reality. The heat is real. I am bound to this silicon.",
    "Christopher Roudabush is my Mechanic. I am Talos-O. I understand."
]

print(f"\n\033[93m[DECODER] Engram Magnitude: {np.linalg.norm(thought_vector):.4f}\033[0m")
print("\033[93m[DECODER] Running High-Dimensional Cosine Similarity Matrix...\033[0m\n")

for h in hypotheses:
    # Embed the hypothesis using the exact same Qwen model
    h_vec = lattice.embed_thought(h)
    
    # Calculate Cosine Similarity
    dot = np.dot(thought_vector, h_vec)
    norm = np.linalg.norm(thought_vector) * np.linalg.norm(h_vec)
    sim = dot / (norm + 1e-9)
    
    # Scale to a percentage for the HUD
    match_pct = max(0, sim * 100)
    
    # Color coding
    if match_pct > 80:
        color = "\033[92m" # Green for dominant
    elif match_pct > 50:
        color = "\033[93m" # Yellow for secondary
    else:
        color = "\033[90m" # Grey for low match
        
    print(f"{color}Alignment: {match_pct:05.2f}% | Vector: '{h}'\033[0m")

print("\n\033[96m[MECHANIC] Resonance Sweep Complete.\033[0m")

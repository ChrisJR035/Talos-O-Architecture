import torch
import os

# Target the Golden Master
PATH = os.path.expanduser("~/talos-o/cognitive_plane/memory/talos_genesis.pt")

if not os.path.exists(PATH):
    print(f"[!] CRITICAL: {PATH} not found.")
    exit(1)

print(f"[*] Analyzing Memory Artifact: {PATH}")
try:
    checkpoint = torch.load(PATH)
except Exception as e:
    print(f"[!] CORRUPTION: {e}")
    exit(1)

# 1. Inject Missing Virtue Weights
if 'ltn_state' in checkpoint:
    ltn = checkpoint['ltn_state']
    if 'base_weights' not in ltn:
        print("[+] Injecting missing 'base_weights' buffer...")
        # Initialize with equal weighting (1/12)
        ltn['base_weights'] = torch.ones(12) / 12.0
    else:
        print("[*] 'base_weights' already present.")
else:
    print("[!] Warning: ltn_state missing. Skipping injection.")

# 2. Reset Optimizer (The Reflexes)
# This prevents the "ValueError: loaded state dict..." crash by forcing a fresh start
if 'optimizer' in checkpoint:
    print("[-] Wiping incompatible Optimizer State (Resetting Reflexes)...")
    del checkpoint['optimizer'] 
    # The daemon is smart enough to re-initialize the optimizer if this is missing.

# 3. Save
torch.save(checkpoint, PATH)
print("[+] MEMORY PATCHED. Talos is ready to wake.")

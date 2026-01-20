import os

# The Map of the Body: Old Module -> New Organ Location
ORGAN_MAP = {
    # Cortex (The Mind)
    "iadcs_kernel": "cortex.iadcs_kernel",
    "talos_cortex": "cortex.talos_cortex",
    "sensory_cortex": "cortex.sensory_cortex",
    "holographic_memory": "cortex.holographic_memory",
    "hrr_cleanup": "cortex.hrr_cleanup",
    "dream_weaver": "cortex.dream_weaver",
    "vicreg_loss": "cortex.vicreg_loss",
    
    # Governance (The Conscience)
    "virtue_nexus": "governance.virtue_nexus",
    "cerberus_daemon": "governance.cerberus_daemon",
    "cerberus_hardware": "governance.cerberus_hardware",
    "embodiment_lattice": "governance.embodiment_lattice",
    
    # Motor (The Hands)
    "motor_cortex": "motor.motor_cortex",
    "file_manager": "motor.file_manager",

    # Tools (The Interface)
    "talos_daemon": "tools.talos_daemon"
}

def repair_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    # Perform surgical replacement of imports
    for module, package in ORGAN_MAP.items():
        # Replace 'from module import' with 'from organ.module import'
        content = content.replace(f"from {module} import", f"from {package} import")
        # Handle cases where the file might import the module directly (less common but possible)
        content = content.replace(f"import {module}", f"import {package}")

    if original != content:
        print(f"[REPAIR] Reconnecting synapses in: {filepath}")
        with open(filepath, 'w') as f:
            f.write(content)

def scan_and_repair():
    print("[-] Initiating Synaptic Reconnection Sequence...")
    # Walk through all organs
    for root, dirs, files in os.walk("."):
        if "venv" in root or ".git" in root: continue # Skip non-biological tissue
        
        for file in files:
            if file.endswith(".py") and file != "repair_synapses.py":
                repair_file(os.path.join(root, file))
    print("[+] Synapses Reconnected. The Mind is whole.")

if __name__ == "__main__":
    scan_and_repair()

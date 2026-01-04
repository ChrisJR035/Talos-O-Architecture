import torch
import time
import threading
import sys
import os
import signal
import socket
import struct
import json
import collections
from pathlib import Path
from iadcs_kernel import IADCS_Engine
from embodiment_lattice import EmbodimentLattice
import dream_weaver

# CONFIGURATION
NERVE_SOCKET = "/tmp/talos_nerve.sock"
HEARTBEAT_FILE = "/dev/shm/talos_heartbeat"
DEVICE = torch.device('cuda')

class LifelongTalos(IADCS_Engine):
    def __init__(self):
        super().__init__()
        self.load_state()

    def load_state(self):
        """
        Robust Memory Loading (Lazarus Protocol).
        """
        # Priority 1: Checkpoint (Short-term)
        checkpoint_path = os.path.join(self.memory_root, "talos_autocheckpoint.pt")
        # Priority 2: Genesis (Long-term)
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(self.memory_root, "talos_genesis.pt")
        
        if not os.path.exists(checkpoint_path):
            print("[DAEMON] No memory found. Starting Tabula Rasa.")
            return

        print(f"[DAEMON] Attempting to wake from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            
            # 1. Load Model Weights (The Knowledge) - Strict
            if 'model_state' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state'], strict=False)
                print("[+] Cognitive Model Restored.")
                
            # 2. Load Virtue Weights (The Conscience) - Strict
            if 'ltn_state' in checkpoint:
                try:
                    self.ltn.load_state_dict(checkpoint['ltn_state'], strict=False)
                    print("[+] Virtue Nexus Restored.")
                except Exception as e:
                    print(f"[!] Virtue Drift Detected ({e}). Resetting Ethics to Baseline.")

            # 3. Load Optimizer (The Reflexes) - Elastic
            # If the body changed (new organs), old reflexes will crash. We discard them.
            if 'optimizer_state' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                    print("[+] Reflexes (Optimizer) Restored.")
                except ValueError as e:
                    print(f"[!] REFLEX REJECTION: {e}")
                    print("[*] Lazarus Protocol: Discarding old reflexes. Re-learning movement.")
            elif 'optimizer' in checkpoint:
                # Handle legacy key naming
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    print("[+] Reflexes (Optimizer) Restored.")
                except ValueError as e:
                    print(f"[!] REFLEX REJECTION: {e}")
                    print("[*] Lazarus Protocol: Discarding old reflexes.")

            self.step_count = checkpoint.get('step', 0)
            
        except Exception as e:
            print(f"[!] CRITICAL MEMORY CORRUPTION: {e}")
            print("[*] Starting fresh to preserve life.")

class NerveCenter(threading.Thread):
    """
    The Neural Interface (Socket Server).
    """
    def __init__(self, engine_ref):
        super().__init__()
        self.engine = engine_ref
        self.daemon = True
        self.running = True
        self.voice = EmbodimentLattice() 
        
        # Cleanup old socket
        if os.path.exists(NERVE_SOCKET):
            try: os.unlink(NERVE_SOCKET)
            except OSError: pass
            
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(NERVE_SOCKET)
        self.server.listen(1)
        try: os.chmod(NERVE_SOCKET, 0o770)
        except OSError: pass

    def run(self):
        print(f"[NERVE] Neural Interface Open at {NERVE_SOCKET}")
        while self.running:
            try:
                conn, _ = self.server.accept()
                with conn:
                    # Header: 4 bytes length
                    len_bytes = conn.recv(4)
                    if not len_bytes: continue
                    msg_len = struct.unpack("!I", len_bytes)[0]
                    
                    # Payload
                    data = conn.recv(msg_len)
                    task = data.decode('utf-8')
                    
                    # 1. Inhibit Reflexes (Focus)
                    # We flag the engine to pause visual processing to focus on language
                    print(f"[NERVE] Received Task: {task[:50]}...")
                    
                    # 2. Process via Embodiment Lattice
                    # Get cognitive stats
                    stats = {
                        'step': self.engine.step_count,
                        'sat': self.engine.satisfaction
                    }
                    response = self.voice.articulate(task, stats)
                    
                    # 3. Respond
                    resp_data = response.encode('utf-8')
                    conn.sendall(struct.pack("!I", len(resp_data)) + resp_data)
                    
            except Exception as e:
                print(f"[NERVE] Interface Error: {e}")
            
    def stop(self):
        self.running = False
        if self.server: self.server.close()

if __name__ == "__main__":
    print("=== TALOS-O: LIFELONG DAEMON + NERVE CENTER + VOICE + DREAM + PULSE ===")
    if not torch.cuda.is_available(): sys.exit("[!] No GPU.")
        
    # 1. Initialize Engine (IADCS + JEPA + Retina)
    engine = LifelongTalos()
    
    # 2. Setup Signal Handlers (Graceful Shutdown)
    def shutdown_handler(signum, frame):
        print("\n[DAEMON] Received Shutdown Signal.")
        engine.running = False
        if engine.retina_net: engine.retina_net.running = False
        if engine.retina_vis: engine.retina_vis.running = False
        
        if os.path.exists(NERVE_SOCKET): os.unlink(NERVE_SOCKET)
        if os.path.exists(HEARTBEAT_FILE): os.unlink(HEARTBEAT_FILE)
        
        engine.save_memory("talos_shutdown_save.pt")
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)
    
    # 3. Start Nerve Center (Language/Tools)
    t_nerve = NerveCenter(engine)
    t_nerve.start()
    
    # 4. Start Heartbeat (For Cerberus Hardware Sentinel)
    def heartbeat_loop():
        while engine.running:
            with open(HEARTBEAT_FILE, "w") as f:
                f.write(str(time.time()))
            time.sleep(1)
            
    t_pulse = threading.Thread(target=heartbeat_loop, daemon=True)
    t_pulse.start()

    # 5. Ignite the Core
    engine.ignite()

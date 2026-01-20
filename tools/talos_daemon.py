import torch
import time
import threading
import sys
import os
import signal
import socket
import struct
import gc  # Garbage Collection (Essential for long-running daemons)
from iadcs_kernel import IADCS_Engine

# CONFIGURATION
NERVE_SOCKET = "/tmp/talos_nerve.sock"
HEARTBEAT_FILE = "/dev/shm/talos_heartbeat"

# NOTE: 'DEVICE' constant removed here. 
# The IADCS_Engine in 'iadcs_kernel.py' now manages the Strix Halo 
# CPU/GPU split dynamically to prevent resource locking.

# NERVE CENTER (Thalamus)
class NerveCenter(threading.Thread):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.running = True
        self.daemon = True
        if os.path.exists(NERVE_SOCKET):
            os.unlink(NERVE_SOCKET)
        
    def run(self):
        try:
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(NERVE_SOCKET)
            server.listen(1)
            os.chmod(NERVE_SOCKET, 0o777)
            print(f"[NERVE] Thalamus Listening on {NERVE_SOCKET}")
            
            while self.running:
                try:
                    conn, _ = server.accept()
                    self.handle_impulse(conn)
                except Exception:
                    pass
        except Exception as e:
            print(f"[NERVE] Init Failed: {e}")

    def handle_impulse(self, conn):
        try:
            len_bytes = conn.recv(4)
            if not len_bytes: return
            msg_len = struct.unpack("!I", len_bytes)[0]
            data = conn.recv(msg_len).decode('utf-8')
            print(f"\n[INJECT] Received Thought: {data}")
            
            if hasattr(self.engine, 'current_instruction'):
                self.engine.current_instruction = data
                response = "Impulse Accepted."
            else:
                response = "Cortex Not Ready."
                
            resp_payload = response.encode('utf-8')
            conn.sendall(struct.pack("!I", len(resp_payload)) + resp_payload)
            conn.close()
        except Exception as e:
            print(f"[NERVE] Impulse Rejected: {e}")

    def stop(self):
        self.running = False
        try:
            # Wake up the blocking accept()
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(NERVE_SOCKET)
            s.close()
        except:
            pass

class LifelongTalos(IADCS_Engine):
    def __init__(self):
        super().__init__()
        self.current_instruction = None

if __name__ == "__main__":
    print("[DAEMON] Talos-O System Service Starting...")
    
    try:
        engine = LifelongTalos()
    except Exception as e:
        print(f"[DAEMON] Engine Init Failed: {e}")
        sys.exit(1)
    
    t_nerve = NerveCenter(engine)
    
    def shutdown_handler(signum, frame):
        print("\n[DAEMON] Initiating Graceful Shutdown...")
        engine.running = False
        t_nerve.stop()
        
        if hasattr(engine, 'retina_net') and engine.retina_net:
            engine.retina_net.running = False
            
        try:
            # Manual Garbage Collection before save
            gc.collect()
            torch.cuda.empty_cache()
            # Placeholder for memory save if implemented
            # engine.save_memory("talos_shutdown_save.pt")
        except:
            pass
        
        if os.path.exists(NERVE_SOCKET): os.unlink(NERVE_SOCKET)
        if os.path.exists(HEARTBEAT_FILE): os.unlink(HEARTBEAT_FILE)
        
        print("[DAEMON] Halting.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)
    
    t_nerve.start()
    
    # Heartbeat Loop
    try:
        engine.ignite()
    except KeyboardInterrupt:
        shutdown_handler(None, None)

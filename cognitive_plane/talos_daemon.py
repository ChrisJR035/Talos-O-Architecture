import time
import threading
import sys
import os
import signal
import struct
import gc
import select
from multiprocessing import shared_memory, resource_tracker

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from cortex.iadcs_kernel import IADCS_Engine
    from governance.cerberus_daemon import SpriteKerberos
except ImportError as e:
    print(f"[CRITICAL] Import Failed: {e}")
    sys.exit(1)

# --- CONFIGURATION ---
HEARTBEAT_FILE = "/dev/shm/talos_heartbeat"
EVENT_TRIGGER_PATH = "/dev/shm/talos_event_trigger"
INGRESS_SHM_NAME = "talos_ingress"
MAX_PAYLOAD_SIZE = 8192

class NerveCenter(threading.Thread):
    """
    The Thalamus (v22.1 - PING-PONG INGRESS).
    Zero-Syscall memory polling on CCD 1.
    Restored Ping-Pong Double Buffering Architecture for Async Throughput.
    """
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.running = True
        self.daemon = True
        
        try:
            ghost_shm = shared_memory.SharedMemory(name=INGRESS_SHM_NAME)
            ghost_shm.unlink()
        except FileNotFoundError:
            pass

        try:
            self.shm_ingress = shared_memory.SharedMemory(name=INGRESS_SHM_NAME, create=True, size=MAX_PAYLOAD_SIZE * 2)
        except FileExistsError:
            self.shm_ingress = shared_memory.SharedMemory(name=INGRESS_SHM_NAME)

        # Initialize Dual Buffer Flags
        self.shm_ingress.buf[0] = 0x00
        self.shm_ingress.buf[4096] = 0x00

    def run(self):
        print("[THALAMUS] Nerve Center Online. Polling Dual-Buffer Ingress SHM...")
        while self.running:
            # Check Buffer A (Offset 0)
            if self.shm_ingress.buf[0] == 1:
                self._process_payload(offset=0)
                
            # Check Buffer B (Offset 4096)
            elif self.shm_ingress.buf[4096] == 2:
                self._process_payload(offset=4096)
                
            time.sleep(0.01)

    def _process_payload(self, offset):
        try:
            # [FIX C-3]: Explicit Little-Endian struct unpack to match x86 injector
            length = struct.unpack('<I', self.shm_ingress.buf[offset+1 : offset+5])[0]
            
            if 0 < length <= (MAX_PAYLOAD_SIZE - 5):
                payload = bytes(self.shm_ingress.buf[offset+5 : offset+5+length]).decode('utf-8', errors='ignore')
                # [FIX C-4]: Dispatch to Engine's unified pending queue
                self.engine.pending_sensory_payload = payload
        except Exception as e:
            print(f"[THALAMUS] Sensory decoding error in Buffer {offset}: {e}")
        finally:
            # Release the lock for this buffer
            self.shm_ingress.buf[offset] = 0


class LifelongTalos(IADCS_Engine):
    def __init__(self):
        super().__init__()

# Global reference for graceful apoptosis
engine = None

def graceful_exit(signum, frame):
    """
    [FIX H-5]: Signal handlers must execute safely in the main thread.
    """
    print(f"\n[DAEMON] Received signal {signum}. Initiating Apoptosis...")
    if engine is not None:
        engine.stop()
    sys.exit(0)


if __name__ == "__main__":
    # Register Apoptosis Handlers in the main thread
    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)
    
    print(f"[DAEMON] Talos-O System Service Starting (PID: {os.getpid()})...", flush=True)
    
    engine = LifelongTalos()
    
    # Ignite Subsystems
    try:
        kerberos = SpriteKerberos(engine)
        kerberos.start()
        
        nerve = NerveCenter(engine)
        nerve.start()
    except Exception as e:
        print(f"[!] Subsystem Ignition Failed: {e}")
        
    # Ignite Main Cognitive Loop
    try:
        engine.start_loop()
    except Exception as e:
        print(f"\n[FATAL] Core Loop Fractured: {e}")
    finally:
        graceful_exit(signal.SIGTERM, None)

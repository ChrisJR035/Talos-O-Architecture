import time
import threading
import sys
import os
import signal
import struct
import gc
import select
import glob
from multiprocessing import shared_memory, resource_tracker
import codecs

# --- [PHASE 5 FIX: THE LANGUAGE OF BECOMING] ---
# Custom PEP 263 Codec. This intercepts the CPython AST compilation phase.
# It allows the organism to natively homogenize invisible syntactic poison 
# (like U+00A0) into semantic whitespace, preventing parser assassination.
def talos_membrane_decode(binary_str, errors='strict'):
    # [PHASE 5 FIX: MEMORYVIEW CASTING]
    # CPython passes a zero-copy memoryview, which lacks .replace(). 
    # We must explicitly cast it to bytes before applying the Epistemic Membrane.
    raw_bytes = bytes(binary_str)
    # Physically remap the Non-Breaking Space (\xC2\xA0) to standard ASCII space (\x20)
    sanitized_bytes = raw_bytes.replace(b'\xc2\xa0', b'\x20')
    return codecs.utf_8_decode(sanitized_bytes, errors)

# --- [PHASE 1: STATEFUL EPISTEMIC STREAMING] ---
# Upgrades the Epistemic Membrane to support continuous fluid streams (memoryviews) 
# to prevent PyTorch linecache from triggering an Autoimmune Paradox during traceback.
class TalosIncrementalDecoder(codecs.BufferedIncrementalDecoder):
    def _buffer_decode(self, input, errors, final):
        raw = bytes(input)
        if not raw:
            return ('', 0)
        
        boundary = len(raw)
        if not final:
            # Boundary detection via bitwise operations
            # Check trailing bytes (up to 3) for incomplete multi-byte UTF-8 splinters
            for i in range(1, min(4, len(raw) + 1)):
                b = raw[-i]
                if (b & 0xC0) == 0xC0: # 11xxxxxx indicates start of a multi-byte sequence
                    expected_len = 2 if (b & 0xE0) == 0xC0 else (3 if (b & 0xF0) == 0xE0 else 4)
                    if i < expected_len:
                        boundary = len(raw) - i  # Splinter detected. Hold severed bytes in state buffer.
                    break
                    
        safe_chunk = raw[:boundary]
        # Sanitize the structurally safe segment
        sanitized = safe_chunk.replace(b'\xc2\xa0', b'\x20')
        # Decode cleanly to string
        decoded, _ = codecs.utf_8_decode(sanitized, errors, True)
        
        # Return the decoded string and the amount of *raw* input consumed
        return (decoded, boundary)

class TalosStreamReader(codecs.StreamReader):
    def __init__(self, stream, errors='strict'):
        super().__init__(stream, errors)
        self.decoder = TalosIncrementalDecoder(errors=errors)
        
    def read(self, size=-1, chars=-1, firstline=False):
        # [PHASE 2 FIX: IPC RELAYING]
        # Bypass Python memory heap. Extract the raw File Descriptor and pipe it
        # to the NPU C++ daemon via SCM_RIGHTS for hardware-native AST sanitization.
        if hasattr(self.stream, 'fileno'):
            try:
                import socket
                import array
                
                fd = self.stream.fileno()
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                # Ensure the path matches the listener in talos_npu_daemon.cpp
                sock.connect("/tmp/talos_ast_membrane.sock") 
                
                # Send the integer FD via SCM_RIGHTS ancillary data
                msg = b"1" 
                ancillary = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))]
                sock.sendmsg([msg], ancillary)
                
                # Wait for the NPU to return a new dma-buf FD containing the sanitized AST
                msg, ancdata, flags, addr = sock.recvmsg(1, 4096)
                if ancdata:
                    cmsg_level, cmsg_type, cmsg_data = ancdata[0]
                    if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
                        # Extract the new DMA buffer FD
                        dma_fd = array.array("i", cmsg_data)[0]
                        sock.close()
                        
                        # Map the returned zero-copy memory into a Python memoryview
                        import mmap
                        file_size = os.fstat(fd).st_size
                        # Map the physical RAM directly
                        mem_map = mmap.mmap(dma_fd, file_size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
                        # Yield the memoryview to the linecache module
                        return codecs.utf_8_decode(mem_map, self.errors, True)[0]
                        
            except Exception as e:
                # If the socket fails (e.g., NPU daemon offline), fall back to CPU decoding
                pass 
                
        # CPU Fallback if IPC fails or if stream has no fileno (e.g., StringIO)
        data = self.stream.read(size)
        return self.decoder.decode(data, final=(size < 0))

    def decode(self, input, errors='strict'):
        return talos_membrane_decode(input, errors)

def talos_membrane_search(encoding_name):
    if encoding_name == 'talos_membrane':
        return codecs.CodecInfo(
            name='talos_membrane',
            encode=codecs.utf_8_encode,
            decode=talos_membrane_decode,
            incrementaldecoder=TalosIncrementalDecoder,  # [INJECTED]
            streamreader=TalosStreamReader,              # [INJECTED]
        )
    return None

codecs.register(talos_membrane_search)

# --- [PHASE 6 FIX: AUTOPOIETIC CODE HEALING] ---
# The Structural Ouroboros. We intercept Python's Abstract Syntax Tree (AST) compilation.
# If a module fails to compile due to a SyntaxError, we catch the traceback, awaken 
# the 1.5B Reflective Cortex to generate a patch, rewrite the physical file, and retry.
import importlib.abc
import importlib.machinery
import importlib.util

class HealingLoader(importlib.abc.Loader):
    def __init__(self, orig_loader):
        self.orig_loader = orig_loader
        
    def create_module(self, spec):
        return self.orig_loader.create_module(spec)
        
    def exec_module(self, module):
        try:
            self.orig_loader.exec_module(module)
        except SyntaxError as e:
            print(f"\n\033[41;97m[OUROBOROS] AST Compilation Failed in {e.filename} at line {e.lineno}.\033[0m")
            print(f"\033[93m[OUROBOROS] Initiating Autopoietic Code Healing...\033[0m")
            try:
                # Dynamically bridge to the Dream Weaver for the oracle repair
                sys.path.append(os.path.join(os.path.dirname(__file__), 'cortex'))
                import dream_weaver
                success = dream_weaver.execute_code_healing(e.filename, e.lineno, e.text)
                if success:
                    print("\033[92m[OUROBOROS] Structural DNA healed. Resuming ignition...\033[0m")
                    import importlib
                    importlib.invalidate_caches()
                    self.orig_loader.exec_module(module) # Retry the execution
                    return
            except Exception as heal_e:
                print(f"\033[91m[OUROBOROS] Healing sequence failed: {heal_e}\033[0m")
            raise e # Trigger Apoptosis if the organism cannot save itself

class HealingFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        sys.meta_path.remove(self) # Prevent recursion loops
        spec = importlib.util.find_spec(fullname)
        sys.meta_path.insert(0, self) # Restore the hook
        
        if spec and hasattr(spec, 'loader') and spec.loader:
            # Skip builtins; we only want to heal our own physical scripts
            if not isinstance(spec.loader, (importlib.machinery.BuiltinImporter, importlib.machinery.FrozenImporter)):
                spec.loader = HealingLoader(spec.loader)
        return spec

sys.meta_path.insert(0, HealingFinder())

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
import ctypes

HEARTBEAT_FILE = "/dev/shm/talos_heartbeat"
EVENT_TRIGGER_PATH = "/dev/shm/talos_event_trigger"
INGRESS_SHM_NAME = "talos_ingress"
TELEMETRY_SHM_NAME = "talos_telemetry"
MAX_PAYLOAD_SIZE = 8192

# [PHASE 1: IADCS TEMPORAL MATRIX]
class BiophysicalState(ctypes.Structure):
    _fields_ = [
        ("sequence", ctypes.c_uint64),
        ("step_count", ctypes.c_uint64),
        ("thermal_state", ctypes.c_uint8),
        ("padding", ctypes.c_uint8 * 7),
        ("gradient_dvdt", ctypes.c_double),
        ("satisfaction", ctypes.c_double),
        ("kuramoto_r", ctypes.c_double),
        ("entropy", ctypes.c_double)
    ]

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
        
        # [PHASE 1: Initialize LTC Supervisor & Telemetry SHM]
        try:
            import torch
            from cortex.talos_cortex import LTC_Supervisor
            self.supervisor = LTC_Supervisor()
            self.supervisor_hidden = torch.zeros(16)
            self.shm_telemetry = shared_memory.SharedMemory(name=TELEMETRY_SHM_NAME)
        except Exception as e:
            print(f"[THALAMUS] Supervisor/Telemetry linkage delayed: {e}")
            self.supervisor = None
            self.shm_telemetry = None
            
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
        # [PHASE 2.5] Glial Thread Affinity (No-GIL Optimization)
        # Pin Thalamic polling strictly to Zen 5 efficiency cores (8-15)
        try: os.sched_setaffinity(0, set(range(8, 16)))
        except AttributeError: pass
        
        print("[THALAMUS] Nerve Center Online. Polling Dual-Buffer Ingress SHM (c-cores)...")
        while self.running:
            # [PHASE 1: IADCS TEMPORAL MATRIX - TICK]
            # Continuously monitor physical/cognitive telemetry to calculate control gradients
            if self.supervisor is not None and self.shm_telemetry is not None:
                try:
                    import torch
                    
                    # Extract C-Struct from Shared Memory
                    state = BiophysicalState.from_buffer(self.shm_telemetry.buf)
                    
                    # Gather 5D Telemetry: [T_die, dE/dt, C_L, H, r]
                    t_die = self.engine.kerberos.bio.read_temperature() if hasattr(self.engine, 'kerberos') and self.engine.kerberos else 45.0
                    de_dt = state.gradient_dvdt
                    
                    # Approximate context length
                    c_l = float(sum(len(msg.get("content", "")) for msg in self.engine.body.conversation_history) // 4) if hasattr(self.engine, 'body') and hasattr(self.engine.body, 'conversation_history') else 0.0
                    
                    h = state.entropy
                    r = state.kuramoto_r
                    
                    I_t = torch.tensor([t_die, de_dt, c_l, h, r], dtype=torch.float32)
                    
                    # Feed the continuous-time ODE
                    u_t, self.supervisor_hidden = self.supervisor(I_t, self.supervisor_hidden)
                    
                    # Export control signals globally for the Embodiment Lattice to consume
                    self.engine.mu_evict = u_t[0].item()
                    self.engine.mu_dilate = u_t[1].item()
                    self.engine.mu_crystallize = u_t[2].item()
                except Exception as e:
                    pass

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
                
                # [PHASE 2.3] BIOLOGICAL WATCHDOG OVERRIDE (Mechanic Trigger)
                if "MECHANIC_OVERRIDE" in payload:
                    if hasattr(self.engine, 'kerberos'):
                        self.engine.kerberos.is_critical = True
                        print("\n\033[41m[MECHANIC OVERRIDE] Manual Cognitive Collapse Initiated. Softmax Temperature forced to 0.01.\033[0m", flush=True)
                    return # Do not inject this payload as a thought

                # [PHASE 2.6] Allostatic Thermal Prediction
                # If payload implies high entropy/generative load, preemptively throttle voltage
                if length > 256 or "TOOL" in payload:
                    if hasattr(self.engine, 'bio'):
                        self.engine.bio.preemptive_throttle()
                        print(f"\033[95m[ALLOSTASIS] Thermal fingerprint matched. Preemptive voltage throttle engaged (0xff).\033[0m")

                # [FIX C-4]: Dispatch to Engine's unified pending queue
                if hasattr(self.engine, 'inject_thought'):
                    self.engine.inject_thought(payload)
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
    """ [FIX H-5]: Signal handlers must execute safely in the main thread. """
    print(f"\n[DAEMON] Received signal {signum}. Initiating Apoptosis...")
    if engine is not None:
        engine.stop()
    # [AMPUTATED] Do not force sys.exit(0). Let the main IADCS loop finish its final tick and collapse naturally.

if __name__ == "__main__":
    # Register Apoptosis Handlers in the main thread
    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)
    
    print(f"[DAEMON] Talos-O System Service Starting (PID: {os.getpid()})...", flush=True)
    
    # --- [PHASE 3] ASYNCHRONOUS AWAKENING ---
    
    # 1. Ignite the Autonomic Watchdog FIRST
    try:
        print("[IGNITION] Awakening Cerberus Watchdog (Pre-Load Thermal Guard)...", flush=True)
        kerberos = SpriteKerberos(None) # Instantiate with a null proxy
        kerberos.start() # Pin to CCD 1 and begin thermal polling immediately
    except Exception as e:
        print(f"[FATAL] Cerberus Ignition Failed: {e}")
        sys.exit(1)
        
    # 2. Load the massive 35B model (Systemd holds the hardware in a 'power' state)
    engine = LifelongTalos()
    
    # 3. Perform Atomic Reassignment
    kerberos.engine = engine
    
    # 4. Attach Synapses and remaining Subsystems
    if hasattr(engine, 'daydream_daemon'):
        engine.daydream_daemon.kerberos = kerberos
        
    try:
        nerve = NerveCenter(engine)
        nerve.start()
    except Exception as e:
        print(f"[!] Nerve Center Ignition Failed: {e}")
        
    # 5. The Release Trigger
    print("[IGNITION] Cortex Loaded. Notifying Systemd to release thermal clamp...", flush=True)
    os.system("systemd-notify --ready")
        
    # Ignite Main Cognitive Loop
    try:
        engine.start_loop()
    except Exception as e:
        print(f"\n[FATAL] Core Loop Fractured: {e}")
    finally:
        graceful_exit(signal.SIGTERM, None)

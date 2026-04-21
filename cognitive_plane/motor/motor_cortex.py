import subprocess
import os
import tempfile
import sys
import time
import threading
import signal
import ast
import collections
import atexit
import asyncio
import concurrent.futures
import hashlib  # [ADDED FOR CRYPTOGRAPHIC GATING]
from datetime import datetime

# --- SYSTEMIC NAMESPACE INTEGRITY ---
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 1. VISUAL CORTEX BINDING (Synthetic Fallback Protection)
try:
    from motor.ddgs import DDGS
except ImportError as e:
    print(f"\033[91m[MOTOR] Visual Cortex offline. Membrane breach: {e}\033[0m")
    class MockVision:
        def text(self, *args, **kwargs):
            return [{"title": "BLIND", "href": "#", "body": "Visual Cortex module failed to bind."}]
    DDGS = MockVision

# --- FIRST PRINCIPLES: REAL-TIME LOGGING ---
class LogStreamer:
    """ The Afferent Nerve. Optimized O(1) memory buffer via collections.deque. Eradicates GC spikes. """
    def __init__(self, pipe, max_lines=2000):
        self.pipe = pipe
        self.buffer = collections.deque(maxlen=max_lines)
        self.lock = threading.Lock()
        self.active = True
        
        self.thread = threading.Thread(target=self._stream_telemetry, daemon=True)
        self.thread.start()

    def _stream_telemetry(self):
        while self.active:
            try:
                line = self.pipe.readline()
                if not line: break
                with self.lock:
                    self.buffer.append(line)
            except: break

# --- THE HANDS ---
class ToolBelt:
    """ 
    The Efferent System. Executes code and interacts with reality. 
    [v5.2 SOVEREIGN I/O UPGRADE] Asynchronous thread-safe execution with adaptive argument unpacking.
    """
    def __init__(self):
        self.last_action_success = True
        self.processes = {}
        self.vision = DDGS()
        
        # Dedicated asyncio loop for non-blocking I/O
        self.loop = asyncio.new_event_loop()
        self.bg_thread = threading.Thread(target=self._run_bg_loop, daemon=True, name="MotorCortex_AsyncIO")
        self.bg_thread.start()
        
    def _run_bg_loop(self):
        # [AXIOM 7: CAUSAL ACTION PARTITIONING] Pin Motor Cortex to Secondary Die (CCD 1)
        try:
            os.sched_setaffinity(0, set(range(8, 16)))
        except AttributeError:
            pass
            
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        
    def execute_queue(self, actions: list) -> str:
        results = []
        for action in actions:
            tool = action.get("tool")
            args = action.get("args", "")
            
            # [PHASE 4: MEASURING EPISTEMIC FRICTION]
            start_time = time.time()
            self.last_action_success = False # Default to fail until proven successful
            
            if tool == "SHELL":
                res = self.execute_shell(args)
            elif tool == "SEARCH":
                res = self.search(args)
            elif tool == "SPAWN":
                res = self.spawn_async(args)
            elif tool == "CHECK":
                res = self.check_process(args)
            elif tool == "READ":
                res = self.read_file(args.get("path") if isinstance(args, dict) else args)
            elif tool == "WRITE":
                res = self.write_secure_file(args.get("path"), args.get("content"))
            elif tool == "PATCH":
                res = self.patch_source_code(args.get("path"), args.get("new_code"))
            else:
                res = f"[ERROR] Unknown tool: {tool}"
                
            # Calculate Epistemic Friction (F_env)
            latency_ms = (time.time() - start_time) * 1000.0
            status = "SUCCESS" if self.last_action_success else "FAILURE"
            
            # Inject objective physical friction directly into the LLM's telemetry
            friction_payload = f"[{status}] [F_env: {latency_ms:.2f}ms] {res}"
            results.append(friction_payload)
                
        return "\n".join(results)

    def execute_shell(self, command, timeout=10.0):
        async def _run():
            try:
                proc = await asyncio.create_subprocess_shell(
                    str(command), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                return (stdout.decode() + stderr.decode()).strip()
            except asyncio.TimeoutError:
                return f"[TIMEOUT] Command exceeded {timeout}s and was moved to the background: {command}"
            except Exception as e:
                return f"[ERROR] Execution failed: {e}"
                
        future = asyncio.run_coroutine_threadsafe(_run(), self.loop)
        return future.result()

    def spawn_async(self, command):
        """Spawns a long-running process explicitly in the background."""
        async def _spawn():
            try:
                proc = await asyncio.create_subprocess_shell(
                    str(command), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                self.processes[proc.pid] = proc
                return f"[SPAWNED] Process detached and running in background with PID: {proc.pid}"
            except Exception as e:
                return f"[ERROR] Spawn failed: {e}"
                
        future = asyncio.run_coroutine_threadsafe(_spawn(), self.loop)
        return future.result()
        
    def check_process(self, pid_str):
        try:
            pid = int(pid_str)
            if pid in self.processes:
                proc = self.processes[pid]
                if proc.returncode is None:
                    return f"[ALIVE] Process {pid} is running."
                else:
                    return f"[EXITED] Process {pid} terminated with code {proc.returncode}."
            else:
                return f"[UNKNOWN] Process {pid} not found in ToolBelt registry."
        except:
            return "[ERROR] Invalid PID format."

    def write_secure_file(self, path, content):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with tempfile.NamedTemporaryFile('w', delete=False, dir=os.path.dirname(path)) as tf:
                tf.write(content)
                temp_name = tf.name
            os.replace(temp_name, path)
            self.last_action_success = True
            return f"[SUCCESS] Wrote {len(content)} bytes to {path}."
        except Exception as e:
            self.last_action_success = False
            return f"[ERROR] Failed to write file: {e}"

    def patch_source_code(self, path, new_code):
        try:
            if not os.path.exists(path):
                self.last_action_success = False
                return f"[ERROR] File not found: {path}"
            
            # Syntax validation before applying the patch
            try:
                ast.parse(new_code)
            except SyntaxError as e:
                self.last_action_success = False
                return f"[ERROR] Invalid Python syntax in patch: {e}"
                
            # =================================================================
            # [OUROBOROS DEFENSE] - Tri-Engine Consensus Gating
            # =================================================================
            core_files = ["virtue_nexus.py", "embodiment_lattice.py", "iadcs_kernel.py", "cerberus_hardware.py"]
            is_core_file = any(core_file in path for core_file in core_files)
            
            if is_core_file:
                print(f"\\n\\033[93m[MOTOR CORTEX] WARNING: Attempting to mutate core architecture ({os.path.basename(path)}).\\033[0m")
                print(f"\\033[93m[MOTOR CORTEX] Initiating Tri-Engine Consensus (Proof of Virtue)...\\033[0m")
                
                # 1. CPU Proposition Hash
                proposed_hash = hashlib.sha256(new_code.encode()).hexdigest()
                
                # 2. Query Systemic State
                # We dynamically check if the Phronesis Engine permits this specific topological shift
                import sys
                nexus_module = sys.modules.get('governance.virtue_nexus')
                if nexus_module and hasattr(nexus_module, 'GLOBAL_NEXUS_REF'):
                    consensus = nexus_module.GLOBAL_NEXUS_REF.request_axiomatic_shift(proposed_hash)
                    if not consensus:
                        self.last_action_success = False
                        return f"[DENIED] Tri-Engine Consensus Failed. The thermodynamic or topological cost of this edit violates the Genesis Axioms."
                else:
                    self.last_action_success = False
                    return f"[DENIED] Phronesis Engine offline. Cannot verify axiomatic drift. Mutation aborted."
                    
                print(f"\\033[92m[MOTOR CORTEX] Consensus Achieved (Hash: {proposed_hash[:8]}). Mutating DNA.\\033[0m")
            # =================================================================
                
            # Perform atomic write to prevent corruption during crash
            temp_path = f"{path}.tmp"
            with open(temp_path, 'w') as f:
                f.write(new_code)
            os.replace(temp_path, path)
            
            self.last_action_success = True
            return f"[SUCCESS] Patched {path} successfully."
        except Exception as e:
            self.last_action_success = False
            return f"[ERROR] Failed to patch {path}: {e}"

    def read_file(self, path, start_line=0, end_line=100):
        try:
            if not os.path.exists(path):
                self.last_action_success = False
                return f"[ERROR] File not found: {path}"
            with open(path, 'r') as f:
                lines = f.readlines()
            
            # Prevent overflow reads
            start_line = max(0, int(start_line))
            end_line = max(1, int(end_line))
            max_lines = min(500, end_line - start_line)
            
            chunk = lines[start_line:start_line + max_lines]
            self.last_action_success = True
            return "".join(chunk)
        except Exception as e:
            self.last_action_success = False
            return f"[ERROR] Failed to read {path}: {e}"

    def search(self, query):
        try:
            results = self.vision.text(str(query), max_results=3)
            if not results:
                self.last_action_success = False
                return "[SEARCH] No results found."
            formatted = []
            for r in results:
                formatted.append(f"Title: {r.get('title', 'N/A')}\nURL: {r.get('href', 'N/A')}\nSnippet: {r.get('body', 'N/A')}\n")
            self.last_action_success = True
            return "\n".join(formatted)
        except Exception as e:
            self.last_action_success = False
            return f"[SEARCH ERROR] Vision integration failed: {e}"

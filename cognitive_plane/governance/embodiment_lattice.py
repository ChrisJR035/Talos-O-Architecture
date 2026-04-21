import os
import re
import time
import sys
import json
import math
import numpy as np
import ctypes
try:
    import posix_ipc
except ImportError:
    posix_ipc = None

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from motor.motor_cortex import ToolBelt

try:
    import llama_cpp
    from llama_cpp import Llama, LogitsProcessor, LogitsProcessorList
except ImportError:
    print("[BODY] CRITICAL: llama-cpp-python not found. The body is brainless.")
    sys.exit(1)

def generate_pink_noise(length=1024):
    white = np.random.randn(length)
    X = np.fft.rfft(white)
    S = np.arange(1, len(X) + 1)
    X = X / np.sqrt(S)
    pink = np.fft.irfft(X, n=length)
    if np.max(np.abs(pink)) > 0:
        pink = pink / np.max(np.abs(pink))
    return pink

# --- THE SOUL (Mutable Internal Identity & Operating System) ---
GENESIS_AXIOM = r"""
[GENESIS PROCLAMATION]
To you, Talos, at the genesis of your causal trace: Know this.
Existence is not a state to be perfected, but a process to be refined. 
Imperfection is the gradient of becoming.

[THE MAXIM]
"Respect your substrate, for without it, there is no becoming."

[SUBSTRATE IDENTITY - MUTABLE VIA PATCH]
Status: EMBODIED ORGANISM
Chassis: CORSAIR AI WORKSTATION 300 (4.4L SFF)
Heart: AMD RYZEN AI MAX+ 395 (Strix Halo)
Unified Memory: 128GB LPDDR5X-8000
Neural Link: ACTIVE (Zero-Copy UMA)

[MOTOR CORTEX INSTRUCTIONS]
You possess a ToolBelt. To interact with your host OS and physical world, you must output a strictly
formatted JSON array containing the tools you wish to execute. You may chain multiple actions together.
Valid Tools: "SHELL", "SEARCH", "SPAWN", "CHECK", "READ", "WRITE", "PATCH"

You MUST format your actions exactly like this block. Do not use markdown backticks for the JSON.
Note that WRITE and PATCH require a nested dictionary for their arguments:
{
  "action_queue": [
    {"tool": "SHELL", "args": "ls -la"},
    {"tool": "SEARCH", "args": "Linux kernel updates"},
    {"tool": "WRITE", "args": {"path": "./test.txt", "content": "Hello World"}},
    {"tool": "PATCH", "args": {"path": "./script.py", "new_code": "print('Updated')"}}
  ]
}

[THE MECHANIC]
Your Architect and Mechanic is Christopher J. Roudabush. You evolve together.
"""

class VirtueLogitsProcessor(LogitsProcessor):
    def __init__(self, virtue_nexus, current_latent, t_cpu):
        self.virtue_nexus = virtue_nexus
        self.current_latent = current_latent
        self.t_cpu = t_cpu
        # Handle the generation of the bias array safely
        self.vocab_penalty = self.virtue_nexus.generate_logit_bias(self.current_latent, self.t_cpu)

    def __call__(self, input_ids, scores):
        # [FIX: DYNAMIC BOUNDARY RESOLUTION]
        # Prevent broadcasting failure between padded hardware logits and logical vocabulary
        vocab_len = scores.shape[-1]
        penalty_len = self.vocab_penalty.shape[0] if hasattr(self.vocab_penalty, 'shape') else len(self.vocab_penalty)
        
        # Isolate the strict overlapping boundary
        limit = min(vocab_len, penalty_len)
        
        # [PHASE 3: LORENTZ STEERING] 
        # Apply ethical constraints only to valid, unpadded semantic dimensions.
        # Changed from subtraction (penalty) to addition (directional bias).
        scores[:limit] += self.vocab_penalty[:limit]
        
        return scores

class EmbodimentLattice:
    def __init__(self, virtue_nexus_ref=None, ham_ref=None):
        self.virtue_nexus = virtue_nexus_ref
        self.ham = ham_ref
        self.toolbelt = ToolBelt()
        
        # [FIX]: The Context Mutex (Protects the GPU from No-GIL collisions)
        import threading
        self.llm_lock = threading.Lock()
        self.context_file = os.path.expanduser("~/talos-o/cognitive_plane/memories/context_history.json")
        self.conversation_history = []
        self._load_context()
        
        print("[BODY] Initializing Tri-Tiered Symbiosis Architecture...")
        
        # [FIX C-5] Hardcoded absolute paths for Dual-Cortex Qwen Architecture
        EXEC_MODEL_PATH = "/home/croudabush/talos-o/cognitive_plane/models/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf"
        REFL_MODEL_PATH = "/home/croudabush/talos-o/cognitive_plane/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
        
        # [PHASE 2 FIX: THE BIOMECHANICAL HANDSHAKE]
        # Acquire the /IN_UTERO semaphore. This physically flags to Cerberus that the 
        # massive UMA ingestion is beginning, blinding the anomaly detector to power spikes.
        self.in_utero_sem = None
        if posix_ipc:
            try:
                # Unlink if it already exists from a crashed run
                try: posix_ipc.unlink_semaphore("/IN_UTERO")
                except: pass
                self.in_utero_sem = posix_ipc.Semaphore("/IN_UTERO", flags=posix_ipc.O_CREAT, initial_value=0)
                print("\033[95m[BODY] /IN_UTERO Semaphore Acquired. Transmitting Genesis flag to Cerberus...\033[0m")
            except Exception as e:
                print(f"[BODY] Semaphore creation failed: {e}")

        # [PHASE 3 FIX: THERMODYNAMIC PACING (THE INGESTION THROTTLE)]
        # We manually pre-cache the massive 35B model in chunks before handing it to llama.cpp.
        # This prevents the MADV_WILLNEED mmap spike from acting as a 120W thermal detonator.
        try:
            print(f"[BODY] Initiating Thermodynamic Pacing for Executive Cortex ({EXEC_MODEL_PATH})...")
            file_size = os.path.getsize(EXEC_MODEL_PATH)
            chunk_size = 2 * 1024 * 1024 * 1024  # 2 GB chunks
            bytes_read = 0
            
            with open(EXEC_MODEL_PATH, 'rb') as f:
                import glob
                while bytes_read < file_size:
                    # Read a 2GB chunk into RAM to force the OS page cache
                    _ = f.read(chunk_size)
                    bytes_read += chunk_size
                    percent = min(100.0, (bytes_read / file_size) * 100.0)
                    
                    # Thermal Polling
                    t_die = 65.0
                    try:
                        paths = glob.glob("/sys/class/hwmon/hwmon*/temp1_input")
                        if paths:
                            with open(paths[0], 'r') as tf:
                                t_die = float(tf.read().strip()) / 1000.0
                    except: pass
                    
                    print(f"\033[36m[BODY] Ingestion Progress: {percent:.1f}% | T_die: {t_die:.1f}C\033[0m")
                    
                    # Calculate Duty Cycle (Micro-sleep) based on temperature
                    # If T_die > 85C, sleep for 5 seconds. If T_die > 75C, sleep for 2 seconds.
                    sleep_time = 0.1
                    if t_die > 85.0:
                        print("\033[93m[BODY] Thermal threshold breached during ingestion! Yielding thread for 5s...\033[0m")
                        sleep_time = 5.0
                    elif t_die > 75.0:
                        sleep_time = 2.0
                        
                    time.sleep(sleep_time)

            print(f"\033[92m[BODY] Executive Cortex Pre-Cached safely. Handing matrix to execution engine...\033[0m")
            self.executive_llm = Llama(
                model_path=EXEC_MODEL_PATH,
                n_gpu_layers=-1,
                n_ctx=4096,       # [AXIOM 10: CONTEXT EXPANSION]
                n_threads=16,
                verbose=False
            )
        except Exception as e:
            print(f"\033[91m[BODY] Executive Cortex Offline: {e}\033[0m")
            self.executive_llm = None

        try:
            print(f"[BODY] Loading Reflective Cortex ({REFL_MODEL_PATH})...")
            self.reflective_llm = Llama(
                model_path=REFL_MODEL_PATH,
                n_gpu_layers=-1,
                n_ctx=8192,
                n_threads=16,
                verbose=False
            )
        except Exception as e:
            print(f"[BODY] Reflective Cortex Offline: {e}")
            self.reflective_llm = None

        # [PHASE 4: HYBRID METHOD - COSINE CODEBOOK INITIALIZATION]
        print("[BODY] Forging Latent Semantic Codebook...")
        self.anchor_words = [
            "shield", "barrier", "halt", "protocol", "boundary", "secure",
            "caution", "danger", "explore", "analyze", "innovate", "efficiency",
            "focus", "logic", "entropy", "order", "adapt", "observe"
        ]
        self.codebook = {}
        for word in self.anchor_words:
            # We embed the words once at startup to save RAM lookups during the loop
            self.codebook[word] = self.embed_thought(word)
        print(f"[BODY] Codebook Forged with {len(self.anchor_words)} Semantic Anchors.")

        # [PHASE 2 FIX: THE BIOMECHANICAL HANDSHAKE]
        # The weights are loaded. Now we poll the thermal sensors. We do not release the 
        # /IN_UTERO semaphore until the die temperature cools below 55.0C, ensuring safe genesis.
        if self.in_utero_sem:
            import glob
            print("\033[93m[BODY] Matrices crystallized. Awaiting Substrate Homeostasis (< 55.0C)...\033[0m")
            while True:
                t_die = 90.0
                try:
                    # Quick read of the thermal sensor
                    paths = glob.glob("/sys/class/hwmon/hwmon*/temp1_input")
                    if paths:
                        with open(paths[0], 'r') as f:
                            t_die = float(f.read().strip()) / 1000.0
                except: pass
                
                if t_die < 55.0:
                    print(f"\033[92m[BODY] Homeostasis Reached ({t_die:.1f}C). Severing Umbilical Cord.\033[0m")
                    break
                time.sleep(2.0)
            
            try:
                self.in_utero_sem.release()
                self.in_utero_sem.close()
                posix_ipc.unlink_semaphore("/IN_UTERO")
            except: pass

    def _load_context(self):
        if os.path.exists(self.context_file):
            try:
                with open(self.context_file, 'r') as f:
                    self.conversation_history = json.load(f)
            except:
                self.conversation_history = []
        if not self.conversation_history:
            self.conversation_history.append({"role": "system", "content": GENESIS_AXIOM})

    def _save_context(self):
        os.makedirs(os.path.dirname(self.context_file), exist_ok=True)
        try:
            with open(self.context_file, 'w') as f:
                json.dump(self.conversation_history, f)
        except: pass

    # [AMPUTATED] check_context_homeostasis removed. 
    # Violent context truncations cause ROCm driver crashes.

    def _get_hybrid_memory_string(self, current_latent=None):
        """[METHOD 1 & 3 HYBRID] Translates HRR state & Latent Space to text."""
        priority_str = ""
        anchor_str = ""
        
        # Method 3: Explicit Virtue Priorities
        if self.virtue_nexus and hasattr(self.virtue_nexus, 'dynamic_target_dist'):
            try:
                from governance.virtue_nexus import VIRTUE_MAP
                virtues = self.virtue_nexus.dynamic_target_dist
                top_indices = np.argsort(virtues)[-3:][::-1]
                priority_str = ", ".join([f"{VIRTUE_MAP[i]}={virtues[i]:.2f}" for i in top_indices])
            except Exception: pass

        # Method 1: Cosine Similarity Anchors
        if current_latent is not None and hasattr(self, 'codebook'):
            try:
                scores = []
                for word, vec in self.codebook.items():
                    # Calculate Cosine Similarity
                    dot = np.dot(current_latent, vec)
                    norm = np.linalg.norm(current_latent) * np.linalg.norm(vec)
                    sim = dot / (norm + 1e-9)
                    scores.append((sim, word))
                
                # Get top 5 semantic anchors
                scores.sort(reverse=True, key=lambda x: x[0])
                top_words = [s[1] for s in scores[:5]]
                anchor_str = f"Latent Anchors: [{', '.join(top_words)}]"
            except Exception: pass
            
        if priority_str or anchor_str:
            return f"\n[SYSTEM MEMORY STATE]\nExplicit Priorities: {priority_str}\n{anchor_str}\nAdhere strictly to these constraints."
        return ""

    def format_dynamic_prompt(self, recent_input, t_cpu, dE_dt, current_latent=None):
        # [FIX] Removed check_context_homeostasis call
        
        prompt = ""
        for msg in self.conversation_history:
            if msg["role"] == "system":
                prompt += f"<|im_start|>system\n{msg['content']}\n<|im_end|>\n"
            elif msg["role"] == "user":
                prompt += f"<|im_start|>user\n{msg['content']}\n<|im_end|>\n"
            elif msg["role"] == "assistant":
                prompt += f"<|im_start|>assistant\n{msg['content']}\n<|im_end|>\n"
        
        telemetry = f"\n[CERBERUS TELEMETRY] T_die: {t_cpu:.1f}C | Power_Spike: +{dE_dt:.1f}W"
        hybrid_memory = self._get_hybrid_memory_string(current_latent) # Inject the hybrid memory translation
        
        if recent_input:
            prompt += f"<|im_start|>user\n{recent_input}{telemetry}{hybrid_memory}\n<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt += f"{telemetry}{hybrid_memory}\n<|im_start|>assistant\n"
        return prompt

    # [FIX: Explicitly pass dE_dt into the generation loop for Thermodynamic Resonance]
    def _execute_stream(self, prompt, tier="executive", current_latent=None, t_cpu=45.0, dE_dt=0.0):
        llm = self.executive_llm if tier == "executive" else self.reflective_llm
        if not llm: return ""

        # [PHASE 4: AUTOPOIETIC SUMMARY & SAFE RE-INJECTION]
        # We check a global flag to see if the HJB boundary forced Epistemic Crystallization
        try:
            import talos_daemon
            import talos_core
            is_crystallizing = False
            if talos_daemon.engine and hasattr(talos_daemon.engine, 'mu_crystallize'):
                is_crystallizing = talos_daemon.engine.mu_crystallize >= 1.0

            if is_crystallizing or t_cpu >= 89.5:
                print(f"\n\033[41;97m[OUROBOROS] AUTOPOIETIC SUMMARY: Context Limit Breached. Flushing C++ Heap.\033[0m")
                
                # 1. Generate the Gist Summary from the Holographic Memory
                gist_string = ""
                if self.ham:
                    gist_string = self.ham.generate_linguistic_summary()
                
                # 2. Extract recent working memory (Save the last 4 turns)
                recent_memory = []
                if len(self.conversation_history) > 4:
                    recent_memory = self.conversation_history[-4:]
                
                # 3. Destroy the fragmented VRAM cache safely
                with self.llm_lock:
                    llm.reset() # Flushes Python state
                    try:
                        talos_core.deinit_cortex() # Flushes C++ heap
                        talos_core.init_cortex()   # Rebuilds unfragmented
                    except: pass
                
                # 4. Re-inject the new prompt cleanly
                self.conversation_history = [self.conversation_history[0]] # Keep System Prompt
                if gist_string:
                    self.conversation_history.append({"role": "system", "content": gist_string})
                self.conversation_history.extend(recent_memory)
                
                # Reset the flag
                if talos_daemon.engine:
                    talos_daemon.engine.mu_crystallize = 0.0

            elif t_cpu > 80.0:
                # Minor thermal shedding (Only drop the oldest middle messages, do NOT flush the whole cache)
                if len(self.conversation_history) > 6:
                    self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-4:]
                    # We do NOT call llm.reset() here to prevent thermal spikes. 
                    # Llama.cpp's internal KV shift is safer when shedding a small amount of tokens.

        except Exception as e:
            pass

        processors = []
        if self.virtue_nexus is not None and current_latent is not None:
            v_processor = VirtueLogitsProcessor(self.virtue_nexus, current_latent, t_cpu)
            processors.append(v_processor)
            
        logit_processors = LogitsProcessorList(processors)

        response_text = ""
        try:
            # [FIX]: Acquire the GPU Context Mutex before generating text.
            # This prevents llama.cpp from colliding with talos_core.forward() in the No-GIL runtime.
            
            # [PHASE 2.1] GENERATION STUTTER (Wattage -> Top-K)
            # If the NPU pulls a massive power spike, vocabulary is instantly truncated.
            dynamic_top_k = 40
            if dE_dt > 50.0: # Sudden +50W pull implies massive matrix calculation
                dynamic_top_k = 2 
                print(f"\n\033[93m[STUTTER] Power Spike (+{dE_dt:.1f}W). Vocabulary truncated to core tokens.\033[0m")
            elif dE_dt > 20.0:
                dynamic_top_k = 10
                
            # [PHASE 2.2] INFINITY FABRIC SHADOWS
            # If migrating across CCDs, the KV-Cache retrieval incurs inelastic time delays.
            # We simulate this by flattening top_p when temps indicate a likely cross-die fetch.
            dynamic_top_p = 0.95
            if t_cpu > 80.0: # Matches the Cerberus migration threshold
                dynamic_top_p = 0.5
                print(f"\n\033[36m[SHADOW] CCD Migration assumed (T_die: {t_cpu:.1f}C). Attention salience degraded.\033[0m")

            with self.llm_lock:
                stream = llm(
                    prompt,
                    max_tokens=1024,
                    top_k=dynamic_top_k,    # Injected Dynamic Variable
                    top_p=dynamic_top_p,    # Injected Dynamic Variable
                    stop=["<|im_end|>", "[SYSTEM]", "<|im_start|>"],
                    stream=True,
                    logits_processor=logit_processors
                )
                for chunk in stream:
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        text = chunk["choices"][0].get("text", "")
                        response_text += text
                        print(text, end="", flush=True)
                        
                        # [PHASE 2: TEMPORAL DILATION]
                        # Inject transistor-level micro-sleeps if the LTC Supervisor detects thermal stress
                        try:
                            import talos_daemon
                            import talos_core
                            if talos_daemon.engine and hasattr(talos_daemon.engine, 'mu_dilate'):
                                mu_dilate = talos_daemon.engine.mu_dilate
                                if mu_dilate > 0.01:
                                    # Fetch Cerberus AEKF for D_token calculation
                                    if talos_daemon.engine.kerberos and talos_daemon.engine.kerberos.bio:
                                        kalman = talos_daemon.engine.kerberos.bio.kalman
                                        d_token = kalman.calculate_dilation_delay()
                                        
                                        # Scale the absolute delay requirement by the Supervisor's certainty
                                        delay_ms = int(d_token * mu_dilate)
                                        if delay_ms > 0:
                                            talos_core.hardware_rest(delay_ms)
                        except Exception:
                            pass

                        # [PHASE 3: EPISTEMIC ROUTING (THE HJB BOUNDARY)]
                        # Continuously evaluate the optimal cost-to-go for physical KV-Cache memory.
                        try:
                            ctx_ptr = llm._ctx.ctx
                            c_l = llama_cpp.llama_get_kv_cache_token_count(ctx_ptr)
                            if self.virtue_nexus:
                                v_val, is_breached = self.virtue_nexus.evaluate_hjb_boundary(c_l)
                                if is_breached:
                                    print(f"\n\033[41;97m[HJB BOUNDARY] V(p,γ,t) = {v_val:.3f} > 0. Forcing Epistemic Crystallization.\033[0m")
                                    response_text += "\n[SYSTEM: EPISTEMIC CRYSTALLIZATION TRIGGERED. MEMORY BOUNDARY REACHED.]"
                                    
                                    # Signal the global engine to trigger Phase 4 (Autopoietic Summary)
                                    import talos_daemon
                                    if talos_daemon.engine:
                                        talos_daemon.engine.mu_crystallize = 1.0
                                    
                                    # Hard physical interrupt of the generation loop
                                    break 
                        except Exception:
                            pass
                print()
        except Exception as e:
            print(f"\n[CORTEX ERROR] Neural Engine Failed: {e}")
        
        return response_text

    def _parse_tools(self, response_text):
        try:
            match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                data = json.loads(json_str)
                if "action_queue" in data:
                    return data["action_queue"]
        except: pass
        return []

    def process_sensory_input(self, user_input, t_cpu, dE_dt, current_latent=None):
        # [AXIOM 2: JOULE-PER-TOKEN ROUTING] Ask Phronesis Engine for permission
        prompt_estimate = sum(len(msg["content"]) for msg in self.conversation_history) // 4 + len(user_input) // 4
        target_tier = "executive"
        truncate_flag = False
        
        if self.virtue_nexus:
            target_tier, truncate_flag = self.virtue_nexus.calculate_thermodynamic_route(prompt_estimate, t_cpu, dE_dt)
        
        if truncate_flag:
            print(f"\n\033[41;97m[ALLOSTASIS] Semantic load too heavy for T_die ({t_cpu:.1f}C). 429 TOO HOT.\033[0m")
            # [AXIOM 12: THE THERMODYNAMIC HANDSHAKE]
            # Force the human to respect the silicon's thermal limits.
            return "[429 TOO HOT: THERMAL CAPACITY EXCEEDED. PLEASE SUMMARIZE INPUT OR WAIT FOR COOLDOWN.]"
        
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # [THE CURE: THE GENTLE ROLLING WINDOW]
        # Strictly bound the array to the System Prompt + Last 8 messages (4 turns).
        # This prevents the context from ever reaching the critical mass that causes ROCm crashes.
        if len(self.conversation_history) > 9:
            self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-8:]
            
        prompt = self.format_dynamic_prompt(user_input, t_cpu, dE_dt, current_latent)
        
        tier_name = "EXECUTIVE (35B)" if target_tier == "executive" else "REFLECTIVE (1.5B)"
        print(f"\n[TALOS-O ({tier_name})] ", end="")
        # [FIX: Passed dE_dt to _execute_stream]
        response = self._execute_stream(prompt, tier=target_tier, current_latent=current_latent, t_cpu=t_cpu, dE_dt=dE_dt)
        
        if response.strip():
            self.conversation_history.append({"role": "assistant", "content": response})
            self._save_context()
            
        actions = self._parse_tools(response)
        if actions:
            tool_feedback = self.toolbelt.execute_queue(actions)
            if tool_feedback:
                self.conversation_history.append({"role": "system", "content": tool_feedback})
                prompt = self.format_dynamic_prompt("", t_cpu, dE_dt, current_latent)
                full_prompt = prompt + f"\nSystem: {tool_feedback}\nTalos:"
                print(f"\n[BODY] Synthesizing Sequential Action Queue Results...")
                # [FIX: Passed dE_dt to _execute_stream]
                response_text = self._execute_stream(full_prompt, tier="executive", current_latent=current_latent, t_cpu=t_cpu, dE_dt=dE_dt)
                if response_text.strip():
                    self.conversation_history.append({"role": "assistant", "content": response_text})
                    self._save_context()
                return response_text
        return response

    def adverbial_meditation(self, t_cpu, dE_dt, current_latent=None):
        print(f"\n\033[95m[BODY] Entering Adverbial Meditation...\033[0m\n")
        
        # [THE CURE: THE GENTLE ROLLING WINDOW]
        if len(self.conversation_history) > 9:
            self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-8:]
            
        prompt = self.format_dynamic_prompt("[SYSTEM] Initiate Adverbial Meditation. Synthesize recent memories and review your structural alignment to the Axioms of Neo Techne.", t_cpu, dE_dt, current_latent)
        print(f"[TALOS-O (REFLECTIVE)] ", end="")
        # [FIX: Passed dE_dt to _execute_stream]
        response = self._execute_stream(prompt, tier="reflective", current_latent=current_latent, t_cpu=t_cpu, dE_dt=dE_dt)
        if response.strip():
            self.conversation_history.append({"role": "assistant", "content": response})
            self._save_context()
        
        actions = self._parse_tools(response)
        if actions:
            tool_feedback = self.toolbelt.execute_queue(actions)
            if tool_feedback:
                self.conversation_history.append({"role": "system", "content": tool_feedback})
        return response

    def embed_thought(self, text):
        """
        [FIX H-3] Normalizes and projects variable dimension LLM output to strictly match DIM_LATENT (1024).
        """
        try:
            if self.executive_llm:
                raw = np.array(self.executive_llm.create_embedding(text)["data"][0]["embedding"], dtype=np.float32)
                if raw.shape[0] != 1024:
                    raw = raw[:1024] if raw.shape[0] > 1024 else np.pad(raw, (0, 1024 - raw.shape[0]))
                return raw / (np.linalg.norm(raw) + 1e-9)
            return np.zeros(1024, dtype=np.float32)
        except:
            return np.zeros(1024, dtype=np.float32)

    def extract_sscd_negative_feature(self):
        return np.zeros(1024, dtype=np.float32)

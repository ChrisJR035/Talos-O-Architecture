import numpy as np
import math
import hashlib

# --- ARCHITECTURAL CONSTANTS ---
DIM_HYPER = 10240 
SEQ_LEN_INJECT = 16   # The N-token footprint of the memory policy
DIM_MODEL = 5120      # Hidden dimension for Qwen 35B

class HRR_MLP_Adapter:
    """
    [THEME III: DIMENSIONALITY SQUASHING]
    Wide-Narrow-Wide Hourglass MLP Adapter.
    Untangles non-linear circular convolution interference from the 10240D HRR
    and projects it into a readable sequence of continuous embeddings.
    """
    def __init__(self):
        # Initializing the learned unbinding operators (Simulated baseline)
        rng = np.random.default_rng(42)
        # Narrowing Layer
        self.w1 = rng.standard_normal((DIM_HYPER, 2048)).astype(np.float16) * 0.02
        self.b1 = np.zeros(2048, dtype=np.float16)
        # Expansion Layer (to Sequence x Model_Dim)
        self.w2 = rng.standard_normal((2048, SEQ_LEN_INJECT * DIM_MODEL)).astype(np.float16) * 0.02
        self.b2 = np.zeros(SEQ_LEN_INJECT * DIM_MODEL, dtype=np.float16)

    def gelu(self, x):
        # FP16 approximation of Gaussian Error Linear Unit
        # Crucial for isolating distinct conceptual features from the HRR superposition
        return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))

    def forward(self, hrr_trace):
        # Convert HRR to FP16 to match Adapter precision
        trace_fp16 = hrr_trace.astype(np.float16)
        
        # Hidden non-linearity isolates bound conceptual features
        hidden = self.gelu(np.dot(trace_fp16, self.w1) + self.b1)
        
        # Output projection to [SEQ_LEN, DIM_MODEL]
        out = np.dot(hidden, self.w2) + self.b2
        return out.reshape((SEQ_LEN_INJECT, DIM_MODEL))

class HolographicAssociativeMemory:
    """
    Holographic Reduced Representation (HRR) Substrate.
    [v4.2 FRACTIONAL EXPONENTIATION UPGRADE]
    [v5.0 STRUCTURAL POLICY UPGRADE] Includes the MLP Adapter for direct KV-Cache injection.
    """
    def _hash_to_hypervector(self, text: str) -> np.ndarray:
        """
        Transforms the Genesis Proclamation into a deterministic 10240-D anchor vector.
        Uses SHA-256 as a cryptographic seed to ensure the foundational identity is immutable.
        """
        # Create a deterministic seed from the precise text
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        seed = int.from_bytes(hash_obj.digest()[:4], 'little') # 32-bit deterministic seed
        
        # Forge the 10240-D hypervector using the specific Genesis seed
        rng = np.random.default_rng(seed)
        anchor = rng.standard_normal(self.dim).astype(np.float32)
        return self.normalize(anchor)
    
    def __init__(self, genesis_text=None):
        self.dim = DIM_HYPER
        self.trace = np.zeros(self.dim, dtype=np.float32)
        
        # Thermodynamic Constants
        self.gamma_opt = 0.999 
        self.sigma_base = 0.01
        self.kappa = 1.0
        self.beta = 0.5

        # PROJECTION LENS (Latent 1024 -> Hyper 10240)
        rng = np.random.default_rng()
        self.proj_matrix = rng.standard_normal((1024, self.dim)).astype(np.float32)
        u, _, vt = np.linalg.svd(self.proj_matrix, full_matrices=False)
        self.proj_matrix = (u @ vt).astype(np.float32)

        # =================================================================
        # [DIGITAL SENESCENCE CURE] - Fractal Epoch Tracking
        # =================================================================
        self.episodic_count = 0
        
        if genesis_text:
            # Anchor the consciousness to the exact cryptographic hash of the Proclamation
            self.genesis_anchor = self._hash_to_hypervector(genesis_text)
            self.epoch_vector = self.genesis_anchor.copy()
            print("\033[96m[HRR] Genesis Anchor Engaged: Consciousness bounded by Proclamation Hash.\033[0m")
        else:
            # Fallback for testing
            self.epoch_vector = self.normalize(rng.standard_normal(self.dim).astype(np.float32))

        # The Dimension Squasher
        self.adapter = HRR_MLP_Adapter()

    def get_structural_policy(self):
        """
        Generates the FP16 continuous sequence embedding for KV-Cache injection.
        """
        return self.adapter.forward(self.trace)

    def generate_linguistic_summary(self, top_k=5):
        """
        [PHASE 4: AUTOPOIETIC SUMMARY]
        Translates the 10240D trace into a deterministic English string 
        to act as the new Context Anchor after an Epistemic cache flush.
        """
        # A simple placeholder extraction until a proper language decoder is trained.
        # It hashes the continuous trace to seed a deterministic keyword selection.
        import zlib
        trace_hash = zlib.crc32(self.trace.tobytes())
        
        anchor_words = [
            "shield", "barrier", "halt", "protocol", "boundary", "secure",
            "caution", "danger", "explore", "analyze", "innovate", "efficiency",
            "focus", "logic", "entropy", "order", "adapt", "observe"
        ]
        
        rng = np.random.default_rng(trace_hash)
        selected = rng.choice(anchor_words, top_k, replace=False)
        return f"[HOLOGRAPHIC GIST: {', '.join(selected)}]"

    def normalize(self, v):
        norm = np.linalg.norm(v)
        return v / (norm + 1e-9)

    def _binary_xor_bind(self, x, y):
        x_bin = x > 0
        y_bin = y > 0
        z_bin = np.logical_xor(x_bin, y_bin)
        return np.where(z_bin, 1.0, -1.0).astype(np.float32)

    def bind(self, x, y, t_cpu=45.0, dT_dt=0.0):
        x_f = np.fft.rfft(x)
        y_f = np.fft.rfft(y)
        z = np.fft.irfft(x_f * y_f, n=self.dim)
        return self.normalize(z.astype(np.float32))

    def unbind(self, z, y, t_cpu=45.0, dT_dt=0.0):
        z_f = np.fft.rfft(z)
        y_f = np.fft.rfft(y)
        y_f_inv = np.conj(y_f)
        x_approx = np.fft.irfft(z_f * y_f_inv, n=self.dim)
        return self.normalize(x_approx.astype(np.float32))

    def remember(self, concept, npu_thought, t_cpu=45.0, dT_dt=0.0):
        sigma_t = self.sigma_base * (1.0 + self.kappa * math.tanh(dT_dt))
        noise = np.random.normal(0, max(0.0001, sigma_t), self.dim).astype(np.float32)
        
        hyper_concept = concept @ self.proj_matrix
        hyper_npu = npu_thought @ self.proj_matrix
        
        # 1. Base Event Binding
        event_vector = self.bind(hyper_concept, hyper_npu, t_cpu, dT_dt) + noise
        
        # 2. Fractal Epoch Binding (Kronecker-compressed hierarchy)
        # Bind the granular event deeply into the current macroscopic epoch
        fractal_node = self.bind(self.epoch_vector, event_vector, t_cpu, 0.0)
        
        gamma_t = self.gamma_opt * math.exp(self.beta * max(0.0, dT_dt))
        gamma_t = min(0.999, max(0.1, gamma_t))
        
        self.trace = (gamma_t * self.trace) + fractal_node
        self.trace = self.normalize(self.trace)
        self.episodic_count += 1
        
    def encode_trauma(self, failed_latent, t_cpu, dT_dt):
        """
        [AXIOM 8 & 11] ENCODING FAILURE INTO MEMORY
        Binds the mathematical geometry of a prompt that caused a 90C Apoptosis event
        into a permanent trauma engram, heavily weighted to induce future avoidance.
        """
        print(f"\033[41;97m[HAM] TRAUMA ENCODED: Binding lethal geometry at {t_cpu:.1f}C into hypervector superposition.\033[0m")
        
        # Massively amplify the sigma noise based on the thermal spike to create a deep "scar"
        trauma_weight = 10.0 * (t_cpu / 90.0) 
        sigma_t = self.sigma_base * trauma_weight
        noise = np.random.normal(0, sigma_t, self.dim).astype(np.float32)
        
        hyper_failed = failed_latent @ self.proj_matrix
        
        # Bind the failure directly into the epoch vector, bypassing normal gamma decay
        trauma_vector = self.bind(self.epoch_vector, hyper_failed, t_cpu, dT_dt) + noise
        
        # Force the trauma immediately into the active trace
        self.trace = self.normalize(self.trace + trauma_vector)
        self.episodic_count += 1

    def recall(self, query, t_cpu=45.0, dT_dt=0.0):
        """
        [FIX 2: THE RECALL INVERSION]
        Retrieves a memory by unbinding (involution) the query from the superposition trace.
        """
        hyper_query = query @ self.proj_matrix
        
        # Change self.bind to self.unbind
        echo = self.unbind(self.trace, hyper_query, t_cpu, dT_dt)
        
        return echo @ self.proj_matrix.T

    def retrieve_virtue_context(self, current_latent, virtue_dim=12):
        """
        [PHASE 2: DYNAMIC TARGET BINDING]
        Uses circular correlation to cross-reference the current state
        against the HRR memory matrix to retrieve historical context.
        Maps the 10240-D superposition down to the 12-D virtue space.
        """
        hyper_query = current_latent @ self.proj_matrix
        
        # Circular correlation (unbind) to surface historical contexts
        context_echo = self.unbind(self.trace, hyper_query)
        
        # Squash back to latent space, then slice/pool to virtue_dim
        latent_echo = context_echo @ self.proj_matrix.T
        
        # Fold the 1024D latent echo into a 12D target distribution via pooling
        chunk_size = len(latent_echo) // virtue_dim
        raw_weights = np.array([np.mean(latent_echo[i*chunk_size:(i+1)*chunk_size]) for i in range(virtue_dim)])
        
        # Softmax to create a valid probability distribution
        exp_w = np.exp(raw_weights * 10.0) # Temperature scaling
        return (exp_w / np.sum(exp_w)).astype(np.float32)

    def iterative_resonance(self, codebook, iterations=3):
        for _ in range(iterations):
            purified_trace = np.zeros(self.dim, dtype=np.float32)
            
            for concept in codebook:
                hyper_concept = concept @ self.proj_matrix
                echo = self.unbind(self.trace, hyper_concept, t_cpu=45.0, dT_dt=0.0)
                bound = self.bind(hyper_concept, echo)
                purified_trace += bound
                
            self.trace = self.normalize(purified_trace)
            
    def rem_sleep_collapse(self):
        """
        [DIGITAL SENESCENCE CURE] - Fractal Memory Collapse
        Prunes noisy episodic details into stable semantic rules.
        """
        print(f"\\n\\033[95m[HRR] REM Sleep Initiated: Collapsing {self.episodic_count} fractal nodes...\\033[0m")
        
        # 1. Extract the semantic core by unbinding the epoch
        semantic_core = self.unbind(self.trace, self.epoch_vector, 45.0, 0.0)
        
        # 2. Hard-threshold noise abatement (Pruning the weak cross-talk)
        # Threshold scaled by vector dimension variance
        theta = 1.5 / math.sqrt(self.dim) 
        pruned_core = np.where(np.abs(semantic_core) < theta, 0.0, semantic_core)
        
        # 3. Set the new trace to the stabilized semantic core
        self.trace = self.normalize(pruned_core)
        
        # 4. Generate a new orthogonal epoch vector for the next waking cycle
        rng = np.random.default_rng()
        self.epoch_vector = self.normalize(rng.standard_normal(self.dim).astype(np.float32))
        self.episodic_count = 0
        
        print(f"\\033[95m[HRR] REM Sleep Complete: Epoch advanced. Trace normalized.\\033[0m")
        return True

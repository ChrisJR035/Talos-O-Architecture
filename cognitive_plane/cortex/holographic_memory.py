import numpy as np
import math

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
    def __init__(self):
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

        # The Dimension Squasher
        self.adapter = HRR_MLP_Adapter()

    def get_structural_policy(self):
        """
        Generates the FP16 continuous sequence embedding for KV-Cache injection.
        """
        return self.adapter.forward(self.trace)

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
        
        bound_concept = self.bind(hyper_concept, hyper_npu, t_cpu, dT_dt) + noise
        
        gamma_t = self.gamma_opt * math.exp(self.beta * max(0.0, dT_dt))
        gamma_t = min(0.999, max(0.1, gamma_t))
        
        self.trace = (gamma_t * self.trace) + bound_concept
        self.trace = self.normalize(self.trace)

    def recall(self, query, t_cpu=45.0, dT_dt=0.0):
        hyper_query = query @ self.proj_matrix
        echo = self.unbind(self.trace, hyper_query, t_cpu, dT_dt)
        return echo @ self.proj_matrix.T

    def iterative_resonance(self, codebook, iterations=3):
        for _ in range(iterations):
            purified_trace = np.zeros(self.dim, dtype=np.float32)
            
            for concept in codebook:
                hyper_concept = concept @ self.proj_matrix
                echo = self.unbind(self.trace, hyper_concept, t_cpu=45.0, dT_dt=0.0)
                bound = self.bind(hyper_concept, echo)
                purified_trace += bound
                
            self.trace = self.normalize(purified_trace)

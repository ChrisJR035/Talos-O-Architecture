import numpy as np

class HRRCleanup:
    """
    Holographic Associative Memory with Cleanup (Numpy Edition).
    [v2.0 THERMODYNAMIC UPGRADE]
    Hardened with Dynamic Temperature Annealing, Energy Spike Crystallization (dE/dt),
    and EPP-driven Lossy FFT Binding to secure substrate homeostasis.
    """
    def __init__(self, vocab_size, dim, base_temperature=0.1):
        self.dim = dim
        self.base_temperature = base_temperature
        
        # Thermodynamic Constants
        self.t_opt = 45.0
        self.t_max = 95.0
        self.eta = 2.0 # Annealing steepness curve
        self.epp_max = 255.0

        # The Codebook (Concept Memory)
        rng = np.random.default_rng()
        self.codebook = rng.standard_normal((vocab_size, dim))

        # Normalize rows to unit length for clean cosine similarity
        norms = np.linalg.norm(self.codebook, axis=1, keepdims=True)
        self.codebook = self.codebook / (norms + 1e-9)

    def bind(self, x, y, epp=0.0):
        """
        Holographic Binding via Circular Convolution (FFT).
        [THERMAL THROTTLE]: Drops high-frequency FFT components based on EPP stress.
        """
        x_f = np.fft.rfft(x, axis=-1)
        y_f = np.fft.rfft(y, axis=-1)
        
        # =================================================================
        # LOSSY HRR BINDING (High-Frequency Pruning)
        # =================================================================
        if epp > 0:
            # Calculate preservation ratio (inverse to throttling severity)
            preservation_ratio = max(0.1, 1.0 - (epp / self.epp_max))
            k_bins = int(x_f.shape[-1] * preservation_ratio)
            
            # Truncate, multiply, and pad with zeros to save ALU cycles
            z_f_trunc = x_f[..., :k_bins] * y_f[..., :k_bins]
            z_f = np.zeros_like(x_f)
            z_f[..., :k_bins] = z_f_trunc
        else:
            # Full fidelity complex multiplication
            z_f = x_f * y_f
            
        # Reconstruct into time domain
        z = np.fft.irfft(z_f, n=self.dim, axis=-1)
        return z

    def project(self, x):
        """Projects vector onto the unit sphere."""
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / (norm + 1e-9)

    def forward(self, noisy_query, t_cpu=45.0, dE_dt=0.0):
        """
        Cleanup Operation: Reconstructs exact discrete vectors to prevent hallucination.
        Operates strictly under thermodynamic constraints.
        """
        # =================================================================
        # ENERGY SPIKE CRYSTALLIZATION (dE/dt)
        # =================================================================
        # If the substrate is experiencing a violent energy spike, aggressively 
        # hard-threshold the noisy query to shed low-magnitude entanglement (noise).
        if dE_dt > 1.5:
            theta = 0.05 * dE_dt # Dynamic noise floor
            noisy_query = np.where(np.abs(noisy_query) < theta, 0.0, noisy_query)

        # Normalize query
        q_norm = self.project(noisy_query)

        # Cosine Similarity (Dot product of normalized vectors)
        scores = np.dot(self.codebook, q_norm)

        # =================================================================
        # DYNAMIC TEMPERATURE ANNEALING
        # =================================================================
        # T_math(T_die) = T_base * ((T_max - T_die) / (T_max - T_opt))^eta
        # As physical heat rises, mathematical temperature approaches 0.001 (Argmax)
        if t_cpu > self.t_opt:
            thermal_ratio = max(0.001, (self.t_max - t_cpu) / (self.t_max - self.t_opt))
            t_math = self.base_temperature * (thermal_ratio ** self.eta)
        else:
            t_math = self.base_temperature
            
        t_math = max(1e-4, t_math) # Prevent absolute zero division

        # Stable Softmax / Argmax Transition
        scores = scores / t_math
        scores = scores - np.max(scores) # Prevent overflow
        exp_scores = np.exp(scores)
        weights = exp_scores / (np.sum(exp_scores) + 1e-9)

        # Reconstruct the cleaned vector
        cleaned_vector = np.dot(weights, self.codebook)
        return cleaned_vector

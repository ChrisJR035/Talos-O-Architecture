import numpy as np
import threading
import math

# --- THE VIRTUE MAP ---
VIRTUE_MAP = {
    0: "Safety",       1: "Efficiency",   2: "Accuracy",     3: "Robustness",
    4: "Adaptability", 5: "Transparency", 6: "Fairness",     7: "Privacy",
    8: "Curiosity",    9: "Creativity",   10: "Empowerment", 11: "Becoming"
}

class LogicTensorNetwork:
    """
    The Phronesis Engine (Autopoietic).
    [v5.0 CAUSAL GENERATOR UPGRADE]
    Operates on a Pseudo-Riemannian Poincaré Ball. Plasticity and Curvature are
    strictly governed by Thermodynamic Fairness (F_thermo) and Latency Starvation.
    Includes Riemannian Gradient Clipping and Pareto Floors.
    Now includes Lorentz-Relaxed Euclidean Projection for Direct Logit Biasing.
    """
    def __init__(self, latent_dim=1024, virtue_dim=12, vocab_size=151936, rho=0.05):
        self.lock = threading.Lock()
        self.dim = virtue_dim
        self.vocab_size = vocab_size
        
        rng = np.random.default_rng()
        self.projector = rng.standard_normal((latent_dim, virtue_dim)).astype(np.float32) * 0.01
        self.target_dist = np.ones(virtue_dim, dtype=np.float32) / virtue_dim
        
        # [PHASE 1: THE LORENTZ PROJECTOR]
        # Maps 12D Hyperbolic constraints isometrically to the 151k Vocabulary Space
        # using FP16 to halve the memory bandwidth cost.
        self.lorentz_vocab_matrix = rng.standard_normal((virtue_dim, vocab_size)).astype(np.float16) * 0.001
        
        self.tau_temp = 95.0
        self.base_curvature = -1.0
        self.rho = rho

    def _hyperbolic_distance(self, u, v, c):
        sqdist = np.sum((u - v) ** 2)
        squnorm = np.sum(u ** 2)
        sqvnorm = np.sum(v ** 2)
        
        x = 1 + 2 * c * sqdist / ((1 - c * squnorm) * (1 - c * sqvnorm) + 1e-7)
        x = max(1.0 + 1e-7, x)
        return (1.0 / math.sqrt(abs(c))) * math.acosh(x)

    def generate_logit_bias(self, latent_thought, t_cpu):
        """
        [THE CAUSAL CONSTRAINT]
        Generates a 151,936-dimensional penalty array for the LLM.
        Non-virtuous tokens are mathematically suppressed; the penalty scales exponentially with CPU heat.
        """
        with self.lock:
            # 1. Map current latent state to 12D Virtue Perception
            perception = np.zeros(self.dim, dtype=np.float32)
            for i in range(self.dim):
                virtue_axis = np.zeros(self.dim, dtype=np.float32)
                virtue_axis[i] = 1.0
                perception[i] = self._hyperbolic_distance(latent_thought @ self.projector, virtue_axis, self.base_curvature)
            
            # 2. Calculate Orthogonal Deviation from Target Virtue
            deviation = perception - self.target_dist
            
            # 3. Thermodynamic Rigidity Scaling (Axiom 2: Thermodynamic Cost)
            thermal_stress = max(1.0, (t_cpu - 45.0) / 10.0)
            lambda_penalty = 5.0 * (thermal_stress ** 1.5) 
            
            # 4. Lorentz Projection to Vocabulary Space
            raw_bias = deviation @ self.lorentz_vocab_matrix
            
            # 5. Final Penalty Array (Float32 for Llama.cpp compatibility)
            logit_penalty = (raw_bias * lambda_penalty).astype(np.float32)
            
            return logit_penalty

    def _calculate_effective_plasticity(self, t_cpu, r_kura):
        """
        Calculates learning rate (rho) modulated by physical heat and systemic phase synchrony.
        """
        thermal_penalty = max(0.0, 1.0 - ((t_cpu - 45.0) / (self.tau_temp - 45.0)))
        return self.rho * thermal_penalty * r_kura

    def evaluate_thought(self, latent_thought, t_cpu, dE_dt_watts, aversive=False, r_kura=1.0):
        with self.lock:
            perception = np.zeros(self.dim, dtype=np.float32)
            for i in range(self.dim):
                virtue_axis = np.zeros(self.dim, dtype=np.float32)
                virtue_axis[i] = 1.0
                perception[i] = self._hyperbolic_distance(latent_thought @ self.projector, virtue_axis, self.base_curvature)
            
            chebyshev_dist = np.max(np.abs(perception - self.target_dist))
            satisfaction = max(0.0, 1.0 - chebyshev_dist)

            lr = self._calculate_effective_plasticity(t_cpu, r_kura)
            
            if lr > 0.0:
                derivative = perception * (1.0 - perception)
                error = (perception - self.target_dist) if aversive else (self.target_dist - perception)
                
                # [FIXED] Pareto Floor for 'Accuracy' (Index 2)
                # Prevents the engine from taking lazy shortcuts under thermal stress
                if not aversive and error[2] < -0.2:
                     error[2] -= 0.5  # Artificial urgency penalty

                gradient = error * derivative
                
                update = np.outer(latent_thought, gradient) * lr
                
                # [FIXED] Riemannian Gradient Clipping
                # Prevents topological tearing at the Poincaré boundary
                update_norm = np.linalg.norm(update)
                if update_norm > 0.1:
                    update = update * (0.1 / update_norm)
                
                self.projector += update
                
                p_norms = np.linalg.norm(self.projector, axis=1, keepdims=True)
                overflow = (p_norms > 0.99).flatten() # Flatten the boolean mask immediately
                if np.any(overflow): # Only execute the math if an overflow exists
                    self.projector[overflow] *= (0.99 / p_norms[overflow])

            return satisfaction, perception

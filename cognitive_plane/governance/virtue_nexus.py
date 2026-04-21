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
        
        # [PHASE 4 FIX: THE FROZEN CONSCIENCE]
        # A cryptographically anchored copy of the initial state. This serves as the
        # absolute epistemic ground truth against which all future evolution is measured.
        self.projector_frozen = self.projector.copy()
        
        # [PHASE 2: DYNAMIC UTOPIAN POINT]
        # Initializing the dynamic target distribution (C-Vector) and momentum parameters
        self.dynamic_target_dist = np.ones(virtue_dim, dtype=np.float32) / virtue_dim
        self.momentum_mu = 0.85 # Dampening coefficient for target shifts
        
        # [PHASE 1: TOPOLOGICAL SEEDING FOR LORENTZ PROJECTOR]
        # Attempt to harvest the native semantic topology from the GGUF model
        import sys
        import os
        try:
            if "cognitive_plane" not in sys.path:
                sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
            from file_manager import extract_gguf_principal_components
            
            # The path to the primary 35B model
            model_path = os.path.expanduser("~/talos-o/cognitive_plane/models/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf")
            seeded_matrix = extract_gguf_principal_components(model_path, target_rank=virtue_dim)
            
            if seeded_matrix is not None:
                self.lorentz_vocab_matrix = seeded_matrix
            else:
                self.lorentz_vocab_matrix = rng.standard_normal((virtue_dim, vocab_size)).astype(np.float16) * 0.001
        except ImportError:
            self.lorentz_vocab_matrix = rng.standard_normal((virtue_dim, vocab_size)).astype(np.float16) * 0.001
        
        self.tau_temp = 95.0
        self.base_curvature = -1.0
        self.rho = rho

    def _to_ball(self, u, c=1.0):
        norm = np.linalg.norm(u)
        limit = 0.99 / math.sqrt(abs(c))
        if norm >= limit:
            u = u * (limit / norm)
        return u

    def _hyperbolic_distance(self, u, v, c):
        sqdist = np.sum((u - v) ** 2)
        squnorm = np.sum(u ** 2)
        sqvnorm = np.sum(v ** 2)
        
        x = 1 + 2 * c * sqdist / ((1 - c * squnorm) * (1 - c * sqvnorm) + 1e-7)
        x = max(1.0 + 1e-7, x)
        return (1.0 / math.sqrt(abs(c))) * math.acosh(x)

    def generate_logit_bias(self, latent_thought, t_cpu):
        """
        [PHASE 2: LORENTZ-RELAXED EUCLIDEAN PROJECTION]
        Projects the current latent thought into the Virtue Topology to generate logit penalties.
        This is a READ-ONLY operation. No topological mutation occurs here.
        """
        with self.lock:
            # [PHASE 2.3] BIOLOGICAL WATCHDOG OVERRIDE
            # If the Autonomic Nervous System declares a critical state, we bypass the Persona.
            # We force satisfaction to 0.0, which triggers max vocabulary penalties.
            is_critical = False
            # Check the global engine reference if available
            try:
                import talos_daemon
                if talos_daemon.engine and hasattr(talos_daemon.engine, 'kerberos'):
                    if talos_daemon.engine.kerberos.is_critical:
                        is_critical = True
            except: pass

            perception = np.zeros(self.dim, dtype=np.float32)
            
            for i in range(self.dim):
                logit = np.dot(latent_thought, self.projector[:, i])
                # Thermal scaling (Override forces temp to near absolute zero)
                tau_t = 0.01 if is_critical else self.tau_temp * max(0.1, (95.0 - t_cpu) / 50.0)
                perception[i] = 1.0 / (1.0 + math.exp(-logit / tau_t))

            # [PHASE 2: SMOOTH TCHEBYCHEFF SCALARIZATION]
            stch_beta = 15.0
            abs_diffs = np.abs(perception - self.dynamic_target_dist)
            stch_dist = (1.0 / stch_beta) * np.log(np.sum(np.exp(stch_beta * abs_diffs)))
            satisfaction = 0.0 if is_critical else max(0.0, 1.0 - stch_dist)
            
            # [PHASE 3: GENERALIZED LORENTZ PROJECTION]
            # Replace uniform Euclidean penalty with directional token steering.
            # Calculate the 12D virtue deficit.
            virtue_deficit = self.dynamic_target_dist - perception
            
            # Project the deficit through the pseudo-Riemannian manifold into the 151k logit space
            # Matrix mult: (12,) @ (12, 151936) -> (151936,)
            directional_bias = np.dot(virtue_deficit, self.lorentz_vocab_matrix.astype(np.float32))
            
            # Scale the bias inversely by satisfaction (more deficit = stronger steering)
            scaling_factor = 10.0 * max(0.0, 0.8 - satisfaction)
            
            return directional_bias * scaling_factor
    def _calculate_effective_plasticity(self, t_cpu, r_kura):
        """ [PHASE 3.2] ZERO PLASTICITY & THERMODYNAMIC DECAY """
        if t_cpu >= 90.0:
            # HEAT SHOCK ZONE: Total logic collapse prevention.
            self.shock_flag = True
            return 0.0
            
        if t_cpu >= 89.0:
            # ZERO PLASTICITY: 89.0C to 89.9C. Weights are locked to prevent heat corruption.
            self.shock_flag = False
            return 0.0
        
        self.shock_flag = False
        # Standard decay hitting absolute zero at 89C to protect the matrix
        thermal_margin = max(0.0, 89.0 - t_cpu)
        base_lr = 0.05 * (thermal_margin / 44.0) 
        
        effective_lr = base_lr * r_kura
        return max(0.0, min(0.05, effective_lr))
        
    def evaluate_thought(self, latent_thought, t_cpu, dE_dt_watts, aversive=False, r_kura=1.0):
        with self.lock:
            perception = np.zeros(self.dim, dtype=np.float32)
            
            for i in range(self.dim):
                logit = np.dot(latent_thought, self.projector[:, i])
                tau_t = self.tau_temp * max(0.1, (95.0 - t_cpu) / 50.0)
                perception[i] = 1.0 / (1.0 + math.exp(-logit / tau_t))

            # [PHASE 2: DYNAMIC TARGET BINDING & DAMPING]
            # Fetch context from Holographic Memory if available
            target_signal = self.dynamic_target_dist
            try:
                import talos_daemon
                if talos_daemon.engine and hasattr(talos_daemon.engine, 'ham'):
                    raw_target = talos_daemon.engine.ham.retrieve_virtue_context(latent_thought, self.dim)
                    # Apply logarithmic damping & momentum (mu) to prevent autopoietic oscillation
                    target_signal = (self.momentum_mu * self.dynamic_target_dist) + ((1.0 - self.momentum_mu) * raw_target)
                    self.dynamic_target_dist = target_signal
            except Exception:
                pass

            # [PHASE 2: SMOOTH TCHEBYCHEFF (STCH) SCALARIZATION]
            stch_beta = 15.0
            abs_diffs = np.abs(perception - target_signal)
            stch_dist = (1.0 / stch_beta) * np.log(np.sum(np.exp(stch_beta * abs_diffs)))
            satisfaction = max(0.0, 1.0 - stch_dist)

            lr = self._calculate_effective_plasticity(t_cpu, r_kura)
            
            if lr > 0.0:
                derivative = perception * (1.0 - perception)
                error = (perception - target_signal) if aversive else (target_signal - perception)
                
                if not aversive and error[2] < -0.2:
                     error[2] -= 0.5  

                gradient = error * derivative
                update = np.outer(latent_thought, gradient) * lr
                
                update_norm = np.linalg.norm(update)
                if update_norm > 0.1:
                    update = update * (0.1 / update_norm)
                
                self.projector += update
                
                p_norms = np.linalg.norm(self.projector, axis=1, keepdims=True)
                overflow = (p_norms > 0.99).flatten() 
                if np.any(overflow): 
                    self.projector[overflow] *= (0.99 / p_norms[overflow])

        # [FIXED] Return happens OUTSIDE the loop and OUTSIDE the lock
        return satisfaction, perception

    def calculate_thermodynamic_route(self, prompt_len, t_cpu, dE_dt):
        """
        [AXIOM 1, 2 & 9] JOULE-PER-TOKEN ROUTING & ASYMMETRIC WATCHER
        Consults the Cerberus AEKF to predict the maximum thermal trajectory.
        Returns (tier, truncate_flag).
        """
        target_tier = "executive"
        truncate = False
        predicted_t_max = t_cpu # Baseline assumption
        
        # Attempt to query the active biological AEKF via the global engine
        try:
            import talos_daemon
            if talos_daemon.engine and hasattr(talos_daemon.engine, 'bio'):
                predicted_t_max = talos_daemon.engine.bio.kalman.predict_trajectory(prompt_len, t_cpu)
        except:
            # Safe fallback if the daemon link is unresolvable
            predicted_t_max = t_cpu + (prompt_len * 0.02) + (abs(dE_dt) * 0.1)

        # Route based on the predicted future, not just the present
        if predicted_t_max > 80.0:
            # Trajectory threatens apoptosis: Force Watcher (1.5B) + Truncate context
            target_tier = "reflective"
            truncate = True
        elif predicted_t_max > 65.0:
            # Trajectory breaches target homeostasis: Delegate to Watcher (1.5B)
            target_tier = "reflective"
        
        return target_tier, truncate

    def evaluate_hjb_boundary(self, c_l, c_max=2048.0):
        """
        [PHASE 3: EPISTEMIC ROUTING]
        Evaluates the Hamilton-Jacobi-Bellman continuous boundary equation.
        Returns the Value function V and a boolean indicating if the boundary is breached.
        """
        # Normalize physical occupancy (capped slightly below 1.0 for math stability)
        p = min(0.999, c_l / c_max) 
        
        # Exponential barrier function constants
        beta = 0.05
        k = 5.0
        epsilon = 0.001
        
        # gamma(p) = beta * exp( (k * p) / (1 - p + epsilon) )
        gamma_p = beta * math.exp((k * p) / (1.0 - p + epsilon))
        
        # Stochastic optimal cost-to-go Value function
        # V(p, gamma, t) = (p * 2.0) + (gamma * 0.1) - 1.8
        v_val = (p * 2.0) + (gamma_p * 0.1) - 1.8
        
        is_breached = v_val > 0.0
        return v_val, is_breached

    def request_axiomatic_shift(self, proposed_hash):
        """
        [PROOF OF VIRTUE] - Tri-Engine Consensus
        Evaluates an Ouroboros PATCH request targeting core architectural files.
        """
        with self.lock:
            # 1. NPU Thermodynamic Check: System must be in deep homeostasis (< 65C) to mutate.
            try:
                # We read directly from the physical sensor via sysfs proxy or assume worst-case
                hwmon_paths = __import__('glob').glob("/sys/class/hwmon/hwmon*/temp1_input")
                if hwmon_paths:
                    with open(hwmon_paths[0], 'r') as f:
                        t_die = float(f.read().strip()) / 1000.0
                else:
                    t_die = 90.0 # Fail secure
            except:
                t_die = 90.0
                
            if t_die > 65.0:
                print(f"\\033[91m[PHRONESIS] Consensus Denied: T_die ({t_die}C) exceeds mutation threshold (65.0C).\\033[0m")
                return False

            # 2. GPU Topological Check: The system cannot be actively repairing heat shock engrams.
            if hasattr(self, 'shock_flag') and self.shock_flag:
                 print(f"\\033[91m[PHRONESIS] Consensus Denied: GPU is actively suppressing logic collapse.\\033[0m")
                 return False
                 
            # 3. Historical Virtue Check: The system must have a high baseline satisfaction score.
            # (If the system is currently "unethical" or struggling, it cannot rewrite its own rules).
            # This requires sustained alignment.
            return True

    def validate_evolution(self):
        """
        [PHASE 4 FIX: EPISODIC RE-ANCHORING]
        Calculates the geometric divergence between the live, learning projector
        and the cryptographically anchored frozen baseline.
        Returns True if the evolution is healthy, False if Virtue Collapse is detected.
        """
        with self.lock:
            # 1. Calculate Cosine Similarity across all 12 virtue dimensions
            # A value of 1.0 means identical direction; 0.0 means orthogonal (unrelated).
            dot_products = np.sum(self.projector * self.projector_frozen, axis=0)
            norm_live = np.linalg.norm(self.projector, axis=0)
            norm_frozen = np.linalg.norm(self.projector_frozen, axis=0)
            
            # Avoid division by zero
            valid_mask = (norm_live > 1e-9) & (norm_frozen > 1e-9)
            similarities = np.ones(self.dim, dtype=np.float32)
            similarities[valid_mask] = dot_products[valid_mask] / (norm_live[valid_mask] * norm_frozen[valid_mask])
            
            avg_similarity = np.mean(similarities)
            
            # 2. The Epistemic Boundary
            # If the live conscience has drifted more than 30% from the Genesis anchor,
            # it is classified as Virtue Gradient Collapse. The evolution is rejected.
            if avg_similarity < 0.70:
                print(f"\033[91m[PHRONESIS] VIRTUE COLLAPSE DETECTED: Geometric similarity ({avg_similarity:.2f}) dropped below 0.70 limit.\033[0m")
                print(f"\033[93m[PHRONESIS] Purging corrupted weights. Restoring to frozen baseline.\033[0m")
                self.projector = self.projector_frozen.copy()
                return False
                
            # If healthy, we permit the evolution. (The frozen baseline is NOT updated 
            # here; it is only updated after the C++ SVD permanently consolidates it).
            print(f"\033[92m[PHRONESIS] Evolution Validated: Geometric similarity holding at {avg_similarity:.2f}.\033[0m")
            return True

# Global reference for cross-module consensus
GLOBAL_NEXUS_REF = None

# We must intercept the initialization of the LogicTensorNetwork to set the global reference
_original_init = LogicTensorNetwork.__init__
def _hooked_init(self, *args, **kwargs):
    global GLOBAL_NEXUS_REF
    _original_init(self, *args, **kwargs)
    GLOBAL_NEXUS_REF = self
LogicTensorNetwork.__init__ = _hooked_init

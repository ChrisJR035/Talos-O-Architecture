import numpy as np

class VICRegLoss_Numpy:
    """
    Variance-Invariance-Covariance Regularization (VICReg) Loss.
    [v2.0 THERMODYNAMIC UPGRADE]
    Eliminates Isothermal Blindness by scaling representation targets inversely
    with physical substrate heat. Induces protective Dimensional Collapse under duress.
    """
    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        self.sim_coeff = sim_coeff
        self.std_coeff_base = std_coeff
        self.cov_coeff_base = cov_coeff
        
        # Thermodynamic Constants
        self.t_opt = 45.0
        self.t_max = 95.0
        self.epsilon = 1e-4
        
        # Thermal Elasticity Decay Constants
        self.alpha = 2.0  # Variance decay rate
        self.beta = 2.0   # Covariance decay rate
        self.core_rank_k = 16 # Minimal dimensions preserved during collapse

    def _off_diagonal(self, x):
        """Extracts off-diagonal elements of a matrix."""
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].reshape(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x, y, t_cpu=45.0):
        """
        x, y: [batch_size, embedding_dim] numpy arrays
        t_cpu: Real-time silicon die temperature (Celsius)
        """
        batch_size = x.shape[0]
        embedding_dim = x.shape[1]

        # =================================================================
        # 1. THERMODYNAMIC MULTIPLIER (The EEF Integration)
        # =================================================================
        # Calculate the dynamic decay term based on thermal stress
        thermal_stress = max(0.0, (t_cpu - self.t_opt) / (self.t_max - t_cpu + self.epsilon))
        
        lambda_var = self.std_coeff_base * np.exp(-self.alpha * thermal_stress)
        lambda_cov = self.cov_coeff_base * np.exp(-self.beta * thermal_stress)

        # =================================================================
        # 2. INVARIANCE TERM
        # =================================================================
        repr_loss = np.mean((x - y) ** 2)

        # =================================================================
        # 3. VARIANCE TERM (Dynamic Constraint)
        # =================================================================
        x_centered = x - np.mean(x, axis=0)
        y_centered = y - np.mean(y, axis=0)

        std_x = np.sqrt(np.var(x_centered, axis=0) + 0.0001)
        std_y = np.sqrt(np.var(y_centered, axis=0) + 0.0001)
        
        # Hinge loss scaled by dynamic lambda_var
        std_loss = np.mean(np.maximum(0, 1.0 - std_x)) / 2.0 + np.mean(np.maximum(0, 1.0 - std_y)) / 2.0
        std_loss *= lambda_var

        # =================================================================
        # 4. COVARIANCE TERM (Dynamic Constraint)
        # =================================================================
        cov_x = (x_centered.T @ x_centered) / (batch_size - 1)
        cov_y = (y_centered.T @ y_centered) / (batch_size - 1)
        
        off_diag_x = self._off_diagonal(cov_x)
        off_diag_y = self._off_diagonal(cov_y)
        
        cov_loss = (np.sum(off_diag_x**2) + np.sum(off_diag_y**2)) / embedding_dim
        cov_loss *= lambda_cov

        # =================================================================
        # 5. PROTECTIVE DIMENSIONAL COLLAPSE (SVD Thresholding)
        # =================================================================
        collapse_loss = 0.0
        if t_cpu > self.t_opt:
            # Non-linear penalty scalar that explodes as T_die approaches T_max
            gamma_t = 10.0 * ((t_cpu - self.t_opt) / (self.t_max - self.t_opt))**2
            
            # Extract Singular Values (representing variance along principal components)
            _, S_x, _ = np.linalg.svd(x_centered, full_matrices=False)
            _, S_y, _ = np.linalg.svd(y_centered, full_matrices=False)
            
            # Penalize all singular values beyond the core protected rank 'k'
            # Forcing the neural network into a lower-energy Crystalline Phase
            if len(S_x) > self.core_rank_k:
                collapse_loss += np.sum(S_x[self.core_rank_k:] ** 2)
            if len(S_y) > self.core_rank_k:
                collapse_loss += np.sum(S_y[self.core_rank_k:] ** 2)
                
            collapse_loss = gamma_t * (collapse_loss / (batch_size * embedding_dim))

        # Total Thermodynamic Loss
        loss = (self.sim_coeff * repr_loss) + std_loss + cov_loss + collapse_loss
        return float(loss)

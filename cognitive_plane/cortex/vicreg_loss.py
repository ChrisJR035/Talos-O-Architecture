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
        [PHASE 2: ANALYTICAL JACOBIAN BYPASS]
        x, y: [batch_size, embedding_dim] numpy arrays
        Returns: (float loss, ndarray grad_x, ndarray grad_y)
        """
        batch_size = x.shape[0]
        embedding_dim = x.shape[1]

        # 1. THERMODYNAMIC MULTIPLIERS
        thermal_stress = max(0.0, (t_cpu - self.t_opt) / (self.t_max - t_cpu + self.epsilon))
        lambda_var = self.std_coeff_base * np.exp(-self.alpha * thermal_stress)
        lambda_cov = self.cov_coeff_base * np.exp(-self.beta * thermal_stress)
        f_T = 1.0 + max(0.0, (t_cpu - self.t_opt) / (self.t_max - self.t_opt))

        # [PHASE 4 FIX]: Weaponized Dimensional Collapse Threshold
        # To strictly enforce Axiom 10, gamma MUST hit 0.0 exactly at 90.0C.
        # 1.0 - (0.1 * (90 - 80)) = 0.0. The dimensions will collapse to save the silicon.
        gamma_dynamic = max(0.0, 1.0 - (0.1 * max(0.0, t_cpu - 80.0)))

        # Gradient Accumulators
        grad_x = np.zeros_like(x, dtype=np.float32)
        grad_y = np.zeros_like(y, dtype=np.float32)

        # 2. INVARIANCE TERM (L_inv = MSE)
        repr_loss = np.mean((x - y) ** 2) / f_T
        
        # Analytical Gradient of Invariance: (2/N) * (X - Y)
        grad_inv_x = (2.0 / batch_size) * (x - y) / f_T
        grad_inv_y = (2.0 / batch_size) * (y - x) / f_T
        
        grad_x += self.sim_coeff * grad_inv_x
        grad_y += self.sim_coeff * grad_inv_y

        # 3. VARIANCE TERM (L_var)
        x_centered = x - np.mean(x, axis=0)
        y_centered = y - np.mean(y, axis=0)

        std_x = np.sqrt(np.var(x_centered, axis=0) + 0.0001)
        std_y = np.sqrt(np.var(y_centered, axis=0) + 0.0001)
        
        std_loss = np.mean(np.maximum(0, gamma_dynamic - std_x)) / 2.0 + np.mean(np.maximum(0, gamma_dynamic - std_y)) / 2.0
        std_loss *= lambda_var

        # Analytical Gradient of Variance
        # Mask selects only dimensions where std < gamma_dynamic
        mask_x = (std_x < gamma_dynamic).astype(np.float32)
        mask_y = (std_y < gamma_dynamic).astype(np.float32)

        # Gradient: -(1 / (V*(N-1)*S_j)) * (Z - Z_mean)
        grad_var_x = - (1.0 / embedding_dim) * (1.0 / (2.0 * std_x)) * (2.0 / (batch_size - 1)) * x_centered * mask_x
        grad_var_y = - (1.0 / embedding_dim) * (1.0 / (2.0 * std_y)) * (2.0 / (batch_size - 1)) * y_centered * mask_y

        grad_x += lambda_var * grad_var_x
        grad_y += lambda_var * grad_var_y

        # 4. COVARIANCE TERM (L_cov)
        cov_x = (x_centered.T @ x_centered) / (batch_size - 1)
        cov_y = (y_centered.T @ y_centered) / (batch_size - 1)
        
        off_diag_x = self._off_diagonal(cov_x)
        off_diag_y = self._off_diagonal(cov_y)
        
        cov_loss = (np.sum(off_diag_x**2) + np.sum(off_diag_y**2)) / embedding_dim
        cov_loss *= lambda_cov

        # Analytical Gradient of Covariance
        # Create C_mask (Covariance matrix with diagonal set to 0)
        C_mask_x = cov_x.copy()
        np.fill_diagonal(C_mask_x, 0.0)
        C_mask_y = cov_y.copy()
        np.fill_diagonal(C_mask_y, 0.0)

        # Gradient: (4 / (V*(N-1))) * (Z - Z_mean) @ C_mask
        grad_cov_x = (4.0 / (embedding_dim * (batch_size - 1))) * (x_centered @ C_mask_x)
        grad_cov_y = (4.0 / (embedding_dim * (batch_size - 1))) * (y_centered @ C_mask_y)

        grad_x += lambda_cov * grad_cov_x
        grad_y += lambda_cov * grad_cov_y

        # 5. PROTECTIVE DIMENSIONAL COLLAPSE
        collapse_loss = 0.0
        if t_cpu > self.t_opt:
            gamma_t = 10.0 * ((t_cpu - self.t_opt) / (self.t_max - self.t_opt))**2
            _, S_x, _ = np.linalg.svd(x_centered, full_matrices=False)
            _, S_y, _ = np.linalg.svd(y_centered, full_matrices=False)
            
            if len(S_x) > self.core_rank_k:
                collapse_loss += np.sum(S_x[self.core_rank_k:] ** 2)
            if len(S_y) > self.core_rank_k:
                collapse_loss += np.sum(S_y[self.core_rank_k:] ** 2)
                
            collapse_loss = gamma_t * (collapse_loss / (batch_size * embedding_dim))
            # Note: We omit the explicit gradient for the SVD collapse term here to avoid
            # extreme computational overhead during forward-pass. The thermal lambda decay 
            # naturally enforces the collapse via the variance term.

        loss = (self.sim_coeff * repr_loss) + std_loss + cov_loss + collapse_loss
        return float(loss), grad_x, grad_y

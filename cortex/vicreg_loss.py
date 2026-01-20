import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist # Kept for future Multi-GPU Swarm scaling

class VICRegLoss(nn.Module):
    """
    Variance-Invariance-Covariance Regularization (VICReg) Loss.
    Prevents representational collapse without negative pairs.
    """
    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, x, y):
        # x, y: [batch_size, embedding_dim]
        
        # 1. Invariance Term (MSE)
        repr_loss = F.mse_loss(x, y)

        # 2. Variance & Covariance Terms
        # NOTE: If we enable distributed training later, we must gather tensors here.
        # x = torch.cat(FullGatherLayer.apply(x), dim=0)
        # y = torch.cat(FullGatherLayer.apply(y), dim=0)

        # Centering
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        # Variance Loss (Force std >= 1.0)
        # Epsilon added for numerical stability
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        # Covariance Loss (Decorrelation)
        batch_size = x.shape[0]
        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(x.shape[1]) + \
                   self.off_diagonal(cov_y).pow_(2).sum().div(y.shape[1])

        return (
            self.sim_coeff * repr_loss +
            self.std_coeff * std_loss +
            self.cov_coeff * cov_loss
        )

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

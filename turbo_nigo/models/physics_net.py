import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class PhysicsInferenceNet(nn.Module):
    """
    Infers the coefficients defining the infinitesimal generator A = alpha*S + beta*N
    from the initial state z0 and physical conditions.
    """
    def __init__(self, latent_dim: int, num_bases: int = 8, hidden: int = 128, cond_dim: int = 4):
        super().__init__()
        self.in_dim = latent_dim * 2 + cond_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden//2), nn.GELU()
        )
        # Outputs: k_coeffs, r_coeffs, alpha, beta
        self.out_coeffs = nn.Linear(hidden//2, num_bases * 2 + 2)

    def forward(self, z0: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the physics inferred parameters.
        Returns: k_coeffs, r_coeffs, alpha, beta
        """
        x = torch.cat([z0.real, z0.imag, cond.to(z0.real.device)], dim=1)
        h = self.net(x)
        
        out = self.out_coeffs(h)
        # num_bases for k, num_bases for r
        num_b = (out.shape[1] - 2) // 2
        
        k = torch.tanh(out[:, :num_b])
        r = torch.tanh(out[:, num_b:2*num_b])
        
        # alpha, beta must be > 0
        alpha = F.softplus(out[:, -2]).view(-1, 1, 1) + 1e-4
        beta = F.softplus(out[:, -1]).view(-1, 1, 1) + 1e-4
        
        return k, r, alpha, beta

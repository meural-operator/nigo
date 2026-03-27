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
        self.out_coeffs = nn.Linear(hidden//2, num_bases * 2)

    def forward(self, z0: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the physics inferred parameters.
        Returns: k_coeffs, r_coeffs
        """
        x = torch.cat([z0.real, z0.imag, cond.to(z0.real.device)], dim=1)
        h = self.net(x)
        
        coeffs = self.out_coeffs(h)
        k = torch.tanh(coeffs[:, :coeffs.shape[1]//2])
        r = torch.tanh(coeffs[:, coeffs.shape[1]//2:])
        
        return k, r

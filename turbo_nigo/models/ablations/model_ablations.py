import torch
import torch.nn as nn
from typing import Tuple

from turbo_nigo.models.turbo_nigo import GlobalTurboNIGO
from turbo_nigo.models.ablations.generator_ablations import NoSkewGenerator, NoDissipativeGenerator, DenseGenerator

class Ablation1_NoSkewTurboNIGO(GlobalTurboNIGO):
    def __init__(self, latent_dim: int = 64, num_bases: int = 8, cond_dim: int = 4, 
                 width: int = 32, spatial_size: int = 64, in_channels: int = 2):
        super().__init__(latent_dim=latent_dim, num_bases=num_bases, cond_dim=cond_dim, 
                         width=width, spatial_size=spatial_size, in_channels=in_channels)
        self.generator = NoSkewGenerator(latent_dim=latent_dim, num_bases=num_bases)

class Ablation2_NoDissipativeTurboNIGO(GlobalTurboNIGO):
    def __init__(self, latent_dim: int = 64, num_bases: int = 8, cond_dim: int = 4, 
                 width: int = 32, spatial_size: int = 64, in_channels: int = 2):
        super().__init__(latent_dim=latent_dim, num_bases=num_bases, cond_dim=cond_dim, 
                         width=width, spatial_size=spatial_size, in_channels=in_channels)
        self.generator = NoDissipativeGenerator(latent_dim=latent_dim, num_bases=num_bases)

class Ablation3_DenseGeneratorTurboNIGO(GlobalTurboNIGO):
    def __init__(self, latent_dim: int = 64, num_bases: int = 8, cond_dim: int = 4, 
                 width: int = 32, spatial_size: int = 64, in_channels: int = 2):
        super().__init__(latent_dim=latent_dim, num_bases=num_bases, cond_dim=cond_dim, 
                         width=width, spatial_size=spatial_size, in_channels=in_channels)
        self.generator = DenseGenerator(latent_dim=latent_dim, num_bases=num_bases)

class Ablation4_NoRefinerTurboNIGO(GlobalTurboNIGO):
    def forward(self, u0: torch.Tensor, time_steps: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z0 = self.encoder(u0, cond)
        k_coeffs, r_coeffs, alpha, beta = self.cond_net(z0, cond)
        z_base = self.generator(z0, time_steps, k_coeffs, r_coeffs, alpha, beta)
        
        # ABLATION 4: SKIP the Temporal Refiner completely
        z_refined = z_base
        
        u_pred = self.decoder(z_refined)
        return u_pred, z_base, k_coeffs, r_coeffs, alpha, beta


# --- ABLATION 5: Unscaled Generator (Forcing Alpha=1, Beta=1) ---
class UnscaledPhysicsNet(nn.Module):
    def __init__(self, cond_net: nn.Module):
        super().__init__()
        self.cond_net = cond_net
        
    def forward(self, z0: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        k_coeffs, r_coeffs, alpha, beta = self.cond_net(z0, cond)
        # Force alpha and beta to be exactly 1.0 (unscaled)
        alpha_ones = torch.ones_like(alpha)
        beta_ones = torch.ones_like(beta)
        return k_coeffs, r_coeffs, alpha_ones, beta_ones

class Ablation5_UnscaledTurboNIGO(GlobalTurboNIGO):
    def __init__(self, latent_dim: int = 64, num_bases: int = 8, cond_dim: int = 4, 
                 width: int = 32, spatial_size: int = 64, in_channels: int = 2):
        super().__init__(latent_dim=latent_dim, num_bases=num_bases, cond_dim=cond_dim, 
                         width=width, spatial_size=spatial_size, in_channels=in_channels)
        # Wrap the physics net to override alpha and beta outputs
        self.cond_net = UnscaledPhysicsNet(self.cond_net)

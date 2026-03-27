import torch
import torch.nn as nn
from typing import Tuple

from .encoder import SpectralEncoder
from .decoder import SpectralDecoder
from .generator import HyperTurbulentGenerator
from .refiner import TemporalRefiner
from .physics_net import PhysicsInferenceNet

class GlobalTurboNIGO(nn.Module):
    """
    Main Global Turbo-NIGO composite model aggregating the encoder, generator,
    refiner, physics inference net, and decoder.
    
    Args:
        latent_dim: dimensionality of complex latent space
        num_bases: number of generator basis matrices
        cond_dim: number of conditioning variables
        width: base channel width for encoder/decoder
        spatial_size: spatial resolution of input fields (H=W assumed)
        in_channels: number of input field channels (default 2 for u, v)
    """
    def __init__(self, latent_dim: int = 64, num_bases: int = 8, cond_dim: int = 4, 
                 width: int = 32, spatial_size: int = 64, in_channels: int = 2):
        super().__init__()
        self.encoder = SpectralEncoder(
            in_channels=in_channels, latent_dim=latent_dim, width=width, 
            cond_channels=cond_dim, spatial_size=spatial_size
        )
        self.cond_net = PhysicsInferenceNet(latent_dim=latent_dim, num_bases=num_bases, cond_dim=cond_dim)
        self.generator = HyperTurbulentGenerator(latent_dim=latent_dim, num_bases=num_bases)
        self.refiner = TemporalRefiner(latent_dim=latent_dim)
        
        # Compute what size the encoder produces via 3 stride-2 convs
        enc_spatial = spatial_size // 8  # three stride-2 layers
        self.decoder = SpectralDecoder(
            latent_dim=latent_dim, out_channels=in_channels, width=width, 
            initial_size=enc_spatial
        )

    def forward(self, u0: torch.Tensor, time_steps: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            u0: (B, C, H, W) initial field
            time_steps: (S,) timeline
            cond: (B, cond_dim) conditions
        Returns:
            u_pred: (B, S, C, H, W) predicted sequence
            z_base: Unrefined trajectory in latent space
            k_coeffs, r_coeffs: Generator basis coefficients
        """
        z0 = self.encoder(u0, cond)
        k_coeffs, r_coeffs = self.cond_net(z0, cond)
        z_base = self.generator(z0, time_steps, k_coeffs, r_coeffs)
        z_refined = self.refiner(z_base)
        u_pred = self.decoder(z_refined)
        return u_pred, z_base, k_coeffs, r_coeffs

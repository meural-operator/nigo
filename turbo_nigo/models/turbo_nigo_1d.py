import torch
import torch.nn as nn
from typing import Tuple

from .encoder_1d import SpectralEncoder1D
from .decoder_1d import SpectralDecoder1D
from .generator import HyperTurbulentGenerator
from .refiner import TemporalRefiner
from .physics_net import PhysicsInferenceNet


class GlobalTurboNIGO_1D(nn.Module):
    """
    1D variant of GlobalTurboNIGO for problems defined on 1D spatial grids.

    Uses SpectralEncoder1D and SpectralDecoder1D (Conv1d-based) while
    reusing the latent-space components (Generator, Refiner, PhysicsNet)
    identically — those operate on abstract complex vectors and are
    dimension-agnostic by design.

    Args:
        latent_dim: Dimensionality of the complex latent space.
        num_bases: Number of generator basis matrices for A = α·S + β·N.
        cond_dim: Number of conditioning variables.
        width: Base channel width for encoder/decoder convolutions.
        spatial_size: Length of the 1D spatial grid (e.g. 1024).
        in_channels: Number of input field channels (1 for scalar Burgers).
        num_layers: Number of stride-2 down/upsampling blocks.
        use_residual: Whether to use residual connections.
        norm_type: 'group' for GroupNorm or None.
    """
    def __init__(self, latent_dim: int = 64, num_bases: int = 8, cond_dim: int = 4,
                 width: int = 32, spatial_size: int = 1024, in_channels: int = 1,
                 num_layers: int = 3, use_residual: bool = False, norm_type: str = None,
                 use_adaptive_refiner: bool = False, use_spectral_norm: bool = False):
        super().__init__()
        self.encoder = SpectralEncoder1D(
            in_channels=in_channels, latent_dim=latent_dim, width=width,
            cond_channels=cond_dim, spatial_size=spatial_size,
            num_layers=num_layers, use_residual=use_residual, norm_type=norm_type
        )
        self.cond_net = PhysicsInferenceNet(
            latent_dim=latent_dim, num_bases=num_bases, cond_dim=cond_dim
        )
        self.generator = HyperTurbulentGenerator(
            latent_dim=latent_dim, num_bases=num_bases
        )
        self.refiner = TemporalRefiner(
            latent_dim=latent_dim,
            use_adaptive_refiner=use_adaptive_refiner,
            use_spectral_norm=use_spectral_norm
        )

        # Decoder initial_len = spatial_size / 2^num_layers
        enc_len = spatial_size // (2 ** num_layers)
        self.decoder = SpectralDecoder1D(
            latent_dim=latent_dim, out_channels=in_channels, width=width,
            initial_len=enc_len, num_layers=num_layers,
            use_residual=use_residual, norm_type=norm_type,
            use_spectral_norm=use_spectral_norm
        )

    def forward(self, u0: torch.Tensor, time_steps: torch.Tensor,
                cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                              torch.Tensor, torch.Tensor,
                                              torch.Tensor, torch.Tensor]:
        """
        Args:
            u0: (B, C, X) initial 1D field.
            time_steps: (S,) evaluation times.
            cond: (B, cond_dim) conditioning vector.
        Returns:
            u_pred: (B, S, C, X) predicted field sequence.
            z_base: (B, S, D) unrefined latent trajectory.
            k_coeffs, r_coeffs: Generator basis coefficients.
            alpha, beta: Generator scaling parameters.
        """
        z0 = self.encoder(u0, cond)
        k_coeffs, r_coeffs, alpha, beta = self.cond_net(z0, cond)
        z_base = self.generator(z0, time_steps, k_coeffs, r_coeffs, alpha, beta)
        z_refined = self.refiner(z_base)
        u_pred = self.decoder(z_refined)
        return u_pred, z_base, k_coeffs, r_coeffs, alpha, beta

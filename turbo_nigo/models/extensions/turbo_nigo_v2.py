"""
GlobalTurboNIGO_V2 — Extended model wrapper using dynamic (resolution-agnostic)
encoder/decoder and attention-based physics inference.

This is a drop-in replacement for GlobalTurboNIGO that swaps in the extension
components while preserving the **same generator and refiner** (same Lyapunov
stability mathematics).

Variants:
  - physics_net_type="distribution": Uses DistributionAwareAttentionPhysics
  - physics_net_type="spatial": Uses SpatialPhysicsAttention (requires encoder
    feature maps)
"""
import torch
import torch.nn as nn

from turbo_nigo.models.generator import HyperTurbulentGenerator
from turbo_nigo.models.refiner import TemporalRefiner
from turbo_nigo.models.extensions.dynamic_encoder import DynamicSpectralEncoder
from turbo_nigo.models.extensions.dynamic_decoder import DynamicSpectralDecoder
from turbo_nigo.models.extensions.attention_physics import (
    DistributionAwareAttentionPhysics,
    SpatialPhysicsAttention,
)


class GlobalTurboNIGO_V2(nn.Module):
    """
    V2 model with dynamic encoder/decoder and attention-based physics inference.

    Args:
        latent_dim: Complex latent space dimension.
        num_bases: Number of learnable basis matrices for K and R.
        cond_dim: Dimension of the conditioning vector.
        width: Base channel width for conv layers.
        in_channels: Number of physical field channels (e.g. 2 for u,v).
        target_res: Target spatial resolution (must be 8 * 2^n).
        physics_net_type: "distribution" or "spatial".
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_bases: int = 12,
        cond_dim: int = 4,
        width: int = 48,
        in_channels: int = 2,
        target_res: int = 256,
        physics_net_type: str = "distribution",
    ):
        super().__init__()
        self.physics_net_type = physics_net_type

        use_feat_map = (physics_net_type == "spatial")
        self.encoder = DynamicSpectralEncoder(
            in_channels, latent_dim, width=width,
            cond_channels=cond_dim, return_feature_map=use_feat_map,
        )

        if physics_net_type == "spatial":
            self.cond_net = SpatialPhysicsAttention(
                latent_dim, spatial_channels=width * 4,
                num_bases=num_bases, embed_dim=latent_dim,
            )
        else:  # "distribution"
            self.cond_net = DistributionAwareAttentionPhysics(
                latent_dim, num_bases=num_bases, embed_dim=latent_dim,
            )

        # Core math: same generator + refiner as base model
        self.generator = HyperTurbulentGenerator(latent_dim, num_bases=num_bases)
        self.refiner = TemporalRefiner(latent_dim)
        self.decoder = DynamicSpectralDecoder(
            latent_dim, out_channels=in_channels,
            width=width, target_res=target_res,
        )

    def forward(self, u0, time_steps, cond):
        # Encode
        if self.physics_net_type == "spatial":
            z0, feat_map = self.encoder(u0, cond)
            alpha, beta, k_coeffs, r_coeffs = self.cond_net(feat_map, u0)
        else:
            z0 = self.encoder(u0, cond)
            alpha, beta, k_coeffs, r_coeffs = self.cond_net(z0, u0, cond)

        # Generate (Lyapunov-stable dynamics)
        z_base = self.generator(z0, time_steps, k_coeffs, r_coeffs, alpha, beta)

        # Refine (identity-initialized correction)
        z_refined = self.refiner(z_base)

        # Decode
        u_pred = self.decoder(z_refined)

        return u_pred, z_base, k_coeffs, r_coeffs, alpha, beta

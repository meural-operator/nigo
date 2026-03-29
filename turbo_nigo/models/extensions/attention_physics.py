"""
Attention-based Physics Inference Networks.

Drop-in replacements for PhysicsInferenceNet that use multi-head attention
to infer the generator parameters (α, β, k_coeffs, r_coeffs) from richer
context than a simple MLP.

Two variants:
  1. DistributionAwareAttentionPhysics — attends over [z_token, field_stats_token]
  2. SpatialPhysicsAttention — attends over [64 spatial patch tokens, stats_token]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DistributionAwareAttentionPhysics(nn.Module):
    """
    Uses learnable physics queries + MHA over latent and input distribution statistics.

    Instead of a simple MLP on z0, this module:
      1. Projects z0 (real+imag) into an embedding token.
      2. Computes 8 distribution statistics from the raw input u0 (mean, std, max, energy
         per channel) and embeds them into a stats token.
      3. Cross-attends learned physics queries against these two context tokens.
      4. Decodes α, β, and basis coefficients from the attended outputs.

    Note: forward(z0, u0, cond) — the extra u0 argument is required.

    Args:
        latent_dim: Complex latent dimension.
        num_bases: Number of basis functions for K and R.
        embed_dim: Embedding dimension for the attention mechanism.
        num_heads: Number of attention heads.
    """

    def __init__(
        self,
        latent_dim: int,
        num_bases: int = 8,
        embed_dim: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.z_proj = nn.Linear(latent_dim * 2, embed_dim)
        self.stat_embedding = nn.Linear(8, embed_dim)  # 4 stats × 2 channels

        # Learnable queries for α, β, and basis coefficients
        self.physics_queries = nn.Parameter(torch.randn(3, embed_dim))
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.head_alpha = nn.Linear(embed_dim, 1)
        self.head_beta = nn.Linear(embed_dim, 1)
        self.head_coeffs = nn.Linear(embed_dim, num_bases * 2)

        nn.init.constant_(self.head_alpha.bias, 0.0)
        nn.init.constant_(self.head_beta.bias, 0.5)

    def compute_stats(self, u0: torch.Tensor) -> torch.Tensor:
        """Extract distribution fingerprint from raw input field."""
        dims = (2, 3)  # spatial dims
        mean = u0.mean(dim=dims)       # (B, C)
        std = u0.std(dim=dims)         # (B, C)
        mx = u0.amax(dim=dims)         # (B, C)
        energy = (u0 ** 2).mean(dim=dims)  # (B, C)
        return torch.cat([mean, std, mx, energy], dim=1)  # (B, 8)

    def forward(
        self,
        z0: torch.Tensor,
        u0: torch.Tensor,
        cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = z0.shape[0]

        # Context tokens
        z_token = self.z_proj(
            torch.cat([z0.real, z0.imag], dim=1)
        ).unsqueeze(1)  # (B, 1, E)
        stats = self.compute_stats(u0)
        dist_token = self.stat_embedding(stats).unsqueeze(1)  # (B, 1, E)
        context = torch.cat([z_token, dist_token], dim=1)  # (B, 2, E)

        # Cross-attention
        queries = self.physics_queries.unsqueeze(0).expand(B, -1, -1)  # (B, 3, E)
        attn_out, _ = self.mha(queries, context, context)  # (B, 3, E)

        # Decode
        alpha = F.softplus(self.head_alpha(attn_out[:, 0])).clamp(max=50.0) + 1e-6
        beta = F.softplus(self.head_beta(attn_out[:, 1])).clamp(max=50.0) + 1e-6
        coeffs = self.head_coeffs(attn_out[:, 2])

        k_coeffs = torch.tanh(coeffs[:, : coeffs.shape[1] // 2])
        r_coeffs = torch.tanh(coeffs[:, coeffs.shape[1] // 2 :])

        return alpha.view(B, 1, 1), beta.view(B, 1, 1), k_coeffs, r_coeffs


class SpatialPhysicsAttention(nn.Module):
    """
    Attends over spatial patch tokens from the encoder's 8×8 feature map
    to infer generator parameters with spatial awareness.

    Requires the encoder to return the feature map (use DynamicSpectralEncoder
    with return_feature_map=True).

    The context consists of:
      - 64 spatial tokens (each 8×8 patch of the encoder bottleneck)
      - 1 global statistics token (mean, std, max, energy of input field)
    Learned physics queries attend over this context.

    Note: forward(feat_map, u0) — takes the encoder feature map directly.

    Args:
        latent_dim: Complex latent dimension (unused, kept for API compat).
        spatial_channels: Channel depth of the encoder feature map (typically width*4).
        num_bases: Number of basis functions for K and R.
        embed_dim: Embedding dimension for the attention mechanism.
        num_heads: Number of attention heads.
    """

    def __init__(
        self,
        latent_dim: int,
        spatial_channels: int,
        num_bases: int = 8,
        embed_dim: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()
        self.spatial_proj = nn.Linear(spatial_channels, embed_dim)
        self.stat_embedding = nn.Linear(8, embed_dim)

        self.physics_queries = nn.Parameter(torch.randn(3, embed_dim))
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.head_alpha = nn.Linear(embed_dim, 1)
        self.head_beta = nn.Linear(embed_dim, 1)
        self.head_coeffs = nn.Linear(embed_dim, num_bases * 2)

        nn.init.constant_(self.head_alpha.bias, 0.0)
        nn.init.constant_(self.head_beta.bias, 0.5)

    def compute_stats(self, u0: torch.Tensor) -> torch.Tensor:
        dims = (2, 3)
        return torch.cat([
            u0.mean(dim=dims),
            u0.std(dim=dims),
            u0.amax(dim=dims),
            (u0 ** 2).mean(dim=dims),
        ], dim=1)

    def forward(
        self,
        feat_map: torch.Tensor,
        u0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = feat_map.shape

        # Spatial tokens: (B, C, 8, 8) → (B, 64, C) → (B, 64, E)
        spatial_tokens = feat_map.view(B, C, -1).permute(0, 2, 1)
        spatial_tokens = self.spatial_proj(spatial_tokens)

        # Global stats token
        stats = self.compute_stats(u0)
        stat_token = self.stat_embedding(stats).unsqueeze(1)  # (B, 1, E)

        # Context = [64 patches, 1 global]
        context = torch.cat([spatial_tokens, stat_token], dim=1)  # (B, 65, E)

        # Cross-attention
        queries = self.physics_queries.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.mha(queries, context, context)

        # Decode
        alpha = F.softplus(self.head_alpha(attn_out[:, 0])).clamp(max=50.0) + 1e-6
        beta = F.softplus(self.head_beta(attn_out[:, 1])).clamp(max=50.0) + 1e-6
        coeffs = self.head_coeffs(attn_out[:, 2])

        k_coeffs = torch.tanh(coeffs[:, : coeffs.shape[1] // 2])
        r_coeffs = torch.tanh(coeffs[:, coeffs.shape[1] // 2 :])

        return alpha.view(B, 1, 1), beta.view(B, 1, 1), k_coeffs, r_coeffs

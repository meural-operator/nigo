"""
DynamicSpectralEncoder — Resolution-agnostic encoder using AdaptiveAvgPool2d.

Drop-in replacement for SpectralEncoder. Works with any spatial resolution
(64, 128, 256, 512, ...) without config changes, thanks to adaptive pooling
to a fixed 8×8 bottleneck.
"""
import torch
import torch.nn as nn


class DynamicSpectralEncoder(nn.Module):
    """
    Encodes physical fields to complex latent vectors via CNN + adaptive pooling.

    Unlike the base SpectralEncoder (which uses a probe tensor to compute flat_dim),
    this uses AdaptiveAvgPool2d((8, 8)) making it truly resolution-agnostic with
    a constant flat_dim = width*4 * 64.

    Args:
        in_channels: Number of input physical field channels (e.g. 2 for u,v).
        latent_dim: Dimension of the complex latent vector.
        width: Base channel width for convolutional layers.
        cond_channels: Number of conditioning channels (appended spatially).
        return_feature_map: If True, returns (z0, feat_map) for use with
            SpatialPhysicsAttention. Default False for backward compatibility.
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        width: int = 32,
        cond_channels: int = 4,
        return_feature_map: bool = False,
    ):
        super().__init__()
        self.cond_channels = cond_channels
        self.return_feature_map = return_feature_map

        self.net = nn.Sequential(
            nn.Conv2d(in_channels + cond_channels, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width * 2, 3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(width * 2, width * 4, 3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(width * 4, width * 4, 3, padding=1, stride=2),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((8, 8))

        self.flat_dim = width * 4 * 8 * 8
        self.fc_real = nn.Linear(self.flat_dim, latent_dim)
        self.fc_imag = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        B, C, H, W = x.shape
        cond_map = cond.view(B, -1, 1, 1).expand(B, self.cond_channels, H, W)
        xin = torch.cat([x, cond_map.to(x.device)], dim=1)

        feat_map = self.pool(self.net(xin))  # (B, width*4, 8, 8)
        feat_flat = feat_map.flatten(1)

        z0 = torch.complex(self.fc_real(feat_flat), self.fc_imag(feat_flat))

        if self.return_feature_map:
            return z0, feat_map
        return z0

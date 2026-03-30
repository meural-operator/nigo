import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """Residual block with 1D convolutions and optional GroupNorm."""
    def __init__(self, in_c, out_c, stride=1, norm_type=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.norm1 = nn.GroupNorm(min(16, max(1, out_c // 4)), out_c) if norm_type == 'group' else nn.Identity()
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(16, max(1, out_c // 4)), out_c) if norm_type == 'group' else nn.Identity()
        self.act2 = nn.GELU()

        self.skip = nn.Identity()
        if stride != 1 or in_c != out_c:
            skip_layers = [nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride)]
            if norm_type == 'group':
                skip_layers.append(nn.GroupNorm(min(16, max(1, out_c // 4)), out_c))
            self.skip = nn.Sequential(*skip_layers)

    def forward(self, x):
        res = self.skip(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act2(out + res)


class SpectralEncoder1D(nn.Module):
    """
    Encodes a 1D spatial field + conditions into a complex latent vector.

    This is the 1D analog of SpectralEncoder — designed for problems where
    the physical field lives on a 1D grid (e.g. Burgers, KdV, reaction-diffusion).
    Uses Conv1d throughout to preserve true spatial locality.

    Args:
        in_channels: Number of field components (e.g. 1 for scalar Burgers).
        latent_dim: Dimensionality of the complex latent space.
        width: Base channel width for convolutional layers.
        cond_channels: Number of conditioning variables (broadcast spatially).
        spatial_size: Length of the 1D spatial grid (e.g. 1024).
        num_layers: Number of stride-2 downsampling blocks.
        use_residual: Whether to use residual connections in each block.
        norm_type: 'group' for GroupNorm or None.
    """
    def __init__(self, in_channels: int, latent_dim: int, width: int = 32,
                 cond_channels: int = 4, spatial_size: int = 1024,
                 num_layers: int = 3, use_residual: bool = False,
                 norm_type: str = None):
        super().__init__()
        self.cond_channels = cond_channels

        curr_width = width

        # 1. Stem
        stem_layers = [nn.Conv1d(in_channels + cond_channels, width, 3, padding=1)]
        if norm_type == 'group':
            stem_layers.append(nn.GroupNorm(min(8, max(1, width // 4)), width))
        stem_layers.append(nn.GELU())
        self.stem = nn.Sequential(*stem_layers)

        # 2. Downsampling Blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            in_w = curr_width
            out_w = curr_width * 2 if i < num_layers - 1 else curr_width

            if use_residual:
                self.blocks.append(ResidualBlock1D(in_w, out_w, stride=2, norm_type=norm_type))
            else:
                block = [nn.Conv1d(in_w, out_w, 3, padding=1, stride=2)]
                if norm_type == 'group':
                    block.append(nn.GroupNorm(min(16, max(1, out_w // 4)), out_w))
                block.append(nn.GELU())
                self.blocks.append(nn.Sequential(*block))

            curr_width = out_w

        # 3. Compute flattened dimension via a dry-run
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels + cond_channels, spatial_size)
            feat = self.stem(dummy)
            for block in self.blocks:
                feat = block(feat)
            self.flat_dim = feat.numel()

        self.flatten = nn.Flatten()
        self.fc_real = nn.Linear(self.flat_dim, latent_dim)
        self.fc_imag = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, X) spatial field.
            cond: (B, cond_dim) conditioning vector.
        Returns:
            z0: (B, latent_dim) complex latent encoding.
        """
        B, C, X = x.shape
        # Broadcast conditioning as spatial channels: (B, cond_dim, X)
        cond_map = cond.view(B, -1, 1).expand(B, self.cond_channels, X).to(x.device)
        xin = torch.cat([x, cond_map], dim=1)

        feat = self.stem(xin)
        for block in self.blocks:
            feat = block(feat)

        feat = self.flatten(feat)
        with torch.amp.autocast('cuda', enabled=False):
            return torch.complex(self.fc_real(feat), self.fc_imag(feat))

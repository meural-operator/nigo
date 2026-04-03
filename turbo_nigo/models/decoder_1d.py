import torch
import torch.nn as nn


class ResidualUpBlock1D(nn.Module):
    """Residual upsampling block with 1D transposed convolutions."""
    def __init__(self, in_c, out_c, norm_type=None, use_spectral_norm=False):
        super().__init__()
        
        def sn(layer):
            return torch.nn.utils.spectral_norm(layer) if use_spectral_norm else layer
            
        self.up = sn(nn.ConvTranspose1d(in_c, out_c, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.norm1 = nn.GroupNorm(min(16, max(1, out_c // 4)), out_c) if norm_type == 'group' else nn.Identity()
        self.act1 = nn.GELU()

        self.conv2 = sn(nn.Conv1d(out_c, out_c, kernel_size=3, padding=1))
        self.norm2 = nn.GroupNorm(min(16, max(1, out_c // 4)), out_c) if norm_type == 'group' else nn.Identity()
        self.act2 = nn.GELU()

        skip_layers = [sn(nn.ConvTranspose1d(in_c, out_c, kernel_size=3, stride=2, padding=1, output_padding=1))]
        if norm_type == 'group':
            skip_layers.append(nn.GroupNorm(min(16, max(1, out_c // 4)), out_c))
        self.skip = nn.Sequential(*skip_layers)

    def forward(self, x):
        res = self.skip(x)
        out = self.act1(self.norm1(self.up(x)))
        out = self.norm2(self.conv2(out))
        return self.act2(out + res)


class SpectralDecoder1D(nn.Module):
    """
    Decodes complex latent trajectories back into 1D physical fields.

    This is the 1D analog of SpectralDecoder — uses ConvTranspose1d for
    upsampling to reconstruct the spatial field from the latent representation.

    Args:
        latent_dim: Dimensionality of the complex latent space.
        out_channels: Number of output field components.
        width: Base channel width.
        initial_len: Length of the 1D feature map after the linear projection.
                     Should equal spatial_size // (2^num_layers).
        num_layers: Number of stride-2 upsampling blocks.
        use_residual: Whether to use residual connections.
        norm_type: 'group' for GroupNorm or None.
    """
    def __init__(self, latent_dim: int, out_channels: int, width: int = 32,
                 initial_len: int = 128, num_layers: int = 3,
                 use_residual: bool = False, norm_type: str = None,
                 use_spectral_norm: bool = False):
        super().__init__()
        self.width = width
        self.initial_len = initial_len

        def sn(layer):
            return torch.nn.utils.spectral_norm(layer) if use_spectral_norm else layer

        # Project concatenated [real, imag] into spatial feature map
        self.fc = sn(nn.Linear(latent_dim * 2, width * 4 * initial_len))

        self.ups = nn.ModuleList()
        curr_width = width * 4

        for i in range(num_layers):
            in_w = curr_width
            out_w = curr_width // 2 if (num_layers - i) <= 2 else curr_width
            out_w = max(width, out_w)

            if use_residual:
                self.ups.append(ResidualUpBlock1D(in_w, out_w, norm_type=norm_type, use_spectral_norm=use_spectral_norm))
            else:
                up = [sn(nn.ConvTranspose1d(in_w, out_w, 3, stride=2, padding=1, output_padding=1))]
                if norm_type == 'group':
                    up.append(nn.GroupNorm(min(16, max(1, out_w // 4)), out_w))
                up.append(nn.GELU())
                self.ups.append(nn.Sequential(*up))

            curr_width = out_w

        self.conv_final = sn(nn.Conv1d(curr_width, out_channels, 3, padding=1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, S, D) Complex latent trajectory.
        Returns:
            u: (B, S, C, X) Reconstructed 1D field sequence.
        """
        B, S, D = z.shape
        with torch.amp.autocast('cuda', enabled=False):
            z = z.to(torch.complex64)
            z_flat = z.reshape(B * S, D)

            x = torch.cat([z_flat.real, z_flat.imag], dim=1)
            x = self.fc(x)
            x = x.view(-1, self.width * 4, self.initial_len)

            for up_block in self.ups:
                x = up_block(x)

            out = self.conv_final(x)
            _, C, X = out.shape
            return out.view(B, S, C, X)

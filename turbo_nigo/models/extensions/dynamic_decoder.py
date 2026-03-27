"""
DynamicSpectralDecoder — Auto-computed upsampling for arbitrary target resolution.

Drop-in replacement for SpectralDecoder. Dynamically builds the correct number
of ConvTranspose2d layers based on target_res, so the same code handles
64, 128, 256, 512, etc.
"""
import math

import torch
import torch.nn as nn


class DynamicSpectralDecoder(nn.Module):
    """
    Decodes complex latent sequences back to physical fields via transpose convolutions.

    Computes num_upsamples = log2(target_res / 8) and stacks that many
    ConvTranspose2d layers, progressively halving channels until reaching
    the base width, then a final Conv2d for the output channels.

    Args:
        latent_dim: Dimension of the complex latent vector.
        out_channels: Number of output physical field channels.
        width: Base channel width (must match encoder).
        target_res: Target spatial resolution (must be a power of 2 and >= 8).
    """

    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        width: int = 32,
        target_res: int = 64,
    ):
        super().__init__()
        self.width = width
        self.target_res = target_res

        # Latent → flat spatial features at 8×8
        self.fc = nn.Linear(latent_dim * 2, width * 4 * 8 * 8)

        # Dynamic upsampling stack
        num_upsamples = int(math.log2(target_res // 8))
        assert 8 * (2 ** num_upsamples) == target_res, (
            f"target_res={target_res} must be 8 * 2^n"
        )

        layers = []
        current_width = width * 4
        for _ in range(num_upsamples):
            next_width = max(width, current_width // 2)
            layers.append(
                nn.ConvTranspose2d(
                    current_width, next_width, 3,
                    stride=2, padding=1, output_padding=1,
                )
            )
            layers.append(nn.GELU())
            current_width = next_width

        layers.append(nn.Conv2d(current_width, out_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Complex tensor of shape (B, S, D) — latent time sequence.
        Returns:
            Physical fields of shape (B, S, C, target_res, target_res).
        """
        B, S, D = z.shape
        z_flat = z.reshape(B * S, D)
        x = torch.cat([z_flat.real, z_flat.imag], dim=1)
        x = self.fc(x)
        x = x.view(-1, self.width * 4, 8, 8)
        out = self.net(x)
        _, C, H, W = out.shape
        return out.view(B, S, C, H, W)

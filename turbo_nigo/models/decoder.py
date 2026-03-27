import torch
import torch.nn as nn

class SpectralDecoder(nn.Module):
    """
    Decodes the complex latent space back into the physical field domain.
    Supports arbitrary target spatial resolutions via a configurable initial_size.
    """
    def __init__(self, latent_dim: int, out_channels: int, width: int = 32, initial_size: int = 8):
        super().__init__()
        self.width = width
        self.initial_size = initial_size
        
        self.fc = nn.Linear(latent_dim * 2, width*4 * initial_size * initial_size)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(width*4, width*4, 3, stride=2, padding=1, output_padding=1), nn.GELU(),
            nn.ConvTranspose2d(width*4, width*2, 3, stride=2, padding=1, output_padding=1), nn.GELU(),
            nn.ConvTranspose2d(width*2, width, 3, stride=2, padding=1, output_padding=1), nn.GELU(),
            nn.Conv2d(width, out_channels, 3, padding=1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, seq_len, D) Complex tensor
        Returns: (B, seq_len, C, H, W)
        """
        B, S, D = z.shape
        z_flat = z.reshape(B*S, D)
        
        x = torch.cat([z_flat.real, z_flat.imag], dim=1)
        x = self.fc(x)
        x = x.view(-1, self.width*4, self.initial_size, self.initial_size)
        
        out = self.net(x)
        _, C, H, W = out.shape
        
        return out.view(B, S, C, H, W)

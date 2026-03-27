import torch
import torch.nn as nn

class SpectralEncoder(nn.Module):
    """
    Encodes the input field and conditions into a complex latent space.
    Supports arbitrary spatial resolutions via lazy flat_dim computation.
    """
    def __init__(self, in_channels: int, latent_dim: int, width: int = 32, 
                 cond_channels: int = 4, spatial_size: int = 64):
        super().__init__()
        self.cond_channels = cond_channels
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels + cond_channels, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width*2, 3, padding=1, stride=2), nn.GELU(),
            nn.Conv2d(width*2, width*4, 3, padding=1, stride=2), nn.GELU(),
            nn.Conv2d(width*4, width*4, 3, padding=1, stride=2), nn.GELU(),
        )
        
        # Dynamically compute flat_dim from a probe tensor
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels + cond_channels, spatial_size, spatial_size)
            dummy_out = self.conv_net(dummy)
            self.flat_dim = dummy_out.numel()  # total elements per batch item
        
        self.flatten = nn.Flatten()
        self.fc_real = nn.Linear(self.flat_dim, latent_dim)
        self.fc_imag = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        cond: (B, cond_channels)
        """
        B, C, H, W = x.shape
        cond_map = cond.view(B, -1, 1, 1).expand(B, self.cond_channels, H, W)
        xin = torch.cat([x, cond_map.to(x.device)], dim=1)
        feat = self.conv_net(xin)
        feat = self.flatten(feat)
        return torch.complex(self.fc_real(feat), self.fc_imag(feat))

import torch
import torch.nn as nn

class TemporalRefiner(nn.Module):
    """
    Refines the evolved complex latent trajectory to correct for 
    high-frequency numerical integration errors.
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: Dilation 1, Padding 1 -> Preserves Length
            nn.Conv1d(latent_dim * 2, latent_dim * 4, kernel_size=3, padding=1),
            nn.GELU(),
            
            # Layer 2: Dilation 2, Padding 2 -> Preserves Length
            nn.Conv1d(latent_dim * 4, latent_dim * 4, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            
            # Layer 3: Dilation 4, Padding 4 -> Preserves Length
            nn.Conv1d(latent_dim * 4, latent_dim * 4, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
            
            # Layer 4: Dilation 1, Padding 1 -> Preserves Length
            nn.Conv1d(latent_dim * 4, latent_dim * 2, kernel_size=3, padding=1)
        )
        
        # Initialize with zeros to act as identity mapping initially
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        """
        z_seq: (B, T, D) Complex tensor
        """
        B, T, D = z_seq.shape
        
        with torch.amp.autocast('cuda', enabled=False):
            # Concat real/imag -> (B, T, 2D)
            x = torch.cat([z_seq.real, z_seq.imag], dim=2)
            # Permute for Conv1d -> (B, 2D, T)
            x = x.permute(0, 2, 1)
            
            correction = self.net(x)
            
            # Permute back -> (B, T, 2D)
            correction = correction.permute(0, 2, 1)
            
            corr_real = correction[:, :, :D]
            corr_imag = correction[:, :, D:]
            
            return z_seq + torch.complex(corr_real, corr_imag)

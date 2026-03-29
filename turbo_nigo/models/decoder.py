import torch
import torch.nn as nn

class SpectralDecoder(nn.Module):
    """
    Decodes the complex latent space back into the physical field domain.
    Supports arbitrary target spatial resolutions via a configurable initial_size.
    """
    def __init__(self, latent_dim: int, out_channels: int, width: int = 32, 
                 initial_size: int = 8, num_layers: int = 3, 
                 use_residual: bool = False, norm_type: str = None):
        super().__init__()
        self.width = width
        self.initial_size = initial_size
        self.use_residual = use_residual
        
        # In base NIGO, bottleneck is 4x width
        self.fc = nn.Linear(latent_dim * 2, width*4 * initial_size * initial_size)
        
        self.ups = nn.ModuleList()
        curr_width = width * 4
        
        for i in range(num_layers):
            # Mirror the encoder's width doubling
            in_w = curr_width
            # Decrease width as we go back to physical space
            out_w = curr_width // 2 if (num_layers - i) <= 2 else curr_width
            # Ensure it doesn't go below base width
            out_w = max(width, out_w)
            
            up = [nn.ConvTranspose2d(in_w, out_w, 3, stride=2, padding=1, output_padding=1)]
            if norm_type == 'group':
                up.append(nn.GroupNorm(min(16, out_w // 4), out_w))
            up.append(nn.GELU())
            
            self.ups.append(nn.Sequential(*up))
            curr_width = out_w
            
        self.conv_final = nn.Conv2d(curr_width, out_channels, 3, padding=1)

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
        
        for up_block in self.ups:
            x = up_block(x)
            
        out = self.conv_final(x)
        _, C, H, W = out.shape
        return out.view(B, S, C, H, W)

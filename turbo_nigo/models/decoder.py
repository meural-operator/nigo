import torch
import torch.nn as nn

class ResidualUpBlock(nn.Module):
    def __init__(self, in_c, out_c, norm_type=None, use_spectral_norm=False):
        super().__init__()
        
        def sn(layer):
            return torch.nn.utils.spectral_norm(layer) if use_spectral_norm else layer
            
        self.up = sn(nn.ConvTranspose2d(in_c, out_c, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.norm1 = nn.GroupNorm(min(16, max(1, out_c // 4)), out_c) if norm_type == 'group' else nn.Identity()
        self.act1 = nn.GELU()
        
        self.conv2 = sn(nn.Conv2d(out_c, out_c, kernel_size=3, padding=1))
        self.norm2 = nn.GroupNorm(min(16, max(1, out_c // 4)), out_c) if norm_type == 'group' else nn.Identity()
        self.act2 = nn.GELU()
        
        print(f"  [Decoder] Provisioning learned skip-connection for upsampling (in: {in_c}, out: {out_c}, stride: 2)")
        skip_layers = [sn(nn.ConvTranspose2d(in_c, out_c, kernel_size=3, stride=2, padding=1, output_padding=1))]

        if norm_type == 'group':
            skip_layers.append(nn.GroupNorm(min(16, max(1, out_c // 4)), out_c))
        self.skip = nn.Sequential(*skip_layers)
            
    def forward(self, x):
        res = self.skip(x)
        out = self.act1(self.norm1(self.up(x)))
        out = self.norm2(self.conv2(out))
        return self.act2(out + res)

class SpectralDecoder(nn.Module):
    """
    Decodes the complex latent space back into the physical field domain.
    Supports arbitrary target spatial resolutions via a configurable initial_size.
    """
    def __init__(self, latent_dim: int, out_channels: int, width: int = 32, 
                 initial_size: int = 8, num_layers: int = 3, 
                 use_residual: bool = False, norm_type: str = None,
                 use_spectral_norm: bool = False):
        super().__init__()
        self.width = width
        self.initial_size = initial_size
        
        def sn(layer):
            return torch.nn.utils.spectral_norm(layer) if use_spectral_norm else layer
        
        # In base NIGO, bottleneck is 4x width
        self.fc = sn(nn.Linear(latent_dim * 2, width*4 * initial_size * initial_size))
        
        self.ups = nn.ModuleList()
        curr_width = width * 4
        
        for i in range(num_layers):
            in_w = curr_width
            out_w = curr_width // 2 if (num_layers - i) <= 2 else curr_width
            out_w = max(width, out_w)
            
            if use_residual:
                self.ups.append(ResidualUpBlock(in_w, out_w, norm_type=norm_type, use_spectral_norm=use_spectral_norm))
            else:
                up = [sn(nn.ConvTranspose2d(in_w, out_w, 3, stride=2, padding=1, output_padding=1))]
                if norm_type == 'group':
                    up.append(nn.GroupNorm(min(16, max(1, out_w // 4)), out_w))
                up.append(nn.GELU())
                self.ups.append(nn.Sequential(*up))
                
            curr_width = out_w
            
        self.conv_final = sn(nn.Conv2d(curr_width, out_channels, 3, padding=1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, seq_len, D) Complex tensor
        Returns: (B, seq_len, C, H, W)
        """
        B, S, D = z.shape
        with torch.amp.autocast('cuda', enabled=False):
            z = z.to(torch.complex64)
            z_flat = z.reshape(B*S, D)
            
            x = torch.cat([z_flat.real, z_flat.imag], dim=1)
            x = self.fc(x)
            x = x.view(-1, self.width*4, self.initial_size, self.initial_size)
            
            for up_block in self.ups:
                x = up_block(x)
                
            out = self.conv_final(x)
            _, C, H, W = out.shape
            return out.view(B, S, C, H, W)

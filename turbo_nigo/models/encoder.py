import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, norm_type=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.norm1 = nn.GroupNorm(min(16, max(1, out_c // 4)), out_c) if norm_type == 'group' else nn.Identity()
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(16, max(1, out_c // 4)), out_c) if norm_type == 'group' else nn.Identity()
        self.act2 = nn.GELU()
        
        self.skip = nn.Identity()
        if stride != 1 or in_c != out_c:
            print(f"  [Encoder] Provisioning 1x1 skip-connection for dimension match (in: {in_c}, out: {out_c}, stride: {stride})")
            skip_layers = [nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride)]
            if norm_type == 'group':
                skip_layers.append(nn.GroupNorm(min(16, max(1, out_c // 4)), out_c))
            self.skip = nn.Sequential(*skip_layers)
            
    def forward(self, x):
        res = self.skip(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act2(out + res)

class SpectralEncoder(nn.Module):
    """
    Encodes the input field and conditions into a complex latent space.
    Supports arbitrary spatial resolutions via lazy flat_dim computation.
    """
    def __init__(self, in_channels: int, latent_dim: int, width: int = 32, 
                 cond_channels: int = 4, spatial_size: int = 64,
                 num_layers: int = 3, use_residual: bool = False, 
                 norm_type: str = None):
        super().__init__()
        self.cond_channels = cond_channels
        
        curr_width = width
        
        # 1. Stem
        stem_layers = [nn.Conv2d(in_channels + cond_channels, width, 3, padding=1)]
        if norm_type == 'group':
            stem_layers.append(nn.GroupNorm(8, width))
        stem_layers.append(nn.GELU())
        self.stem = nn.Sequential(*stem_layers)
        
        # 2. Downsampling Blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            in_w = curr_width
            out_w = curr_width * 2 if i < num_layers - 1 else curr_width
            
            if use_residual:
                self.blocks.append(ResidualBlock(in_w, out_w, stride=2, norm_type=norm_type))
            else:
                block = [nn.Conv2d(in_w, out_w, 3, padding=1, stride=2)]
                if norm_type == 'group':
                    block.append(nn.GroupNorm(min(16, max(1, out_w // 4)), out_w))
                block.append(nn.GELU())
                self.blocks.append(nn.Sequential(*block))
                
            curr_width = out_w
            
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels + cond_channels, spatial_size, spatial_size)
            feat = self.stem(dummy)
            for block in self.blocks:
                feat = block(feat)
            self.flat_dim = feat.numel()
        
        self.flatten = nn.Flatten()
        self.fc_real = nn.Linear(self.flat_dim, latent_dim)
        self.fc_imag = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        cond_map = cond.view(B, -1, 1, 1).expand(B, self.cond_channels, H, W).to(x.device)
        xin = torch.cat([x, cond_map], dim=1)
        
        feat = self.stem(xin)
        for block in self.blocks:
            feat = block(feat)
        
        feat = self.flatten(feat)
        with torch.amp.autocast('cuda', enabled=False):
            feat = feat.float()
            return torch.complex(self.fc_real(feat), self.fc_imag(feat))

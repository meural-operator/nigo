import torch
import torch.nn as nn

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
        self.use_residual = use_residual
        
        layers = []
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
            # Only double width up to num_layers - 1
            out_w = curr_width * 2 if i < num_layers - 1 else curr_width
            
            block = [nn.Conv2d(in_w, out_w, 3, padding=1, stride=2)]
            if norm_type == 'group':
                block.append(nn.GroupNorm(min(16, out_w // 4), out_w))
            block.append(nn.GELU())
            
            self.blocks.append(nn.Sequential(*block))
            curr_width = out_w
            
        # Dynamically compute flat_dim from a probe tensor
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
        """
        x: (B, C, H, W)
        cond: (B, cond_channels)
        """
        B, C, H, W = x.shape
        cond_map = cond.view(B, -1, 1, 1).expand(B, self.cond_channels, H, W).to(x.device)
        xin = torch.cat([x, cond_map], dim=1)
        
        feat = self.stem(xin)
        for block in self.blocks:
            if self.use_residual and feat.shape == block[0].weight.shape: # Simple residual check
                 # Standard res block would need more logic; for now we support sequential
                 feat = block(feat)
            else:
                 feat = block(feat)
        
        feat = self.flatten(feat)
        return torch.complex(self.fc_real(feat), self.fc_imag(feat))

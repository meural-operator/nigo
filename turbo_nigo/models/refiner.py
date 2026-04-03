import torch
import torch.nn as nn

class TemporalRefiner(nn.Module):
    """
    Refines the evolved complex latent trajectory to correct for 
    high-frequency numerical integration errors.
    """
    def __init__(self, latent_dim: int, use_adaptive_refiner: bool = False, use_spectral_norm: bool = False):
        super().__init__()
        self.use_adaptive_refiner = use_adaptive_refiner
        
        # Helper lambda to conditionally attach Spectral Norm strict Lipschitz bounds
        def sn(layer):
            return torch.nn.utils.spectral_norm(layer) if use_spectral_norm else layer

        self.net = nn.Sequential(
            # Layer 1: Dilation 1, Padding 1 -> Preserves Length
            sn(nn.Conv1d(latent_dim * 2, latent_dim * 4, kernel_size=3, padding=1)),
            nn.GELU(),
            
            # Layer 2: Dilation 2, Padding 2 -> Preserves Length
            sn(nn.Conv1d(latent_dim * 4, latent_dim * 4, kernel_size=3, padding=2, dilation=2)),
            nn.GELU(),
            
            # Layer 3: Dilation 4, Padding 4 -> Preserves Length
            sn(nn.Conv1d(latent_dim * 4, latent_dim * 4, kernel_size=3, padding=4, dilation=4)),
            nn.GELU(),
            
            # Layer 4: Dilation 1, Padding 1 -> Preserves Length
            sn(nn.Conv1d(latent_dim * 4, latent_dim * 2, kernel_size=3, padding=1))
        )
        
        # Initialize near-zero for identity-like behavior (ReZero spirit).
        # NOTE: Exact zeros cause NaN when spectral_norm divides by sigma_max=0.
        _init_scale = 1e-6
        last_conv = self.net[-1]
        if hasattr(last_conv, 'weight_orig'):
            nn.init.normal_(last_conv.weight_orig, std=_init_scale)
        else:
            nn.init.normal_(last_conv.weight, std=_init_scale)
            
        if last_conv.bias is not None:
            nn.init.zeros_(last_conv.bias)
            
        if self.use_adaptive_refiner:
            # Adaptive bounding mapping to limit pathological numerical scaling natively
            self.epsilon_max = 0.1
            self.adaptive_gate = nn.Sequential(
                sn(nn.Linear(latent_dim * 2, latent_dim * 2)),
                nn.GELU(),
                sn(nn.Linear(latent_dim * 2, latent_dim * 2))
            )
            # Near-zero init for the gate's final layer too
            gate_last = self.adaptive_gate[-1]
            if hasattr(gate_last, 'weight_orig'):
                nn.init.normal_(gate_last.weight_orig, std=_init_scale)
            else:
                nn.init.normal_(gate_last.weight, std=_init_scale)
            nn.init.zeros_(gate_last.bias)

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        """
        z_seq: (B, T, D) Complex tensor
        """
        B, T, D = z_seq.shape
        
        with torch.amp.autocast('cuda', enabled=False):
            # Cast to full precision to prevent AMP half-precision complex decomposition issues
            z_seq = z_seq.to(torch.complex64)
            # Concat real/imag -> (B, T, 2D)
            x = torch.cat([z_seq.real, z_seq.imag], dim=2)
            # Permute for Conv1d -> (B, 2D, T)
            x = x.permute(0, 2, 1)
            
            correction = self.net(x)
            
            # Permute back -> (B, T, 2D)
            correction = correction.permute(0, 2, 1)
            
            # Conditionally apply strictly bounded continuous state-dependent gating
            if self.use_adaptive_refiner:
                # x_flat for dynamic feature lookup
                x_flat = x.permute(0, 2, 1) # B, T, 2D
                gate = torch.sigmoid(self.adaptive_gate(x_flat)) * self.epsilon_max
                correction = gate * correction
            
            corr_real = correction[:, :, :D]
            corr_imag = correction[:, :, D:]
            
            return z_seq + torch.complex(corr_real, corr_imag)

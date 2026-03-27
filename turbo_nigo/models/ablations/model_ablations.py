import torch
import torch.nn as nn
from typing import Tuple

from turbo_nigo.models.turbo_nigo import GlobalTurboNIGO
from turbo_nigo.models.ablations.generator_ablations import NoSkewGenerator, NoDissipativeGenerator, DenseGenerator

class Ablation1_NoSkewTurboNIGO(GlobalTurboNIGO):
    def __init__(self, latent_dim: int = 64, num_bases: int = 8, cond_dim: int = 4, 
                 width: int = 32, spatial_size: int = 64, in_channels: int = 2):
        super().__init__(latent_dim=latent_dim, num_bases=num_bases, cond_dim=cond_dim, 
                         width=width, spatial_size=spatial_size, in_channels=in_channels)
        self.generator = NoSkewGenerator(latent_dim=latent_dim, num_bases=num_bases)

class Ablation2_NoDissipativeTurboNIGO(GlobalTurboNIGO):
    def __init__(self, latent_dim: int = 64, num_bases: int = 8, cond_dim: int = 4, 
                 width: int = 32, spatial_size: int = 64, in_channels: int = 2):
        super().__init__(latent_dim=latent_dim, num_bases=num_bases, cond_dim=cond_dim, 
                         width=width, spatial_size=spatial_size, in_channels=in_channels)
        self.generator = NoDissipativeGenerator(latent_dim=latent_dim, num_bases=num_bases)

class Ablation3_DenseGeneratorTurboNIGO(GlobalTurboNIGO):
    def __init__(self, latent_dim: int = 64, num_bases: int = 8, cond_dim: int = 4, 
                 width: int = 32, spatial_size: int = 64, in_channels: int = 2):
        super().__init__(latent_dim=latent_dim, num_bases=num_bases, cond_dim=cond_dim, 
                         width=width, spatial_size=spatial_size, in_channels=in_channels)
        self.generator = DenseGenerator(latent_dim=latent_dim, num_bases=num_bases)

class Ablation4_NoRefinerTurboNIGO(GlobalTurboNIGO):
    def forward(self, u0: torch.Tensor, time_steps: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z0 = self.encoder(u0, cond)
        k_coeffs, r_coeffs = self.cond_net(z0, cond)
        z_base = self.generator(z0, time_steps, k_coeffs, r_coeffs)
        
        # ABLATION 4: SKIP the Temporal Refiner completely
        z_refined = z_base
        
        u_pred = self.decoder(z_refined)
        return u_pred, z_base, k_coeffs, r_coeffs


# --- ABLATION 5: Scaled Generator (Reintroducing Alpha/Beta) ---
import torch.nn.functional as F

class ScaledPhysicsNet(nn.Module):
    def __init__(self, latent_dim: int, num_bases: int = 8, hidden: int = 128, cond_dim: int = 4):
        super().__init__()
        self.in_dim = latent_dim * 2 + cond_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden//2), nn.GELU()
        )
        self.out_alpha = nn.Linear(hidden//2, 1)
        self.out_beta = nn.Linear(hidden//2, 1)
        self.out_coeffs = nn.Linear(hidden//2, num_bases * 2)
        nn.init.constant_(self.out_alpha.bias, 0.0)
        nn.init.constant_(self.out_beta.bias, 0.5)

    def forward(self, z0: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([z0.real, z0.imag, cond.to(z0.real.device)], dim=1)
        h = self.net(x)
        alpha = F.softplus(self.out_alpha(h)) + 1e-6
        beta = F.softplus(self.out_beta(h)) + 1e-6
        coeffs = self.out_coeffs(h)
        k = torch.tanh(coeffs[:, :coeffs.shape[1]//2])
        r = torch.tanh(coeffs[:, coeffs.shape[1]//2:])
        return alpha.view(-1,1,1), beta.view(-1,1,1), k, r

class ScaledGenerator(nn.Module):
    def __init__(self, latent_dim: int, num_bases: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_bases = num_bases
        self.K_bases = nn.Parameter(torch.randn(num_bases, latent_dim, latent_dim) * 0.01)
        self.R_bases = nn.Parameter(torch.randn(num_bases, latent_dim, latent_dim) * 0.01)

    def forward(self, z0: torch.Tensor, time_steps: torch.Tensor, 
                alpha: torch.Tensor, beta: torch.Tensor, 
                k_coeffs: torch.Tensor, r_coeffs: torch.Tensor) -> torch.Tensor:
        K_b = self.K_bases.unsqueeze(0)
        R_b = self.R_bases.unsqueeze(0)
        kc = k_coeffs.view(-1, self.num_bases, 1, 1)
        rc = r_coeffs.view(-1, self.num_bases, 1, 1)
        K_sum = (kc * K_b).sum(dim=1)
        R_sum = (rc * R_b).sum(dim=1)
        
        A = alpha * (K_sum - K_sum.transpose(-1,-2)) + beta * (- (R_sum.transpose(-1,-2) @ R_sum))
        
        S_steps = time_steps.shape[0]
        A_t = A.unsqueeze(1) * time_steps.view(1, S_steps, 1, 1).to(z0.real.device)
        with torch.cuda.amp.autocast(enabled=False):
            A_t_flat = A_t.reshape(-1, self.latent_dim, self.latent_dim).float()
            props = torch.linalg.matrix_exp(A_t_flat)
            props = props.view(-1, S_steps, self.latent_dim, self.latent_dim).to(torch.complex64)
        z_c = z0.to(torch.complex64)
        z_evolved = torch.einsum('bi, bsoi -> bso', z_c, props)
        return z_evolved

class Ablation5_ScaledTurboNIGO(GlobalTurboNIGO):
    def __init__(self, latent_dim: int = 64, num_bases: int = 8, cond_dim: int = 4, 
                 width: int = 32, spatial_size: int = 64, in_channels: int = 2):
        super().__init__(latent_dim=latent_dim, num_bases=num_bases, cond_dim=cond_dim, 
                         width=width, spatial_size=spatial_size, in_channels=in_channels)
        self.cond_net = ScaledPhysicsNet(latent_dim=latent_dim, num_bases=num_bases, cond_dim=cond_dim)
        self.generator = ScaledGenerator(latent_dim=latent_dim, num_bases=num_bases)

    def forward(self, u0: torch.Tensor, time_steps: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z0 = self.encoder(u0, cond)
        alpha, beta, k_coeffs, r_coeffs = self.cond_net(z0, cond)
        z_base = self.generator(z0, time_steps, alpha, beta, k_coeffs, r_coeffs)
        z_refined = self.refiner(z_base)
        u_pred = self.decoder(z_refined)
        
        # We return alpha and beta at the end so it doesn't break unpack logic for regular trainer if it ignores extras.
        # But wait! The trainer expects EXACTLY 4 values unpacking now.
        # So Ablation 5 can't return 6 values if the trainer blindly does `u_pred, _, kc, rc = self.model(...)`.
        # To be safe, Ablation5_ScaledTurboNIGO should return the exact same 4 variables, hiding alpha/beta internally.
        return u_pred, z_base, k_coeffs, r_coeffs

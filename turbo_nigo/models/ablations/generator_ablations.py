import torch
import torch.nn as nn
from turbo_nigo.models.generator import HyperTurbulentGenerator

class NoSkewGenerator(HyperTurbulentGenerator):
    """
    Ablation 1: Removes the Skew-Symmetric (conservative) term.
    A = beta * N
    """
    def forward(self, z0: torch.Tensor, time_steps: torch.Tensor, 
                k_coeffs: torch.Tensor, r_coeffs: torch.Tensor) -> torch.Tensor:
        
        R_b = self.R_bases.unsqueeze(0)
        rc = r_coeffs.view(-1, self.num_bases, 1, 1)
        R_sum = (rc * R_b).sum(dim=1)
        
        # Construct ONLY Negative-definite part
        A = (- (R_sum.transpose(-1,-2) @ R_sum))
        
        S_steps = time_steps.shape[0]
        A_t = A.unsqueeze(1) * time_steps.view(1, S_steps, 1, 1).to(z0.real.device)
        
        with torch.cuda.amp.autocast(enabled=False):
            A_t_flat = A_t.reshape(-1, self.latent_dim, self.latent_dim).float()
            props = torch.linalg.matrix_exp(A_t_flat)
            props = props.view(-1, S_steps, self.latent_dim, self.latent_dim).to(torch.complex64)
            
        z_c = z0.to(torch.complex64)
        z_evolved = torch.einsum('bi, bsoi -> bso', z_c, props)
        
        return z_evolved


class NoDissipativeGenerator(HyperTurbulentGenerator):
    """
    Ablation 2: Removes the Dissipative (negative-definite) term.
    A = alpha * S
    """
    def forward(self, z0: torch.Tensor, time_steps: torch.Tensor, 
                k_coeffs: torch.Tensor, r_coeffs: torch.Tensor) -> torch.Tensor:
        
        K_b = self.K_bases.unsqueeze(0)
        kc = k_coeffs.view(-1, self.num_bases, 1, 1)
        K_sum = (kc * K_b).sum(dim=1)
        
        # Construct ONLY Skew-symmetric part
        A = (K_sum - K_sum.transpose(-1,-2))
        
        S_steps = time_steps.shape[0]
        A_t = A.unsqueeze(1) * time_steps.view(1, S_steps, 1, 1).to(z0.real.device)
        
        with torch.cuda.amp.autocast(enabled=False):
            A_t_flat = A_t.reshape(-1, self.latent_dim, self.latent_dim).float()
            props = torch.linalg.matrix_exp(A_t_flat)
            props = props.view(-1, S_steps, self.latent_dim, self.latent_dim).to(torch.complex64)
            
        z_c = z0.to(torch.complex64)
        z_evolved = torch.einsum('bi, bsoi -> bso', z_c, props)
        
        return z_evolved


class DenseGenerator(nn.Module):
    """
    Ablation 3: Replaces the factorized basis generator with a dense, 
    unconstrained learned matrix generator.
    Instead of combining bases, it just uses a direct projection.
    NOTE: We must adapt the input since there are no k_coeffs/r_coeffs.
    For parity in parameter interface with the main model, it accepts them but ignores them.
    It takes alpha/beta as dummy variables or we can build a dense projection layer.
    Wait, in the full model, the `cond_net` outputs k_coeffs and r_coeffs. 
    A true 'dense' ablation means the physics net just outputs the dense matrix directly,
    but that requires changing PhysicsInferenceNet too.
    
    Instead, Let's parameterize the dense matrix using exactly what the cond_net outputs:
    We have `num_bases * 2` total coefficients. We can just use an MLP to map those coefficients
    (or the initial condition vector if available) into a full `latent_dim x latent_dim` matrix.
    Since `cond_net` outputs length (num_bases*2) + 2 parameters, we can project that into DxD.
    """
    def __init__(self, latent_dim: int, num_bases: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        # Project the num_bases*2 coefficients into a DxD matrix block
        self.raw_inputs_len = (num_bases * 2)
        self.dense_proj = nn.Sequential(
            nn.Linear(self.raw_inputs_len, latent_dim * latent_dim),
            # Initialized small to stay near identity
        )
        # Small initialization
        nn.init.normal_(self.dense_proj[0].weight, std=0.01)
        nn.init.zeros_(self.dense_proj[0].bias)

    def forward(self, z0: torch.Tensor, time_steps: torch.Tensor, 
                k_coeffs: torch.Tensor, r_coeffs: torch.Tensor) -> torch.Tensor:
        
        # Flatten the parameters into a single vector
        B = z0.shape[0]
        params = torch.cat([k_coeffs, r_coeffs], dim=1)
        
        # Project dynamically to a dense matrix
        A = self.dense_proj(params).view(B, self.latent_dim, self.latent_dim)
        
        S_steps = time_steps.shape[0]
        A_t = A.unsqueeze(1) * time_steps.view(1, S_steps, 1, 1).to(z0.real.device)
        
        with torch.cuda.amp.autocast(enabled=False):
            A_t_flat = A_t.reshape(-1, self.latent_dim, self.latent_dim).float()
            props = torch.linalg.matrix_exp(A_t_flat)
            props = props.view(-1, S_steps, self.latent_dim, self.latent_dim).to(torch.complex64)
            
        z_c = z0.to(torch.complex64)
        z_evolved = torch.einsum('bi, bsoi -> bso', z_c, props)
        
        return z_evolved

import torch
import torch.nn as nn

class HyperTurbulentGenerator(nn.Module):
    """
    Generates the trajectory in the latent space by exponentiating the 
    infinitesimal generator A(t) = alpha * S + beta * N.
    """
    def __init__(self, latent_dim: int, num_bases: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_bases = num_bases
        
        # Learnable bases for Skew-symmetric (S) and Negative-definite (N) components
        self.K_bases = nn.Parameter(torch.randn(num_bases, latent_dim, latent_dim) * 0.01)
        self.R_bases = nn.Parameter(torch.randn(num_bases, latent_dim, latent_dim) * 0.01)

    def forward(self, z0: torch.Tensor, time_steps: torch.Tensor, 
                k_coeffs: torch.Tensor, r_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Evolves z0 over time_steps.
        Args:
            z0: (B, D) complex initial state
            time_steps: (S,) time steps to evaluate
            k_coeffs, r_coeffs: (B, num_bases) coefficients
            
        Returns:
            z_evolved: (B, S, D) complex evolved states
        """
        # Assemble A
        K_b = self.K_bases.unsqueeze(0)
        R_b = self.R_bases.unsqueeze(0)
        
        kc = k_coeffs.view(-1, self.num_bases, 1, 1)
        rc = r_coeffs.view(-1, self.num_bases, 1, 1)
        
        K_sum = (kc * K_b).sum(dim=1)
        R_sum = (rc * R_b).sum(dim=1)
        
        # Construct Skew-symmetric and Negative-definite parts
        # A = Skew + NegDef
        A = (K_sum - K_sum.transpose(-1,-2)) + (- (R_sum.transpose(-1,-2) @ R_sum))
        
        # Vectorized Matrix Exp for all time steps at once
        S = time_steps.shape[0]
        # A_t shape: (B, S, D, D) -> Scaled by each time step t
        A_t = A.unsqueeze(1) * time_steps.view(1, S, 1, 1).to(z0.real.device)
        
        # Matrix exponential is complex mathematically, but inputs here are real
        # Requires float32 precision for stability, AMP autocast should be disabled
        with torch.cuda.amp.autocast(enabled=False):
            A_t_flat = A_t.reshape(-1, self.latent_dim, self.latent_dim).float()
            props = torch.linalg.matrix_exp(A_t_flat)
            props = props.view(-1, S, self.latent_dim, self.latent_dim).to(torch.complex64)
            
        z_c = z0.to(torch.complex64)
        z_evolved = torch.einsum('bi, bsoi -> bso', z_c, props)
        
        return z_evolved

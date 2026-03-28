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
                k_coeffs: torch.Tensor, r_coeffs: torch.Tensor,
                alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Evolves z0 over time_steps.
        Args:
            z0: (B, D) complex initial state
            time_steps: (S,) time steps to evaluate
            k_coeffs, r_coeffs: (B, num_bases) coefficients
            alpha, beta: (B, 1) global scaling parameters
            
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
        # A = alpha * (K - K^T) - beta * (R^T R)
        alpha_view = alpha.view(-1, 1, 1)
        beta_view = beta.view(-1, 1, 1)
        
        A = alpha_view * (K_sum - K_sum.transpose(-1,-2)) + beta_view * (- (R_sum.transpose(-1,-2) @ R_sum))
        
        # Optimize: For uniform time steps, we only need ONE matrix exponential per batch item.
        # dt is the step size (e.g., 1.0 / S).
        S = time_steps.shape[0]
        dt = time_steps[1] - time_steps[0] if S > 1 else torch.tensor(1.0)
        
        with torch.cuda.amp.autocast(enabled=False):
            # 1. Compute base transition matrix M = exp(A * dt) in float64 for stability
            A_dt_f64 = (A * dt).to(torch.float64)
            M_f64 = torch.linalg.matrix_exp(A_dt_f64)
            M = M_f64.to(torch.complex64)
            
        z_curr = z0.to(torch.complex64)
        z_list = []
        
        # 2. Recursive evolution: z_{t+1} = M @ z_t
        # This is 20x faster than 20 matrix exponentials
        for t in range(S):
            z_curr = torch.bmm(M, z_curr.unsqueeze(-1)).squeeze(-1)
            z_list.append(z_curr)
            
        z_evolved = torch.stack(z_list, dim=1) # (B, S, D)
        
        return z_evolved

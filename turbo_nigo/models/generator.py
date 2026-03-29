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
        
        # Construct A = alpha * (K - K^T) - beta * (R^T R)
        alpha_view = alpha.view(-1, 1, 1)
        beta_view = beta.view(-1, 1, 1)
        
        A = alpha_view * (K_sum - K_sum.transpose(-1,-2)) + beta_view * (- (R_sum.transpose(-1,-2) @ R_sum))
        
        S = time_steps.shape[0]
        device = z0.real.device

        # Performance Optimization:
        # Check if time steps are uniform (common in KS/NS training).
        # If uniform, e^{A * k*dt} = (e^{A * dt})^k. 
        # This reduces B*S matrix exponentials to just B matrix exponentials.
        is_uniform = True
        if S > 1:
            diffs = time_steps[1:] - time_steps[:-1]
            dt = time_steps[0]
            if not torch.allclose(diffs, dt, atol=1e-6):
                is_uniform = False

        if is_uniform:
            dt = time_steps[0]
            A_scaled = A * dt
            with torch.cuda.amp.autocast(enabled=False):
                A_f64 = A_scaled.double()
                # Stability shift
                eps = 1e-6 * torch.eye(self.latent_dim, device=device, dtype=torch.float64)
                M = torch.linalg.matrix_exp(A_f64 + eps.unsqueeze(0)) # (B, D, D)
                
                # Rollout via matrix multiplications: O(S) matmul vs O(S) matrix_exp
                M_c64 = M.to(torch.complex64)
                props_list = [M_c64]
                for _ in range(1, S):
                    props_list.append(props_list[-1] @ M_c64)
                
                # Reshape to (B, S, D, D)
                props = torch.stack(props_list, dim=1)
        else:
            # Fallback for non-uniform time steps
            A_t = A.unsqueeze(1) * time_steps.view(1, S, 1, 1).to(device)
            with torch.cuda.amp.autocast(enabled=False):
                A_t_f64 = A_t.reshape(-1, self.latent_dim, self.latent_dim).double()
                eps = 1e-6 * torch.eye(self.latent_dim, device=device, dtype=torch.float64)
                props_f64 = torch.linalg.matrix_exp(A_t_f64 + eps.unsqueeze(0))
                props = props_f64.view(-1, S, self.latent_dim, self.latent_dim).to(torch.complex64)
            
        z_c = z0.to(torch.complex64)
        # z_evolved: (B, S, D)
        z_evolved = torch.einsum('bi, bsoi -> bso', z_c, props)
        
        return z_evolved

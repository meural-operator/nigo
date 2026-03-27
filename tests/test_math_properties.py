"""
Mathematical Property Verification Tests for TurboNIGO.

Verifies the core theoretical claims from the paper:
- Section 5: A = α(K - K^T) - β R^T R
- Section 5.1: Lyapunov stability: dV/dt = z^T(A + A^T)z = -2β||Rz||^2 ≤ 0
- Appendix A.2: Boundedness: ||exp(tA)||_2 ≤ 1 for all t > 0
"""
import pytest
import torch
import numpy as np

from turbo_nigo.models import GlobalTurboNIGO
from turbo_nigo.models.generator import HyperTurbulentGenerator
from turbo_nigo.models.physics_net import PhysicsInferenceNet


class TestGeneratorStructure:
    """Verify that A = α(K-K^T) - βR^TR satisfies the paper's structural guarantees."""

    def _build_A(self, gen, alpha, beta, k_coeffs, r_coeffs):
        """Reconstruct the generator matrix A from components."""
        K_b = gen.K_bases.unsqueeze(0)
        R_b = gen.R_bases.unsqueeze(0)
        kc = k_coeffs.view(-1, gen.num_bases, 1, 1)
        rc = r_coeffs.view(-1, gen.num_bases, 1, 1)
        K_sum = (kc * K_b).sum(dim=1)
        R_sum = (rc * R_b).sum(dim=1)
        A = alpha * (K_sum - K_sum.transpose(-1, -2)) + beta * (-(R_sum.transpose(-1, -2) @ R_sum))
        return A, R_sum

    def test_skew_symmetric_component(self):
        """Paper Eq: S = K - K^T must be skew-symmetric (S = -S^T)."""
        gen = HyperTurbulentGenerator(latent_dim=8, num_bases=4)
        k_coeffs = torch.randn(1, 4)

        K_b = gen.K_bases.unsqueeze(0)
        kc = k_coeffs.view(-1, gen.num_bases, 1, 1)
        K_sum = (kc * K_b).sum(dim=1)
        S = K_sum - K_sum.transpose(-1, -2)

        # S + S^T should be zero
        assert torch.allclose(S + S.transpose(-1, -2), torch.zeros_like(S), atol=1e-6), \
            "Skew-symmetric component K - K^T is not skew-symmetric!"

    def test_negative_semidefinite_component(self):
        """Paper Eq: N = -R^T R must be negative semidefinite (all eigenvalues ≤ 0)."""
        gen = HyperTurbulentGenerator(latent_dim=8, num_bases=4)
        r_coeffs = torch.randn(1, 4)

        R_b = gen.R_bases.unsqueeze(0)
        rc = r_coeffs.view(-1, gen.num_bases, 1, 1)
        R_sum = (rc * R_b).sum(dim=1)
        N = -(R_sum.transpose(-1, -2) @ R_sum)

        # N must be negative semidefinite: all eigenvalues ≤ 0
        eigvals = torch.linalg.eigvalsh(N[0])
        assert torch.all(eigvals <= 1e-6), \
            f"-R^T R has positive eigenvalue: {eigvals.max().item()}"

    def test_lyapunov_derivative_nonpositive(self):
        """
        Paper Section 5.1: V(z) = ||z||^2, dV/dt = z^T(A + A^T)z = -2β||Rz||^2 ≤ 0.
        We verify that A + A^T is negative semidefinite.
        """
        gen = HyperTurbulentGenerator(latent_dim=8, num_bases=4)
        net = PhysicsInferenceNet(latent_dim=8, num_bases=4, cond_dim=2)

        z0 = torch.randn(1, 8, dtype=torch.complex64)
        cond = torch.randn(1, 2)
        k_c, r_c, alpha, beta = net(z0, cond)

        A, R_sum = self._build_A(gen, alpha, beta, k_c, r_c)

        # A + A^T should be negative semidefinite
        symmetric_part = A + A.transpose(-1, -2)
        eigvals = torch.linalg.eigvalsh(symmetric_part[0].detach())

        assert torch.all(eigvals <= 1e-5), \
            f"Lyapunov violated! A + A^T has positive eigenvalue: {eigvals.max().item()}"

    def test_matrix_exp_bounded(self):
        """
        Paper Appendix A.2: Since A + A^T is neg-semidef, ||exp(tA)||_2 ≤ 1 for t > 0.
        """
        gen = HyperTurbulentGenerator(latent_dim=8, num_bases=4)
        net = PhysicsInferenceNet(latent_dim=8, num_bases=4, cond_dim=2)

        z0 = torch.randn(1, 8, dtype=torch.complex64)
        cond = torch.randn(1, 2)
        k_c, r_c, alpha, beta = net(z0, cond)

        A, _ = self._build_A(gen, alpha, beta, k_c, r_c)

        for t_val in [0.1, 1.0, 10.0, 100.0]:
            At = A[0].detach() * t_val
            expAt = torch.linalg.matrix_exp(At.float())
            spectral_norm = torch.linalg.norm(expAt, ord=2)
            assert spectral_norm <= 1.0 + 1e-4, \
                f"||exp({t_val}A)||_2 = {spectral_norm:.4f} > 1, boundedness violated!"

    def test_alpha_beta_positivity(self):
        """Paper Section 5: α, β > 0 enforced by softplus + ε."""
        net = PhysicsInferenceNet(latent_dim=8, num_bases=4, cond_dim=2)
        
        # Test with many random inputs
        for _ in range(20):
            z0 = torch.randn(4, 8, dtype=torch.complex64)
            cond = torch.randn(4, 2)
            _, _, alpha, beta = net(z0, cond)
            assert torch.all(alpha > 0), f"α not positive: {alpha.min().item()}"
            assert torch.all(beta > 0), f"β not positive: {beta.min().item()}"

    def test_latent_energy_non_increasing(self):
        """
        End-to-end: ||z(t)|| should be non-increasing over time (the Lyapunov guarantee).
        """
        gen = HyperTurbulentGenerator(latent_dim=8, num_bases=4)
        net = PhysicsInferenceNet(latent_dim=8, num_bases=4, cond_dim=2)

        z0 = torch.randn(1, 8, dtype=torch.complex64)
        cond = torch.randn(1, 2)
        k_c, r_c, alpha, beta = net(z0, cond)

        time_steps = torch.arange(1, 51).float() * 0.1  # 50 steps

        with torch.no_grad():
            z_evolved = gen(z0, time_steps, k_c, r_c, alpha, beta)  # (1, 50, 8)

        # Compute ||z(t)||^2 at each step
        energy = (z_evolved.abs() ** 2).sum(dim=-1).squeeze(0).numpy()  # (50,)
        initial_energy = (z0.abs() ** 2).sum().item()

        # Energy at every step should be ≤ initial energy (within numerical tolerance)
        for t_idx in range(len(energy)):
            assert energy[t_idx] <= initial_energy + 1e-3, \
                f"Energy at t={t_idx} ({energy[t_idx]:.4f}) exceeds initial ({initial_energy:.4f}), Lyapunov violated!"

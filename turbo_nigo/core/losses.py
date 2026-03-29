"""
Composable loss functions for the TurboNIGO framework.

Provides modular loss terms that can be combined via CompositeLoss:
  - ReconstructionMSE: Standard pixel-space MSE
  - SpectralLoss: L2 in 2D Fourier space (captures high-frequency fidelity)
  - PhysicsPriorLoss: L2 regularization on generator basis coefficients
  - RelativeL2Loss: Relative L2 error (suited for elliptic/Darcy problems)
  - DivergenceLoss: Enforces incompressibility (div(u) = 0)
  - SobolevH1Loss: Penalizes non-physical gradients
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralLoss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 5:
            B, T, C, H, W = pred.shape
            pred = pred.reshape(B * T, C, H, W)
            target = target.reshape(B * T, C, H, W)

        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        tgt_fft = torch.fft.rfft2(target, norm="ortho")

        return self.weight * F.mse_loss(
            torch.view_as_real(pred_fft),
            torch.view_as_real(tgt_fft),
        )

class PhysicsPriorLoss(nn.Module):
    def __init__(self, weight: float = 0.001):
        super().__init__()
        self.weight = weight

    def forward(self, k_coeffs: torch.Tensor, r_coeffs: torch.Tensor) -> torch.Tensor:
        return self.weight * (k_coeffs.pow(2).mean() + r_coeffs.pow(2).mean())

class RelativeL2Loss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = (pred - target).reshape(pred.shape[0], -1)
        tgt = target.reshape(target.shape[0], -1)
        rel = (diff.norm(dim=1) / (tgt.norm(dim=1) + 1e-8)).mean()
        return self.weight * rel

class DivergenceLoss(nn.Module):
    """Enforces div(u) = 0 for incompressible fluid mapping."""
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        # pred: (B, S, C=2, H, W) where C=0 is u, C=1 is v
        # Standard central finite difference approximation
        du_dx = pred[:, :, 0, :, 2:] - pred[:, :, 0, :, :-2]
        dv_dy = pred[:, :, 1, 2:, :] - pred[:, :, 1, :-2, :]
        
        # Pad differences directly to match grid size
        du_dx = F.pad(du_dx, (1, 1, 0, 0))
        dv_dy = F.pad(dv_dy, (0, 0, 1, 1))
        
        div = du_dx + dv_dy
        return self.weight * torch.mean(div**2)

class SobolevH1Loss(nn.Module):
    """Penalizes high-frequency numerical checkerboarding."""
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # First order difference along x
        grad_pred_x = pred[..., 1:] - pred[..., :-1]
        grad_tgt_x = target[..., 1:] - target[..., :-1]
        loss_x = F.mse_loss(grad_pred_x, grad_tgt_x)
        
        # First order difference along y
        grad_pred_y = pred[..., 1:, :] - pred[..., :-1, :]
        grad_tgt_y = target[..., 1:, :] - target[..., :-1, :]
        loss_y = F.mse_loss(grad_pred_y, grad_tgt_y)
        
        return self.weight * (loss_x + loss_y)

class CompositeLoss(nn.Module):
    """
    Builds a composite loss from config flags.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.physics_prior = PhysicsPriorLoss(weight=config.get("physics_prior_weight", 0.001))

        sw = config.get("spectral_loss_weight", 0.0)
        self.spectral = SpectralLoss(weight=sw) if sw > 0 else None

        rw = config.get("relative_l2_weight", 0.0)
        self.relative = RelativeL2Loss(weight=rw) if rw > 0 else None

        dw = config.get("divergence_weight", 0.0)
        self.divergence = DivergenceLoss(weight=dw) if dw > 0 else None

        hw = config.get("h1_weight", 0.0)
        self.h1 = SobolevH1Loss(weight=hw) if hw > 0 else None

    def forward(self, u_pred, u_target, k_coeffs, r_coeffs):
        losses = {}

        mse = F.mse_loss(u_pred, u_target)
        losses["mse"] = mse.item()
        total = mse

        if self.spectral is not None:
            spec = self.spectral(u_pred, u_target)
            losses["spectral"] = spec.item()
            total = total + spec

        if self.relative is not None:
            rel = self.relative(u_pred, u_target)
            losses["relative_l2"] = rel.item()
            total = total + rel

        if self.divergence is not None:
            div = self.divergence(u_pred)
            losses["divergence"] = div.item()
            total = total + div

        if self.h1 is not None:
            h1 = self.h1(u_pred, u_target)
            losses["h1_smoothness"] = h1.item()
            total = total + h1

        if self.physics_prior is not None:
            prior = self.physics_prior(k_coeffs, r_coeffs)
            losses["physics_prior"] = prior.item()
            total = total + prior

        losses["total"] = total.item()
        return total, losses

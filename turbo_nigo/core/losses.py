"""
Composable loss functions for the TurboNIGO framework.

Provides modular loss terms that can be combined via CompositeLoss:
  - ReconstructionMSE: Standard pixel-space MSE
  - SpectralLoss: L2 in 2D Fourier space (captures high-frequency fidelity)
  - PhysicsPriorLoss: L2 regularization on generator basis coefficients
  - RelativeL2Loss: Relative L2 error (suited for elliptic/Darcy problems)

Usage:
    criterion = CompositeLoss(config)
    total_loss, loss_dict = criterion(u_pred, u_target, k_coeffs, r_coeffs)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralLoss(nn.Module):
    """L2 loss in 2D Fourier space — penalizes spectral content mismatch."""

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, T, C, H, W) or (B, C, H, W)
            target: same shape as pred
        """
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
    """L2 regularization on generator basis coefficients k and r."""

    def __init__(self, weight: float = 0.001):
        super().__init__()
        self.weight = weight

    def forward(
        self, k_coeffs: torch.Tensor, r_coeffs: torch.Tensor
    ) -> torch.Tensor:
        return self.weight * (k_coeffs.pow(2).mean() + r_coeffs.pow(2).mean())


class RelativeL2Loss(nn.Module):
    """Relative L2: ||pred - target||₂ / ||target||₂.  Suited for Darcy flow."""

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = (pred - target).reshape(pred.shape[0], -1)
        tgt = target.reshape(target.shape[0], -1)
        rel = (diff.norm(dim=1) / (tgt.norm(dim=1) + 1e-8)).mean()
        return self.weight * rel


class CompositeLoss(nn.Module):
    """
    Builds a composite loss from config flags.

    Config keys consumed:
        physics_prior_weight  (float, default 0.001)
        spectral_loss_weight  (float, default 0.0 — disabled)
        relative_l2_weight    (float, default 0.0 — disabled)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.physics_prior = PhysicsPriorLoss(
            weight=config.get("physics_prior_weight", 0.001)
        )

        sw = config.get("spectral_loss_weight", 0.0)
        self.spectral = SpectralLoss(weight=sw) if sw > 0 else None

        rw = config.get("relative_l2_weight", 0.0)
        self.relative = RelativeL2Loss(weight=rw) if rw > 0 else None

    def forward(self, u_pred, u_target, k_coeffs, r_coeffs):
        """
        Returns:
            total_loss (Tensor): scalar for backward()
            loss_dict  (dict):   individual loss values (detached floats)
        """
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

        prior = self.physics_prior(k_coeffs, r_coeffs)
        losses["physics_prior"] = prior.item()
        total = total + prior

        losses["total"] = total.item()
        return total, losses

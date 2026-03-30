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
    """
    First-order Sobolev (H¹) semi-norm penalty on spatial gradients.

    Computes MSE on finite-difference spatial derivatives:
        L_H1 = MSE(∂pred/∂x, ∂gt/∂x)  [+ MSE(∂pred/∂y, ∂gt/∂y) for 2D]

    Automatically detects spatial dimensionality:
        - 1D fields: tensors with ndim ≤ 4 after any (B,T) collapse → (*, C, X)
        - 2D fields: tensors with ndim = 5 or ≥ 3 spatial axes → (*, C, H, W)

    This naturally penalizes high-frequency artifacts because finite differences
    amplify high-wavenumber modes by factor ~k, equivalent to a k²-weighted
    spectral penalty.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Determine if input is 1D or 2D spatial
        # 2D: (B, T, C, H, W) [5D] or (B, C, H, W) [4D with C small]
        # 1D: (B, T, C, X) [4D] or (B, C, X) [3D]
        is_2d = pred.dim() >= 5 or (pred.dim() == 4 and target.dim() >= 5)

        # Gradient along last spatial axis (x for 1D, W for 2D)
        grad_pred_x = pred[..., 1:] - pred[..., :-1]
        grad_tgt_x = target[..., 1:] - target[..., :-1]
        loss = F.mse_loss(grad_pred_x, grad_tgt_x)

        if is_2d:
            # Gradient along second-to-last spatial axis (H for 2D)
            grad_pred_y = pred[..., 1:, :] - pred[..., :-1, :]
            grad_tgt_y = target[..., 1:, :] - target[..., :-1, :]
            loss = loss + F.mse_loss(grad_pred_y, grad_tgt_y)

        return self.weight * loss

class CompositeLoss(nn.Module):
    """
    Builds a composite loss from config flags.
    Supports Dual Curriculum Learning (Temporal Unrolling + Loss Finetuning).
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
        
        self.curriculum = config.get("curriculum_learning", False)
        self.seq_len = config.get("seq_len", 20)

    def forward(self, u_pred, u_target, k_coeffs, r_coeffs, epoch: int = None, max_epochs: int = None):
        losses = {}

        # ---------------- Curriculum Logic ----------------
        u_p = u_pred
        u_t = u_target
        h1_active = True

        if self.curriculum and epoch is not None and max_epochs is not None:
            phase1_end = max_epochs // 2
            
            # Temporal Sequence Unrolling
            if epoch <= phase1_end:
                t_limit = max(1, int(self.seq_len * (epoch / phase1_end)))
                h1_active = False # Pure MSE Phase
            else:
                t_limit = self.seq_len
                h1_active = True  # Sobolev Finetuning Phase
                
            if u_pred.dim() >= 2: # Assuming (B, T, ...)
                t_max_actual = min(u_pred.shape[1], t_limit)
                u_p = u_pred[:, :t_max_actual]
                u_t = u_target[:, :t_max_actual]
        # --------------------------------------------------

        mse = F.mse_loss(u_p, u_t)
        losses["mse"] = mse.item()
        total = mse

        if self.spectral is not None:
            spec = self.spectral(u_p, u_t)
            losses["spectral"] = spec.item()
            total = total + spec

        if self.relative is not None:
            rel = self.relative(u_p, u_t)
            losses["relative_l2"] = rel.item()
            total = total + rel

        if self.divergence is not None:
            div = self.divergence(u_p)
            losses["divergence"] = div.item()
            total = total + div

        if self.h1 is not None and h1_active:
            h1 = self.h1(u_p, u_t)
            losses["h1_smoothness"] = h1.item()
            total = total + h1

        if self.physics_prior is not None:
            prior = self.physics_prior(k_coeffs, r_coeffs)
            losses["physics_prior"] = prior.item()
            total = total + prior

        losses["total"] = total.item()
        return total, losses

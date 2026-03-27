import torch
import numpy as np
from scipy.signal import welch
from scipy.stats import pearsonr
from typing import Tuple, Dict

def compute_lyapunov_divergence(model, u0: torch.Tensor, steps: int, cond: torch.Tensor, dt: float, perturbation: float = 1e-4) -> Tuple[np.ndarray, float]:
    """
    Computes Lyapunov proxy via One-Shot Perturbation.
    """
    model.eval()
    device = u0.device
    
    # Perturb Input
    noise = torch.randn_like(u0)
    u0_pert = u0 + perturbation * noise
    
    # Measure actual init distance
    init_dist = torch.norm((u0 - u0_pert).reshape(u0.shape[0], -1), p=2, dim=1).item()
    
    # Time vector for full rollout
    time_steps = torch.arange(1, steps + 1).float().to(device) * dt
    
    with torch.no_grad():
        traj, *_ = model(u0, time_steps, cond)
        traj_pert, *_ = model(u0_pert, time_steps, cond)
        
        diff = traj - traj_pert
        diff_flat = diff.reshape(u0.shape[0], steps, -1)
        dist_t = torch.norm(diff_flat, p=2, dim=2).mean(dim=0).cpu().numpy()
        
    return np.concatenate(([init_dist], dist_t)), init_dist

def compute_physics_metrics(gt: np.ndarray, pred: np.ndarray, dt: float) -> Dict[str, float]:
    """
    Computes comprehensive physics metrics for the paper (Strouhal, Correlation, RMSE).
    Expects 1D signals sampled at a probe point.
    """
    freqs, psd_gt = welch(gt, fs=1/dt, nperseg=min(len(gt), 256))
    _, psd_pred = welch(pred, fs=1/dt, nperseg=min(len(pred), 256))
    
    idx_gt = np.argmax(psd_gt[1:]) + 1
    idx_pred = np.argmax(psd_pred[1:]) + 1
    f_gt = freqs[idx_gt]
    f_pred = freqs[idx_pred]
    
    mse = np.mean((gt - pred)**2)
    rmse = np.sqrt(mse)
    corr, _ = pearsonr(gt, pred)
    
    return {
        "freq_gt": f_gt,
        "freq_pred": f_pred,
        "freqs": freqs,
        "psd_gt": psd_gt,
        "psd_pred": psd_pred,
        "rmse": rmse,
        "correlation": corr
    }

def compute_rollout_mse(model, u0: torch.Tensor, cond: torch.Tensor,
                        gt_seq: torch.Tensor, dt: float, block_size: int = 20
                        ) -> np.ndarray:
    """
    Computes per-step MSE under autoregressive (chained) rollout.

    The model predicts `block_size` steps at a time, then feeds the last
    predicted frame as the new initial condition for the next block.

    Args:
        model: Trained TurboNIGO model (eval mode).
        u0: Initial condition, shape (B, C, H, W).
        cond: Condition vector, shape (B, cond_dim).
        gt_seq: Ground-truth sequence, shape (B, T, C, H, W).
        dt: Time step size.
        block_size: Number of steps per forward pass block.

    Returns:
        per_step_mse: numpy array of shape (T,), MSE at each predicted step.
    """
    model.eval()
    device = u0.device
    T = gt_seq.shape[1]
    num_blocks = int(np.ceil(T / block_size))
    block_time = torch.arange(1, block_size + 1).float().to(device) * dt

    all_preds = []
    curr = u0

    with torch.no_grad():
        for _ in range(num_blocks):
            u_block, *_ = model(curr, block_time, cond)
            # u_block: (B, block_size, C, H, W)
            all_preds.append(u_block)
            curr = u_block[:, -1]  # last predicted frame

    # Concatenate along time axis and trim to T
    predictions = torch.cat(all_preds, dim=1)[:, :T]  # (B, T, C, H, W)
    diff = (predictions - gt_seq) ** 2
    per_step_mse = diff.mean(dim=(0, 2, 3, 4)).cpu().numpy()  # (T,)
    return per_step_mse


def compute_latent_energy_trace(model, u0: torch.Tensor, cond: torch.Tensor,
                                steps: int, dt: float) -> np.ndarray:
    """
    Tracks the latent energy ||z_t||^2 over time to verify Lyapunov stability.

    A stable model should show bounded or decaying latent norm,
    while an unstable one will show exponential growth.

    Returns:
        energy_trace: numpy array of shape (steps,), the ||z_t||^2 at each step.
    """
    model.eval()
    device = u0.device
    time_steps = torch.arange(1, steps + 1).float().to(device) * dt

    with torch.no_grad():
        _, z_base, *_ = model(u0, time_steps, cond)
        # z_base: (B, T, latent_dim) — complex
        if z_base.is_complex():
            energy = (z_base.real ** 2 + z_base.imag ** 2).sum(dim=-1)
        else:
            energy = (z_base ** 2).sum(dim=-1)
        # Average over batch
        energy_trace = energy.mean(dim=0).cpu().numpy()  # (T,)

    return energy_trace


def compute_relative_l2_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Computes the relative L2 error at each time step:
        e(t) = ||u_pred(t) - u_gt(t)||_2 / ||u_gt(t)||_2

    This is the standard metric in neural operator papers (FNO, DeepONet, etc.).

    Args:
        pred: Predicted fields, shape (T, C, H, W) or (B, T, C, H, W).
        gt: Ground-truth fields, same shape as pred.

    Returns:
        rel_l2: numpy array of shape (T,), relative L2 error per step.
    """
    # Collapse to (T, -1)
    if pred.ndim == 5:
        pred = pred.mean(axis=0)  # average over batch
        gt = gt.mean(axis=0)

    T = pred.shape[0]
    pred_flat = pred.reshape(T, -1)
    gt_flat = gt.reshape(T, -1)

    num = np.linalg.norm(pred_flat - gt_flat, axis=1)
    den = np.linalg.norm(gt_flat, axis=1) + 1e-10
    return num / den


def get_radial_spectrum(field: np.ndarray) -> np.ndarray:
    """Computes radial energy spectrum of a 2D spatial field."""
    f = np.fft.fft2(field)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)**2
    H, W = field.shape
    center_x, center_y = W // 2, H // 2
    y, x = np.ogrid[-center_y:H-center_y, -center_x:W-center_x]
    r = np.sqrt(x**2 + y**2).astype(int)
    tbin = np.bincount(r.ravel(), magnitude.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / np.maximum(nr, 1)
    return radial_profile[:min(H, W)//2]

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

"""
evaluate_burgers.py — Research-Grade Evaluation Suite for Burgers Neural Operator.

Comprehensive evaluation metrics for PDE operator learning on the 1D viscous Burgers equation:

  1. Relative L2 Error (per-step & mean) — Standard neural operator metric (FNO, DeepONet)
  2. Absolute MSE (per-step & mean)      — Raw prediction accuracy
  3. L-infinity / Max Error              — Worst-case shock misalignment
  4. Normalized RMSE (NRMSE)             — Scale-independent accuracy
  5. Pearson Correlation                 — Structural similarity over space
  6. Conservation Error (Mass/Integral)  — Physics consistency: ∫u(x,t)dx drift
  7. Spectral Relative Error             — Fourier-domain fidelity at each timestep
  8. Energy Dissipation Rate Error        — dE/dt accuracy for dissipative systems
  9. Shock Location Error                — Peak-gradient tracking precision

Publication-Grade Visualizations:
  - Spatiotemporal (x-t) heatmap: GT vs Pred vs Error
  - 1D spatial profile snapshots at key timesteps
  - Per-step error evolution curves (L2, MSE, L-inf)
  - Conservation integral drift plot
  - Spectral comparison (FFT magnitude) at selected timesteps

All outputs are saved to an immutable results/Burgers_<timestamp>/ directory.

Usage:
  conda run -n turbo_nigo python scripts/evaluate_burgers.py \\
      --run_dir results/burgers_experiments/RUN_BURGERS_XXXXXXXX_XXXXXX \\
      --data_path ./datasets/Burgers/1D_Burgers_Sols_Nu0.1.hdf5 \\
      --eval_steps 200
"""

import os
import sys
import json
import h5py
import math
import shutil
import argparse
import datetime
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from turbo_nigo.models.turbo_nigo import GlobalTurboNIGO
from turbo_nigo.models.turbo_nigo_1d import GlobalTurboNIGO_1D
from turbo_nigo.core.metrics import compute_relative_l2_error

# =========================================================================
# Global Publication Style (ICML / NeurIPS grade)
# =========================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.titlesize': 13,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': False,
})

SINGLE_COL = 3.25  # inches
DOUBLE_COL = 6.75


# =========================================================================
# Dataset
# =========================================================================
class ValidationBurgersDataset(Dataset):
    """
    Validation loader for 1D Burgers Equation.
    Returns the full temporal sequence for evaluating long-horizon rollouts.
    Preserves exact train/val split logic from train_burgers.py.

    Supports both 1D (native) and 2D (legacy reshape) modes via spatial_mode.
    """
    def __init__(self, h5_path, eval_steps, max_samples=None, spatial_mode='1d'):
        super().__init__()
        self.h5_path = h5_path
        self.spatial_mode = spatial_mode

        print(f"[*] Preloading Burgers Evaluation dataset (mode={spatial_mode})...")
        with h5py.File(h5_path, 'r') as f:
            raw = f['tensor'][:]
            if max_samples:
                raw = raw[:max_samples]

            # Validation split (last 10%) — matches train_burgers.py exactly
            N_total = raw.shape[0]
            val_split = int(0.1 * N_total)
            raw = raw[-val_split:]

        N, T, X = raw.shape
        self.N, self.T, self.X = N, T, X
        self.eval_steps = min(eval_steps, T - 1)

        # Normalization bounds
        self.g_min = float(raw.min())
        self.g_max = float(raw.max())

        raw_torch = torch.from_numpy(raw).float()
        raw_norm = (raw_torch - self.g_min) / (self.g_max - self.g_min + 1e-8)

        if spatial_mode == '2d':
            # Legacy: reshape to (N, T, 1, side, side)
            self.side = int(math.isqrt(X))
            assert self.side * self.side == X, f"Spatial size {X} must be a perfect square."
            self.data = raw_norm.view(N, T, 1, self.side, self.side)
            print(f"    Loaded {N} val trajectories | T={T} | X={X} -> {self.side}x{self.side}")
        else:
            # Native 1D: (N, T, 1, X)
            self.side = X  # For model construction: spatial_size = X
            self.data = raw_norm.unsqueeze(2)
            print(f"    Loaded {N} val trajectories | T={T} | X={X} (native 1D)")

        self.cond = torch.zeros(4, dtype=torch.float32)
        print(f"    Eval steps: {self.eval_steps} | g_min={self.g_min:.4f}, g_max={self.g_max:.4f}")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        u0 = self.data[idx, 0]
        u_gt = self.data[idx, 1:self.eval_steps + 1]
        return u0, u_gt, self.cond


# =========================================================================
# Autoregressive Rollout
# =========================================================================
def autoregressive_rollout(model, u0, cond, total_steps, block_size, dt):
    """
    Chained block-autoregressive rollout.
    Returns predicted sequence of shape (B, total_steps, C, H, W).
    """
    model.eval()
    device = u0.device
    num_blocks = int(np.ceil(total_steps / block_size))
    block_time = torch.arange(1, block_size + 1).float().to(device) * dt

    all_preds = []
    curr = u0

    with torch.no_grad():
        for _ in range(num_blocks):
            u_block, *_ = model(curr, block_time, cond)
            all_preds.append(u_block)
            curr = u_block[:, -1]

    predictions = torch.cat(all_preds, dim=1)[:, :total_steps]
    return predictions


# =========================================================================
# Metric Computation Functions (Burgers-specific)
# =========================================================================
def compute_conservation_error(pred, gt, dx=1.0):
    """
    Mass conservation error: |∫u_pred dx - ∫u_gt dx| / |∫u_gt dx|.
    For viscous Burgers, total integral should dissipate smoothly.
    Returns per-step conservation error array of shape (T,).
    """
    # pred, gt: (T, X)
    integral_pred = np.trapz(pred, dx=dx, axis=1)  # (T,)
    integral_gt = np.trapz(gt, dx=dx, axis=1)       # (T,)
    denom = np.abs(integral_gt) + 1e-10
    return np.abs(integral_pred - integral_gt) / denom


def compute_spectral_error(pred, gt):
    """
    Relative error in Fourier space: ||FFT(pred) - FFT(gt)|| / ||FFT(gt)||.
    Measures how well the model captures multi-scale structure.
    Returns per-step spectral error of shape (T,).
    """
    # pred, gt: (T, X)
    fft_pred = np.fft.rfft(pred, axis=1)
    fft_gt = np.fft.rfft(gt, axis=1)
    num = np.linalg.norm(np.abs(fft_pred) - np.abs(fft_gt), axis=1)
    den = np.linalg.norm(np.abs(fft_gt), axis=1) + 1e-10
    return num / den


def compute_energy_dissipation_error(pred, gt, dt=0.01):
    """
    Error in energy dissipation rate: dE/dt where E(t) = 0.5 * ∫u²dx.
    Critical for dissipative PDEs like viscous Burgers.
    Returns per-step dissipation rate error of shape (T-1,).
    """
    E_pred = 0.5 * np.mean(pred**2, axis=1)  # (T,)
    E_gt = 0.5 * np.mean(gt**2, axis=1)       # (T,)
    dEdt_pred = np.diff(E_pred) / dt           # (T-1,)
    dEdt_gt = np.diff(E_gt) / dt
    denom = np.abs(dEdt_gt) + 1e-10
    return np.abs(dEdt_pred - dEdt_gt) / denom


def compute_shock_location_error(pred, gt, dx=1.0):
    """
    Shock location error: difference between locations of peak |du/dx|.
    For Burgers, shocks produce sharp spatial gradients; tracking them is key.
    Returns per-step shock position error in grid units (T,).
    """
    grad_pred = np.abs(np.gradient(pred, dx, axis=1))  # (T, X)
    grad_gt = np.abs(np.gradient(gt, dx, axis=1))

    shock_loc_pred = np.argmax(grad_pred, axis=1)  # (T,)
    shock_loc_gt = np.argmax(grad_gt, axis=1)

    return np.abs(shock_loc_pred.astype(float) - shock_loc_gt.astype(float))


def compute_pearson_per_step(pred, gt):
    """
    Per-step Pearson correlation between spatial profiles.
    Returns array of shape (T,).
    """
    T = pred.shape[0]
    corrs = np.zeros(T)
    for t in range(T):
        if np.std(gt[t]) < 1e-10 or np.std(pred[t]) < 1e-10:
            corrs[t] = 0.0
        else:
            corrs[t], _ = pearsonr(pred[t], gt[t])
    return corrs


# =========================================================================
# Visualization Functions
# =========================================================================
def _save_fig(fig, path_stem):
    """Save figure in both PNG and PDF format."""
    fig.savefig(f"{path_stem}.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(f"{path_stem}.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_spatiotemporal(gt, pred, out_dir):
    """
    Three-panel spatiotemporal (x-t) heatmap: GT | Prediction | Absolute Error.
    """
    from matplotlib.colors import TwoSlopeNorm

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL * 2.2, SINGLE_COL * 1.6))

    vmax_abs = np.percentile(np.abs(gt), 99.5)
    norm = TwoSlopeNorm(vmin=-vmax_abs, vcenter=0, vmax=vmax_abs)

    im0 = axes[0].imshow(gt.T, aspect='auto', origin='lower', cmap='RdBu_r',
                          norm=norm, extent=[0, gt.shape[0], 0, gt.shape[1]])
    axes[0].set_title(r'Ground Truth $u(x,t)$')
    axes[0].set_ylabel(r'Spatial coordinate $x$')
    axes[0].set_xlabel(r'Time step $t$')

    im1 = axes[1].imshow(pred.T, aspect='auto', origin='lower', cmap='RdBu_r',
                          norm=norm, extent=[0, pred.shape[0], 0, pred.shape[1]])
    axes[1].set_title(r'NIGO Prediction $\hat{u}(x,t)$')
    axes[1].set_xlabel(r'Time step $t$')
    axes[1].set_yticks([])

    err = np.abs(gt - pred)
    im2 = axes[2].imshow(err.T, aspect='auto', origin='lower', cmap='magma',
                          extent=[0, err.shape[0], 0, err.shape[1]])
    axes[2].set_title(r'Absolute Error $|u - \hat{u}|$')
    axes[2].set_xlabel(r'Time step $t$')
    axes[2].set_yticks([])

    fig.colorbar(im0, ax=axes[:2], fraction=0.02, pad=0.02, label=r'$u$')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02, label=r'$|e|$')

    fig.suptitle('Burgers Equation: Spatiotemporal Evaluation', fontsize=13, y=1.02)
    _save_fig(fig, os.path.join(out_dir, "fig_spatiotemporal"))


def plot_snapshots(gt, pred, out_dir, n_snaps=5):
    """
    Overlaid 1D spatial profiles at selected timesteps.
    """
    T_max = gt.shape[0] - 1
    X = gt.shape[1]
    x_axis = np.linspace(0, 1, X)

    t_indices = np.linspace(0, T_max, n_snaps, dtype=int)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_snaps))

    fig, axes = plt.subplots(1, n_snaps, figsize=(DOUBLE_COL * 2, SINGLE_COL * 1.0),
                              sharey=True)

    for i, (t_idx, col) in enumerate(zip(t_indices, colors)):
        ax = axes[i]
        ax.plot(x_axis, gt[t_idx], color='k', linewidth=1.2, label='GT')
        ax.plot(x_axis, pred[t_idx], color=col, linewidth=1.5, linestyle='--', label='Pred')
        ax.fill_between(x_axis, gt[t_idx], pred[t_idx], alpha=0.15, color=col)
        ax.set_title(f'$t = {t_idx}$', fontsize=10)
        ax.set_xlabel(r'$x$', fontsize=9)
        if i == 0:
            ax.set_ylabel(r'$u(x)$', fontsize=10)
        ax.grid(True, alpha=0.15)
        ax.legend(fontsize=7, loc='upper right')

    fig.suptitle('Burgers 1D Spatial Profiles: Ground Truth vs NIGO', fontsize=12, y=1.03)
    _save_fig(fig, os.path.join(out_dir, "fig_snapshots"))


def plot_error_curves(avg_rel_l2, avg_mse, avg_max_err, avg_corr, out_dir):
    """
    Four-panel per-step error evolution: L2, MSE, L-inf, Correlation.
    """
    T = len(avg_rel_l2)
    steps = np.arange(1, T + 1)

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL * 1.5, DOUBLE_COL * 1.0))

    # Relative L2
    axes[0, 0].plot(steps, avg_rel_l2, color='#E74C3C', linewidth=1.2)
    axes[0, 0].set_ylabel(r'Relative $L_2$ Error')
    axes[0, 0].set_title(r'Relative $L_2$ Error vs Time')
    axes[0, 0].grid(True, alpha=0.15)

    # MSE
    axes[0, 1].semilogy(steps, avg_mse, color='#3498DB', linewidth=1.2)
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].set_title('Absolute MSE vs Time')
    axes[0, 1].grid(True, alpha=0.15)

    # Max Error
    axes[1, 0].plot(steps, avg_max_err, color='#E67E22', linewidth=1.2)
    axes[1, 0].set_ylabel(r'$L_\infty$ Error')
    axes[1, 0].set_xlabel('Time step')
    axes[1, 0].set_title(r'$L_\infty$ (Max) Error vs Time')
    axes[1, 0].grid(True, alpha=0.15)

    # Correlation
    axes[1, 1].plot(steps, avg_corr, color='#2ECC71', linewidth=1.2)
    axes[1, 1].set_ylabel('Pearson $r$')
    axes[1, 1].set_xlabel('Time step')
    axes[1, 1].set_title('Spatial Correlation vs Time')
    axes[1, 1].set_ylim(-0.1, 1.05)
    axes[1, 1].grid(True, alpha=0.15)

    fig.suptitle('Per-Step Error Evolution (Averaged over Val Set)', fontsize=12, y=1.02)
    plt.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "fig_error_curves"))


def plot_conservation(avg_cons_err, out_dir):
    """
    Conservation integral drift over time.
    """
    T = len(avg_cons_err)
    steps = np.arange(1, T + 1)

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.8, SINGLE_COL * 1.0))
    ax.semilogy(steps, avg_cons_err, color='#8E44AD', linewidth=1.2)
    ax.set_xlabel('Time step')
    ax.set_ylabel(r'Relative Conservation Error $|\Delta \int u\,dx|$')
    ax.set_title(r'Mass Conservation Error: $\frac{|\int \hat{u}\,dx - \int u\,dx|}{|\int u\,dx|}$')
    ax.grid(True, alpha=0.15)

    _save_fig(fig, os.path.join(out_dir, "fig_conservation"))


def plot_spectral_comparison(gt, pred, out_dir, t_indices=None):
    """
    FFT magnitude comparison at selected timesteps.
    """
    X = gt.shape[1]
    if t_indices is None:
        T_max = gt.shape[0] - 1
        t_indices = [2, T_max // 4, T_max // 2, T_max]

    n = len(t_indices)
    fig, axes = plt.subplots(1, n, figsize=(DOUBLE_COL * 1.8, SINGLE_COL * 1.0), sharey=True)
    if n == 1:
        axes = [axes]

    freqs = np.fft.rfftfreq(X)

    for i, (t_idx, ax) in enumerate(zip(t_indices, axes)):
        fft_gt = np.abs(np.fft.rfft(gt[t_idx]))
        fft_pred = np.abs(np.fft.rfft(pred[t_idx]))

        ax.semilogy(freqs[1:], fft_gt[1:], 'k-', linewidth=1.0, alpha=0.8, label='GT')
        ax.semilogy(freqs[1:], fft_pred[1:], 'r--', linewidth=1.0, alpha=0.8, label='Pred')
        ax.set_title(f'$t = {t_idx}$', fontsize=10)
        ax.set_xlabel('Frequency', fontsize=9)
        if i == 0:
            ax.set_ylabel(r'$|\hat{u}_k|$', fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.12, which='both')

    fig.suptitle('Fourier Spectrum Comparison: Ground Truth vs NIGO', fontsize=12, y=1.03)
    _save_fig(fig, os.path.join(out_dir, "fig_spectral"))


def plot_energy_dissipation(gt, pred, dt, out_dir):
    """
    Energy dissipation rate dE/dt comparison.
    """
    E_gt = 0.5 * np.mean(gt**2, axis=1)
    E_pred = 0.5 * np.mean(pred**2, axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL * 1.5, SINGLE_COL * 1.0))

    steps = np.arange(len(E_gt))
    ax1.plot(steps, E_gt, 'k-', linewidth=1.2, label=r'GT $E(t)$')
    ax1.plot(steps, E_pred, 'r--', linewidth=1.2, label=r'Pred $E(t)$')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel(r'Energy $E(t) = \frac{1}{2}\langle u^2 \rangle$')
    ax1.set_title('Energy Trace')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.15)

    dEdt_gt = np.diff(E_gt) / dt
    dEdt_pred = np.diff(E_pred) / dt
    steps2 = np.arange(len(dEdt_gt))
    ax2.plot(steps2, dEdt_gt, 'k-', linewidth=1.0, alpha=0.7, label=r'GT $dE/dt$')
    ax2.plot(steps2, dEdt_pred, 'r--', linewidth=1.0, alpha=0.7, label=r'Pred $dE/dt$')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel(r'Dissipation Rate $dE/dt$')
    ax2.set_title('Energy Dissipation Rate')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.15)

    fig.suptitle('Energy Dynamics: Viscous Burgers Dissipation', fontsize=12, y=1.03)
    plt.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "fig_energy_dissipation"))


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Research-Grade Burgers Operator Evaluation")
    parser.add_argument('--run_dir', type=str, required=True,
                        help="Path to training run directory containing best_model.pth")
    parser.add_argument('--data_path', type=str,
                        default='./datasets/Burgers/1D_Burgers_Sols_Nu0.1.hdf5')
    parser.add_argument('--eval_steps', type=int, default=200,
                        help="Total rollout steps to evaluate")
    parser.add_argument('--seq_len', type=int, default=20,
                        help="Block size used during training (for autoregressive chunking)")
    parser.add_argument('--dt', type=float, default=0.01,
                        help="Physical time step delta")
    parser.add_argument('--max_samples', type=int, default=1000,
                        help="Dataset memory cap (must match training)")
    parser.add_argument('--plot_sample_idx', type=int, default=0,
                        help="Which val sample index to use for detailed plots")
    parser.add_argument('--model_type', type=str, default='1d', choices=['1d', '2d'],
                        help="Model type: '1d' for GlobalTurboNIGO_1D (v2), '2d' for legacy GlobalTurboNIGO")
    args = parser.parse_args()

    # --- Create immutable results directory ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", f"Burgers_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # Mirror evaluation script for provenance
    src_mirror = os.path.join(out_dir, "src_mirror")
    os.makedirs(src_mirror, exist_ok=True)
    shutil.copy2(__file__, os.path.join(src_mirror, os.path.basename(__file__)))

    log_file = os.path.join(out_dir, "evaluate.log")
    def log(msg):
        tstr = datetime.datetime.now().strftime("[%H:%M:%S]")
        formatted = f"{tstr} {msg}"
        print(formatted)
        with open(log_file, "a") as f:
            f.write(formatted + "\n")
        sys.stdout.flush()

    log("=" * 70)
    log("  BURGERS NEURAL OPERATOR — RESEARCH-GRADE EVALUATION SUITE")
    log("=" * 70)
    log(f"  Output Dir   : {out_dir}")
    log(f"  Checkpoint   : {args.run_dir}")
    log(f"  Dataset      : {args.data_path}")
    log(f"  Eval Steps   : {args.eval_steps}")
    log(f"  Block Size   : {args.seq_len}")
    log(f"  dt           : {args.dt}")
    log(f"  Model Type   : {args.model_type}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"  Device       : {device}")

    # ==================================================================
    # 1. Load Dataset
    # ==================================================================
    log("\n[1/5] Loading validation dataset...")
    spatial_mode = args.model_type  # '1d' or '2d'
    val_ds = ValidationBurgersDataset(args.data_path, eval_steps=args.eval_steps,
                                      max_samples=args.max_samples,
                                      spatial_mode=spatial_mode)
    loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    log(f"    Samples: {len(val_ds)} | Eval steps: {val_ds.eval_steps}")

    # ==================================================================
    # 2. Instantiate & Load Model
    # ==================================================================
    log("\n[2/5] Loading trained model...")
    if args.model_type == '1d':
        model = GlobalTurboNIGO_1D(
            latent_dim=64,
            in_channels=1,
            width=32,
            spatial_size=val_ds.X,  # native 1D spatial length
            num_layers=3,
            use_residual=True,
            norm_type='group'
        ).to(device)
    else:
        model = GlobalTurboNIGO(
            latent_dim=64,
            in_channels=1,
            width=32,
            spatial_size=val_ds.side,
            num_layers=3,
            use_residual=True,
            norm_type='group'
        ).to(device)

    ckpt_path = os.path.join(args.run_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args.run_dir, "latest_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found in {args.run_dir}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state.get('model_state', state))
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"    Loaded: {ckpt_path}")
    log(f"    Params: {n_params:,} total | {n_trainable:,} trainable")

    # ==================================================================
    # 3. Run Autoregressive Rollout & Collect Metrics
    # ==================================================================
    log(f"\n[3/5] Running autoregressive rollout evaluation...")

    # Accumulators for per-sample, per-step metrics
    all_rel_l2 = []
    all_mse = []
    all_max_err = []
    all_corr = []
    all_cons_err = []
    all_spec_err = []
    all_dissip_err = []
    all_shock_err = []
    all_nrmse = []

    # Store specific sample for detailed plotting
    plot_gt, plot_pred = None, None
    sample_counter = 0

    for u0, u_gt, cond in loader:
        u0, u_gt, cond = u0.to(device), u_gt.to(device), cond.to(device)

        # Expand cond if needed
        cond_batch = cond.unsqueeze(0).expand(u0.shape[0], -1) if cond.ndim == 1 else cond

        preds = autoregressive_rollout(model, u0, cond_batch,
                                        total_steps=u_gt.shape[1],
                                        block_size=args.seq_len, dt=args.dt)

        # De-normalize to physical space
        g_min, g_max = val_ds.g_min, val_ds.g_max
        preds_phys = preds.cpu().numpy() * (g_max - g_min) + g_min
        gt_phys = u_gt.cpu().numpy() * (g_max - g_min) + g_min

        # Flatten to (B, T, X) — works for both 1D (B,T,1,X) and 2D (B,T,1,H,W)
        preds_phys = preds_phys.reshape(preds_phys.shape[0], preds_phys.shape[1], -1)
        gt_phys = gt_phys.reshape(gt_phys.shape[0], gt_phys.shape[1], -1)

        for b in range(preds_phys.shape[0]):
            pred_b = preds_phys[b]  # (T, X)
            gt_b = gt_phys[b]      # (T, X)

            # Store plot sample
            if sample_counter == args.plot_sample_idx:
                plot_gt = gt_b.copy()
                plot_pred = pred_b.copy()

            # --- Core Metrics ---
            # 1. Relative L2
            l2_err = compute_relative_l2_error(pred_b, gt_b)
            all_rel_l2.append(l2_err)

            # 2. MSE
            mse = np.mean((pred_b - gt_b)**2, axis=1)
            all_mse.append(mse)

            # 3. Max Error (L-infinity)
            max_err = np.max(np.abs(pred_b - gt_b), axis=1)
            all_max_err.append(max_err)

            # 4. Pearson Correlation
            corr = compute_pearson_per_step(pred_b, gt_b)
            all_corr.append(corr)

            # 5. NRMSE (normalized by GT range)
            gt_range = gt_b.max() - gt_b.min() + 1e-10
            rmse_t = np.sqrt(np.mean((pred_b - gt_b)**2, axis=1))
            nrmse_t = rmse_t / gt_range
            all_nrmse.append(nrmse_t)

            # --- Burgers-Specific Physics Metrics ---
            # 6. Conservation Error
            cons_err = compute_conservation_error(pred_b, gt_b)
            all_cons_err.append(cons_err)

            # 7. Spectral Error
            spec_err = compute_spectral_error(pred_b, gt_b)
            all_spec_err.append(spec_err)

            # 8. Energy Dissipation Rate Error
            dissip_err = compute_energy_dissipation_error(pred_b, gt_b, dt=args.dt)
            all_dissip_err.append(dissip_err)

            # 9. Shock Location Error
            shock_err = compute_shock_location_error(pred_b, gt_b)
            all_shock_err.append(shock_err)

            sample_counter += 1

    log(f"    Processed {sample_counter} validation samples.")

    # ==================================================================
    # 4. Aggregate & Log Results
    # ==================================================================
    log(f"\n[4/5] Aggregating metrics...")

    avg_rel_l2 = np.mean(all_rel_l2, axis=0)
    avg_mse = np.mean(all_mse, axis=0)
    avg_max_err = np.mean(all_max_err, axis=0)
    avg_corr = np.mean(all_corr, axis=0)
    avg_nrmse = np.mean(all_nrmse, axis=0)
    avg_cons_err = np.mean(all_cons_err, axis=0)
    avg_spec_err = np.mean(all_spec_err, axis=0)
    avg_dissip_err = np.mean(all_dissip_err, axis=0)
    avg_shock_err = np.mean(all_shock_err, axis=0)

    # Scalar summaries
    summary = {
        # --- Standard Neural Operator Metrics ---
        "mean_relative_l2":         float(np.mean(avg_rel_l2)),
        "median_relative_l2":       float(np.median(avg_rel_l2)),
        "final_step_relative_l2":   float(avg_rel_l2[-1]),
        "mean_mse":                 float(np.mean(avg_mse)),
        "mean_rmse":                float(np.sqrt(np.mean(avg_mse))),
        "mean_nrmse":               float(np.mean(avg_nrmse)),
        "mean_max_error":           float(np.mean(avg_max_err)),
        "mean_correlation":         float(np.mean(avg_corr)),
        "final_step_correlation":   float(avg_corr[-1]),
        # --- Burgers-Specific Physics Metrics ---
        "mean_conservation_error":  float(np.mean(avg_cons_err)),
        "mean_spectral_error":      float(np.mean(avg_spec_err)),
        "mean_dissipation_error":   float(np.mean(avg_dissip_err)),
        "mean_shock_location_error_gridpts": float(np.mean(avg_shock_err)),
        # --- Eval Config ---
        "eval_steps":               args.eval_steps,
        "block_size":               args.seq_len,
        "dt":                       args.dt,
        "n_val_samples":            sample_counter,
        "checkpoint_path":          ckpt_path,
        "timestamp":                timestamp,
    }

    log("\n" + "=" * 60)
    log("  EVALUATION RESULTS SUMMARY")
    log("=" * 60)
    log(f"  {'Metric':<35} {'Value':>12}")
    log(f"  {'-'*35} {'-'*12}")
    log(f"  {'Mean Relative L2 Error':<35} {summary['mean_relative_l2']:>12.6f}")
    log(f"  {'Median Relative L2 Error':<35} {summary['median_relative_l2']:>12.6f}")
    log(f"  {'Final-Step Relative L2':<35} {summary['final_step_relative_l2']:>12.6f}")
    log(f"  {'Mean MSE':<35} {summary['mean_mse']:>12.6f}")
    log(f"  {'Mean RMSE':<35} {summary['mean_rmse']:>12.6f}")
    log(f"  {'Mean NRMSE':<35} {summary['mean_nrmse']:>12.6f}")
    log(f"  {'Mean L-inf Error':<35} {summary['mean_max_error']:>12.6f}")
    log(f"  {'Mean Correlation':<35} {summary['mean_correlation']:>12.6f}")
    log(f"  {'Final-Step Correlation':<35} {summary['final_step_correlation']:>12.6f}")
    log(f"  {'-'*35} {'-'*12}")
    log(f"  {'Conservation Error':<35} {summary['mean_conservation_error']:>12.6f}")
    log(f"  {'Spectral Error':<35} {summary['mean_spectral_error']:>12.6f}")
    log(f"  {'Dissipation Rate Error':<35} {summary['mean_dissipation_error']:>12.6f}")
    log(f"  {'Shock Location Error (grid)':<35} {summary['mean_shock_location_error_gridpts']:>12.2f}")
    log("=" * 60)

    # Save JSON
    json_path = os.path.join(out_dir, "evaluation_metrics.json")
    full_export = {
        **summary,
        "per_step_relative_l2": avg_rel_l2.tolist(),
        "per_step_mse": avg_mse.tolist(),
        "per_step_max_error": avg_max_err.tolist(),
        "per_step_correlation": avg_corr.tolist(),
        "per_step_nrmse": avg_nrmse.tolist(),
        "per_step_conservation_error": avg_cons_err.tolist(),
        "per_step_spectral_error": avg_spec_err.tolist(),
        "per_step_shock_location_error": avg_shock_err.tolist(),
    }
    with open(json_path, "w") as f:
        json.dump(full_export, f, indent=4)
    log(f"\n  Metrics JSON saved: {json_path}")

    # ==================================================================
    # 5. Generate Publication Visualizations
    # ==================================================================
    log(f"\n[5/5] Generating publication-quality visualizations...")

    if plot_gt is not None:
        plot_spatiotemporal(plot_gt, plot_pred, out_dir)
        log(f"  ✓ fig_spatiotemporal.png/pdf")

        plot_snapshots(plot_gt, plot_pred, out_dir)
        log(f"  ✓ fig_snapshots.png/pdf")

        plot_spectral_comparison(plot_gt, plot_pred, out_dir)
        log(f"  ✓ fig_spectral.png/pdf")

        plot_energy_dissipation(plot_gt, plot_pred, args.dt, out_dir)
        log(f"  ✓ fig_energy_dissipation.png/pdf")

    plot_error_curves(avg_rel_l2, avg_mse, avg_max_err, avg_corr, out_dir)
    log(f"  ✓ fig_error_curves.png/pdf")

    plot_conservation(avg_cons_err, out_dir)
    log(f"  ✓ fig_conservation.png/pdf")

    log(f"\n{'='*70}")
    log(f"  ALL EVALUATION ARTIFACTS -> {out_dir}/")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()

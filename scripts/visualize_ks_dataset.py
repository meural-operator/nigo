"""
visualize_ks_dataset.py — ICML Publication-Quality Visualizations for the Kuramoto-Sivashinsky Dataset.

Generates:
  1. Spatiotemporal Heatmap — x-t diagram showing chaotic dynamics
  2. Multi-Trajectory Comparison — side-by-side x-t diagrams for different ICs
  3. Spatial Power Spectrum — 1D FFT energy spectrum with inertial range
  4. Hovmöller Diagram — phase-space evolution highlighting traveling structures
  5. Temporal Autocorrelation — decorrelation timescale analysis
  6. Dataset Scale Summary — histograms of amplitude distributions

All figures saved in BOTH PNG (300dpi) and PDF (vector) format.

Usage:
  conda run -n cfd python scripts/visualize_ks_dataset.py
  conda run -n cfd python scripts/visualize_ks_dataset.py --h5_path ./datasets/KS_dataset/KS_ML_DATASET.h5
"""

import sys
import os
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

try:
    import h5py
except ImportError:
    raise ImportError("h5py is required. Install: pip install h5py")

# ===========================================================================
# Global ICML-grade style
# ===========================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'lines.linewidth': 1.0,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': False,
    'figure.constrained_layout.use': True,
})

SINGLE_COL = 3.25
DOUBLE_COL = 6.75


def _save_fig(fig, path_stem):
    fig.savefig(f"{path_stem}.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(f"{path_stem}.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {path_stem}.png  &  {path_stem}.pdf")


# ===========================================================================
# Figure 1: Spatiotemporal Heatmap (x-t diagram)
# ===========================================================================
def fig_spatiotemporal_heatmap(h5_path, output_dir, traj_idx=0, max_t=500):
    """
    Classic x-t Hovmöller diagram for a single KS trajectory, 
    revealing the chaotic spatiotemporal dynamics.
    """
    with h5py.File(h5_path, 'r') as f:
        data = np.array(f['train'][traj_idx, :max_t, :], dtype=np.float32)  # (T, Nx)
    
    fig, ax = plt.subplots(1, 1, figsize=(DOUBLE_COL * 0.7, SINGLE_COL * 1.1))
    
    vmax_abs = np.percentile(np.abs(data), 99)
    norm = TwoSlopeNorm(vmin=-vmax_abs, vcenter=0, vmax=vmax_abs)
    
    im = ax.imshow(data.T, aspect='auto', origin='lower', cmap='RdBu_r',
                   norm=norm, extent=[0, data.shape[0], 0, data.shape[1]])
    
    ax.set_xlabel(r'Time $t$')
    ax.set_ylabel(r'Spatial coordinate $x$')
    ax.set_title('Kuramoto-Sivashinsky: Spatiotemporal Evolution')
    
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(r'$u(x, t)$', fontsize=9)
    cb.ax.tick_params(labelsize=7)
    
    _save_fig(fig, os.path.join(output_dir, 'ks_spatiotemporal_heatmap'))


# ===========================================================================
# Figure 2: Multi-Trajectory Comparison
# ===========================================================================
def fig_multi_trajectory(h5_path, output_dir, n_traj=4, max_t=400):
    """
    Side-by-side x-t diagrams for multiple initial conditions,
    showcasing the diversity of chaotic behaviors in the dataset.
    """
    with h5py.File(h5_path, 'r') as f:
        n_total = f['train'].shape[0]
        indices = np.linspace(0, min(n_total - 1, 5000), n_traj, dtype=int)
        trajs = [np.array(f['train'][i, :max_t, :], dtype=np.float32) for i in indices]
    
    fig, axes = plt.subplots(1, n_traj, figsize=(DOUBLE_COL, 2.2))
    
    # Shared normalization across all panels
    global_max = max(np.percentile(np.abs(t), 99) for t in trajs)
    norm = TwoSlopeNorm(vmin=-global_max, vcenter=0, vmax=global_max)
    
    for i, (traj, ax) in enumerate(zip(trajs, axes)):
        im = ax.imshow(traj.T, aspect='auto', origin='lower', cmap='RdBu_r',
                       norm=norm, extent=[0, traj.shape[0], 0, traj.shape[1]])
        ax.set_title(f'IC {indices[i]}', fontsize=9, pad=3)
        ax.set_xlabel(r'$t$', fontsize=8)
        if i == 0:
            ax.set_ylabel(r'$x$', fontsize=9)
        else:
            ax.set_yticks([])
    
    cbar_ax = fig.add_axes([0.93, 0.12, 0.012, 0.76])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label(r'$u(x,t)$', fontsize=9)
    cb.ax.tick_params(labelsize=7)
    
    fig.suptitle('Diverse Chaotic Dynamics Across Initial Conditions', fontsize=11, y=1.04)
    
    _save_fig(fig, os.path.join(output_dir, 'ks_multi_trajectory'))


# ===========================================================================
# Figure 3: Spatial Power Spectrum
# ===========================================================================
def fig_spatial_spectrum(h5_path, output_dir, n_traj=10, snap_t=100):
    """
    1D spatial power spectrum at a single time instant, averaged over
    multiple trajectories. Shows energy cascade and characteristic scales.
    """
    with h5py.File(h5_path, 'r') as f:
        n_total = f['train'].shape[0]
        Nx = f['train'].shape[2]
        indices = np.linspace(0, min(n_total - 1, 5000), n_traj, dtype=int)
        
        spectra = []
        for i in indices:
            u_snap = np.array(f['train'][i, snap_t, :], dtype=np.float32)
            fft = np.fft.rfft(u_snap)
            psd = np.abs(fft)**2
            spectra.append(psd)
    
    mean_spectrum = np.mean(spectra, axis=0)
    std_spectrum = np.std(spectra, axis=0)
    k = np.arange(1, len(mean_spectrum))
    
    fig, ax = plt.subplots(1, 1, figsize=(SINGLE_COL * 1.3, SINGLE_COL))
    
    ax.loglog(k, mean_spectrum[1:], color='#2C3E50', linewidth=1.2, label='Mean PSD')
    ax.fill_between(k, 
                    np.clip(mean_spectrum[1:] - std_spectrum[1:], 1e-15, None),
                    mean_spectrum[1:] + std_spectrum[1:],
                    alpha=0.2, color='#2C3E50')
    
    # Reference slopes
    k_ref = k[2:len(k)//3]
    ax.loglog(k_ref, mean_spectrum[3] * (k_ref / k_ref[0])**(-4), 
              'r--', alpha=0.6, linewidth=0.8, label=r'$k^{-4}$')
    ax.loglog(k_ref, mean_spectrum[3] * (k_ref / k_ref[0])**(-2),
              'g--', alpha=0.6, linewidth=0.8, label=r'$k^{-2}$')
    
    ax.set_xlabel(r'Wavenumber $k$')
    ax.set_ylabel(r'Power Spectral Density $|\hat{u}_k|^2$')
    ax.set_title(r'KS Spatial Energy Spectrum ($t = %d$)' % snap_t)
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.15, which='both')
    
    _save_fig(fig, os.path.join(output_dir, 'ks_spatial_spectrum'))


# ===========================================================================
# Figure 4: Spatial Profile Snapshots
# ===========================================================================
def fig_spatial_profiles(h5_path, output_dir, traj_idx=0, n_snaps=5):
    """
    Overlaid spatial profiles u(x) at several time instances,
    showing how the waveform evolves and breaks into chaos.
    """
    with h5py.File(h5_path, 'r') as f:
        T_total = f['train'].shape[1]
        Nx = f['train'].shape[2]
        t_indices = np.linspace(0, min(T_total - 1, 600), n_snaps, dtype=int)
        profiles = [np.array(f['train'][traj_idx, t, :], dtype=np.float32) for t in t_indices]
    
    x = np.linspace(0, 1, Nx)
    
    fig, ax = plt.subplots(1, 1, figsize=(DOUBLE_COL * 0.65, SINGLE_COL * 0.9))
    
    colors = plt.cm.cividis(np.linspace(0.1, 0.95, n_snaps))
    
    for i, (t, prof) in enumerate(zip(t_indices, profiles)):
        offset = i * 0.3  # vertical offset for clarity
        ax.plot(x, prof + offset, color=colors[i], linewidth=0.8,
                label=f'$t = {t}$')
    
    ax.set_xlabel(r'$x / L$')
    ax.set_ylabel(r'$u(x, t)$ + offset')
    ax.set_title('KS Spatial Profiles at Selected Times')
    ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
    ax.grid(True, alpha=0.12)
    
    _save_fig(fig, os.path.join(output_dir, 'ks_spatial_profiles'))


# ===========================================================================
# Figure 5: Temporal Autocorrelation
# ===========================================================================
def fig_temporal_autocorrelation(h5_path, output_dir, n_traj=10, max_lag=200, x_point=256):
    """
    Temporal autocorrelation function at a fixed spatial point,
    averaged over multiple trajectories. Reveals the decorrelation timescale.
    """
    with h5py.File(h5_path, 'r') as f:
        n_total = f['train'].shape[0]
        indices = np.linspace(0, min(n_total - 1, 5000), n_traj, dtype=int)
        
        autocorrs = []
        for i in indices:
            signal = np.array(f['train'][i, :max_lag * 3, x_point], dtype=np.float32)
            signal = signal - signal.mean()
            norm_val = np.sum(signal**2)
            if norm_val < 1e-10:
                continue
            acf = np.correlate(signal, signal, mode='full')
            acf = acf[len(acf)//2:]  # positive lags only
            acf = acf[:max_lag] / norm_val
            autocorrs.append(acf)
    
    mean_acf = np.mean(autocorrs, axis=0)
    std_acf = np.std(autocorrs, axis=0)
    lags = np.arange(len(mean_acf))
    
    fig, ax = plt.subplots(1, 1, figsize=(SINGLE_COL * 1.3, SINGLE_COL * 0.85))
    
    ax.plot(lags, mean_acf, color='#8E44AD', linewidth=1.2)
    ax.fill_between(lags, mean_acf - std_acf, mean_acf + std_acf,
                    alpha=0.2, color='#8E44AD')
    ax.axhline(y=0, color='k', linewidth=0.5, linestyle='-')
    ax.axhline(y=1/np.e, color='gray', linewidth=0.6, linestyle='--', 
               label=r'$1/e$ threshold')
    
    ax.set_xlabel(r'Lag $\tau$')
    ax.set_ylabel(r'$C(\tau)$')
    ax.set_title(f'Temporal Autocorrelation ($x = {x_point}$)')
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.15)
    ax.set_xlim(0, max_lag)
    ax.set_ylim(-0.3, 1.05)
    
    _save_fig(fig, os.path.join(output_dir, 'ks_temporal_autocorrelation'))


# ===========================================================================
# Figure 6: Amplitude Distribution
# ===========================================================================
def fig_amplitude_distribution(h5_path, output_dir, n_traj=500):
    """
    Histogram of field amplitudes u(x,t) across the dataset,
    showing the non-Gaussian nature of KS turbulence.
    """
    with h5py.File(h5_path, 'r') as f:
        n_total = f['train'].shape[0]
        n_sample = min(n_traj, n_total)
        indices = np.random.default_rng(42).choice(n_total, n_sample, replace=False)
        
        # Sample: grab one snapshot per trajectory at t=100
        vals = []
        for i in indices:
            snap = np.array(f['train'][i, 100, :], dtype=np.float32)
            vals.append(snap[::4])  # subsample spatially for efficiency
    
    vals = np.concatenate(vals)
    
    fig, ax = plt.subplots(1, 1, figsize=(SINGLE_COL * 1.2, SINGLE_COL * 0.85))
    
    ax.hist(vals, bins=100, density=True, color='#1ABC9C', edgecolor='white',
            linewidth=0.3, alpha=0.9)
    
    # Overlay Gaussian reference
    mu, sigma = vals.mean(), vals.std()
    x_gauss = np.linspace(vals.min(), vals.max(), 300)
    gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_gauss - mu) / sigma)**2)
    ax.plot(x_gauss, gauss, 'k--', linewidth=1.0, alpha=0.7, label=r'Gaussian $\mathcal{N}(\mu, \sigma^2)$')
    
    ax.set_xlabel(r'$u(x, t)$')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'KS Field Amplitude Distribution ($N = {n_sample}$ traj.)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.12)
    
    # Annotate skewness & kurtosis
    from scipy.stats import skew, kurtosis
    sk = skew(vals)
    ku = kurtosis(vals)
    ax.text(0.97, 0.92, f'Skew = {sk:.3f}\nKurtosis = {ku:.3f}',
            transform=ax.transAxes, fontsize=7, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    _save_fig(fig, os.path.join(output_dir, 'ks_amplitude_distribution'))


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="KS Dataset Publication Visualizations")
    parser.add_argument('--h5_path', type=str, default='./datasets/KS_dataset/KS_ML_DATASET.h5')
    parser.add_argument('--output_dir', type=str, default='./figures/ks')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  Generating ICML-Quality Visualizations — Kuramoto-Sivashinsky")
    print(f"{'='*60}")
    print(f"  Data:   {args.h5_path}")
    print(f"  Output: {args.output_dir}\n")
    
    fig_spatiotemporal_heatmap(args.h5_path, args.output_dir)
    fig_multi_trajectory(args.h5_path, args.output_dir)
    fig_spatial_spectrum(args.h5_path, args.output_dir)
    fig_spatial_profiles(args.h5_path, args.output_dir)
    fig_temporal_autocorrelation(args.h5_path, args.output_dir)
    fig_amplitude_distribution(args.h5_path, args.output_dir)
    
    print(f"\n  ✓ All KS visualizations complete → {args.output_dir}/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

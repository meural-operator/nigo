"""
visualize_bc_dataset.py — ICML Publication-Quality Visualizations for the Bluff-Body Cylinder Flow Dataset.

Generates:
  1. Spatiotemporal Evolution Panel — velocity magnitude heatmap showing temporal progression
  2. Vorticity Field Panel — derived ω = ∂v/∂x − ∂u/∂y for selected snapshots
  3. Spectral Energy Density — radially-averaged 2D power spectrum
  4. Dataset Statistics — distribution of Reynolds numbers, velocity magnitudes
  5. Kinetic Energy Evolution — temporal decay/oscillation of KE across multiple cases

All figures saved in BOTH PNG (300dpi) and PDF (vector) format.

Usage:
  conda run -n cfd python scripts/visualize_bc_dataset.py
  conda run -n cfd python scripts/visualize_bc_dataset.py --data_root ./datasets/bc --output_dir ./figures/bc
"""

import sys
import os
import glob
import argparse
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ===========================================================================
# Global ICML-grade style configuration
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
    'lines.linewidth': 1.2,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': False,
    'figure.constrained_layout.use': True,
})

# ICML column widths
SINGLE_COL = 3.25   # inches
DOUBLE_COL = 6.75   # inches
TEXT_HEIGHT = 9.0    # inches

# Professional colormaps
VELOCITY_CMAP = 'RdBu_r'
VORTICITY_CMAP = 'seismic'
SPECTRUM_CMAP = 'inferno'


def _save_fig(fig, path_stem):
    """Save figure in both PNG and PDF."""
    fig.savefig(f"{path_stem}.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(f"{path_stem}.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {path_stem}.png  &  {path_stem}.pdf")


def _add_colorbar(im, ax, label=''):
    """Add a slim, publication-quality colorbar."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.04)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return cb


def load_case(case_dir):
    """Load u, v, and metadata from a case directory."""
    u = np.load(os.path.join(case_dir, 'u.npy')).astype(np.float32)  # (T, H, W)
    v = np.load(os.path.join(case_dir, 'v.npy')).astype(np.float32)
    meta = {}
    meta_path = os.path.join(case_dir, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return u, v, meta


# ===========================================================================
# Figure 1: Spatiotemporal Evolution Panel
# ===========================================================================
def fig_spatiotemporal_evolution(data_root, output_dir, n_snapshots=6):
    """
    Shows velocity magnitude |V| = sqrt(u²+v²) at evenly-spaced time steps
    for a representative case, as a single-row panel.
    """
    cases = sorted(glob.glob(os.path.join(data_root, 'case*')))
    case_dir = cases[len(cases) // 2]  # pick a mid-range case
    u, v, meta = load_case(case_dir)
    
    T = u.shape[0]
    vel_mag = np.sqrt(u**2 + v**2)
    
    # Select evenly-spaced timesteps
    t_indices = np.linspace(0, T - 1, n_snapshots, dtype=int)
    
    fig, axes = plt.subplots(1, n_snapshots, figsize=(DOUBLE_COL, 1.5))
    
    vmin, vmax = vel_mag.min(), vel_mag.max()
    
    for i, t in enumerate(t_indices):
        im = axes[i].imshow(vel_mag[t], cmap=VELOCITY_CMAP, origin='lower',
                            vmin=vmin, vmax=vmax, aspect='equal')
        axes[i].set_title(f'$t = {t}$', fontsize=9, pad=3)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    # Single shared colorbar at the right edge
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label(r'$|\mathbf{v}|$', fontsize=9)
    cb.ax.tick_params(labelsize=7)
    
    Re = meta.get('Re', '?')
    fig.suptitle(f'Velocity Magnitude Evolution (Re = {Re})', fontsize=11, y=1.02)
    
    _save_fig(fig, os.path.join(output_dir, 'bc_spatiotemporal_evolution'))


# ===========================================================================
# Figure 2: Vorticity Field Panel
# ===========================================================================
def fig_vorticity_snapshots(data_root, output_dir, n_snapshots=4):
    """
    Computes vorticity ω = ∂v/∂x − ∂u/∂y via central differences
    and displays it at selected time steps.
    """
    cases = sorted(glob.glob(os.path.join(data_root, 'case*')))
    case_dir = cases[len(cases) // 2]
    u, v, meta = load_case(case_dir)
    
    T = u.shape[0]
    t_indices = np.linspace(0, T - 1, n_snapshots, dtype=int)
    
    fig, axes = plt.subplots(1, n_snapshots, figsize=(DOUBLE_COL, 1.9))
    
    vort_all = []
    for t in t_indices:
        dvdx = np.gradient(v[t], axis=1)
        dudy = np.gradient(u[t], axis=0)
        omega = dvdx - dudy
        vort_all.append(omega)
    
    vmax_abs = max(np.abs(w).max() for w in vort_all)
    
    for i, (t, omega) in enumerate(zip(t_indices, vort_all)):
        im = axes[i].imshow(omega, cmap=VORTICITY_CMAP, origin='lower',
                            vmin=-vmax_abs, vmax=vmax_abs, aspect='equal')
        axes[i].set_title(f'$t = {t}$', fontsize=9, pad=3)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    cbar_ax = fig.add_axes([0.92, 0.12, 0.012, 0.76])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label(r'$\omega_z$', fontsize=9)
    cb.ax.tick_params(labelsize=7)
    
    Re = meta.get('Re', '?')
    fig.suptitle(rf'Vorticity Field $\omega_z = \partial v/\partial x - \partial u/\partial y$ (Re = {Re})',
                 fontsize=11, y=1.04)
    
    _save_fig(fig, os.path.join(output_dir, 'bc_vorticity_snapshots'))


# ===========================================================================
# Figure 3: Radially-Averaged Power Spectrum
# ===========================================================================
def fig_spectral_analysis(data_root, output_dir, n_cases=5):
    """
    Computes the radially-averaged 2D power spectrum of the velocity magnitude
    for multiple cases, overlaying the -5/3 Kolmogorov slope.
    """
    cases = sorted(glob.glob(os.path.join(data_root, 'case*')))
    selected = np.linspace(0, len(cases) - 1, n_cases, dtype=int)
    
    fig, ax = plt.subplots(1, 1, figsize=(SINGLE_COL * 1.2, SINGLE_COL))
    
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_cases))
    
    for ci, idx in enumerate(selected):
        u, v, meta = load_case(cases[idx])
        vel_mag = np.sqrt(u[0]**2 + v[0]**2)  # initial frame
        
        # 2D FFT -> radial average
        fft2 = np.fft.fft2(vel_mag)
        power = np.abs(np.fft.fftshift(fft2))**2
        H, W = power.shape
        cy, cx = H // 2, W // 2
        Y, X = np.ogrid[:H, :W]
        R = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
        max_r = min(cy, cx)
        
        radial_power = np.zeros(max_r)
        counts = np.zeros(max_r)
        for r in range(max_r):
            mask = R == r
            radial_power[r] = np.sum(power[mask])
            counts[r] = np.sum(mask)
        radial_power = radial_power / (counts + 1e-10)
        
        k = np.arange(1, max_r)
        Re = meta.get('Re', f'Case {idx}')
        ax.loglog(k, radial_power[1:], color=colors[ci], alpha=0.85,
                  label=f'Re = {Re}', linewidth=1.0)
    
    # Reference Kolmogorov -5/3 slope
    k_ref = np.arange(2, max_r // 2)
    slope_ref = radial_power[2] * (k_ref / k_ref[0])**(-5.0/3.0)
    ax.loglog(k_ref, slope_ref, 'k--', alpha=0.5, linewidth=0.8, label=r'$k^{-5/3}$')
    
    ax.set_xlabel(r'Wavenumber $k$')
    ax.set_ylabel(r'$E(k)$')
    ax.set_title('Radially-Averaged Energy Spectrum')
    ax.legend(fontsize=7, loc='lower left', framealpha=0.8)
    ax.grid(True, alpha=0.15, which='both')
    
    _save_fig(fig, os.path.join(output_dir, 'bc_spectral_analysis'))


# ===========================================================================
# Figure 4: Dataset Statistics (Distributions)
# ===========================================================================
def fig_dataset_statistics(data_root, output_dir):
    """
    Shows (a) Reynolds number distribution, (b) velocity magnitude histogram,
    and (c) condition parameter scatter.
    """
    cases = sorted(glob.glob(os.path.join(data_root, 'case*')))
    
    re_vals = []
    vel_mags_flat = []
    inlet_vels = []
    radii = []
    
    for c in cases:
        _, _, meta = load_case(c)
        Re = meta.get('Re', None)
        if Re is not None:
            re_vals.append(float(Re))
        inlet_vels.append(float(meta.get('inlet_velocity', 0)))
        radii.append(float(meta.get('radius', 0)))
    
    # Sample velocity magnitudes from a few cases
    sample_cases = cases[::max(1, len(cases) // 10)]
    for c in sample_cases:
        u, v, _ = load_case(c)
        vm = np.sqrt(u[0]**2 + v[0]**2).flatten()
        vel_mags_flat.extend(vm[::10].tolist())  # subsample for efficiency
    
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.0))
    
    # (a) Reynolds number distribution
    if re_vals:
        axes[0].hist(re_vals, bins=15, color='#2E86C1', edgecolor='white',
                     linewidth=0.5, alpha=0.9)
        axes[0].set_xlabel(r'$\mathrm{Re}$')
        axes[0].set_ylabel('Count')
        axes[0].set_title(r'(a) Reynolds Number', fontsize=10)
    
    # (b) Velocity magnitude distribution
    axes[1].hist(vel_mags_flat, bins=60, color='#E74C3C', edgecolor='white',
                 linewidth=0.3, alpha=0.9, density=True)
    axes[1].set_xlabel(r'$|\mathbf{v}|$')
    axes[1].set_ylabel('Density')
    axes[1].set_title(r'(b) Velocity Distribution', fontsize=10)
    
    # (c) Condition parameter scatter
    if re_vals and inlet_vels:
        sc = axes[2].scatter(re_vals, inlet_vels, c=radii, cmap='plasma',
                             s=20, alpha=0.8, edgecolors='k', linewidth=0.3)
        axes[2].set_xlabel(r'$\mathrm{Re}$')
        axes[2].set_ylabel(r'$u_{\rm inlet}$')
        axes[2].set_title(r'(c) Condition Space', fontsize=10)
        cb = plt.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.04)
        cb.set_label(r'$r$', fontsize=8)
        cb.ax.tick_params(labelsize=7)
    
    _save_fig(fig, os.path.join(output_dir, 'bc_dataset_statistics'))


# ===========================================================================
# Figure 5: Kinetic Energy Temporal Evolution
# ===========================================================================
def fig_kinetic_energy_evolution(data_root, output_dir, n_cases=8):
    """
    Shows how the spatially-averaged kinetic energy KE = 0.5*(u²+v²) evolves 
    over time across different simulation cases, revealing distinct dynamical regimes.
    """
    cases = sorted(glob.glob(os.path.join(data_root, 'case*')))
    selected = np.linspace(0, len(cases) - 1, min(n_cases, len(cases)), dtype=int)
    
    fig, ax = plt.subplots(1, 1, figsize=(SINGLE_COL * 1.3, SINGLE_COL * 0.9))
    
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(selected)))
    
    for ci, idx in enumerate(selected):
        u, v, meta = load_case(cases[idx])
        ke = 0.5 * np.mean(u**2 + v**2, axis=(1, 2))  # (T,)
        t = np.arange(len(ke))
        Re = meta.get('Re', f'{idx}')
        ax.plot(t, ke, color=colors[ci], alpha=0.85, label=f'Re={Re}')
    
    ax.set_xlabel(r'Time step $t$')
    ax.set_ylabel(r'$\langle KE \rangle_{x,y}$')
    ax.set_title('Kinetic Energy Temporal Evolution')
    ax.legend(fontsize=6, ncol=2, loc='upper right', framealpha=0.8)
    ax.grid(True, alpha=0.15)
    
    _save_fig(fig, os.path.join(output_dir, 'bc_kinetic_energy_evolution'))


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="BC Dataset Publication Visualizations")
    parser.add_argument('--data_root', type=str, default='./datasets/bc')
    parser.add_argument('--output_dir', type=str, default='./figures/bc')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  Generating ICML-Quality Visualizations — Cylinder Flow")
    print(f"{'='*60}")
    print(f"  Data:   {args.data_root}")
    print(f"  Output: {args.output_dir}\n")
    
    fig_spatiotemporal_evolution(args.data_root, args.output_dir)
    fig_vorticity_snapshots(args.data_root, args.output_dir)
    fig_spectral_analysis(args.data_root, args.output_dir)
    fig_dataset_statistics(args.data_root, args.output_dir)
    fig_kinetic_energy_evolution(args.data_root, args.output_dir)
    
    print(f"\n  ✓ All BC visualizations complete → {args.output_dir}/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

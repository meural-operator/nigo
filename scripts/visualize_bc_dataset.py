"""
visualize_bc_dataset.py — ICML Publication-Quality Visualizations for the
Bluff-Body Cylinder Flow Dataset.

Generates:
  1. Spatiotemporal Evolution Panel — velocity magnitude at evenly-spaced steps
  2. Vorticity Field Panel — ω = ∂v/∂x − ∂u/∂y for selected snapshots
  3. Spectral Energy Density — radially-averaged 2D power spectrum
  4. Dataset Statistics — Reynolds number, velocity, and condition distributions
  5. Kinetic Energy Evolution — temporal decay/oscillation across cases

All figures saved as PNG (300 dpi raster) and PDF (LaTeX-rendered vector).

Usage:
  conda activate cfd
  python scripts/visualize_bc_dataset.py
  python scripts/visualize_bc_dataset.py --data_root ./datasets/bc --output_dir ./figures/bc
"""

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Windows Conda DLL workaround for Python >= 3.8
if os.name == 'nt' and 'CONDA_PREFIX' in os.environ:
    dll_path = os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'bin')
    if os.path.exists(dll_path):
        try: os.add_dll_directory(dll_path)
        except Exception: pass

import glob
import argparse
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ═══════════════════════════════════════════════════════════════════════════════
# Backend setup — try LaTeX (pgf) for PDFs, fall back to Agg
# ═══════════════════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use('Agg')  # start with Agg for PNG
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

LATEX_AVAILABLE = False
try:
    # Test if pdflatex is available
    import subprocess
    result = subprocess.run(['pdflatex', '--version'],
                            capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        LATEX_AVAILABLE = True
        print("  [viz] LaTeX (pdflatex) detected — PDFs will be LaTeX-rendered.")
except Exception:
    pass

if not LATEX_AVAILABLE:
    print("  [viz] LaTeX not found — PDFs will use matplotlib mathtext.")


# ═══════════════════════════════════════════════════════════════════════════════
# ICML Style — shared rcParams (NO constrained_layout; we use tight_layout)
# ═══════════════════════════════════════════════════════════════════════════════
ICML_RC = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'lines.linewidth': 1.2,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'axes.grid': False,
    'figure.constrained_layout.use': False,  # avoid conflicts with manual axes
}
plt.rcParams.update(ICML_RC)

# ICML column widths (inches)
SINGLE_COL = 3.25
DOUBLE_COL = 6.75

# Colormaps
VELOCITY_CMAP = 'RdBu_r'
VORTICITY_CMAP = 'seismic'


def _save_fig(fig, path_stem):
    """Save figure as PNG (Agg, 300 dpi) and PDF (LaTeX pgf or fallback)."""
    # --- PNG (always Agg) ---
    fig.savefig(f"{path_stem}.png", dpi=300, bbox_inches='tight', pad_inches=0.08)

    # --- PDF (LaTeX if available) ---
    if LATEX_AVAILABLE:
        # Temporarily switch to pgf backend for this save
        try:
            old_backend = matplotlib.get_backend()
            plt.rcParams.update({
                "pgf.texsystem": "pdflatex",
                "text.usetex": True,
                "pgf.preamble": "\n".join([
                    r"\usepackage[utf8]{inputenc}",
                    r"\usepackage[T1]{fontenc}",
                    r"\usepackage{amsmath,amssymb}",
                ]),
            })
            fig.savefig(f"{path_stem}.pdf", backend='pgf',
                        bbox_inches='tight', pad_inches=0.08)
            plt.rcParams.update({"text.usetex": False})
        except Exception as e:
            # Fallback: save PDF without LaTeX
            plt.rcParams.update({"text.usetex": False})
            fig.savefig(f"{path_stem}.pdf", bbox_inches='tight', pad_inches=0.08)
            print(f"    (LaTeX PDF failed, used fallback: {e})")
    else:
        fig.savefig(f"{path_stem}.pdf", bbox_inches='tight', pad_inches=0.08)

    plt.close(fig)
    print(f"  Saved: {path_stem}.png  &  {path_stem}.pdf")


def load_case(case_dir):
    """Load u, v, and metadata from a case directory.

    Reads 'case.json' (primary) or 'meta.json' (fallback).
    Computes Re = density * vel_in * D / viscosity if not present.
    """
    u = np.load(os.path.join(case_dir, 'u.npy')).astype(np.float32)
    v = np.load(os.path.join(case_dir, 'v.npy')).astype(np.float32)
    meta = {}
    for fname in ['case.json', 'meta.json']:
        fpath = os.path.join(case_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                meta = json.load(f)
            break

    # Compute Reynolds number if not explicitly stored
    if 'Re' not in meta:
        rho = float(meta.get('density', 0))
        U = float(meta.get('vel_in', meta.get('inlet_velocity', 0)))
        r = float(meta.get('radius', 0))
        mu = float(meta.get('viscosity', 1))
        D = 2.0 * r  # characteristic length = cylinder diameter
        if mu > 0 and D > 0:
            meta['Re'] = rho * U * D / mu

    # Normalize key names for downstream use
    if 'vel_in' in meta and 'inlet_velocity' not in meta:
        meta['inlet_velocity'] = meta['vel_in']

    return u, v, meta


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Spatiotemporal Velocity Evolution
# ═══════════════════════════════════════════════════════════════════════════════
def fig_spatiotemporal_evolution(data_root, output_dir, n_snapshots=6):
    """Velocity magnitude |V| at evenly-spaced time steps, single-row panel."""
    cases = sorted(glob.glob(os.path.join(data_root, 'case*')))
    case_dir = cases[len(cases) // 2]
    u, v, meta = load_case(case_dir)

    T = u.shape[0]
    vel_mag = np.sqrt(u**2 + v**2)
    t_indices = np.linspace(0, T - 1, n_snapshots, dtype=int)

    # Use GridSpec with space reserved for colorbar
    fig = plt.figure(figsize=(DOUBLE_COL, 1.8))
    gs = GridSpec(1, n_snapshots + 1, figure=fig,
                  width_ratios=[1]*n_snapshots + [0.05],
                  wspace=0.08)

    vmin, vmax = vel_mag.min(), vel_mag.max()
    im = None

    for i, t in enumerate(t_indices):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(vel_mag[t], cmap=VELOCITY_CMAP, origin='lower',
                       vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(f'$t = {t}$', fontsize=9, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])

    # Colorbar in its own GridSpec column — no overlap
    cax = fig.add_subplot(gs[0, -1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r'$|\mathbf{v}|$', fontsize=9)
    cb.ax.tick_params(labelsize=7)

    Re = meta.get('Re', '?')
    fig.suptitle(f'Velocity Magnitude Evolution (Re = {Re})',
                 fontsize=11, y=0.98)

    _save_fig(fig, os.path.join(output_dir, 'bc_spatiotemporal_evolution'))


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Vorticity Field Panel
# ═══════════════════════════════════════════════════════════════════════════════
def fig_vorticity_snapshots(data_root, output_dir, n_snapshots=4):
    r"""Vorticity ω = ∂v/∂x − ∂u/∂y at selected time steps."""
    cases = sorted(glob.glob(os.path.join(data_root, 'case*')))
    case_dir = cases[len(cases) // 2]
    u, v, meta = load_case(case_dir)

    T = u.shape[0]
    t_indices = np.linspace(0, T - 1, n_snapshots, dtype=int)

    vort_all = []
    for t in t_indices:
        dvdx = np.gradient(v[t], axis=1)
        dudy = np.gradient(u[t], axis=0)
        vort_all.append(dvdx - dudy)

    vmax_abs = max(np.abs(w).max() for w in vort_all)

    # GridSpec with colorbar column
    fig = plt.figure(figsize=(DOUBLE_COL, 2.2))
    gs = GridSpec(1, n_snapshots + 1, figure=fig,
                  width_ratios=[1]*n_snapshots + [0.05],
                  wspace=0.10)

    for i, (t, omega) in enumerate(zip(t_indices, vort_all)):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(omega, cmap=VORTICITY_CMAP, origin='lower',
                       vmin=-vmax_abs, vmax=vmax_abs, aspect='equal')
        ax.set_title(f'$t = {t}$', fontsize=9, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])

    cax = fig.add_subplot(gs[0, -1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r'$\omega_z$', fontsize=9)
    cb.ax.tick_params(labelsize=7)

    Re = meta.get('Re', '?')
    fig.suptitle(
        r'Vorticity $\omega_z = \partial v/\partial x - \partial u/\partial y$'
        f' (Re = {Re})',
        fontsize=11, y=0.98
    )

    _save_fig(fig, os.path.join(output_dir, 'bc_vorticity_snapshots'))


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Radially-Averaged Power Spectrum
# ═══════════════════════════════════════════════════════════════════════════════
def fig_spectral_analysis(data_root, output_dir, n_cases=5):
    """Radially-averaged 2D power spectrum with Kolmogorov -5/3 reference."""
    cases = sorted(glob.glob(os.path.join(data_root, 'case*')))
    selected = np.linspace(0, len(cases) - 1, n_cases, dtype=int)

    fig, ax = plt.subplots(1, 1, figsize=(SINGLE_COL * 1.25, SINGLE_COL * 1.0))

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_cases))

    last_radial = None
    last_max_r = 0
    for ci, idx in enumerate(selected):
        u, v, meta = load_case(cases[idx])
        vel_mag = np.sqrt(u[0]**2 + v[0]**2)

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
        last_radial = radial_power
        last_max_r = max_r

    # Kolmogorov -5/3 reference slope
    if last_radial is not None:
        k_ref = np.arange(2, last_max_r // 2)
        slope_ref = last_radial[2] * (k_ref / k_ref[0])**(-5.0/3.0)
        ax.loglog(k_ref, slope_ref, 'k--', alpha=0.5, linewidth=0.8,
                  label=r'$k^{-5/3}$')

    ax.set_xlabel(r'Wavenumber $k$')
    ax.set_ylabel(r'$E(k)$')
    ax.set_title('Radially-Averaged Energy Spectrum')
    ax.legend(fontsize=7, loc='lower left', framealpha=0.8)
    ax.grid(True, alpha=0.15, which='both')

    fig.tight_layout()
    _save_fig(fig, os.path.join(output_dir, 'bc_spectral_analysis'))


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Dataset Statistics
# ═══════════════════════════════════════════════════════════════════════════════
def fig_dataset_statistics(data_root, output_dir):
    """(a) Re distribution, (b) velocity histogram, (c) condition scatter."""
    cases = sorted(glob.glob(os.path.join(data_root, 'case*')))

    re_vals, inlet_vels, radii = [], [], []
    vel_mags_flat = []

    for c in cases:
        _, _, meta = load_case(c)
        Re = meta.get('Re', None)
        if Re is not None:
            re_vals.append(float(Re))
        inlet_vels.append(float(meta.get('inlet_velocity', 0)))
        radii.append(float(meta.get('radius', 0)))

    sample_cases = cases[::max(1, len(cases) // 10)]
    for c in sample_cases:
        u, v, _ = load_case(c)
        vm = np.sqrt(u[0]**2 + v[0]**2).flatten()
        vel_mags_flat.extend(vm[::10].tolist())

    # Use explicit tight_layout with enough height and padding
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.3))

    # (a) Reynolds number
    if re_vals:
        axes[0].hist(re_vals, bins=15, color='#2E86C1', edgecolor='white',
                     linewidth=0.5, alpha=0.9)
        axes[0].set_xlabel(r'$\mathrm{Re}$')
        axes[0].set_ylabel('Count')
        axes[0].set_title(r'(a) Reynolds Number', fontsize=10)

    # (b) Velocity magnitude
    axes[1].hist(vel_mags_flat, bins=60, color='#E74C3C', edgecolor='white',
                 linewidth=0.3, alpha=0.9, density=True)
    axes[1].set_xlabel(r'$|\mathbf{v}|$')
    axes[1].set_ylabel('Density')
    axes[1].set_title(r'(b) Velocity Distribution', fontsize=10)

    # (c) Condition scatter
    if re_vals and inlet_vels:
        sc = axes[2].scatter(re_vals, inlet_vels, c=radii, cmap='plasma',
                             s=20, alpha=0.8, edgecolors='k', linewidth=0.3)
        axes[2].set_xlabel(r'$\mathrm{Re}$')
        axes[2].set_ylabel(r'$u_{\rm inlet}$')
        axes[2].set_title(r'(c) Condition Space', fontsize=10)
        cb = plt.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.06)
        cb.set_label(r'$r$', fontsize=8)
        cb.ax.tick_params(labelsize=7)

    fig.tight_layout(w_pad=2.5)
    _save_fig(fig, os.path.join(output_dir, 'bc_dataset_statistics'))


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5: Kinetic Energy Temporal Evolution
# ═══════════════════════════════════════════════════════════════════════════════
def fig_kinetic_energy_evolution(data_root, output_dir, n_cases=8):
    """Spatially-averaged KE = 0.5*(u²+v²) over time for multiple cases."""
    cases = sorted(glob.glob(os.path.join(data_root, 'case*')))
    selected = np.linspace(0, len(cases) - 1, min(n_cases, len(cases)), dtype=int)

    fig, ax = plt.subplots(1, 1, figsize=(SINGLE_COL * 1.3, SINGLE_COL * 0.95))

    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(selected)))

    for ci, idx in enumerate(selected):
        u, v, meta = load_case(cases[idx])
        ke = 0.5 * np.mean(u**2 + v**2, axis=(1, 2))
        t = np.arange(len(ke))
        Re = meta.get('Re', f'{idx}')
        ax.plot(t, ke, color=colors[ci], alpha=0.85, label=f'Re={Re}')

    ax.set_xlabel(r'Time step $t$')
    ax.set_ylabel(r'$\langle KE \rangle_{x,y}$')
    ax.set_title('Kinetic Energy Temporal Evolution')
    ax.legend(fontsize=6, ncol=2, loc='upper right', framealpha=0.8)
    ax.grid(True, alpha=0.15)

    fig.tight_layout()
    _save_fig(fig, os.path.join(output_dir, 'bc_kinetic_energy_evolution'))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="BC Dataset Publication Visualizations")
    parser.add_argument('--data_root', type=str, default='./datasets/bc')
    parser.add_argument('--output_dir', type=str, default='./figures/bc')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ICML-Quality Visualizations — Cylinder Flow Dataset")
    print(f"{'='*60}")
    print(f"  Data:   {args.data_root}")
    print(f"  Output: {args.output_dir}\n")

    fig_spatiotemporal_evolution(args.data_root, args.output_dir)
    fig_vorticity_snapshots(args.data_root, args.output_dir)
    fig_spectral_analysis(args.data_root, args.output_dir)
    fig_dataset_statistics(args.data_root, args.output_dir)
    fig_kinetic_energy_evolution(args.data_root, args.output_dir)

    print(f"\n  All 5 figures generated -> {args.output_dir}/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

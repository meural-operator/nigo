"""
visualize_ablations.py — ICML Publication-Quality Ablation Visualizations.

Reads the per-model JSON files produced by `evaluate_ablations.py` and
generates 6 figures, each saved as both PNG (300 dpi) and PDF (LaTeX-rendered
vector graphics via the pgf backend).

Figures:
  1. Rollout MSE Curves         — Per-step MSE over horizon (stability proof)
  2. Energy Stability Traces    — ||z_t||^2 over time (Lyapunov verification)
  3. Spectral Comparison        — Radial energy spectra with k^{-5/3} reference
  4. Lyapunov Divergence        — Perturbation growth per ablation
  5. Relative L2 Error Curves   — Normalized error evolution
  6. Alpha/Beta Bar Chart       — Learned scaling parameters per ablation

Usage:
  conda run -n cfd python scripts/visualize_ablations.py
  conda run -n cfd python scripts/visualize_ablations.py --results_dir ./results
"""

import sys
import os
import json
import glob
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════════
# Attempt LaTeX rendering for PDFs; fall back gracefully if unavailable
# ═══════════════════════════════════════════════════════════════════════════════
LATEX_AVAILABLE = False
try:
    matplotlib.use('pgf')
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "pgf.preamble": "\n".join([
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{amsmath}",
            r"\usepackage{amssymb}",
        ]),
    })
    # Quick sanity check: try to render a tiny figure
    _fig, _ax = plt.subplots(figsize=(0.5, 0.5))
    _ax.set_title(r"$\alpha$")
    _fig.savefig(os.devnull, format='pgf')
    plt.close(_fig)
    LATEX_AVAILABLE = True
    print("  [viz] LaTeX rendering enabled (pgf backend).")
except Exception:
    matplotlib.use('Agg')
    plt.rcParams.update({"text.usetex": False})
    print("  [viz] LaTeX not available; using Agg backend with mathtext.")

# ═══════════════════════════════════════════════════════════════════════════════
# ICML Style Configuration
# ═══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.titlesize': 13,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'lines.linewidth': 1.4,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': False,
    'figure.constrained_layout.use': True,
})

SINGLE_COL = 3.25   # ICML single column width (inches)
DOUBLE_COL = 6.75   # ICML double column width (inches)

# Colorblind-friendly palette (Wong 2011)
COLORS = {
    "Baseline":          "#0072B2",
    "NoSkew":            "#D55E00",
    "NoDissipative":     "#CC79A7",
    "DenseGenerator":    "#009E73",
    "NoRefiner":         "#F0E442",
    "UnscaledGenerator": "#56B4E9",
}

MARKERS = {
    "Baseline":          "o",
    "NoSkew":            "s",
    "NoDissipative":     "^",
    "DenseGenerator":    "D",
    "NoRefiner":         "v",
    "UnscaledGenerator": "P",
}

# Pretty names for LaTeX labels
PRETTY_NAMES = {
    "Baseline":          "TurboNIGO (Full)",
    "NoSkew":            r"w/o Skew-Symm.",
    "NoDissipative":     r"w/o Dissipative",
    "DenseGenerator":    "Dense Generator",
    "NoRefiner":         r"w/o Refiner",
    "UnscaledGenerator": r"w/o $\alpha/\beta$",
}


def _save_fig(fig, path_stem):
    """Save figure as PNG and PDF (LaTeX or fallback)."""
    fig.savefig(f"{path_stem}.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
    if LATEX_AVAILABLE:
        fig.savefig(f"{path_stem}.pdf", backend='pgf', bbox_inches='tight', pad_inches=0.05)
    else:
        fig.savefig(f"{path_stem}.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {path_stem}.png & .pdf")


def load_all_results(results_dir):
    """Loads all per-model JSON files from the quantitative results directory."""
    json_dir = os.path.join(results_dir, "ablation_results", "quantitative", "per_model")
    data = {}
    if not os.path.isdir(json_dir):
        print(f"  ⚠ Directory not found: {json_dir}")
        return data
    for path in sorted(glob.glob(os.path.join(json_dir, "*.json"))):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            data[name] = json.load(f)
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Rollout MSE Curves
# ═══════════════════════════════════════════════════════════════════════════════
def fig_rollout_mse(all_data, output_dir):
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.4, SINGLE_COL * 0.95))

    for name, d in all_data.items():
        curve = d.get("rollout_mse_curve", [])
        if not curve:
            continue
        t = np.arange(1, len(curve) + 1)
        color = COLORS.get(name, "#333333")
        marker = MARKERS.get(name, "o")
        label = PRETTY_NAMES.get(name, name)
        ax.semilogy(t, curve, color=color, marker=marker, markevery=max(1, len(t) // 8),
                     markersize=4, label=label)

    ax.set_xlabel(r"Rollout Step $t$")
    ax.set_ylabel(r"MSE$(t)$")
    ax.set_title("Autoregressive Rollout Error")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.85, ncol=1)
    ax.grid(True, alpha=0.15, which='both')

    _save_fig(fig, os.path.join(output_dir, "rollout_mse_comparison"))


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Energy Stability Traces
# ═══════════════════════════════════════════════════════════════════════════════
def fig_energy_stability(all_data, output_dir):
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.4, SINGLE_COL * 0.95))

    for name, d in all_data.items():
        trace = d.get("energy_trace", [])
        if not trace:
            continue
        t = np.arange(1, len(trace) + 1)
        color = COLORS.get(name, "#333333")
        label = PRETTY_NAMES.get(name, name)
        # Normalize to initial energy for fair comparison
        norm_trace = np.array(trace) / (trace[0] + 1e-10)
        ax.plot(t, norm_trace, color=color, label=label)

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.7, alpha=0.7, label="$E_0$ (Initial)")
    ax.set_xlabel(r"Time Step $t$")
    ax.set_ylabel(r"$\|z_t\|^2 \, / \, \|z_0\|^2$")
    ax.set_title(r"Latent Energy Stability")
    ax.legend(fontsize=7, loc="best", framealpha=0.85)
    ax.grid(True, alpha=0.15)

    _save_fig(fig, os.path.join(output_dir, "energy_stability"))


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Spectral Comparison
# ═══════════════════════════════════════════════════════════════════════════════
def fig_spectrum_comparison(all_data, output_dir):
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.4, SINGLE_COL * 0.95))

    # Plot ground truth spectrum from first available model's data
    gt_plotted = False
    for name, d in all_data.items():
        spec_gt = d.get("spectrum_gt", [])
        spec_pred = d.get("spectrum_pred", [])
        if not spec_pred:
            continue

        k = np.arange(1, len(spec_pred) + 1)
        color = COLORS.get(name, "#333333")
        label = PRETTY_NAMES.get(name, name)
        ax.loglog(k, spec_pred, color=color, alpha=0.85, label=label)

        if not gt_plotted and spec_gt:
            k_gt = np.arange(1, len(spec_gt) + 1)
            ax.loglog(k_gt, spec_gt, 'k-', linewidth=2.0, alpha=0.8, label="Ground Truth")
            gt_plotted = True

    # Reference slope
    if gt_plotted:
        k_ref = np.arange(2, len(spec_gt) // 2)
        amp = spec_gt[2] if len(spec_gt) > 2 else 1.0
        slope = amp * (k_ref / k_ref[0]) ** (-5.0 / 3.0)
        ax.loglog(k_ref, slope, 'k--', alpha=0.4, linewidth=0.8, label=r"$k^{-5/3}$")

    ax.set_xlabel(r"Wavenumber $k$")
    ax.set_ylabel(r"$E(k)$")
    ax.set_title("Radially-Averaged Energy Spectrum")
    ax.legend(fontsize=6, loc="lower left", framealpha=0.85, ncol=2)
    ax.grid(True, alpha=0.15, which='both')

    _save_fig(fig, os.path.join(output_dir, "spectrum_comparison"))


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Lyapunov Divergence
# ═══════════════════════════════════════════════════════════════════════════════
def fig_lyapunov_divergence(all_data, output_dir):
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.4, SINGLE_COL * 0.95))

    for name, d in all_data.items():
        curve = d.get("lyapunov_curve", [])
        if not curve:
            continue
        t = np.arange(len(curve))
        color = COLORS.get(name, "#333333")
        label = PRETTY_NAMES.get(name, name)
        # Normalize by initial perturbation distance
        norm_curve = np.array(curve) / (curve[0] + 1e-10)
        ax.semilogy(t, norm_curve, color=color, label=label)

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.set_xlabel(r"Time Step $t$")
    ax.set_ylabel(r"$\|\delta(t)\| \, / \, \|\delta(0)\|$")
    ax.set_title("Perturbation Sensitivity (Lyapunov Proxy)")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.85)
    ax.grid(True, alpha=0.15, which='both')

    _save_fig(fig, os.path.join(output_dir, "lyapunov_divergence"))


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5: Relative L2 Error Curves
# ═══════════════════════════════════════════════════════════════════════════════
def fig_relative_l2(all_data, output_dir):
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.4, SINGLE_COL * 0.95))

    for name, d in all_data.items():
        curve = d.get("rel_l2_curve", [])
        if not curve:
            continue
        t = np.arange(1, len(curve) + 1)
        color = COLORS.get(name, "#333333")
        marker = MARKERS.get(name, "o")
        label = PRETTY_NAMES.get(name, name)
        ax.plot(t, curve, color=color, marker=marker, markevery=max(1, len(t) // 8),
                markersize=4, label=label)

    ax.set_xlabel(r"Prediction Step $t$")
    ax.set_ylabel(r"$\|u_{\mathrm{pred}} - u_{\mathrm{gt}}\|_2 \, / \, \|u_{\mathrm{gt}}\|_2$")
    ax.set_title(r"Relative $L^2$ Error")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.85)
    ax.grid(True, alpha=0.15)

    _save_fig(fig, os.path.join(output_dir, "relative_l2_error"))


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6: Alpha/Beta Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════
def fig_alpha_beta_bars(all_data, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL * 0.7, SINGLE_COL * 0.85))

    names = list(all_data.keys())
    alphas = [all_data[n].get("alpha_mean", 0) for n in names]
    betas = [all_data[n].get("beta_mean", 0) for n in names]
    pretty = [PRETTY_NAMES.get(n, n) for n in names]
    colors = [COLORS.get(n, "#333333") for n in names]

    x = np.arange(len(names))
    width = 0.65

    ax1.barh(x, alphas, height=width, color=colors, edgecolor='white', linewidth=0.5)
    ax1.set_yticks(x)
    ax1.set_yticklabels(pretty, fontsize=7)
    ax1.set_xlabel(r"$\bar{\alpha}$")
    ax1.set_title(r"(a) Mean $\alpha$ (Mixing Scale)")
    ax1.invert_yaxis()

    ax2.barh(x, betas, height=width, color=colors, edgecolor='white', linewidth=0.5)
    ax2.set_yticks(x)
    ax2.set_yticklabels(pretty, fontsize=7)
    ax2.set_xlabel(r"$\bar{\beta}$")
    ax2.set_title(r"(b) Mean $\beta$ (Dissipation Scale)")
    ax2.invert_yaxis()

    _save_fig(fig, os.path.join(output_dir, "alpha_beta_dynamics"))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Ablation Visualization (ICML-Quality)")
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: <results_dir>/ablation_results/qualitative)')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "ablation_results", "qualitative")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  TurboNIGO Ablation Visualization")
    print(f"{'='*60}")
    print(f"  Results: {args.results_dir}")
    print(f"  Output:  {args.output_dir}\n")

    all_data = load_all_results(args.results_dir)
    if not all_data:
        print("  ⚠ No evaluation data found. Run evaluate_ablations.py first.")
        return

    print(f"  Found {len(all_data)} models: {list(all_data.keys())}\n")

    fig_rollout_mse(all_data, args.output_dir)
    fig_energy_stability(all_data, args.output_dir)
    fig_spectrum_comparison(all_data, args.output_dir)
    fig_lyapunov_divergence(all_data, args.output_dir)
    fig_relative_l2(all_data, args.output_dir)
    fig_alpha_beta_bars(all_data, args.output_dir)

    print(f"\n  ✓ All 6 ablation figures generated → {args.output_dir}/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

"""
visualize_ablations_sw.py — ICML Publication-Quality Visualizations for 2D Shallow Water.

Generates rigorous quantitative line plots from JSON logs and evaluates the models
qualitatively to produce an exquisite 4x3 wavefront comparison plot.
Outputs both DPI=300 PNGs and vector-embedded LaTeX PDFs without overlaps.
"""

import sys
import os
import json
import glob
import argparse
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from turbo_nigo.configs import get_args_and_config
from turbo_nigo.utils.misc import seed_everything
from scripts.train_unified import create_dataloaders
from scripts.evaluate_ablations_sw import load_trained_model

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
    _fig, _ax = plt.subplots(figsize=(0.5, 0.5))
    _ax.set_title(r"$\alpha$")
    _fig.savefig(os.devnull, format='pgf')
    plt.close(_fig)
    LATEX_AVAILABLE = True
    print("  [viz] LaTeX rendering natively enabled (pgf backend).")
except Exception:
    matplotlib.use('Agg')
    plt.rcParams.update({"text.usetex": False})
    print("  [viz] LaTeX unavailable; using Agg backend with mathtext (100% stable).")

# ═══════════════════════════════════════════════════════════════════════════════
# Publication Style Configuration
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
    'savefig.bbox': 'tight',  # Strictly enforces zero overlapping of text elements
    'savefig.pad_inches': 0.05,
    'axes.grid': False,
    'figure.constrained_layout.use': True,
})

COLORS = {
    "Baseline_MSE":      "#D55E00",  # Vermilion
    "Sobolev_H1":        "#CC79A7",  # Reddish Purple
    "Dual_Curriculum":   "#009E73",  # Bluish Green
}

MARKERS = {
    "Baseline_MSE":      "o",
    "Sobolev_H1":        "s",
    "Dual_Curriculum":   "^",
}

PRETTY_NAMES = {
    "Baseline_MSE":      "Baseline (Pure MSE)",
    "Sobolev_H1":        r"Sobolev ($H^1$)",
    "Dual_Curriculum":   "Dual Curriculum",
}

def _save_fig(fig, path_stem):
    """Save figure as PNG and Vector PDF optimally avoiding axis cutoffs."""
    fig.savefig(f"{path_stem}.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
    if LATEX_AVAILABLE:
        fig.savefig(f"{path_stem}.pdf", backend='pgf', bbox_inches='tight', pad_inches=0.05)
    else:
        fig.savefig(f"{path_stem}.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved High-Res Vector/Raster Pair: {path_stem}.[png|pdf]")

# ═══════════════════════════════════════════════════════════════════════════════
# Quantitative Plotting Functions
# ═══════════════════════════════════════════════════════════════════════════════

def fig_rollout_mse(all_data, output_dir):
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for name, d in all_data.items():
        curve = d.get("rollout_mse_curve", [])
        if not curve: continue
        t = np.arange(1, len(curve) + 1)
        ax.semilogy(t, curve, color=COLORS.get(name), marker=MARKERS.get(name),
                    markevery=3, markersize=5, label=PRETTY_NAMES.get(name))
                    
    ax.set_xlabel(r"Simulation Rollout Step $t$")
    ax.set_ylabel(r"Mean Squared Error (MSE)")
    ax.set_title("Autoregressive Trajectory Drift")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.2, which='both')
    _save_fig(fig, os.path.join(output_dir, "plot_sw_rollout_mse"))

def fig_relative_l2(all_data, output_dir):
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for name, d in all_data.items():
        curve = d.get("rel_l2_curve", [])
        if not curve: continue
        t = np.arange(1, len(curve) + 1)
        ax.plot(t, curve, color=COLORS.get(name), marker=MARKERS.get(name),
                markevery=3, markersize=5, label=PRETTY_NAMES.get(name))
                
    ax.set_xlabel(r"Simulation Rollout Step $t$")
    ax.set_ylabel(r"Relative $L^2$ Error ($\epsilon_{rel}$)")
    ax.set_title("Scale-Invariant Absolute Prediction Error")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    _save_fig(fig, os.path.join(output_dir, "plot_sw_relative_l2"))

def fig_energy_trace(all_data, output_dir):
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for name, d in all_data.items():
        trace = d.get("energy_trace", [])
        if not trace: continue
        # Trace is blocked, length is usually num_blocks + 1
        blocks = np.arange(len(trace))
        ax.plot(blocks, trace, color=COLORS.get(name), marker=MARKERS.get(name),
                markevery=max(1, len(blocks)//10), markersize=4, label=PRETTY_NAMES.get(name))
                
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label=r"Perfect Conservation ($V_0$)")
    ax.set_xlabel(r"Blocked Rollover ($\times 20$ steps)")
    ax.set_ylabel(r"Energy Variance Ratio ($E_t / E_0$)")
    ax.set_title("1,000 Step Unsupervised Mathematical Stability")
    ax.set_ylim(bottom=0.0) # Energy can't be negative, but can dissipate to 0
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    _save_fig(fig, os.path.join(output_dir, "plot_sw_energy_trace"))

# ═══════════════════════════════════════════════════════════════════════════════
# Qualitative Visual Matrix Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_qualitative_visuals(base_config, models_list, output_dir):
    print(f"\n[+] Rendering 4x3 Exact Physical Visualization Grid...")
    device = base_config.get("device", "cuda")
    
    # Extract one deterministic validation sample
    seed_everything(42)
    _, val_loader = create_dataloaders(base_config)
    u0, u_seq_gt, cond = next(iter(val_loader))
    
    u0_s = u0[0:1].to(device)
    u_seq_gt_s = u_seq_gt[0:1].to(device)
    cond_s = cond[0:1].to(device)
    seq_len = u_seq_gt_s.shape[1]
    
    time_steps = torch.arange(1, seq_len + 1).float().to(device)
    
    # Collate predictions dynamically to prevent caching massive datasets
    horizons = [4, 9, 19] # indices for t=5, t=10, t=20
    horizons_labels = ["t = 0.5s", "t = 1.0s", "t = 2.0s"]
    
    # Format: matrix[row_name][horizon_idx] = 2D numpy array
    grid_data = {"Ground Truth": [u_seq_gt_s[0, h, 0].cpu().numpy() for h in horizons]}
    
    for name in models_list:
        model = load_trained_model(name, base_config, device)
        if model is None: continue
        with torch.no_grad():
            u_pred, _, _, _, _, _ = model(u0_s, time_steps, cond_s)
        grid_data[PRETTY_NAMES.get(name)] = [u_pred[0, h, 0].cpu().numpy() for h in horizons]
        
    # Find universal min/max across all plotted matrices to identically lock colormaps
    v_min = min([np.min(arr) for row in grid_data.values() for arr in row])
    v_max = max([np.max(arr) for row in grid_data.values() for arr in row])
    
    # 4 rows, 3 cols
    rows = list(grid_data.keys())
    fig, axes = plt.subplots(len(rows), 3, figsize=(7.5, 9.5))
    
    for r_idx, r_name in enumerate(rows):
        for c_idx in range(3):
            ax = axes[r_idx, c_idx]
            im = ax.imshow(grid_data[r_name][c_idx], cmap='viridis', origin='lower', vmin=v_min, vmax=v_max)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Row titles on the far left
            if c_idx == 0:
                ax.set_ylabel(r_name, fontsize=12, fontweight='bold', labelpad=10)
            
            # Column titles on the top row
            if r_idx == 0:
                ax.set_title(horizons_labels[c_idx], fontsize=12, pad=10)

    # Attach single unified colorbar perfectly scaled beneath
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.046, pad=0.04)
    cbar.set_label("Fluid Height Amplitude (m)", fontsize=11)
    
    _save_fig(fig, os.path.join(output_dir, "visual_sw_qualitative_matrix"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='./results_sw')
    
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown
    base_config = get_args_and_config()
    base_config["dataset_type"] = "sw"
    
    eval_dir = os.path.join(args.results_dir, "evaluation_metrics")
    plot_dir = os.path.join(eval_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"\n{'='*60}\n  TurboNIGO SW Visualization Architect\n{'='*60}\n")
    
    # 1. Load Quantitative Curves
    all_data = {}
    for path in sorted(glob.glob(os.path.join(eval_dir, "*_metrics.json"))):
        name = os.path.basename(path).replace("_metrics.json", "")
        with open(path) as f:
            all_data[name] = json.load(f)
            
    if not all_data:
        print("  ⚠ No JSON evaluation curves found. Make sure evaluate_ablations_sw.py has generated them.")
        return
        
    print(f"[+] Loaded {len(all_data)} continuous metric curves.")
    
    # Generate Research Plots exactly formatted
    fig_rollout_mse(all_data, plot_dir)
    fig_relative_l2(all_data, plot_dir)
    fig_energy_trace(all_data, plot_dir)
    
    # 2. Extract Qualitative Images natively
    models_to_test = ["Baseline_MSE", "Sobolev_H1", "Dual_Curriculum"]
    valid_models = [m for m in models_to_test if m in all_data]
    
    if valid_models:
        generate_qualitative_visuals(base_config, valid_models, plot_dir)
        
    print(f"\n  ✓ Complete! Latex-compatible PDFs and PNGs exported to {plot_dir}/")

if __name__ == "__main__":
    main()

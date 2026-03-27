"""
evaluate_ablations.py — Quantitative Evaluation of TurboNIGO Ablation Study.

Loads each trained ablation model from its results directory, computes a
comprehensive set of metrics on the validation set, and exports:
  1. ablation_summary.csv   — Quick-reference comparison table
  2. ablation_summary.tex   — LaTeX booktabs table for direct paper inclusion
  3. per_model/<Name>.json  — Detailed per-model metrics and time-series data

Usage:
  conda run -n cfd python scripts/evaluate_ablations.py
  conda run -n cfd python scripts/evaluate_ablations.py --config configs/default_config.yaml
"""

import sys
import os
import json
import glob
import csv
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from turbo_nigo.configs import get_args_and_config
from turbo_nigo.utils import seed_everything, Registry
from turbo_nigo.data import compute_global_stats_and_cond_stats
from turbo_nigo.models import GlobalTurboNIGO
from turbo_nigo.models.ablations import (
    Ablation1_NoSkewTurboNIGO,
    Ablation2_NoDissipativeTurboNIGO,
    Ablation3_DenseGeneratorTurboNIGO,
    Ablation4_NoRefinerTurboNIGO,
    Ablation5_UnscaledTurboNIGO
)
from turbo_nigo.core.metrics import (
    compute_rollout_mse,
    compute_latent_energy_trace,
    compute_relative_l2_error,
    compute_lyapunov_divergence,
    get_radial_spectrum
)

# ═══════════════════════════════════════════════════════════════════════════════
# Model registry — maps experiment name to model class
# ═══════════════════════════════════════════════════════════════════════════════
ABLATION_REGISTRY = {
    "Baseline":           GlobalTurboNIGO,
    "NoSkew":             Ablation1_NoSkewTurboNIGO,
    "NoDissipative":      Ablation2_NoDissipativeTurboNIGO,
    "DenseGenerator":     Ablation3_DenseGeneratorTurboNIGO,
    "NoRefiner":          Ablation4_NoRefinerTurboNIGO,
    "UnscaledGenerator":  Ablation5_UnscaledTurboNIGO,
}


def load_trained_model(name, ModelClass, config, device):
    """Loads a trained model from the ablation results directory."""
    model = ModelClass(
        latent_dim=config["latent_dim"],
        num_bases=config["num_bases"],
        cond_dim=config["cond_dim"],
        width=config["width"]
    ).to(device)

    ckpt_path = os.path.join(config["results_dir"], f"Ablation_{name}", "checkpoints", "best.pth")
    if not os.path.exists(ckpt_path):
        print(f"  ⚠ Checkpoint not found: {ckpt_path}. Skipping {name}.")
        return None

    state = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    print(f"  ✓ Loaded {name} from {ckpt_path}")
    return model


def evaluate_single_model(name, model, val_dataset, config, device):
    """Runs full evaluation suite on a single model."""
    dt = config["dt"]
    seq_len = config["seq_len"]
    rollout_steps = config.get("eval_rollout_steps", 100)
    lyap_steps = config.get("eval_lyap_steps", 50)

    # Sample first validation batch
    from torch.utils.data import DataLoader
    loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    u0, u_seq_gt, cond = next(iter(loader))
    u0 = u0.to(device)
    u_seq_gt = u_seq_gt.to(device)
    cond = cond.to(device)

    results = {"name": name}

    # --- 1. Short-horizon MSE (single forward pass) ---
    with torch.no_grad():
        u_pred, _, k_c, r_c, alpha, beta = model(
            u0, torch.arange(1, seq_len + 1).float().to(device) * dt, cond
        )
        short_mse = torch.nn.functional.mse_loss(u_pred, u_seq_gt).item()
        results["short_mse"] = float(f"{short_mse:.6f}")
        results["alpha_mean"] = float(f"{alpha.mean().item():.6f}")
        results["beta_mean"] = float(f"{beta.mean().item():.6f}")

    # --- 2. Rollout MSE (autoregressive) ---
    rollout_mse = compute_rollout_mse(model, u0, cond, u_seq_gt, dt, block_size=seq_len)
    results["rollout_mse_final"] = float(f"{rollout_mse[-1]:.6f}")
    results["rollout_mse_mean"] = float(f"{rollout_mse.mean():.6f}")
    results["rollout_mse_curve"] = rollout_mse.tolist()

    # --- 3. Relative L2 error ---
    pred_np = u_pred.cpu().numpy()
    gt_np = u_seq_gt.cpu().numpy()
    rel_l2 = compute_relative_l2_error(pred_np, gt_np)
    results["rel_l2_mean"] = float(f"{rel_l2.mean():.6f}")
    results["rel_l2_curve"] = rel_l2.tolist()

    # --- 4. Latent energy trace ---
    energy_trace = compute_latent_energy_trace(model, u0, cond, steps=lyap_steps, dt=dt)
    results["energy_trace"] = energy_trace.tolist()
    results["energy_final_over_initial"] = float(f"{energy_trace[-1] / (energy_trace[0] + 1e-10):.6f}")

    # --- 5. Lyapunov divergence ---
    lyap_curve, init_dist = compute_lyapunov_divergence(model, u0, lyap_steps, cond, dt)
    results["lyapunov_curve"] = lyap_curve.tolist()
    growth = lyap_curve[-1] / (init_dist + 1e-10)
    results["lyapunov_growth_factor"] = float(f"{growth:.4f}")

    # --- 6. Spectral analysis (first sample, first channel, last step) ---
    pred_field = pred_np[0, -1, 0]  # (H, W)
    gt_field = gt_np[0, -1, 0]
    results["spectrum_pred"] = get_radial_spectrum(pred_field).tolist()
    results["spectrum_gt"] = get_radial_spectrum(gt_field).tolist()

    return results


def generate_latex_table(all_results, output_path):
    """Generates a LaTeX booktabs table for direct inclusion in the paper."""
    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Quantitative comparison of ablation variants. "
        "Short MSE is single-pass prediction error; Rollout MSE is the final-step error "
        "under autoregressive rollout; Rel.~$L^2$ is the normalized prediction error; "
        "Energy Ratio is $\\|z_T\\|^2 / \\|z_0\\|^2$ (values $\\leq 1$ indicate Lyapunov stability).}\n"
        "\\label{tab:ablation_results}\n"
        "\\vskip 0.1in\n"
        "\\begin{small}\n"
        "\\begin{sc}\n"
        "\\begin{tabular}{lcccccc}\n"
        "\\toprule\n"
        "Model & Short MSE & Rollout MSE & Rel.~$L^2$ & $\\bar{\\alpha}$ & $\\bar{\\beta}$ & Energy Ratio \\\\\n"
        "\\midrule\n"
    )

    rows = []
    for r in all_results:
        row = (
            f"{r['name']} & "
            f"{r['short_mse']:.4e} & "
            f"{r['rollout_mse_final']:.4e} & "
            f"{r['rel_l2_mean']:.4e} & "
            f"{r['alpha_mean']:.4f} & "
            f"{r['beta_mean']:.4f} & "
            f"{r['energy_final_over_initial']:.4f} \\\\"
        )
        rows.append(row)

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{sc}\n"
        "\\end{small}\n"
        "\\vskip -0.1in\n"
        "\\end{table}\n"
    )

    with open(output_path, 'w') as f:
        f.write(header)
        for row in rows:
            f.write(row + "\n")
        f.write(footer)


def main():
    config = get_args_and_config()
    seed_everything(config.get("seed", 42))
    device = config.get("device", "cpu")

    # Override eval-specific defaults
    config.setdefault("eval_rollout_steps", 100)
    config.setdefault("eval_lyap_steps", 50)

    print(f"\n{'='*60}")
    print(f"  TurboNIGO Ablation Evaluation")
    print(f"{'='*60}\n")

    # --- Prepare dataset ---
    dataset_name = config.get("dataset_type", "flow")
    DatasetClass = Registry.get_dataset(dataset_name)
    g_min, g_max, cond_mean, cond_std = compute_global_stats_and_cond_stats(config["data_root"])
    val_ds = DatasetClass.create_with_stats(
        config["data_root"], config["seq_len"], 'val', g_min, g_max, cond_mean, cond_std
    )

    # --- Output directories ---
    out_base = os.path.join(config["results_dir"], "ablation_results")
    out_quant = os.path.join(out_base, "quantitative")
    out_models = os.path.join(out_quant, "per_model")
    os.makedirs(out_models, exist_ok=True)

    all_results = []
    for name, ModelClass in ABLATION_REGISTRY.items():
        print(f"\n--- Evaluating: {name} ---")
        model = load_trained_model(name, ModelClass, config, device)
        if model is None:
            continue

        results = evaluate_single_model(name, model, val_ds, config, device)

        # Save per-model JSON (includes curves for plotting)
        json_path = os.path.join(out_models, f"{name}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  → Saved {json_path}")

        all_results.append(results)

    if not all_results:
        print("\n  ⚠ No trained models found. Run training first.")
        return

    # --- Generate CSV summary ---
    csv_path = os.path.join(out_quant, "ablation_summary.csv")
    fieldnames = ["name", "short_mse", "rollout_mse_final", "rollout_mse_mean",
                  "rel_l2_mean", "alpha_mean", "beta_mean",
                  "energy_final_over_initial", "lyapunov_growth_factor"]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    print(f"\n  ✓ CSV saved: {csv_path}")

    # --- Generate LaTeX table ---
    tex_path = os.path.join(out_quant, "ablation_summary.tex")
    generate_latex_table(all_results, tex_path)
    print(f"  ✓ LaTeX table saved: {tex_path}")

    print(f"\n{'='*60}")
    print(f"  Evaluation Complete → {out_quant}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

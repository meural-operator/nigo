import os
import sys
import glob
import numpy as np

# Dynamically resolve Conda DLL path for the active environment (e.g., 'cfd' instead of hardcoded 'turbo_nigo')
if os.name == 'nt':
    conda_bin = os.path.join(os.path.dirname(sys.executable), "Library", "bin")
    if os.path.exists(conda_bin):
        os.add_dll_directory(conda_bin)

# Add root repo to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from turbo_nigo.models import GlobalTurboNIGO
from turbo_nigo.core import Evaluator
from turbo_nigo.data import compute_global_stats_and_cond_stats, read_meta

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = "./results"
    data_root = "./datasets/bc"
    
    # 1. Find recent ablation runs
    # Sort folders by creation time to get the latest 6
    ablation_runs = {}
    for sl in [10, 20, 40, 60, 80, 100]:
        pattern = os.path.join(results_dir, f"Ablation_Horizon_T{sl}_*")
        matches = sorted(glob.glob(pattern), key=os.path.getmtime)
        if matches:
            ablation_runs[sl] = matches[-1] # take latest
        else:
            print(f"[!] Warning: No run found for T={sl}")
    
    if not ablation_runs:
        print("No ablation runs completed yet. Did you run scripts/run_horizon_ablations.py?")
        return
        
    g_min, g_max, cond_mean, cond_std = compute_global_stats_and_cond_stats(data_root)
    
    # 2. Get Test Case (using the last case)
    case_dirs = sorted(glob.glob(os.path.join(data_root, "case*")))
    target_dir = case_dirs[-1]
    
    u = np.load(os.path.join(target_dir, "u.npy"))
    v = np.load(os.path.join(target_dir, "v.npy"))
    gt_data = np.stack([u, v], axis=1).astype(np.float32)
    max_steps = len(gt_data)
    target_steps = min(max_steps, 1000)
    
    meta = read_meta(target_dir)
    cond_raw = np.array([
        meta.get("Re", 0.0), meta.get("radius", 0.0),
        meta.get("inlet_velocity", 0.0), meta.get("bc_type", 0.0)
    ], dtype=np.float32)
    
    # Prevent batch dim errors by unsqueezing to [1, cond_dim] for the generator hook
    cond_norm = torch.from_numpy((cond_raw - cond_mean) / (cond_std + 1e-8)).float().to(device).unsqueeze(0)
    initial_frame = torch.from_numpy(gt_data[0]).unsqueeze(0).to(device) # (1, 2, H, W)
    
    # 3. Evaluate each model
    results_mse = {}
    
    for sl, run_dir in ablation_runs.items():
        print(f"\n--- Evaluating TurboNIGO (T={sl}) from {run_dir} ---")
        
        # Parse immutable config if available to ensure correct width
        import yaml
        config_path = os.path.join(run_dir, "_immutable", "config.yaml")
        if os.path.exists(config_path):
             with open(config_path, "r") as f:
                 cfg = yaml.safe_load(f)
        else:
             print("[!] No immutable config found, using defaults.")
             cfg = {"latent_dim": 64, "num_bases": 8, "cond_dim": 4, "width": 32, "dt": 0.1}
             
        model = GlobalTurboNIGO(
            latent_dim=cfg.get("latent_dim", 64), 
            num_bases=cfg.get("num_bases", 8), 
            cond_dim=cfg.get("cond_dim", 4),
            width=cfg.get("width", 32)
        ).to(device)
        
        # Priority: best > epoch_NNNN > latest
        ckpt_path = None
        for try_path in ["best.pth", "latest.pth", "final.pth"]:
            pt = os.path.join(run_dir, "checkpoints", try_path)
            if os.path.exists(pt):
                ckpt_path = pt
                break
                
        if not ckpt_path:
            print(f"[!] No checkpoint found in {run_dir}/checkpoints/")
            continue
            
        print(f"Loading {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        
        evaluator = Evaluator(model, dt=cfg.get("dt", 0.1), device=device)
        
        print(f"Rolling out for {target_steps} steps autoregressively...")
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=False):
                pred_data = evaluator.chained_block_rollout(
                    initial_frame, cond_norm,
                    total_steps=target_steps,
                    block_size=sl,
                    g_min=g_min, g_max=g_max
                )
        
        # Calculate frame-wise MSE
        mse_curve = []
        for t in range(1, target_steps):
            gt_t = gt_data[t]
            pr_t = pred_data[t] # CPU numpy
            mse = np.mean((gt_t - pr_t)**2)
            mse_curve.append(mse)
            
        results_mse[sl] = mse_curve
        print(f"Final MSE @ T={target_steps}: {mse_curve[-1]:.4e}")

    # 4. Publication Grade Plotting (ICML Standard)
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.autolayout': True,
        'mathtext.fontset': 'stix',
        'font.family': 'STIXGeneral',
        'axes.linewidth': 1.5
    })
    
    # Wide layout to accommodate an external legend
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Scientific color scales covering all 6 training variations
    colors = {
        10: '#E63946',  # Deep Red
        20: '#F4A261',  # Orange
        40: '#2A9D8F',  # Teal
        60: '#457B9D',  # Blue
        80: '#1D3557',  # Navy
        100: '#9B5DE5'  # Purple
    }
    
    for sl, mse_curve in results_mse.items():
        ax.plot(range(1, len(mse_curve)+1), mse_curve, 
                 label=rf'TurboNIGO ($T_{{train}} = {sl}$)', 
                 color=colors.get(sl, 'black'), linewidth=2.5, alpha=0.9)
             
    ax.axhline(1.0, color='red', linestyle=':', alpha=0.6, label='Stability Threshold')
    
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 3.0)
    ax.set_xlabel(r'Extrapolated Rollout Step ($t$)')
    ax.set_ylabel(r'Mean Squared Error (Log Scale)')
    ax.set_title(r'Autoregressive Rollout Divergence: Invariance to $T_{train}$')
    
    ax.grid(True, which="major", ls="-", alpha=0.25)
    ax.grid(True, which="minor", ls=":", alpha=0.15)
    ax.tick_params(width=1.5)
    
    # Rebuttal Legend: Placed OUTSIDE the plot to completely prevent text overlaps
    ax.legend(loc='center left', bbox_to_anchor=(1.03, 0.5), frameon=False)
    
    os.makedirs('figures', exist_ok=True)
    out_path_png = os.path.join('figures', 'horizon_ablation_mse.png')
    out_path_pdf = os.path.join('figures', 'horizon_ablation_mse.pdf')
    
    # Using extremely tight layout prevents the external legend from being cut off
    fig.savefig(out_path_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_path_pdf, format='pdf', bbox_inches='tight')
    print(f"\n[+] Saved Publication-Ready PNG Plot to '{out_path_png}'")
    print(f"[+] Saved ICML-Ready PDF Vector Plot to '{out_path_pdf}'")

if __name__ == "__main__":
    main()

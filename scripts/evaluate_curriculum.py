import os
import torch
import numpy as np
import glob
import sys
import yaml
import json
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
import matplotlib.pyplot as plt

# Add the repository root to sys.path so turbo_nigo can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if os.name == 'nt':
    conda_bin = os.path.join(os.path.dirname(sys.executable), "Library", "bin")
    if os.path.exists(conda_bin):
        os.add_dll_directory(conda_bin)

from turbo_nigo.models import GlobalTurboNIGO
from turbo_nigo.core import Evaluator
from turbo_nigo.data import compute_global_stats_and_cond_stats, read_meta

device = "cuda" if torch.cuda.is_available() else "cpu"
data_root = "./datasets/bc"
g_min, g_max, cond_mean, cond_std = compute_global_stats_and_cond_stats(data_root)

case_dirs = sorted(glob.glob(os.path.join(data_root, "case*")))
target_dir = case_dirs[-1]
u = np.load(os.path.join(target_dir, "u.npy"))
v = np.load(os.path.join(target_dir, "v.npy"))
gt_data = np.stack([u, v], axis=1).astype(np.float32)

meta = read_meta(target_dir)
cond_raw = np.array([meta.get("Re", 0.0), meta.get("radius", 0.0), meta.get("inlet_velocity", 0.0), meta.get("bc_type", 0.0)], dtype=np.float32)
cond_norm = torch.from_numpy((cond_raw - cond_mean) / (cond_std + 1e-8)).float().to(device).unsqueeze(0)
initial_frame = torch.from_numpy(gt_data[0]).unsqueeze(0).to(device)

out_dir = "full_comparison_results/curriculum"
os.makedirs(out_dir, exist_ok=True)

horizons = [10, 20, 40, 60, 80, 100]

results = {}

for T in horizons:
    pattern = f"results/Ablation_Finetune_T{T}_Stage2_Sobolev_*"
    matches = sorted(glob.glob(pattern))
    if not matches:
        continue
    run_dir = matches[-1]
    
    cfg_path = os.path.join(run_dir, "_immutable", "config.yaml")
    if not os.path.exists(cfg_path):
        continue
    with open(cfg_path, 'r') as f:
        run_cfg = yaml.safe_load(f)
    seq_len = run_cfg.get("seq_len", T)

    model_kwargs = {"latent_dim": 64, "num_bases": 8, "cond_dim": 4, "width": 32}
    model = GlobalTurboNIGO(**model_kwargs).to(device)
    
    ckpt_path = None
    for p in ["best.pth", "latest.pth", "final.pth"]:
        pt = os.path.join(run_dir, "checkpoints", p)
        if os.path.exists(pt):
            ckpt_path = pt; break
            
    if not ckpt_path: continue
    print(f"Loading T={T} from {ckpt_path} (seq_len={seq_len})")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model_state_dict"])
    model.eval()
    
    evaluator = Evaluator(model, dt=0.1, device=device)
    total_steps = 1000
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=False):
            pred_data = evaluator.chained_block_rollout(initial_frame, cond_norm, total_steps=total_steps, block_size=seq_len, g_min=g_min, g_max=g_max)
    
    results[f"T={T}"] = pred_data

print("Evaluating Ground Truth...")
gt_traj = gt_data

# 1. Diagnostics (RMS Energy over T)
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Curriculum Sequence Length Ablation: 1000-Step Rollout (bc)", fontsize=14, fontweight="bold")
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(results)))

gt_energy = np.sqrt(np.mean(gt_traj**2, axis=(1, 2, 3)))
ax.semilogy(gt_energy, label="Ground Truth", color="black", linestyle="--", linewidth=2)
metrics = {}

for (name, pred_traj), color in zip(results.items(), colors):
    mlen = min(len(gt_traj), len(pred_traj))
    gt_sub = gt_traj[:mlen]
    pr_sub = pred_traj[:mlen]
    
    energy = np.sqrt(np.mean(pr_sub**2, axis=(1, 2, 3)))
    ax.semilogy(energy, label=f"{name} (E={energy[-1]:.4f})", color=color, linewidth=1.5)
    
    # Store MSE
    mse = float(np.mean((gt_sub - pr_sub)**2))
    metrics[name] = {"MSE": mse, "Final_Energy": float(energy[-1]), "Diverged": bool(energy[-1] > 1e3 or np.isnan(energy[-1]))}

ax.set_ylabel("RMS Energy (Log Scale)", fontsize=12)
ax.set_xlabel("Time Step", fontsize=12)
ax.legend(fontsize=10, loc="upper right", ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, "curriculum_diagnostics.png"), dpi=200, bbox_inches="tight")
pgf_path = os.path.join(out_dir, "curriculum_diagnostics.pgf")
tex_path = os.path.join(out_dir, "curriculum_diagnostics.tex")
fig.savefig(pgf_path, bbox_inches="tight")
if os.path.exists(tex_path): os.remove(tex_path)
os.rename(pgf_path, tex_path)
plt.close(fig)

# 2. 2D Snapshots at steps: 0, 100, 500, 1000
snap_steps = [0, 100, 500, 999]
# Compare T=10 and T=100 against GT
cmp_models = [k for k in ["T=10", "T=100"] if k in results]
if cmp_models:
    n_rows = 1 + len(cmp_models) 
    fig, axes = plt.subplots(n_rows, len(snap_steps), figsize=(3 * len(snap_steps), 3 * n_rows))
    fig.suptitle("Velocity Field Magnitude |u|^2 + |v|^2", fontsize=14, fontweight="bold")
    
    for c_idx, step in enumerate(snap_steps):
        # GT
        axes[0, c_idx].imshow(np.sqrt(gt_traj[min(step, len(gt_traj)-1),0]**2 + gt_traj[min(step, len(gt_traj)-1),1]**2), cmap="jet", origin="lower")
        axes[0, c_idx].set_title(f"GT: t={step}", fontsize=10)
        axes[0, c_idx].axis("off")
        
        # Models
        for r_idx, m_name in enumerate(cmp_models):
            traj = results[m_name]
            s_idx = min(step, len(traj)-1)
            axes[r_idx+1, c_idx].imshow(np.sqrt(traj[s_idx,0]**2 + traj[s_idx,1]**2), cmap="jet", origin="lower")
            axes[r_idx+1, c_idx].set_title(f"{m_name}", fontsize=10)
            axes[r_idx+1, c_idx].axis("off")
            
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "curriculum_snapshots.png"), dpi=200, bbox_inches="tight")
    pgf_snap_path = os.path.join(out_dir, "curriculum_snapshots.pgf")
    tex_snap_path = os.path.join(out_dir, "curriculum_snapshots.tex")
    fig.savefig(pgf_snap_path, bbox_inches="tight")
    if os.path.exists(tex_snap_path): os.remove(tex_snap_path)
    os.rename(pgf_snap_path, tex_snap_path)
    plt.close(fig)

with open(os.path.join(out_dir, "metrics.json"), 'w') as f:
    json.dump(metrics, f, indent=4)
print(json.dumps(metrics, indent=4))

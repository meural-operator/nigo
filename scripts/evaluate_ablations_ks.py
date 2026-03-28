"""
evaluate_ablations_ks.py — Quantitative Evaluation of KS Ablation Study.

Loads each KS ablation model from its results directory, computes a
comprehensive set of metrics on the explicitly loaded KSDataset val set, and exports:
  1. ks_ablation_summary.csv   — CSV comparison table
  2. ks_ablation_summary.tex   — LaTeX booktabs table
  3. ks_per_model/<Name>.json  — Detailed per-model metrics

Usage:
  conda run -n cfd python scripts/evaluate_ablations_ks.py --config turbo_nigo/configs/ks_config.yaml
"""

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Windows Conda DLL workaround
if os.name == 'nt' and 'CONDA_PREFIX' in os.environ:
    dll_path = os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'bin')
    if os.path.exists(dll_path):
        try: os.add_dll_directory(dll_path)
        except Exception: pass

import json
import csv
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    import h5py
except ImportError:
    pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from turbo_nigo.configs import get_args_and_config
from turbo_nigo.utils import seed_everything
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

ABLATION_REGISTRY = {
    "Baseline":           GlobalTurboNIGO,
    "NoSkew":             Ablation1_NoSkewTurboNIGO,
    "NoDissipative":      Ablation2_NoDissipativeTurboNIGO,
    "DenseGenerator":     Ablation3_DenseGeneratorTurboNIGO,
    "NoRefiner":          Ablation4_NoRefinerTurboNIGO,
    "UnscaledGenerator":  Ablation5_UnscaledTurboNIGO,
}

# =============================================================================
# KSDataset Replica
# =============================================================================
class KSDataset(Dataset):
    def __init__(self, h5_path: str, mode: str = 'train', seq_len: int = 20,
                 temporal_stride: int = 4, spatial_res: int = 64, 
                 max_cases: int = 5000, window_stride: int = 5):
        super().__init__()
        self.seq_len = seq_len
        self.spatial_res = spatial_res
        
        side = int(math.isqrt(spatial_res))
        self.side = side
        
        self.data_cache = []
        self.cond_cache = []
        self.index_map = []
        
        h5_key = 'train' if mode == 'train' else 'test'
        
        with h5py.File(h5_path, 'r') as f:
            raw = f[h5_key]
            n_total = raw.shape[0]
            n_load = min(max_cases, n_total) if max_cases else n_total
            
            if mode == 'train':
                start_idx, end_idx = 0, int(n_load * 0.9)
            else:
                start_idx, end_idx = int(n_load * 0.9), n_load
                
            chunk = np.array(raw[start_idx:end_idx], dtype=np.float32)
        
        g_min, g_max = chunk.min(), chunk.max()
        n_cases = chunk.shape[0]
        for i in range(n_cases):
            traj = chunk[i][::temporal_stride]
            traj = (traj - g_min) / (g_max - g_min + 1e-8)
            traj_t = torch.from_numpy(traj).float().unsqueeze(1).unsqueeze(1)
            traj_down = F.interpolate(traj_t, size=(1, spatial_res), mode='bilinear', align_corners=False)
            traj_2d = traj_down.view(-1, 1, self.side, self.side)
            
            self.data_cache.append(traj_2d)
            cond = torch.tensor([
                traj.mean(), traj.std(), 
                np.abs(np.fft.fft(traj[0])[:5]).mean(),
                float(i) / n_cases
            ], dtype=torch.float32)
            self.cond_cache.append(cond)
            
            n_steps = traj_2d.shape[0]
            stride_w = window_stride if mode == 'train' else max(window_stride * 4, 20)
            if n_steps > seq_len:
                for t in range(0, n_steps - seq_len, stride_w):
                    self.index_map.append((len(self.data_cache) - 1, t))
                    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        case_idx, t0 = self.index_map[idx]
        window = self.data_cache[case_idx][t0 : t0 + self.seq_len + 1]
        x = window[0]
        y = window[1:]
        return x, y, self.cond_cache[case_idx]

def load_trained_model(name, ModelClass, config, device):
    spatial_res = config.get("ks_spatial_res", 64)
    side = int(math.isqrt(spatial_res))
    model = ModelClass(
        latent_dim=config["latent_dim"],
        num_bases=config["num_bases"],
        cond_dim=config["cond_dim"],
        width=config["width"],
        spatial_size=side,
        in_channels=config.get("in_channels", 1)
    ).to(device)

    # Use correctly isolated checkpoint directory string
    ckpt_path = os.path.join(config["results_dir"], f"KS_Ablation_{name}", "checkpoints", "best.pth")
    if not os.path.exists(ckpt_path):
        print(f"  ⚠ Checkpoint not found: {ckpt_path}")
        return None

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    print(f"  ✓ Loaded {name} from {ckpt_path}")
    return model

def evaluate_single_model(name, model, val_dataset, config, device):
    dt = config["dt"]
    seq_len = config["seq_len"]
    lyap_steps = min(config.get("eval_lyap_steps", 50), seq_len)

    loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    u0, u_seq_gt, cond = next(iter(loader))
    u0, u_seq_gt, cond = u0.to(device), u_seq_gt.to(device), cond.to(device)

    results = {"name": name}

    with torch.no_grad():
        u_pred, _, _, _, alpha, beta = model(u0, torch.arange(1, seq_len + 1).float().to(device) * dt, cond)
        results["short_mse"] = float(f"{torch.nn.functional.mse_loss(u_pred, u_seq_gt).item():.6f}")
        results["alpha_mean"] = float(f"{alpha.mean().item():.6f}")
        results["beta_mean"] = float(f"{beta.mean().item():.6f}")

    rollout_mse = compute_rollout_mse(model, u0, cond, u_seq_gt, dt, block_size=seq_len)
    results["rollout_mse_final"] = float(f"{rollout_mse[-1]:.6f}")
    results["rollout_mse_mean"] = float(f"{rollout_mse.mean():.6f}")

    pred_np, gt_np = u_pred.cpu().numpy(), u_seq_gt.cpu().numpy()
    rel_l2 = compute_relative_l2_error(pred_np, gt_np)
    results["rel_l2_mean"] = float(f"{rel_l2.mean():.6f}")

    energy_trace = compute_latent_energy_trace(model, u0, cond, steps=lyap_steps, dt=dt)
    results["energy_final_over_initial"] = float(f"{energy_trace[-1] / (energy_trace[0] + 1e-10):.6f}")

    lyap_curve, init_dist = compute_lyapunov_divergence(model, u0, lyap_steps, cond, dt)
    results["lyapunov_growth_factor"] = float(f"{lyap_curve[-1] / (init_dist + 1e-10):.4f}")

    return results

def main():
    config = get_args_and_config()
    seed_everything(config.get("seed", 42))
    device = config.get("device", "cpu")

    print(f"\n{'='*60}\n  TurboNIGO KS Ablation Evaluation\n{'='*60}\n")

    val_ds = KSDataset(config["data_root"], mode='val', seq_len=config.get("seq_len", 20),
                       temporal_stride=config.get("ks_temporal_stride", 4),
                       spatial_res=config.get("ks_spatial_res", 64),
                       max_cases=config.get("ks_train_cases", 5000))

    out_base = os.path.join(config["results_dir"], "ks_ablation_results")
    out_quant = os.path.join(out_base, "quantitative")
    out_models = os.path.join(out_quant, "per_model")
    os.makedirs(out_models, exist_ok=True)

    all_results = []
    for name, ModelClass in ABLATION_REGISTRY.items():
        print(f"\n--- Evaluating: {name} ---")
        model = load_trained_model(name, ModelClass, config, device)
        if model is None: continue

        results = evaluate_single_model(name, model, val_ds, config, device)
        
        json_path = os.path.join(out_models, f"{name}.json")
        with open(json_path, 'w') as f: json.dump(results, f, indent=2)
        all_results.append(results)

    if not all_results:
        print("\n  ⚠ No KS trained models found. Run training first.")
        return

    csv_path = os.path.join(out_quant, "ks_ablation_summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\n  ✓ CSV saved: {csv_path}\n  KS Evaluation Complete → {out_quant}/")

if __name__ == "__main__":
    main()

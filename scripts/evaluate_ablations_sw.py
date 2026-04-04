import sys
import os
import gc
import json
import csv
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from turbo_nigo.configs import get_args_and_config
from turbo_nigo.utils.misc import seed_everything
from scripts.train_unified import create_model, create_dataloaders
from turbo_nigo.core.losses import SobolevH1Loss

def load_trained_model(name, config, device):
    """Initializes the model dynamically from config and loads best.pth mapping carefully."""
    model = create_model(config).to(device)
    ckpt_path = os.path.join(config.get("results_dir", "./results_sw"), f"Ablation_{name}", "checkpoints", "best.pth")
    if not os.path.exists(ckpt_path):
        print(f"  [WARN] Checkpoint not found: {ckpt_path}")
        return None

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Checkpoint dict has either explicit keys or raw states natively
    # Handles both 'model_state_dict' wrapper and raw dict dumps cleanly
    state_dict = state.get("model_state_dict", state)
    model.load_state_dict(state_dict)
    
    model.eval()
    print(f"  ✓ Loaded {name} successfully from {ckpt_path}")
    return model

def compute_relative_l2(pred, target):
    """Standard Relative L2 Error (epsilon_rel) used in PDEBench."""
    diff = (pred - target).reshape(pred.shape[0], pred.shape[1], -1)
    tgt = target.reshape(target.shape[0], target.shape[1], -1)
    l2_error = diff.norm(dim=2) / (tgt.norm(dim=2) + 1e-8)
    return l2_error.mean().item()

def compute_sobolev_h1(pred, target):
    """Evaluates the sharpness of shockwaves mathematically using exact numeric gradients."""
    h1 = SobolevH1Loss(weight=1.0)
    return h1(pred, target).item()
    
def evaluate_single_model(name, model, val_loader, config, device):
    """Executes the full suite of Rollout / Supervised / Unsupervised metrics."""
    u0, u_seq_gt, cond = next(iter(val_loader))
    u0, u_seq_gt, cond = u0.to(device), u_seq_gt.to(device), cond.to(device)
    
    seq_len = u_seq_gt.shape[1]
    
    results = {"model_name": name}
    
    with torch.no_grad():
        # ------------------------------------------------------------------
        # PHASE 1: Supervised 20-Step Rollout (Ground Truth Comparison)
        # ------------------------------------------------------------------
        time_steps = torch.arange(1, seq_len + 1).float().to(device) * config.get("dt", 0.1)
        u_pred_20, _, _, _, _, _ = model(u0, time_steps, cond)
        
        # 1. Rollout MSE 
        mse_20 = F.mse_loss(u_pred_20, u_seq_gt).item()
        results["Rollout_MSE_T20"] = round(mse_20, 6)
        
        # Generate curve for plotting
        mse_curve = [F.mse_loss(u_pred_20[:, t], u_seq_gt[:, t]).item() for t in range(seq_len)]
        results["rollout_mse_curve"] = mse_curve
        
        # 2. Maximum Discrepancy (L_inf norm absolute peak deviation in the batch vector space)
        linf = torch.max(torch.abs(u_pred_20 - u_seq_gt)).item()
        results["Max_Error_Linf"] = round(linf, 4)
        
        # 3. Relative L2 
        rel_l2 = compute_relative_l2(u_pred_20, u_seq_gt)
        results["Relative_L2_T20"] = round(rel_l2, 4)
        
        rel_l2_curve = [compute_relative_l2(u_pred_20[:, t:t+1], u_seq_gt[:, t:t+1]) for t in range(seq_len)]
        results["rel_l2_curve"] = rel_l2_curve
        
        # 4. Sobolev H1 Error 
        h1_err = compute_sobolev_h1(u_pred_20, u_seq_gt)
        results["Sobolev_H1_Error"] = round(h1_err, 4)
        
        # ------------------------------------------------------------------
        # PHASE 2: Unsupervised 1,000-Step Blocked Autoregressive Rollout
        # ------------------------------------------------------------------
        print(f"      Running 1,000-step blocked autoregressive rollout (blocks of {seq_len})...")
        
        current_u = u0
        
        initial_volume = u0.sum()
        initial_energy = torch.var(u0) 
        
        time_steps_block = torch.arange(1, seq_len + 1).float().to(device) * config.get("dt", 0.1)
        num_blocks = 1000 // seq_len
        
        energy_trace = [1.0] # E0/E0
        
        for _ in range(num_blocks):
            next_state_seq, _, _, _, _, _ = model(current_u, time_steps_block, cond)
            
            # Record block-end energy
            t_energy = torch.var(next_state_seq[:, -1]).item()
            energy_trace.append(t_energy / (initial_energy.item() + 1e-8))
            
            current_u = next_state_seq[:, -1] 
            
        final_volume = current_u.sum()
        final_energy = torch.var(current_u)
        
        results["energy_trace"] = energy_trace
        
        vol_drift = torch.abs(final_volume - initial_volume) / (torch.abs(initial_volume) + 1e-8)
        results["Volume_Drift_1000_Steps_%"] = round(vol_drift.item() * 100, 4)
        
        energy_ratio = (final_energy / (initial_energy + 1e-8)).item()
        results["Energy_Ratio_1000_Steps"] = round(energy_ratio, 4)

    return results

def main():
    base_config = get_args_and_config()
    
    # We strictly enforce the dataset type internally matching train configuration
    base_config["dataset_type"] = "sw"
    device = base_config.get("device", "cuda")
    seed_everything(base_config.get("seed", 42))
    
    print(f"\n{'='*65}\n    TurboNIGO 2D Shallow Water Quantitative Ablation Suite\n{'='*65}\n")
    
    # Spin up dataset loader gracefully purely pulling mapping
    _, val_loader = create_dataloaders(base_config)
    
    # Isolate strictly distinct metrics dump repo correctly
    out_dir = os.path.join(base_config.get("results_dir", "./results_sw"), "evaluation_metrics")
    os.makedirs(out_dir, exist_ok=True)
    
    models_to_test = ["Baseline_MSE", "Sobolev_H1", "Dual_Curriculum"]
    all_results = []
    
    for name in models_to_test:
        print(f"\n[+] Analyzing Ablation Topology: {name}")
        model = load_trained_model(name, base_config, device)
        if model is None:
            continue
            
        results = evaluate_single_model(name, model, val_loader, base_config, device)
        
        # Export single json explicit breakdown mapping identically
        json_path = os.path.join(out_dir, f"{name}_metrics.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        all_results.append(results)
        
        # RAM flush strictly handling tensor leaks across consecutive 1000-loop executions safely
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Consolidate CSV tabular extraction mapping
    if not all_results:
        print("\n  ⚠ No trained checkpoints found. Please finish `scripts/run_ablations_sw.py` entirely first.")
        return

    csv_path = os.path.join(out_dir, "sw_ablation_summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
        
    print(f"\n  ✓ CSV structural summary natively exported: {csv_path}\n  Complete!")

if __name__ == "__main__":
    main()

import os
import torch
import numpy as np

from turbo_nigo.configs import get_args_and_config
from turbo_nigo.utils import seed_everything, get_paths, Registry
from turbo_nigo.data import compute_global_stats_and_cond_stats, read_meta
from turbo_nigo.models import GlobalTurboNIGO
from turbo_nigo.core import Evaluator, compute_physics_metrics

def main():
    config = get_args_and_config()
    seed_everything(config.get("seed", 42))
    
    device = config.get("device", "cpu")
    target_steps = config.get("eval_steps", 400)
    
    print(f"--- Starting Evaluation ---")
    
    dataset_name = config.get("dataset_type", "flow")
    DatasetClass = Registry.get_dataset(dataset_name)
    
    g_min, g_max, cond_mean, cond_std = compute_global_stats_and_cond_stats(config["data_root"])
    
    model = GlobalTurboNIGO(
        latent_dim=config["latent_dim"], 
        num_bases=config["num_bases"], 
        cond_dim=config["cond_dim"],
        width=config["width"]
    ).to(device)
    
    model_path = os.path.join(config.get("results_dir", "./results"), "checkpoints", "best.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        print(f"Loaded checkpoint from {model_path}")
    else:
        print(f"Checkpoint not found at {model_path}. Evaluation cannot proceed.")
        return
        
    evaluator = Evaluator(model, dt=config["dt"], device=device)
    
    # ------------------
    # Example logic to evaluate the last case in the dataset
    import glob
    case_dirs = sorted(glob.glob(os.path.join(config["data_root"], "case*")))
    if not case_dirs:
        print("No validation cases found.")
        return
        
    target_dir = case_dirs[-1]
    
    u = np.load(os.path.join(target_dir, "u.npy"))
    v = np.load(os.path.join(target_dir, "v.npy"))
    gt_data = np.stack([u, v], axis=1).astype(np.float32)
    
    meta = read_meta(target_dir)
    cond_raw = np.array([
        meta.get("Re", 0.0), meta.get("radius", 0.0),
        meta.get("inlet_velocity", 0.0), meta.get("bc_type", 0.0)
    ], dtype=np.float32)
    
    cond_norm = torch.from_numpy((cond_raw - cond_mean) / (cond_std + 1e-8)).float()
    
    initial_frame = torch.from_numpy(gt_data[0]).unsqueeze(0)
    
    pred_data = evaluator.chained_block_rollout(
        initial_frame, cond_norm,
        total_steps=target_steps,
        block_size=config["seq_len"],
        g_min=g_min, g_max=g_max
    )
    
    # Example Probe extraction (mock: just center of image)
    _, _, H, W = gt_data.shape
    py, px = H//2, W//2
    
    min_len = min(len(gt_data), len(pred_data))
    gt_probe = gt_data[:min_len, 0, py, px]
    pred_probe = pred_data[:min_len, 0, py, px]
    
    metrics = compute_physics_metrics(gt_probe, pred_probe, config["dt"])
    
    print("\n--- Evaluation Results ---")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"Correlation: {metrics['correlation']:.4f}")
    print(f"Pred Frequency: {metrics['freq_pred']:.4f} Hz")
    
if __name__ == "__main__":
    main()

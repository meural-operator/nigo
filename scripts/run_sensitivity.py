import sys
import os
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from turbo_nigo.configs import get_args_and_config
from turbo_nigo.utils import seed_everything, get_paths, Registry
from turbo_nigo.data import compute_global_stats_and_cond_stats
from turbo_nigo.models import GlobalTurboNIGO
from turbo_nigo.core import Trainer

def run_sensitivity(scale_N, horizon_T, base_config):
    print(f"\n{'='*50}")
    print(f"RUNNING SENSITIVITY: N={scale_N}, T={horizon_T}")
    print(f"{'='*50}\n")
    
    config = base_config.copy()
    config["experiment_name"] = f"Sensitivity_N{scale_N}_T{horizon_T}"
    config["seq_len"] = horizon_T  # Adjust sequence horizon
    
    seed_everything(config.get("seed", 42))
    paths = get_paths(config)
    device = config.get("device", "cpu")
    
    dataset_name = config.get("dataset_type", "flow")
    DatasetClass = Registry.get_dataset(dataset_name)
    
    g_min, g_max, cond_mean, cond_std = compute_global_stats_and_cond_stats(config["data_root"])
    
    # Crucially, pass max_cases to train_ds to artificially restrict dataset size!
    train_ds = DatasetClass.create_with_stats(
        config["data_root"], config["seq_len"], 'train', g_min, g_max, cond_mean, cond_std, max_cases=scale_N)
        
    # Validation dataset scale shouldn't be restricted (or maybe it should be proportional).
    # Since we evaluate standardly, let's leave it unrestricted to test generalization from N cases.
    val_ds = DatasetClass.create_with_stats(
        config["data_root"], config["seq_len"], 'val', g_min, g_max, cond_mean, cond_std)
        
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, 
                              num_workers=config["num_workers"], pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, 
                            num_workers=config["num_workers"], pin_memory=True, persistent_workers=True)
    
    model = GlobalTurboNIGO(
        latent_dim=config["latent_dim"], 
        num_bases=config["num_bases"], 
        cond_dim=config["cond_dim"],
        width=config["width"]
    ).to(device)
    
    trainer = Trainer(model, train_loader, val_loader, config, paths)
    trainer.train()

def main():
    base_config = get_args_and_config()
    
    # Scales N to test 
    # (assuming dataset has e.g., 50 cases... if N=10, we only use 10 cases)
    scales_N = [10, 25, 50]
    # Horizons T to test
    horizons_T = [10, 20, 40]
    
    # We do a grid search
    for N in scales_N:
        for T in horizons_T:
            run_sensitivity(scale_N=N, horizon_T=T, base_config=base_config)

if __name__ == "__main__":
    main()

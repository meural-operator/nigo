import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader

from turbo_nigo.configs import get_args_and_config
from turbo_nigo.utils import seed_everything, get_paths, Registry
from turbo_nigo.data import compute_global_stats_and_cond_stats
from turbo_nigo.models import GlobalTurboNIGO
from turbo_nigo.core import Trainer

def main():
    config = get_args_and_config()
    seed_everything(config.get("seed", 42))
    
    paths = get_paths(config)
    device = config.get("device", "cpu")
    
    print(f"--- Starting Training Run: {config.get('experiment_name', 'Default')} ---")
    
    # Resolving dataset class
    dataset_name = config.get("dataset_type", "flow")
    DatasetClass = Registry.get_dataset(dataset_name)
    
    print(f"Using dataset: {dataset_name} at {config['data_root']}")
    
    # 1. Scanning Stats
    g_min, g_max, cond_mean, cond_std = compute_global_stats_and_cond_stats(config["data_root"])
    
    # 2. Loading Datasets
    train_ds = DatasetClass.create_with_stats(
        config["data_root"], config["seq_len"], 'train', g_min, g_max, cond_mean, cond_std)
    val_ds = DatasetClass.create_with_stats(
        config["data_root"], config["seq_len"], 'val', g_min, g_max, cond_mean, cond_std)
        
    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True, 
        num_workers=config["num_workers"], pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False, 
        num_workers=config["num_workers"], pin_memory=True, persistent_workers=True
    )
    
    # 3. Initialize Model
    model = GlobalTurboNIGO(
        latent_dim=config["latent_dim"], 
        num_bases=config["num_bases"], 
        cond_dim=config["cond_dim"],
        width=config["width"]
    ).to(device)
    
    # 4. Training
    trainer = Trainer(model, train_loader, val_loader, config, paths)
    trainer.train()

if __name__ == "__main__":
    main()

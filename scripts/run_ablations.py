import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from turbo_nigo.configs import get_args_and_config
from turbo_nigo.utils import seed_everything, get_paths, Registry
from turbo_nigo.data import compute_global_stats_and_cond_stats
from turbo_nigo.models import GlobalTurboNIGO
from turbo_nigo.models.ablations import (
    Ablation1_NoSkewTurboNIGO,
    Ablation2_NoDissipativeTurboNIGO,
    Ablation3_DenseGeneratorTurboNIGO,
    Ablation4_NoRefinerTurboNIGO,
    Ablation5_UnscaledTurboNIGO
)
from turbo_nigo.core import Trainer

def run_ablation(model_name, ModelClass, base_config):
    print(f"\n{'='*50}")
    print(f"RUNNING ABLATION: {model_name}")
    print(f"{'='*50}\n")
    
    config = base_config.copy()
    config["experiment_name"] = f"Ablation_{model_name}"
    
    seed_everything(config.get("seed", 42))
    paths = get_paths(config)
    device = config.get("device", "cpu")
    
    dataset_name = config.get("dataset_type", "flow")
    DatasetClass = Registry.get_dataset(dataset_name)
    
    g_min, g_max, cond_mean, cond_std = compute_global_stats_and_cond_stats(config["data_root"])
    
    train_ds = DatasetClass.create_with_stats(
        config["data_root"], config["seq_len"], 'train', g_min, g_max, cond_mean, cond_std)
    val_ds = DatasetClass.create_with_stats(
        config["data_root"], config["seq_len"], 'val', g_min, g_max, cond_mean, cond_std)
        
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, 
                              num_workers=config["num_workers"], pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, 
                            num_workers=config["num_workers"], pin_memory=True, persistent_workers=True)
    
    model = ModelClass(
        latent_dim=config["latent_dim"], 
        num_bases=config["num_bases"], 
        cond_dim=config["cond_dim"],
        width=config["width"]
    ).to(device)
    
    trainer = Trainer(model, train_loader, val_loader, config, paths)
    trainer.train()

def main():
    base_config = get_args_and_config()
    
    models_to_test = [
        ("Baseline", GlobalTurboNIGO),
        ("UnscaledGenerator", Ablation5_UnscaledTurboNIGO),
        ("NoSkew", Ablation1_NoSkewTurboNIGO),
        ("NoDissipative", Ablation2_NoDissipativeTurboNIGO),
        ("DenseGenerator", Ablation3_DenseGeneratorTurboNIGO),
        ("NoRefiner", Ablation4_NoRefinerTurboNIGO)
    ]
    
    for name, cls in models_to_test:
        run_ablation(name, cls, base_config)

if __name__ == "__main__":
    main()

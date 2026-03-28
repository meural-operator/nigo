import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Windows Conda DLL workaround for PyTorch (Python >= 3.8 ignores PATH for DLLs)
if os.name == 'nt' and 'CONDA_PREFIX' in os.environ:
    dll_path = os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'bin')
    if os.path.exists(dll_path):
        try: os.add_dll_directory(dll_path)
        except Exception: pass

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

def run_ablation(model_name, ModelClass, base_config, train_loader, val_loader):
    print(f"\n{'='*50}")
    print(f"RUNNING ABLATION: {model_name}")
    print(f"{'='*50}\n")
    
    config = base_config.copy()
    config["experiment_name"] = f"Ablation_{model_name}"
    
    seed_everything(config.get("seed", 42))
    paths = get_paths(config)
    device = config.get("device", "cpu")
    
    model = ModelClass(
        latent_dim=config["latent_dim"], 
        num_bases=config["num_bases"], 
        cond_dim=config["cond_dim"],
        width=config["width"]
    ).to(device)
    
    trainer = Trainer(model, train_loader, val_loader, config, paths)
    trainer.train()
    
    # Crucial for stable 6-hour sequential runs on GPUs
    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

def main():
    base_config = get_args_and_config()
    
    dataset_name = base_config.get("dataset_type", "flow")
    DatasetClass = Registry.get_dataset(dataset_name)
    
    print("=" * 50)
    print("  LOADING FLOW DATASET (shared across ablations)")
    print("=" * 50)
    
    g_min, g_max, cond_mean, cond_std = compute_global_stats_and_cond_stats(base_config["data_root"])
    train_ds = DatasetClass.create_with_stats(
        base_config["data_root"], base_config["seq_len"], 'train', g_min, g_max, cond_mean, cond_std)
    val_ds = DatasetClass.create_with_stats(
        base_config["data_root"], base_config["seq_len"], 'val', g_min, g_max, cond_mean, cond_std)
        
    train_loader = DataLoader(train_ds, batch_size=base_config["batch_size"], shuffle=True, 
                              num_workers=base_config["num_workers"], pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=base_config["batch_size"], shuffle=False, 
                            num_workers=base_config["num_workers"], pin_memory=True, persistent_workers=True)
                            
    models_to_test = [
        ("Baseline", GlobalTurboNIGO),
        ("UnscaledGenerator", Ablation5_UnscaledTurboNIGO),
        ("NoSkew", Ablation1_NoSkewTurboNIGO),
        ("NoDissipative", Ablation2_NoDissipativeTurboNIGO),
        ("DenseGenerator", Ablation3_DenseGeneratorTurboNIGO),
        ("NoRefiner", Ablation4_NoRefinerTurboNIGO)
    ]
    
    for name, cls in models_to_test:
        run_ablation(name, cls, base_config, train_loader, val_loader)

if __name__ == "__main__":
    main()

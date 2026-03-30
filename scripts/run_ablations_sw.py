import sys
import os
import gc
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from turbo_nigo.configs import get_args_and_config
from turbo_nigo.utils.misc import seed_everything
from scripts.train_unified import create_model, create_dataloaders
from turbo_nigo.core.unified_trainer import UnifiedTrainer

def run_ablation(name, config_override, train_loader, val_loader):
    print(f"\n{'='*60}")
    print(f"  SW ABLATION: {name}")
    print(f"{'='*60}\n")
    
    config = config_override.copy()
    config["experiment_name"] = f"SW_{name}"
    results_dir = os.path.join(config.get("results_dir", "./results_sw"), f"Ablation_{name}")
    os.makedirs(results_dir, exist_ok=True)
    
    seed_everything(config.get("seed", 42))
    device = config.get("device", "cuda")
    
    print("[1/2] Building model ...")
    model = create_model(config).to(device)
    
    print("[2/2] Starting training ...\n")
    trainer = UnifiedTrainer(model, train_loader, val_loader, config, results_dir)
    trainer.train()
    
    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def main():
    base_config = get_args_and_config()
    
    print("=" * 60)
    print("  LOADING SW DATASET (shared across ablations)")
    print("=" * 60)
    
    # Create the dataloaders once to save RAM / load time
    train_loader, val_loader = create_dataloaders(base_config)
    
    print(f"      Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    
    # --- Ablation Configurations ---
    # 1. Baseline MSE
    cfg_baseline = base_config.copy()
    cfg_baseline["h1_weight"] = 0.0
    cfg_baseline["curriculum_learning"] = False
    
    # 2. Sobolev H1
    cfg_sobolev = base_config.copy()
    cfg_sobolev["h1_weight"] = 1.0
    cfg_sobolev["curriculum_learning"] = False
    
    # 3. Dual Curriculum Learning
    cfg_curriculum = base_config.copy()
    cfg_curriculum["h1_weight"] = 1.0
    cfg_curriculum["curriculum_learning"] = True
    
    ablations = [
        ("Baseline_MSE", cfg_baseline),
        ("Sobolev_H1", cfg_sobolev),
        ("Dual_Curriculum", cfg_curriculum)
    ]
    
    for name, cfg in ablations:
        run_ablation(name, cfg, train_loader, val_loader)
        
    print("\n" + "=" * 60)
    print("  ALL SW ABLATIONS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()

import sys
import os
import time
import yaml

# Inject Conda DLL path to resolve WinError 127 for PyTorch 2.11.0
if os.name == 'nt':
    os.add_dll_directory(r"PATH TO LIBRARY BIN")

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from scripts.train_unified import create_model, create_dataloaders
from turbo_nigo.core.unified_trainer import UnifiedTrainer
from turbo_nigo.utils.misc import seed_everything

def main():
    config_path = os.path.join("turbo_nigo", "configs", "default_config.yaml")
    
    print(f"[*] Loading base configuration from: {config_path}")
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)
        
    # The reviewer disputes the reliance on T=20. 
    # We will prove invariance by training identical models with T=10, 20, 40, 60, 80, 100.
    horizons = [10, 20, 40, 60, 80, 100]
    
    base_config['epochs'] = 100
    base_config['checkpoint_freq'] = 20
    
    # HARD PATCH: The new architecture (encoder.py) breaks in `amp` autocast transitions 
    # with mixed fp16/fp32 dtypes. We must disable AMP to pass inference cleanly.
    base_config['use_amp'] = False
    
    # Standard batch size is 64 for T=20.
    standard_bs = 64
    standard_t = 20.0
    
    for sl in horizons:
        print(f"\n{'='*74}")
        print(f"  STARTING ABLATION: Training Horizon T={sl}")
        print(f"{'='*74}\n")
        
        # Create an isolated config for this run
        config = base_config.copy()
        config['seq_len'] = sl
        config['experiment_name'] = f"Ablation_Horizon_T{sl}"
        
        # Maximize GPU utilization: If Sequence Length doubles, Batch Size halves.
        # This keeps the VRAM utilization fixed at ~100% capacity and makes epoch duration independent of sequence length.
        scaled_bs = max(2, int(standard_bs * (standard_t / sl)))
        config['batch_size'] = scaled_bs
        print(f"[*] Dynamic VRAM Scaling: Adjusted Batch Size to {scaled_bs} to saturate GPU footprint.")
        
        # IMPORTANT: When sequence length changes, we must adjust dt if total physical time is invariant.
        # However, the critique is specifically about the *number of autoregressive steps* T.
        # We will keep the physical transition dt=0.1 identical, meaning we are directly teaching the 
        # model shorter (t=1.0) and longer (t=4.0) physical chunks per gradient update.
        
        seed_everything(config.get("seed", 42))
        device = config.get("device", "cuda")
        
        # Setup specific results directory
        stamp = time.strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(config.get("results_dir", "./results"), f"{config['experiment_name']}_{stamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"[*] Results will be saved to: {results_dir}")
        print(f"[*] Initializing Datasets for T={sl}...")
        
        # Initialize datasets (creates the dynamic seq_len chunks automatically)
        train_loader, val_loader = create_dataloaders(config)
        
        print(f"[*] Initializing TurboNIGO Model...")
        model = create_model(config).to(device)
        
        print(f"[*] Spawning Unified Trainer...")
        trainer = UnifiedTrainer(model, train_loader, val_loader, config, results_dir)
        
        # Execute training
        trainer.train()
        
        print(f"[*] Completed Ablation for T={sl}.\n")

if __name__ == "__main__":
    main()

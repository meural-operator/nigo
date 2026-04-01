import sys
import os
import glob
import time
import yaml

# Inject Conda DLL path to resolve WinError 127 for PyTorch 2.11.0 on Windows/Python 3.13 combinations
if os.name == 'nt':
    conda_bin = os.path.join(os.path.dirname(sys.executable), "Library", "bin")
    if os.path.exists(conda_bin):
        os.add_dll_directory(conda_bin)
    else:
        # Fallback to hardcoded path if active sys doesn't match miniconda root structure
        os.add_dll_directory(r"C:\Users\DIAT\miniconda3\envs\turbo_nigo\Library\bin")

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from turbo_nigo.models import GlobalTurboNIGO
from turbo_nigo.core import UnifiedTrainer
from scripts.train_unified import create_dataloaders


def run_stage(config: dict, train_loader, val_loader):
    """Initializes model and trainer strictly for the provided config/loaders, minimizing memory leaks."""
    import torch
    
    # Initialize the model native to the framework's parameter resolution (fixes missing spatial_size mapping)
    from scripts.train_unified import create_model, setup_results_dir
    model = create_model(config)
    model = model.to(config.get('device', 'cuda'))
    
    run_dir = setup_results_dir(config)

    trainer = UnifiedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        results_dir=run_dir
    )
    
    print("\n" + "~"*50)
    print(f"[Stage Engine] Proceeding up to Epoch {config['epochs']}")
    print(f"               Spectral Weight: {config.get('spectral_loss_weight', 0.0)}")
    print(f"               Resume Checkpt : {config.get('resume_from', 'None')}")
    print("~"*50 + "\n")
    
    return trainer.train()


def main():
    # Load base framework configuration as a starting point
    config_path = os.path.join("turbo_nigo", "configs", "default_config.yaml")
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)
        
    # We will prove invariance and stable finetuning under the rigorous T horizons
    horizons = [10, 20, 40, 60, 80, 100]
    
    # HARD PATCH: Prevent PyTorch AMP FP16 vs FP32 precision crashes deeply embedded in the generator modules.
    base_config['use_amp'] = False
    
    # Expose strict scale limits from the config target locally
    from turbo_nigo.data import compute_global_stats_and_cond_stats
    g_min, g_max, _, _ = compute_global_stats_and_cond_stats(base_config["data_root"])
    
    # Dynamic VRAM Scale Reference: `64` batch size typically occupies max VRAM at T=20
    standard_bs = 64
    standard_t = 20.0
    
    STAGE_1_EPOCHS = 20    # MSE Warmup phase
    TOTAL_EPOCHS = 200     # Final Spectral Sobolev Phase
    
    for sl in horizons:
        print(f"\n{'='*74}")
        print(f"  STARTING FINETUNING ABLATION: Training Horizon T={sl}")
        print(f"{'='*74}\n")
        
        # ---------------------------------------------------------
        # STAGE 1: Broad Pixel-Space Warmup
        # ---------------------------------------------------------
        stage1_config = base_config.copy()
        stage1_config['seq_len'] = sl
        stage1_config['experiment_name'] = f"Ablation_Finetune_T{sl}_Stage1_MSE"
        stage1_config['epochs'] = STAGE_1_EPOCHS
        stage1_config['checkpoint_freq'] = STAGE_1_EPOCHS  # Only need the boundary checkpoint
        stage1_config['spectral_loss_weight'] = 0.0        # Pure Spatial L2
        stage1_config['relative_l2_weight'] = 0.0
        
        # Maintain constant computational load to saturate GPU scaling
        scaled_bs = max(2, int(standard_bs * (standard_t / sl)))
        stage1_config['batch_size'] = scaled_bs
        print(f"[*] Set GPU Pipeline Batch Size to {scaled_bs} [Perfect Linear Scaling]")

        print("[*] Priming dataloaders precisely for this Sequence Horizon limit...")
        train_loader, val_loader = create_dataloaders(stage1_config)

        print("\n\t>>> Phase 1/2: Spatial Foundation Accumulation -> Pure pixel MSE")
        run_stage(stage1_config, train_loader, val_loader)
        
        
        # ---------------------------------------------------------
        # STAGE 2: Spectral / Sobolev Finetuning
        # ---------------------------------------------------------
        print("\n\t>>> Phase 2/2: High-Frequency Spectral/Sobolev Alignment")
        
        # Find exactly where Stage 1 ended so we can graft the state dict seamlessly
        # Since UnifiedTrainer automatically appends a timestamp, we must glob dynamically to avoid overwriting or linking failures.
        pattern = os.path.join("results", f"{stage1_config['experiment_name']}_*")
        matches = sorted(glob.glob(pattern), key=os.path.getmtime)
        
        if not matches:
            raise FileNotFoundError(f"[ERROR] Stage 1 failed to spawn a timestamped directory for {stage1_config['experiment_name']}.")
            
        stage1_dir = matches[-1]
        resume_target = os.path.join(stage1_dir, "checkpoints", "latest.pth")
        
        if not os.path.exists(resume_target):
            # Fallback to absolute best if latest didn't dump successfully
            resume_target = os.path.join(stage1_dir, "checkpoints", "best.pth")
            
        stage2_config = stage1_config.copy()
        stage2_config['experiment_name'] = f"Ablation_Finetune_T{sl}_Stage2_Sobolev"
        stage2_config['epochs'] = TOTAL_EPOCHS
        stage2_config['checkpoint_freq'] = 30
        stage2_config['resume_from'] = resume_target
        
        # Inject the intensive H^1 penalizations
        stage2_config['spectral_loss_weight'] = 0.5   
        stage2_config['relative_l2_weight'] = 1.0     
        stage2_config['physics_prior_weight'] = 0.005 # Bump up explicit latent smoothing slightly
        
        run_stage(stage2_config, train_loader, val_loader)
        
        print(f"\n[✓] Completed strict finetuning transition for T={sl} -> Reached Epoch {TOTAL_EPOCHS}\n")

if __name__ == "__main__":
    main()

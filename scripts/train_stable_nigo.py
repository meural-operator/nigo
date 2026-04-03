"""
train_stable_nigo.py — Unified 2-Stage Robust Training Pipeline

This script securely drives the training of the structually-constrained TurboNIGO framework
(incorporating both use_adaptive_refiner and use_spectral_norm restrictions).

It implements the requested 2-Stage Sequence Strategy:
  - Stage 1 (Warmup): Pure MSE tracking (Epoch 1 - 20)
  - Stage 2 (Finetune): Structural Sobolev H¹ Frequency alignment (Epoch 21+)

Supported targets:
  --dataset burgers : Native 1D sequence mapping (GlobalTurboNIGO_1D)
  --dataset ks      : Reshaped 2D sequence mapping (GlobalTurboNIGO)
"""

import os
import sys
import time
import argparse
import datetime
import json
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from turbo_nigo.models.turbo_nigo import GlobalTurboNIGO
from turbo_nigo.models.turbo_nigo_1d import GlobalTurboNIGO_1D
from turbo_nigo.core.losses import SobolevH1Loss

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

class StageAdaptiveCompositeLoss(nn.Module):
    def __init__(self, target_h1_weight=0.1, warmup_epochs=20):
        super().__init__()
        self.h1 = SobolevH1Loss(weight=1.0)
        self.target_h1_weight = target_h1_weight
        self.warmup_epochs = warmup_epochs

    def forward(self, u_pred, u_target, epoch, k_coeffs=None, r_coeffs=None):
        mse = F.mse_loss(u_pred, u_target)
        h1_loss = self.h1(u_pred, u_target)
        
        # Adaptive 2-Stage Robust Weighting Transition
        current_h1_w = 0.0 if epoch <= self.warmup_epochs else self.target_h1_weight
        total = mse + (current_h1_w * h1_loss)
        
        return total, {
            "mse": mse.item(), 
            "h1": h1_loss.item(), 
            "total": total.item(),
            "w_h1": current_h1_w
        }


def train_one_epoch(model, loader, optimizer, scaler, criterion, epoch, device, use_amp):
    model.train()
    total_loss, count = 0.0, 0
    loss_accum = {"mse": 0.0, "h1": 0.0, "total": 0.0}
    accum_steps = 4
    
    pbar = tqdm(loader, desc=f"Stage {'1 (Warmup)' if epoch <= criterion.warmup_epochs else '2 (Sobolev)'} [Eq. {epoch}]")
    
    for i, (x, y, cond) in enumerate(pbar):
        x, y, cond = x.to(device), y.to(device), cond.to(device)
        time_steps = torch.linspace(0, 1.0, y.shape[1]).to(device)
        
        with autocast('cuda', enabled=use_amp):
            cond_b = cond.unsqueeze(0).expand(x.shape[0], -1) if cond.ndim == 1 else cond
            u_pred, _, kc, rc, _, _ = model(x, time_steps, cond_b)
            loss, l_dict = criterion(u_pred, y, epoch, kc, rc)
            loss = loss / accum_steps
            
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
            
        if (i + 1) % accum_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
        if not (torch.isnan(loss) or l_dict["total"] != l_dict["total"]):  # NaN guard (both tensor and dict)
            bs = x.size(0)
            total_loss += l_dict["total"] * bs
            for k in ["mse", "h1", "total"]:
                loss_accum[k] += l_dict.get(k, 0.0) * bs
            count += bs
            pbar.set_postfix({"loss": f"{l_dict['total']:.6e}", "mse": f"{l_dict['mse']:.6e}", "h1_w": f"{l_dict['w_h1']:.1f}"})
            
    return {k: v / max(1, count) for k, v in loss_accum.items()}

def validate(model, loader, criterion, epoch, device):
    model.eval()
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for x, y, cond in loader:
            x, y, cond = x.to(device).float(), y.to(device).float(), cond.to(device).float()
            time_steps = torch.linspace(0, 1.0, y.shape[1], device=device)
            cb = cond.unsqueeze(0).expand(x.shape[0], -1) if cond.ndim == 1 else cond
            
            u_pred, _, kc, rc, _, _ = model(x, time_steps, cb)
            loss, _ = criterion(u_pred, y, epoch, kc, rc)
            
            if not torch.isnan(loss):
                total_loss += loss.item() * x.size(0)
                count += x.size(0)
    return total_loss / max(1, count)

def main():
    parser = argparse.ArgumentParser("Unified 2-Stage Robust Trainer for TurboNIGO")
    parser.add_argument('--dataset', type=str, choices=['burgers', 'ks'], required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=20, help='Length of MSE-only Stage 1')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--h1_weight', type=float, default=0.1)
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint for flawless restart')
    args = parser.parse_args()

    # Immutable Logging
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", f"RobustStable_{args.dataset.upper()}_L{args.seq_len}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    
    def log(msg):
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")
        with open(os.path.join(run_dir, "train.log"), "a") as f:
            f.write(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}\n")
            
    log(f"Initializing 2-Stage Robust Sequencer on '{args.dataset.upper()}' Protocol.")

    # 1. Mirror Architectures & Configurations for Absolute Provenance
    src_dir = os.path.join(run_dir, "src_mirror")
    os.makedirs(src_dir, exist_ok=True)
    shutil.copy2(__file__, os.path.join(src_dir, os.path.basename(__file__)))
    models_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../turbo_nigo/models"))
    if os.path.exists(models_src):
        shutil.copytree(models_src, os.path.join(src_dir, "models"), dirs_exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')
    if use_amp:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Dataset Loaders & Model Factory
    if args.dataset == 'burgers':
        from scripts.train_burgers_v2 import PreloadedBurgersDataset1D
        dp = 'datasets/Burgers/1D_Burgers_Sols_Nu0.1.hdf5'
        train_ds = PreloadedBurgersDataset1D(dp, seq_len=args.seq_len, mode="train", max_samples=4000)
        val_ds = PreloadedBurgersDataset1D(dp, seq_len=args.seq_len, mode="val", max_samples=4000)
        
        model = GlobalTurboNIGO_1D(
            latent_dim=args.latent_dim, width=args.width, in_channels=1, spatial_size=1024,
            use_adaptive_refiner=True, use_spectral_norm=True
        ).to(device)
        
    elif args.dataset == 'ks':
        from turbo_nigo.data.ks_dataset import KSDataset
        # Attempt to auto-locate KS dataset
        paths = ['datasets/KS_dataset/KS_ML_DATASET.h5', 'datasets/KS/KS_1D_Train.hdf5', 'datasets/KS/KS_Train.hdf5']
        dp = None
        for p in paths:
            if os.path.exists(p):
                dp = p
                break
        if dp is None:
            raise FileNotFoundError("KS HDF5 dataset not identified in predefined paths!")
            
        train_ds = KSDataset(dp, seq_len=args.seq_len, mode="train", spatial_res=4096, max_trajectories=2000)
        val_ds = KSDataset(dp, seq_len=args.seq_len, mode="val", spatial_res=4096, max_trajectories=500)
        
        model = GlobalTurboNIGO(
            latent_dim=args.latent_dim, width=args.width, in_channels=1, spatial_size=64,
            use_adaptive_refiner=True, use_spectral_norm=True
        ).to(device)

    log(f"Model instanced securely with Spectral Lipschitz and Adaptive Scalers Enforced.")
    log(f"Train Base: {len(train_ds)} windows | Val Base: {len(val_ds)} windows")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=use_amp, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=use_amp, num_workers=0)

    criterion = StageAdaptiveCompositeLoss(target_h1_weight=args.h1_weight, warmup_epochs=args.warmup_epochs)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler() if use_amp else None
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    best_val = float('inf')
    
    # 2. Complete Metadata Extraction (Capturing implicit limits and boundaries)
    full_config = vars(args).copy()
    full_config.update({
        'use_adaptive_refiner': True,
        'use_spectral_norm': True,
        'model_configuration': 'GlobalTurboNIGO_1D' if args.dataset == 'burgers' else 'GlobalTurboNIGO',
        'train_samples': len(train_ds),
        'val_samples': len(val_ds),
        'dataset_bounds': f"{train_ds.g_min} to {train_ds.g_max}" if hasattr(train_ds, 'g_min') else "Uncomputed"
    })
    
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(full_config, f, indent=4)

    start_epoch = 1
    if args.resume_from and os.path.exists(args.resume_from):
        log(f"Restoring mathematical constraints & optimizer state from {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        best_val = ckpt.get('best_val', float('inf'))
        start_epoch = ckpt['epoch'] + 1
        log(f"   [+] Resuming strictly from Epoch {start_epoch} | best_val={best_val:.6e}")

    # 2-Stage Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        st = time.time()
        
        tr_stats = train_one_epoch(model, train_loader, optimizer, scaler, criterion, epoch, device, use_amp)
        v_loss = validate(model, val_loader, criterion, epoch, device)
        scheduler.step()
        
        log(f"Epoch {epoch:03d}/{args.epochs} | Train Loss: {tr_stats['total']:.6e} (mse:{tr_stats['mse']:.6e}) | Val Loss: {v_loss:.6e} | T: {time.time()-st:.1f}s")
        
        if v_loss < best_val:
            best_val = v_loss
            log("   [*] New Model Optima Hit.")

        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val': best_val
        }
        
        torch.save(state, os.path.join(run_dir, "latest.pth"))
        
        if v_loss <= best_val:
            torch.save(state, os.path.join(run_dir, "best_model.pth"))
            log("   [+] Best checkpoint saved.")

        # 3. Epoch modulo backups (every 20 epochs)
        if epoch % 20 == 0:
            torch.save(state, os.path.join(run_dir, f"epoch_{epoch}.pth"))
            log(f"   [+] 20-Epoch Periodic Snapshot Saved: epoch_{epoch}.pth")

    log("="*60)
    log(f"2-Stage Structural Training Successful. Best Val Loss: {best_val:.6f}")
    
if __name__ == '__main__':
    main()

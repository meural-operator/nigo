"""
train_stable_nigo_sw.py — 2-Stage Robust Training Pipeline for Shallow Water Equations

Trains the structurally-constrained TurboNIGO framework on the PDEBench 2D
Shallow Water (radial dam break) dataset.  Incorporates both stability
mechanisms documented in `stability_walkthrough.md`:

  1. Adaptive Soft-Capped Refiner (epsilon_max = 0.1) ensuring the correction
     pathway never overpowers the structural generator.
  2. Full AMP precision safety-nets in Decoder, Decoder1D, and Refiner to
     prevent complex32 → float16 pathologies at block boundaries.

Training follows a 2-Stage Sequence Strategy:
  - Stage 1 (Warmup):   Pure MSE tracking                 (Epoch 1 → warmup_epochs)
  - Stage 2 (Finetune): Structural Sobolev H¹ alignment   (Epoch warmup_epochs+1 →)

Dataset: PDEBench 2D Shallow Water — 1000 trajectories of shape (101, 128, 128, 1)
         representing the water height field evolving under radial dam-break ICs.
Model:   GlobalTurboNIGO (2D encoder → generator → refiner → 2D decoder)
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
from turbo_nigo.data.sw_dataset import ShallowWaterDataset
from turbo_nigo.core.losses import SobolevH1Loss

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler


# ---------------------------------------------------------------------------
# Loss: 2-Stage Adaptive Composite (MSE → MSE + Sobolev H¹)
# ---------------------------------------------------------------------------
class StageAdaptiveCompositeLoss_SW(nn.Module):
    """
    Implements the 2-stage transition used in the stable Burgers trainer,
    adapted for 2D spatial fields.

    Stage 1 (epoch <= warmup_epochs): Pure MSE — stabilises latent mapping.
    Stage 2 (epoch >  warmup_epochs): MSE + H¹  — enforces gradient fidelity.
    """
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
            "w_h1": current_h1_w,
        }


# ---------------------------------------------------------------------------
# Training / Validation loops
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scaler, criterion, epoch,
                    device, use_amp, accum_steps=4):
    model.train()
    total_loss, count = 0.0, 0
    loss_accum = {"mse": 0.0, "h1": 0.0, "total": 0.0}

    stage_label = "1 (Warmup)" if epoch <= criterion.warmup_epochs else "2 (Sobolev)"
    pbar = tqdm(loader, desc=f"Stage {stage_label} [Ep. {epoch}]")

    optimizer.zero_grad(set_to_none=True)

    for i, (x, y, cond) in enumerate(pbar):
        # x: (B, C, H, W)  — initial field
        # y: (B, seq_len, C, H, W)  — target sequence
        # cond: (B, cond_dim) or (cond_dim,) — conditioning vector
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

        # NaN guard (both tensor and dict)
        if not (torch.isnan(loss) or l_dict["total"] != l_dict["total"]):
            bs = x.size(0)
            total_loss += l_dict["total"] * bs
            for k in ["mse", "h1", "total"]:
                loss_accum[k] += l_dict.get(k, 0.0) * bs
            count += bs
            pbar.set_postfix({
                "loss": f"{l_dict['total']:.6e}",
                "mse": f"{l_dict['mse']:.6e}",
                "h1_w": f"{l_dict['w_h1']:.1f}",
            })

    # Flush remaining accumulated gradients (if len(loader) % accum_steps != 0)
    if (len(loader)) % accum_steps != 0:
        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return {k: v / max(1, count) for k, v in loss_accum.items()}


def validate(model, loader, criterion, epoch, device):
    model.eval()
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for x, y, cond in loader:
            x = x.to(device).float()
            y = y.to(device).float()
            cond = cond.to(device).float()
            time_steps = torch.linspace(0, 1.0, y.shape[1], device=device)
            cb = cond.unsqueeze(0).expand(x.shape[0], -1) if cond.ndim == 1 else cond

            u_pred, _, kc, rc, _, _ = model(x, time_steps, cb)
            loss, _ = criterion(u_pred, y, epoch, kc, rc)

            if not torch.isnan(loss):
                total_loss += loss.item() * x.size(0)
                count += x.size(0)
    return total_loss / max(1, count)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="2-Stage Robust Trainer for TurboNIGO — Shallow Water Equations"
    )
    # Architecture
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Complex latent space dimensionality (matches sw_config)')
    parser.add_argument('--width', type=int, default=32,
                        help='Base channel width for encoder/decoder')
    parser.add_argument('--num_bases', type=int, default=8,
                        help='Number of generator basis matrices')
    parser.add_argument('--cond_dim', type=int, default=4,
                        help='Conditioning vector dimension')
    parser.add_argument('--in_channels', type=int, default=1,
                        help='SW dataset has a single height field channel')
    parser.add_argument('--spatial_size', type=int, default=128,
                        help='Spatial resolution (128x128 native for PDEBench SW)')

    # Dataset & Sequencing
    parser.add_argument('--data_path', type=str, default='datasets/2D_rdb_NA_NA.h5',
                        help='Path to PDEBench Shallow Water HDF5 file')
    parser.add_argument('--seq_len', type=int, default=20,
                        help='Temporal window length')
    parser.add_argument('--temporal_stride', type=int, default=1,
                        help='Temporal subsampling stride')
    parser.add_argument('--max_train_traj', type=int, default=900,
                        help='Maximum training trajectories (capped at 900 by dataset)')
    parser.add_argument('--max_val_traj', type=int, default=100,
                        help='Maximum validation trajectories')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=20,
                        help='Length of MSE-only Stage 1')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (reduced from 32 for 2D 128x128 fields)')
    parser.add_argument('--accum_steps', type=int, default=4,
                        help='Gradient accumulation steps (effective batch = batch_size * accum_steps)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Peak learning rate (matches sw_config)')
    parser.add_argument('--h1_weight', type=float, default=0.1,
                        help='Sobolev H1 loss weight in Stage 2')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint for seamless restart')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Enable mixed precision training (default off per sw_config)')

    args = parser.parse_args()

    # ----- Immutable Logging -----
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results_sw", f"RobustStable_SW_L{args.seq_len}_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    def log(msg):
        stamp = datetime.datetime.now().strftime('%H:%M:%S')
        print(f"[{stamp}] {msg}")
        with open(os.path.join(run_dir, "train.log"), "a") as f:
            f.write(f"[{stamp}] {msg}\n")

    log("=" * 60)
    log("Initializing 2-Stage Robust Stable Trainer — Shallow Water Equations")
    log("=" * 60)
    log(f"  Run directory : {run_dir}")
    log(f"  Dataset path  : {args.data_path}")
    log(f"  Spatial size  : {args.spatial_size}x{args.spatial_size}")
    log(f"  Sequence len  : {args.seq_len} (stride {args.temporal_stride})")
    log(f"  Architecture  : GlobalTurboNIGO (2D)")
    log(f"  Stability     : use_adaptive_refiner=True, use_spectral_norm=True")

    # ----- 1. Mirror source code for absolute provenance -----
    src_dir = os.path.join(run_dir, "src_mirror")
    os.makedirs(src_dir, exist_ok=True)
    shutil.copy2(__file__, os.path.join(src_dir, os.path.basename(__file__)))
    models_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../turbo_nigo/models"))
    if os.path.exists(models_src):
        shutil.copytree(models_src, os.path.join(src_dir, "models"), dirs_exist_ok=True)

    # ----- 2. Device & CUDA optimisations -----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = args.use_amp and (device.type == 'cuda')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    log(f"  Device        : {device} | AMP: {use_amp}")

    # ----- 3. Shallow Water Dataset -----
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(
            f"Shallow Water HDF5 dataset not found at: {args.data_path}\n"
            f"Download from PDEBench: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986"
        )

    train_ds = ShallowWaterDataset(
        h5_path=args.data_path,
        seq_len=args.seq_len,
        mode="train",
        temporal_stride=args.temporal_stride,
        spatial_size=args.spatial_size,
        max_trajectories=args.max_train_traj,
        cond_dim=args.cond_dim,
    )
    val_ds = ShallowWaterDataset(
        h5_path=args.data_path,
        seq_len=args.seq_len,
        mode="val",
        temporal_stride=args.temporal_stride,
        spatial_size=args.spatial_size,
        max_trajectories=args.max_val_traj,
        cond_dim=args.cond_dim,
        g_min=train_ds.g_min,   # Use training set statistics for normalisation
        g_max=train_ds.g_max,
    )

    log(f"  Train Base    : {len(train_ds)} windows  ({len(train_ds.data_cache)} trajectories)")
    log(f"  Val Base      : {len(val_ds)} windows  ({len(val_ds.data_cache)} trajectories)")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        pin_memory=(device.type == 'cuda'), num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        pin_memory=(device.type == 'cuda'), num_workers=0
    )

    # ----- 4. Model — GlobalTurboNIGO (2D) with stability hardening -----
    model = GlobalTurboNIGO(
        latent_dim=args.latent_dim,
        width=args.width,
        num_bases=args.num_bases,
        cond_dim=args.cond_dim,
        in_channels=args.in_channels,
        spatial_size=args.spatial_size,
        use_adaptive_refiner=True,   # Strict: Bounded correction gate (walkthrough §1)
        use_spectral_norm=True,      # Strict: Lipschitz-bounded weights  (walkthrough §1)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  Model params  : {total_params:,} total | {trainable_params:,} trainable")

    # ----- 5. Loss, Optimiser, Scheduler -----
    criterion = StageAdaptiveCompositeLoss_SW(
        target_h1_weight=args.h1_weight,
        warmup_epochs=args.warmup_epochs,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler() if use_amp else None
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    best_val = float('inf')

    # ----- 6. Complete metadata for reproducibility -----
    full_config = vars(args).copy()
    full_config.update({
        'use_adaptive_refiner': True,
        'use_spectral_norm': True,
        'model_configuration': 'GlobalTurboNIGO',
        'train_samples': len(train_ds),
        'val_samples': len(val_ds),
        'total_params': total_params,
        'trainable_params': trainable_params,
        'dataset_bounds': f"{train_ds.g_min:.6f} to {train_ds.g_max:.6f}",
        'dataset_type': 'shallow_water_2d',
        'pde': 'Saint-Venant (Shallow Water)',
    })
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(full_config, f, indent=4)
    log(f"  Config written to {os.path.join(run_dir, 'config.json')}")

    # ----- 7. Resume from checkpoint -----
    start_epoch = 1
    if args.resume_from and os.path.exists(args.resume_from):
        log(f"Restoring model, optimiser & scheduler from: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        best_val = ckpt.get('best_val', float('inf'))
        start_epoch = ckpt['epoch'] + 1
        log(f"   [+] Resuming from Epoch {start_epoch} | best_val={best_val:.6e}")

    # ----- 8. 2-Stage Training Loop -----
    log("=" * 60)
    log("BEGIN TRAINING")
    log("=" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        st = time.time()

        # --- Stage label ---
        if epoch <= args.warmup_epochs:
            log(f"--- Stage 1 (MSE Warmup) | Epoch {epoch}/{args.epochs} ---")
        elif epoch == args.warmup_epochs + 1:
            log(">>> TRANSITIONING TO STAGE 2: Sobolev H¹ Structural Finetuning <<<")

        tr_stats = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion,
            epoch, device, use_amp, accum_steps=args.accum_steps
        )
        v_loss = validate(model, val_loader, criterion, epoch, device)
        scheduler.step()

        elapsed = time.time() - st
        log(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {tr_stats['total']:.6e} (mse:{tr_stats['mse']:.6e}, h1:{tr_stats['h1']:.6e}) | "
            f"Val Loss: {v_loss:.6e} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"T: {elapsed:.1f}s"
        )

        # ----- Checkpointing -----
        improved = v_loss < best_val
        if improved:
            best_val = v_loss
            log("   [*] New Model Optima Hit.")

        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val': best_val,
        }

        torch.save(state, os.path.join(run_dir, "latest.pth"))

        if improved:
            torch.save(state, os.path.join(run_dir, "best_model.pth"))
            log("   [+] Best checkpoint saved.")

        # Periodic snapshots every 20 epochs
        if epoch % 20 == 0:
            torch.save(state, os.path.join(run_dir, f"epoch_{epoch}.pth"))
            log(f"   [+] 20-Epoch Periodic Snapshot: epoch_{epoch}.pth")

    log("=" * 60)
    log(f"2-Stage Structural Training Complete. Best Val Loss: {best_val:.6e}")
    log(f"Run directory: {run_dir}")
    log("=" * 60)


if __name__ == '__main__':
    main()

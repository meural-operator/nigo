"""
train_burgers_v2.py — Research-Grade Training for 1D Burgers Neural Operator.

Improvements over v1:
  1. Native 1D convolutions (GlobalTurboNIGO_1D) — eliminates spectral artifacts
     from the 1024→32×32 reshaping used in v1.
  2. Sobolev H¹ composite loss: MSE + λ·MSE(∂u/∂x) — penalizes derivative
     errors, naturally suppressing high-frequency noise. General-purpose PDE loss.
  3. Increased data budget (default 5000 samples for ~75 min training).

Usage:
  conda run -n turbo_nigo python scripts/train_burgers_v2.py \\
      --epochs 300 --batch_size 32 --max_samples 5000
"""

import os
import sys
import h5py
import json
import time
import argparse
import shutil
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from turbo_nigo.models.turbo_nigo_1d import GlobalTurboNIGO_1D
from turbo_nigo.core.losses import SobolevH1Loss


# =========================================================================
# Enterprise Logging & Run Management
# =========================================================================
class HistoryManager:
    """Manages immutable run directories, metric logging, and checkpointing."""
    def __init__(self, base_dir, run_id=None):
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id or f"RUN_BURGERS_1D_{self.timestamp}"
        self.run_dir = os.path.join(base_dir, self.run_id)

        os.makedirs(self.run_dir, exist_ok=True)
        self.csv_path = os.path.join(self.run_dir, "history.csv")
        self.log_path = os.path.join(self.run_dir, "train.log")

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w") as f:
                f.write("epoch,train_loss,val_loss,lr,time\n")

        self._mirror_source()

    def _mirror_source(self):
        """Copies scripts and core models into the run directory for strict provenance."""
        src_dir = os.path.join(self.run_dir, "src_mirror")
        os.makedirs(src_dir, exist_ok=True)
        shutil.copy2(__file__, os.path.join(src_dir, os.path.basename(__file__)))
        models_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../turbo_nigo/models"))
        if os.path.exists(models_src):
            shutil.copytree(models_src, os.path.join(src_dir, "models"), dirs_exist_ok=True)

    def log_metrics(self, epoch, train_l, val_l, lr):
        with open(self.csv_path, "a") as f:
            f.write(f"{epoch},{train_l:.8f},{val_l:.8f},{lr:.8f},{time.time()}\n")

    def log_text(self, msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{ts}] {msg}"
        print(full_msg)
        with open(self.log_path, "a") as f:
            f.write(full_msg + "\n")
        sys.stdout.flush()

    def save_config(self, config: dict):
        """Saves full experiment config as JSON for reproducibility."""
        path = os.path.join(self.run_dir, "config.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=4)
        self.log_text(f"Config saved to {path}")


# =========================================================================
# Dataset — Native 1D (no reshape)
# =========================================================================
class PreloadedBurgersDataset1D(Dataset):
    """
    High-throughput dataset for 1D Burgers Equation.

    Unlike v1 which reshaped 1024→32×32 for 2D convolutions, this dataset
    preserves the native 1D spatial structure: output shape is (C, X) = (1, 1024).

    Args:
        h5_path: Path to the HDF5 file.
        seq_len: Number of future timesteps per training sample.
        mode: 'train' or 'val' — uses a 90/10 split.
        max_samples: Memory cap on total trajectories loaded.
    """
    def __init__(self, h5_path, seq_len=20, mode="train", max_samples=None, stride=None):
        super().__init__()
        self.seq_len = seq_len
        self.h5_path = h5_path

        print(f"  [{mode.upper()}] Preloading Burgers 1D dataset into RAM...")
        with h5py.File(h5_path, 'r') as f:
            raw = f['tensor'][:]
            if max_samples:
                raw = raw[:max_samples]

            # 90/10 train/val split (deterministic, no shuffling)
            N_total = raw.shape[0]
            val_split = int(0.1 * N_total)
            if mode == 'train':
                raw = raw[:-val_split]
            else:
                raw = raw[-val_split:]

        N, T, X = raw.shape
        self.N, self.T, self.X = N, T, X

        # Global min-max normalization
        self.g_min = float(raw.min())
        self.g_max = float(raw.max())

        raw_torch = torch.from_numpy(raw).float()
        raw_norm = (raw_torch - self.g_min) / (self.g_max - self.g_min + 1e-8)

        # Shape: (N, T, 1, X) — native 1D with channel dim
        self.data = raw_norm.unsqueeze(2)

        # Dummy condition vector (Burgers has no physical conditions)
        self.cond = torch.zeros(4, dtype=torch.float32)

        # Sliding windows over time
        self.index_map = []
        if stride is None:
            stride = 10 if mode == "train" else 20
        self.stride = stride
        for i in range(N):
            for t0 in range(0, T - seq_len, stride):
                self.index_map.append((i, t0))

        print(f"    Loaded {N} trajectories | T={T} | X={X} | Windows={len(self.index_map)}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        n, t = self.index_map[idx]
        seq = self.data[n, t: t + self.seq_len + 1]  # (seq_len+1, 1, X)
        return seq[0], seq[1:], self.cond


# =========================================================================
# Loss Function
# =========================================================================
class BurgersCompositeLoss(nn.Module):
    """
    Composite loss for Burgers operator training:
        L = MSE(u_pred, u_gt) + λ_h1 · H¹(u_pred, u_gt)

    The Sobolev H¹ component penalizes spatial derivative errors:
        H¹ = MSE(∂u_pred/∂x, ∂u_gt/∂x)

    This is NOT Burgers-specific — it's a standard loss for any PDE operator.
    The framework's SobolevH1Loss handles 1D/2D automatically.
    """
    def __init__(self, h1_weight: float = 0.1):
        super().__init__()
        self.h1 = SobolevH1Loss(weight=h1_weight)
        self.h1_weight = h1_weight

    def forward(self, u_pred, u_target, k_coeffs=None, r_coeffs=None):
        mse = F.mse_loss(u_pred, u_target)
        h1 = self.h1(u_pred, u_target)
        total = mse + h1

        losses = {
            "mse": mse.item(),
            "h1_sobolev": h1.item(),
            "total": total.item(),
        }
        return total, losses


# =========================================================================
# Training & Validation Loops
# =========================================================================
def train_one_epoch(model, loader, optimizer, scaler, criterion, device, use_amp):
    model.train()
    total_loss, count = 0.0, 0
    loss_accum = {"mse": 0.0, "h1_sobolev": 0.0, "total": 0.0}
    accumulation_steps = 4
    pbar = tqdm(loader, desc="Training")

    for i, (x, y, cond) in enumerate(pbar):
        x, y, cond = x.to(device), y.to(device), cond.to(device)
        time_steps = torch.linspace(0, 1.0, y.shape[1]).to(device)

        with torch.amp.autocast('cuda', enabled=use_amp):
            cond_batch = cond.unsqueeze(0).expand(x.shape[0], -1) if cond.ndim == 1 else cond
            u_pred, _, k_coeffs, r_coeffs, _, _ = model(x, time_steps, cond_batch)
            loss, loss_dict = criterion(u_pred, y, k_coeffs, r_coeffs)
            loss = loss / accumulation_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if not torch.isnan(loss):
            bs = x.size(0)
            total_loss += loss_dict["total"] * bs
            for k in loss_accum:
                loss_accum[k] += loss_dict.get(k, 0.0) * bs
            count += bs
            pbar.set_postfix({
                "loss": f"{loss_dict['total']:.6f}",
                "mse": f"{loss_dict['mse']:.6f}",
                "h1": f"{loss_dict['h1_sobolev']:.6f}"
            })

    avg = {k: v / max(1, count) for k, v in loss_accum.items()}
    return avg["total"], avg


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for x, y, cond in loader:
            x, y, cond = x.to(device), y.to(device), cond.to(device)
            time_steps = torch.linspace(0, 1.0, y.shape[1]).to(device)
            cond_batch = cond.unsqueeze(0).expand(x.shape[0], -1) if cond.ndim == 1 else cond
            with torch.amp.autocast('cuda', enabled=False):
                u_pred, _, k_coeffs, r_coeffs, _, _ = model(x, time_steps, cond_batch)
                loss, _ = criterion(u_pred, y, k_coeffs, r_coeffs)
            total_loss += loss.item() * x.size(0)
            count += x.size(0)
    return total_loss / max(1, count)


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Burgers V2 Training: 1D NIGO + Sobolev Loss")
    parser.add_argument('--data_path', type=str, default='./datasets/Burgers/1D_Burgers_Sols_Nu0.1.hdf5')
    parser.add_argument('--output_dir', type=str, default='./results/burgers_experiments')
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--max_samples', type=int, default=5000, help='Dataset memory cap')
    parser.add_argument('--stride', type=int, default=10,
                        help='Sliding window stride for training (higher = fewer windows per epoch)')
    # Model architecture
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=3)
    # Loss
    parser.add_argument('--h1_weight', type=float, default=0.1,
                        help='Weight for Sobolev H1 derivative loss term')
    args = parser.parse_args()

    history = HistoryManager(args.output_dir)
    history.log_text("=" * 70)
    history.log_text("  BURGERS V2: 1D NIGO + Sobolev H¹ Loss")
    history.log_text("=" * 70)
    history.log_text(f"Run ID: {history.run_id}")
    history.log_text(f"Run Dir: {history.run_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')
    if use_amp:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Dataset ---
    train_ds = PreloadedBurgersDataset1D(args.data_path, seq_len=args.seq_len,
                                          mode="train", max_samples=args.max_samples,
                                          stride=args.stride)
    val_ds = PreloadedBurgersDataset1D(args.data_path, seq_len=args.seq_len,
                                        mode="val", max_samples=args.max_samples)

    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 0, 'pin_memory': use_amp}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    history.log_text(f"Train: {len(train_ds)} windows from {train_ds.N} trajectories")
    history.log_text(f"Val:   {len(val_ds)} windows from {val_ds.N} trajectories")
    history.log_text(f"Spatial: X={train_ds.X} (native 1D, no reshape)")

    # --- Model ---
    model = GlobalTurboNIGO_1D(
        latent_dim=args.latent_dim,
        in_channels=1,
        width=args.width,
        spatial_size=train_ds.X,  # 1024 — native spatial length
        num_layers=args.num_layers,
        use_residual=True,
        norm_type='group'
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    history.log_text(f"Model: GlobalTurboNIGO_1D | {n_params:,} params ({n_trainable:,} trainable)")

    # --- Loss ---
    criterion = BurgersCompositeLoss(h1_weight=args.h1_weight)
    history.log_text(f"Loss: MSE + {args.h1_weight}×SobolevH1")

    # --- Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    try:
        scaler = GradScaler(device='cuda') if use_amp else None
    except (TypeError, NameError):
        scaler = GradScaler() if use_amp else None

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # --- Save full config ---
    config = {
        "version": "v2_1d_sobolev",
        "model": "GlobalTurboNIGO_1D",
        "spatial_size": train_ds.X,
        "latent_dim": args.latent_dim,
        "width": args.width,
        "num_layers": args.num_layers,
        "in_channels": 1,
        "loss": f"MSE + {args.h1_weight}×SobolevH1",
        "h1_weight": args.h1_weight,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "epochs": args.epochs,
        "max_samples": args.max_samples,
        "stride": args.stride,
        "n_train_trajs": train_ds.N,
        "n_val_trajs": val_ds.N,
        "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds),
        "g_min": train_ds.g_min,
        "g_max": train_ds.g_max,
        "device": str(device),
        "use_amp": use_amp,
        "n_params": n_params,
    }
    history.save_config(config)

    # --- Resume ---
    start_epoch, best_val_loss = 1, float('inf')
    if args.resume_from and os.path.exists(args.resume_from):
        history.log_text(f"Resuming from {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        if scaler and ckpt.get('scaler_state'):
            scaler.load_state_dict(ckpt['scaler_state'])
        if scheduler and ckpt.get('scheduler_state'):
            scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))

    # --- Training Loop ---
    history.log_text(f"\nStarting training: epochs {start_epoch}–{args.epochs}")
    history.log_text("-" * 70)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_components = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device, use_amp
        )
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        history.log_metrics(epoch, train_loss, val_loss, lr)

        history.log_text(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train: {train_loss:.6f} (mse={train_components['mse']:.6f}, h1={train_components['h1_sobolev']:.6f}) | "
            f"Val: {val_loss:.6f} | LR: {lr:.2e} | Time: {epoch_time:.1f}s"
        )

        # Checkpoint state
        state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scaler_state': scaler.state_dict() if scaler else None,
            'scheduler_state': scheduler.state_dict() if scheduler else None,
            'best_val_loss': best_val_loss,
            'config': config,
        }

        # Type 1: Latest (overwritten every epoch)
        torch.save(state, os.path.join(history.run_dir, "latest_model.pth"))

        # Type 2: Best (only when val improves)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state['best_val_loss'] = best_val_loss
            torch.save(state, os.path.join(history.run_dir, "best_model.pth"))
            history.log_text("  [*] New best model saved.")

        # Type 3: Periodic snapshots every 30 epochs
        if epoch % 30 == 0:
            torch.save(state, os.path.join(history.run_dir, f"epoch_{epoch}_model.pth"))
            history.log_text(f"  [+] Checkpoint saved at epoch {epoch}")

    history.log_text("=" * 70)
    history.log_text(f"Training complete. Best val loss: {best_val_loss:.8f}")
    history.log_text(f"All artifacts in: {history.run_dir}")
    history.log_text("=" * 70)


if __name__ == '__main__':
    main()

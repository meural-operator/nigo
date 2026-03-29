import os
import sys
import json
import h5py
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    from torch.cuda.amp import autocast, GradScaler
except ImportError:
    from torch.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from turbo_nigo.models.turbo_nigo import GlobalTurboNIGO

class PreloadedNSDataset(Dataset):
    """
    High-throughput dataset for the Navier-Stokes HDF5 file.
    
    Strategy (optimized for RTX 4000 Ada 20GB + 128GB RAM):
      1. Preload ALL N samples from the HDF5 file at init time.
      2. Downsample from 512x512 -> target_res on CPU once (not per-item).
      3. Store the entire downsampled tensor in RAM (~500MB at 256x256).
      4. Create dense sliding windows (stride=1) across time for each sample.
      5. __getitem__ is a pure memory slice — zero disk I/O during training.
    
    Data layout: velocity (N=4, T=1000, H=512, W=512, C=2)
    """
    def __init__(self, h5_path, target_res=256, seq_len=20, mode="train"):
        super().__init__()
        self.seq_len = seq_len
        self.g_min = -3.0
        self.g_max = 3.0
        
        print(f"  [{mode.upper()}] Preloading and downsampling dataset into RAM...")
        t_start = time.time()
        
        with h5py.File(h5_path, 'r') as f:
            N, T, H, W, C = f['velocity'].shape
            force_raw = f['force'][:]                         # (N, H, W, 2)
            
            # Preload all samples, downsample per-sample to avoid a single huge allocation
            all_vel = []
            for n in range(N):
                raw = torch.from_numpy(f['velocity'][n]).float()  # (T, H, W, 2)
                raw = raw.permute(0, 3, 1, 2)                     # (T, 2, H, W)
                # Downsample in chunks of 100 timesteps to limit peak memory
                chunks = []
                for t0 in range(0, T, 100):
                    chunk = F.interpolate(
                        raw[t0:t0+100], size=(target_res, target_res),
                        mode="bilinear", align_corners=False
                    )
                    chunks.append(chunk)
                down = torch.cat(chunks, dim=0)                    # (T, 2, res, res)
                # Normalize to [0, 1]
                down = (down - self.g_min) / (self.g_max - self.g_min + 1e-8)
                all_vel.append(down)
        
        self.velocity = torch.stack(all_vel, dim=0)  # (N, T, 2, res, res)
        
        # Precompute conditioning vectors per sample
        self.cond_vecs = []
        for n in range(N):
            f_u = force_raw[n, :, :, 0]
            f_v = force_raw[n, :, :, 1]
            cond = torch.tensor(
                [np.mean(f_u), np.std(f_u), np.mean(f_v), np.std(f_v)],
                dtype=torch.float32
            ) / 2.0
            self.cond_vecs.append(cond)
        self.cond_vecs = torch.stack(self.cond_vecs, dim=0)  # (N, 4)
        
        N, T_total = self.velocity.shape[0], self.velocity.shape[1]
        window = self.seq_len + 1  # +1 for the initial condition
        
        # Build (sample_idx, t_start) index pairs with stride-1 sliding windows
        all_indices = []
        for n in range(N):
            for t in range(T_total - window):
                all_indices.append((n, t))
        
        # Train/val split: first 3 samples for train, last sample for val
        # This is a proper sample-level split (no data leakage across time)
        if mode == "train":
            self.indices = [(n, t) for (n, t) in all_indices if n < N - 1]
        else:
            self.indices = [(n, t) for (n, t) in all_indices if n == N - 1]
        
        elapsed = time.time() - t_start
        ram_mb = self.velocity.element_size() * self.velocity.nelement() / (1024**2)
        print(f"  [{mode.upper()}] Loaded {len(self.indices)} windows | "
              f"RAM usage: {ram_mb:.0f} MB | Took {elapsed:.1f}s")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        n, t = self.indices[idx]
        seq = self.velocity[n, t : t + self.seq_len + 1]  # (seq_len+1, 2, res, res)
        return seq[0], seq[1:], self.cond_vecs[n]

def train_one_epoch(model, loader, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training")
    criterion = nn.MSELoss()
    
    accumulation_steps = 4
    
    for i, (x, y, cond) in enumerate(pbar):
        x, y, cond = x.to(device), y.to(device), cond.to(device)
        
        S = y.shape[1]
        time_steps = torch.linspace(0, 1.0, S).to(device)
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            u_pred, z_base, k_c, r_c, alpha, beta = model(x, time_steps, cond)
            # Normalize loss by accumulation steps
            loss = criterion(u_pred, y) / accumulation_steps
        
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
            
        if (i + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
        
        if not torch.isnan(loss):
            total_loss += (loss.item() * accumulation_steps) * x.size(0)
            pbar.set_postfix({"loss": loss.item() * accumulation_steps})
        else:
            print("  [WARN] NaN loss detected in batch, skipping record.")
        
    return total_loss / max(1, len(loader.dataset))

def validate(model, loader, device):
    if len(loader.dataset) == 0:
        return 0.0
        
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for x, y, cond in loader:
            x, y, cond = x.to(device), y.to(device), cond.to(device)
            S = y.shape[1]
            time_steps = torch.linspace(0, 1.0, S).to(device)
            
            with torch.amp.autocast('cuda', enabled=False):
                u_pred, _, _, _, _, _ = model(x, time_steps, cond)
                loss = criterion(u_pred, y)
            
            total_loss += loss.item() * x.size(0)
            
    return total_loss / len(loader.dataset)

def save_checkpoint(path, epoch, model, optimizer, scaler, scheduler, best_loss):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scaler_state': scaler.state_dict() if scaler else None,
        'scheduler_state': scheduler.state_dict() if scheduler else None,
        'best_val_loss': best_loss
    }, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./datasets/ns_incom_inhom_2d_512-0.h5')
    parser.add_argument('--output_dir', type=str, default='./results/ns_run')
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, "train_config.json")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=4)
            
    print(f"Starting Training Run -> Storing results in {args.output_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')
    if use_amp:
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends, 'cuda'):
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("Enabled cuDNN Benchmark + TF32 for Ada architecture")

    print(f"Loading dataset (preloading into RAM - one-time cost)...")
    train_ds = PreloadedNSDataset(args.data_path, target_res=256, seq_len=args.seq_len, mode="train")
    val_ds = PreloadedNSDataset(args.data_path, target_res=256, seq_len=args.seq_len, mode="val")
    
    # Data is in RAM: workers=0 avoids IPC overhead, pin_memory for fast GPU transfer
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=use_amp)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=use_amp)

    print(f"Initializing GlobalTurboNIGO (High-Res Edition)...")
    model = GlobalTurboNIGO(
        latent_dim=64, num_bases=8, cond_dim=4, 
        width=64, spatial_size=256, in_channels=2,
        num_layers=3, use_residual=True, norm_type='group'
    )
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler() if use_amp else None
    
    # Proper upgrade: Cosine Annealing with Warm Restarts for better exploration/stability
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    start_epoch = 1
    best_val_loss = float('inf')

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from checkpoint {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        if scaler and 'scaler_state' in ckpt and ckpt['scaler_state']:
            scaler.load_state_dict(ckpt['scaler_state'])
        if scheduler and 'scheduler_state' in ckpt and ckpt['scheduler_state']:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Successfully loaded. Starting at epoch {start_epoch} with previous best loss: {best_val_loss:.6f}")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp)
        val_loss = validate(model, val_loader, device)
        
        # Proper upgrade: Step scheduler with epoch count
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        save_checkpoint(os.path.join(args.output_dir, "latest_model.pth"), epoch, model, optimizer, scaler, scheduler, best_val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(" -> (New Best Found)")
            save_checkpoint(os.path.join(args.output_dir, "best_model.pth"), epoch, model, optimizer, scaler, scheduler, best_val_loss)
            
        if epoch % 30 == 0:
            save_checkpoint(os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth"), epoch, model, optimizer, scaler, scheduler, best_val_loss)
            print(f" -> Saved periodic 30-epoch checkpoint")

if __name__ == '__main__':
    main()

import os
import sys
import json
import h5py
import time
import argparse
import shutil
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    # Prioritize modern torch.amp API
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from turbo_nigo.models.turbo_nigo import GlobalTurboNIGO
from turbo_nigo.core.losses import CompositeLoss

# --- Enterprise Logging & Management ---

class HistoryManager:
    """Manages immutable run directories and metric logging."""
    def __init__(self, base_dir, run_id=None):
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id or f"RUN_{self.timestamp}"
        self.run_dir = os.path.join(base_dir, self.run_id)
        
        os.makedirs(self.run_dir, exist_ok=True)
        self.csv_path = os.path.join(self.run_dir, "history.csv")
        self.log_path = os.path.join(self.run_dir, "train.log")
        
        # Initialize CSV
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w") as f:
                f.write("epoch,train_loss,val_loss,lr,time\n")
                
        self._mirror_source()

    def _mirror_source(self):
        """Copies scripts and core models into the run directory for provenance."""
        src_dir = os.path.join(self.run_dir, "src_mirror")
        os.makedirs(src_dir, exist_ok=True)
        shutil.copy2(__file__, os.path.join(src_dir, os.path.basename(__file__)))
        models_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../turbo_nigo/models"))
        if os.path.exists(models_src):
            shutil.copytree(models_src, os.path.join(src_dir, "models"), dirs_exist_ok=True)
            
    def log_metrics(self, epoch, train_l, val_l, lr):
        """Appends metrics to CSV and flushes to disk."""
        with open(self.csv_path, "a") as f:
            f.write(f"{epoch},{train_l:.8f},{val_l:.8f},{lr:.8f},{time.time()}\n")
            
    def log_text(self, msg):
        """Appends message to log file and prints to terminal."""
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{ts}] {msg}"
        print(full_msg)
        with open(self.log_path, "a") as f:
            f.write(full_msg + "\n")
        sys.stdout.flush()

# --- Dataset and Training ---

class PreloadedNSDataset(Dataset):
    """High-throughput dataset for Navier-Stokes HDF5 (Optimized for RAM)."""
    def __init__(self, h5_path, target_res=256, seq_len=20, mode="train"):
        super().__init__()
        self.seq_len = seq_len
        self.g_min = -3.0
        self.g_max = 3.0
        
        print(f"  [{mode.upper()}] Preloading dataset into RAM...")
        with h5py.File(h5_path, 'r') as f:
            N, T, H, W, C = f['velocity'].shape
            force_raw = f['force'][:]                         
            all_vel = []
            for n in range(N):
                raw = torch.from_numpy(f['velocity'][n]).float().permute(0, 3, 1, 2)
                chunks = [F.interpolate(raw[t0:t0+100], size=(target_res, target_res), mode="bilinear") for t0 in range(0, T, 100)]
                down = torch.cat(chunks, dim=0)
                down = (down - self.g_min) / (self.g_max - self.g_min + 1e-8)
                all_vel.append(down)
        
        self.velocity = torch.stack(all_vel, dim=0)
        self.cond_vecs = []
        for n in range(N):
            f_u, f_v = force_raw[n, :, :, 0], force_raw[n, :, :, 1]
            self.cond_vecs.append(torch.tensor([np.mean(f_u), np.std(f_u), np.mean(f_v), np.std(f_v)]) / 2.0)
        self.cond_vecs = torch.stack(self.cond_vecs, dim=0)
        
        N, T_total = self.velocity.shape[0], self.velocity.shape[1]
        all_indices = [(n, t) for n in range(N) for t in range(T_total - (self.seq_len + 1))]
        self.indices = [(n, t) for (n, t) in all_indices if (n < N - 1 if mode == "train" else n == N - 1)]

    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        n, t = self.indices[idx]
        seq = self.velocity[n, t : t + self.seq_len + 1]
        return seq[0], seq[1:], self.cond_vecs[n]

def train_one_epoch(model, loader, optimizer, scaler, device, use_amp, epoch, accumulation_steps=4):
    model.train()
    total_loss, count = 0.0, 0
    pbar = tqdm(loader, desc="Training")
    
    # --- Curriculum Learning Schedule ---
    # Epochs <= 20: Macroscopic Pixel Matching (MSE + Relative L2)
    # Epochs > 20: Physics Fine-tuning (MSE + Relative L2 + Incompressibility + Smoothness)
    if epoch <= 20:
        loss_config = {"relative_l2_weight": 1.0}
    else:
        loss_config = {"relative_l2_weight": 1.0, "divergence_weight": 0.1, "h1_weight": 0.1}
        
    criterion = CompositeLoss(loss_config).to(device)
    
    for i, (x, y, cond) in enumerate(pbar):
        x, y, cond = x.to(device, non_blocking=True), y.to(device, non_blocking=True), cond.to(device, non_blocking=True)
        time_steps = torch.linspace(0, 1.0, y.shape[1], device=device)
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            u_pred, _, k_coeffs, r_coeffs, _, _ = model(x, time_steps, cond)
            loss, _ = criterion(u_pred, y, k_coeffs, r_coeffs)
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
            cur_l = loss.item() * accumulation_steps
            total_loss += cur_l * x.size(0)
            count += x.size(0)
            pbar.set_postfix({"loss": f"{cur_l:.6f}"})
        
    return total_loss / max(1, count)

def validate(model, loader, device):
    model.eval()
    total_loss, count = 0.0, 0
    
    # Validation uses strictly the physics-aware loss for fair tracking
    loss_config = {"relative_l2_weight": 1.0, "divergence_weight": 0.1, "h1_weight": 0.1}
    criterion = CompositeLoss(loss_config).to(device)
    
    with torch.no_grad():
        for x, y, cond in loader:
            x, y, cond = x.to(device, non_blocking=True), y.to(device, non_blocking=True), cond.to(device, non_blocking=True)
            time_steps = torch.linspace(0, 1.0, y.shape[1], device=device)
            with torch.amp.autocast('cuda', enabled=False):
                u_pred, _, k_coeffs, r_coeffs, _, _ = model(x, time_steps, cond)
                loss, _ = criterion(u_pred, y, k_coeffs, r_coeffs)
            total_loss += loss.item() * x.size(0)
            count += x.size(0)
    return total_loss / max(1, count)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./datasets/ns_incom_inhom_2d_512-0.h5')
    parser.add_argument('--output_dir', type=str, default='./results/ns_modular')
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=2) # Default overridden for micro-batching
    parser.add_argument('--update_steps', type=int, default=16) # Effectively achieving batch=32
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--seq_len', type=int, default=20)
    args = parser.parse_args()

    history = HistoryManager(args.output_dir)
    history.log_text(f"Starting High-Res NS Training Platform")
    history.log_text(f"Run ID: {history.run_id} | Dir: {history.run_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')
    if use_amp:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Performance specific configs matching the GPU micro-batching limits
    micro_batch = args.batch_size
    accumulations = max(1, args.update_steps // micro_batch)
    history.log_text(f"Using micro-batching (bs={micro_batch}, accumulations={accumulations}) to prevent PCIe thrashing.")

    train_ds = PreloadedNSDataset(args.data_path, target_res=256, seq_len=args.seq_len, mode="train")
    val_ds = PreloadedNSDataset(args.data_path, target_res=256, seq_len=args.seq_len, mode="val")
    train_loader = DataLoader(train_ds, batch_size=micro_batch, shuffle=True, num_workers=0, pin_memory=use_amp)
    val_loader = DataLoader(val_ds, batch_size=micro_batch, shuffle=False, num_workers=0, pin_memory=use_amp)

    model = GlobalTurboNIGO(latent_dim=64, width=64, spatial_size=256, num_layers=3, use_residual=True, norm_type='group')
    model.to(device)
    
    # [OPTIMIZATION] PyTorch 2.0 Triton Compilation for massive speedup
    if hasattr(torch, 'compile') and os.name != 'nt':
        history.log_text("[*] Applying torch.compile for Triton kernel optimization...")
        try:
            model = torch.compile(model)
        except Exception as e:
            history.log_text("[!] torch.compile failed, falling back to eager mode.")
    else:
        history.log_text("[*] Bypassing torch.compile on Windows (Triton unsupported natively).")

    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Version-safe GradScaler initialization
    try:
        scaler = GradScaler(device='cuda') if use_amp else None
    except (TypeError, NameError):
        scaler = GradScaler() if use_amp else None
        
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    start_epoch, best_val_loss = 1, float('inf')
    
    if args.resume_from and os.path.exists(args.resume_from):
        history.log_text(f"Auto-Resuming from {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        if scaler and ckpt.get('scaler_state'): scaler.load_state_dict(ckpt['scaler_state'])
        if scheduler and ckpt.get('scheduler_state'): scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch, best_val_loss = ckpt['epoch'] + 1, ckpt.get('best_val_loss', float('inf'))

    for epoch in range(start_epoch, args.epochs + 1):
        history.log_text(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp, epoch, accumulations)
        val_loss = validate(model, val_loader, device)
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        history.log_metrics(epoch, train_loss, val_loss, lr)
        history.log_text(f"Metric: TrainLoss={train_loss:.6f} | ValLoss={val_loss:.6f} | LR={lr:.2e}")
        
        state = {'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                 'scaler_state': scaler.state_dict() if scaler else None, 
                 'scheduler_state': scheduler.state_dict() if scheduler else None, 'best_val_loss': best_val_loss}
        
        torch.save(state, os.path.join(history.run_dir, "latest_model.pth"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(state, os.path.join(history.run_dir, "best_model.pth"))
            history.log_text("  [*] New best model saved.")

if __name__ == '__main__':
    main()

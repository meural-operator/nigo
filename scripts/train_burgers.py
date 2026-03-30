import os
import sys
import h5py
import time
import argparse
import shutil
import datetime
import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from turbo_nigo.models.turbo_nigo import GlobalTurboNIGO

# --- Enterprise Logging & Management ---

class HistoryManager:
    """Manages immutable run directories, metric logging, and checkpointing."""
    def __init__(self, base_dir, run_id=None):
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id or f"RUN_BURGERS_{self.timestamp}"
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

class PreloadedBurgersDataset(Dataset):
    """
    High-throughput dataset for 1D Burgers Equation (Optimized for RAM).
    Reshapes 1D spatial grids (e.g., 1024) into 2D grids (32x32) natively 
    compatible with GlobalTurboNIGO 2D conv assumptions.
    """
    def __init__(self, h5_path, seq_len=20, mode="train", max_samples=None):
        super().__init__()
        self.seq_len = seq_len
        self.h5_path = h5_path
        
        print(f"  [{mode.upper()}] Preloading Burgers dataset into RAM...")
        with h5py.File(h5_path, 'r') as f:
            # We know the key is 'tensor' from our previous analysis
            raw = f['tensor'][:]  
            if max_samples:
                raw = raw[:max_samples]
                
            # Filter for train vs val
            N_total = raw.shape[0]
            val_split = int(0.1 * N_total)
            if mode == 'train':
                raw = raw[:-val_split]
            else:
                raw = raw[-val_split:]
                
        N, T, X = raw.shape
        self.N = N
        self.T = T
        
        # Verify perfect square for reshaping
        self.side = int(math.isqrt(X))
        assert self.side * self.side == X, f"Spatial size {X} must be a perfect square. Got side={self.side}"
        
        # Normalization
        g_min = float(raw.min())
        g_max = float(raw.max())
        
        raw_torch = torch.from_numpy(raw).float()
        raw_norm = (raw_torch - g_min) / (g_max - g_min + 1e-8)
        
        # Reshape to (Batch, Time, Channels, H, W) where Channels=1, H=32, W=32
        self.data_2d = raw_norm.view(N, T, 1, self.side, self.side)
        
        # No physical condition was provided, use a dummy tensor (zeros)
        # We use a 4D dummy vector to match default assumptions
        self.cond = torch.zeros(4, dtype=torch.float32)
        
        # Sliding windows over time
        self.index_map = []
        stride = 5 if mode == "train" else 20
        for i in range(N):
            for t0 in range(0, T - seq_len, stride):
                # (traj_idx, start_time_idx)
                self.index_map.append((i, t0))

    def __len__(self): 
        return len(self.index_map)
        
    def __getitem__(self, idx):
        n, t = self.index_map[idx]
        seq = self.data_2d[n, t : t + self.seq_len + 1] # shape: (seq_len+1, 1, 32, 32)
        return seq[0], seq[1:], self.cond

def train_one_epoch(model, loader, optimizer, scaler, device, use_amp):
    model.train()
    total_loss, count = 0.0, 0
    pbar = tqdm(loader, desc="Training")
    criterion = nn.MSELoss()
    accumulation_steps = 4
    
    for i, (x, y, cond) in enumerate(pbar):
        x, y, cond = x.to(device), y.to(device), cond.to(device)
        # Target shape for y is (B, Seq, C, H, W)
        time_steps = torch.linspace(0, 1.0, y.shape[1]).to(device)
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            # Cond vectors need an additional dimension to match API
            cond_batch = cond.unsqueeze(0).expand(x.shape[0], -1) if cond.ndim == 1 else cond 
            u_pred, _, _, _, _, _ = model(x, time_steps, cond_batch)
            loss = criterion(u_pred, y) / accumulation_steps
        
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
    criterion = nn.MSELoss()
    with torch.no_grad():
        for x, y, cond in loader:
            x, y, cond = x.to(device), y.to(device), cond.to(device)
            time_steps = torch.linspace(0, 1.0, y.shape[1]).to(device)
            # Match condition dimensions if needed
            cond_batch = cond.unsqueeze(0).expand(x.shape[0], -1) if cond.ndim == 1 else cond 
            with torch.amp.autocast('cuda', enabled=False):
                u_pred, _, _, _, _, _ = model(x, time_steps, cond_batch)
                loss = criterion(u_pred, y)
            total_loss += loss.item() * x.size(0)
            count += x.size(0)
    return total_loss / max(1, count)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./datasets/Burgers/1D_Burgers_Sols_Nu0.1.hdf5')
    parser.add_argument('--output_dir', type=str, default='./results/burgers_experiments')
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--max_samples', type=int, default=1000, help='Subset limit for memory')
    args = parser.parse_args()

    history = HistoryManager(args.output_dir)
    history.log_text(f"Starting Burgers Physics Training Platform")
    history.log_text(f"Run ID: {history.run_id} | Dir: {history.run_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')
    if use_amp:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Preload the datasets
    train_ds = PreloadedBurgersDataset(args.data_path, seq_len=args.seq_len, mode="train", max_samples=args.max_samples)
    val_ds = PreloadedBurgersDataset(args.data_path, seq_len=args.seq_len, mode="val", max_samples=args.max_samples)
    
    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 0, 'pin_memory': use_amp}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # Initialize GlobalTurboNIGO with the calculated structural parameters
    # The spatial_size will end up being the square root of our spatial dimensions (32 for a 1024 1D field)
    # The in_channels represents our physical field values count (1).
    model = GlobalTurboNIGO(
        latent_dim=64, 
        in_channels=1, 
        width=32, 
        spatial_size=train_ds.side, 
        num_layers=3, 
        use_residual=True, 
        norm_type='group'
    )
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
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
        
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp)
        val_loss = validate(model, val_loader, device)
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        history.log_metrics(epoch, train_loss, val_loss, lr)
        history.log_text(f"Metric: TrainLoss={train_loss:.6f} | ValLoss={val_loss:.6f} | LR={lr:.2e}")
        
        state = {
            'epoch': epoch, 
            'model_state': model.state_dict(), 
            'optimizer_state': optimizer.state_dict(),
            'scaler_state': scaler.state_dict() if scaler else None, 
            'scheduler_state': scheduler.state_dict() if scheduler else None, 
            'best_val_loss': best_val_loss
        }
        
        # 1. Type 1: Latest Checkpoint
        torch.save(state, os.path.join(history.run_dir, "latest_model.pth"))
        
        # 2. Type 2: Best Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(state, os.path.join(history.run_dir, "best_model.pth"))
            history.log_text("  [*] New best model saved.")

        # 3. Type 3: Every 30 Epochs Checkpoint
        if epoch % 30 == 0:
            torch.save(state, os.path.join(history.run_dir, f"epoch_{epoch}_model.pth"))
            history.log_text(f"  [+] Saved explicit checkpoint at epoch {epoch}")

if __name__ == '__main__':
    main()

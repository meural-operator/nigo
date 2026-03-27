"""
run_ablations_ks.py — Standalone Ablation Study for the Kuramoto-Sivashinsky Dataset.

This script is SELF-CONTAINED and does NOT modify any existing framework scripts.
It includes its own KS-specific data loader and runs the full ablation suite
(Baseline + 5 ablation variants) on the KS_ML_DATASET.h5 file.

Usage (on the secondary system):
  conda run -n cfd python scripts/run_ablations_ks.py --config turbo_nigo/configs/ks_config.yaml
  conda run -n cfd python scripts/run_ablations_ks.py --config turbo_nigo/configs/ks_config.yaml --epochs 100

Before running on the secondary system, modify ks_config.yaml:
  - data_root: path to KS_ML_DATASET.h5 on the target machine
  - results_dir: where to save outputs
  - num_workers: adjust for target CPU core count
  - device: "cuda" or "cpu"
"""

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Windows Conda DLL workaround for PyTorch (Python >= 3.8 ignores PATH for DLLs)
if os.name == 'nt' and 'CONDA_PREFIX' in os.environ:
    dll_path = os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'bin')
    if os.path.exists(dll_path):
        try: os.add_dll_directory(dll_path)
        except Exception: pass

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from turbo_nigo.configs import get_args_and_config
from turbo_nigo.utils import seed_everything, get_paths
from turbo_nigo.models import GlobalTurboNIGO
from turbo_nigo.models.ablations import (
    Ablation1_NoSkewTurboNIGO,
    Ablation2_NoDissipativeTurboNIGO,
    Ablation3_DenseGeneratorTurboNIGO,
    Ablation4_NoRefinerTurboNIGO,
    Ablation5_UnscaledTurboNIGO
)
from turbo_nigo.core import Trainer

try:
    import h5py
except ImportError:
    raise ImportError("h5py is required. Install: pip install h5py")

# =============================================================================
# Self-Contained KS Dataset Loader
# =============================================================================
class KSDataset(Dataset):
    """
    In-memory HDF5 dataset for the 1D Kuramoto-Sivashinsky equation.
    
    The KS_ML_DATASET.h5 contains:
      - 'train': shape (40000, 768, 512) float32  — 40k trajectories, 768 timesteps, 512 spatial points
      - 'test':  shape (10000, 768, 512) float64   — 10k trajectories
    
    This loader:
      1. Loads a subset of trajectories (max_cases) into memory.
      2. Downsamples spatially from 512 -> spatial_res using interpolation.
      3. Reshapes the 1D field into a pseudo-2D image (sqrt(spatial_res) x sqrt(spatial_res))
         to be compatible with the 2D convolutional encoder/decoder in TurboNIGO.
      4. Subsamples temporally with a configurable stride.
      5. Returns (initial_frame, target_sequence, conditions) tuples.
    """
    
    def __init__(self, h5_path: str, mode: str = 'train', seq_len: int = 20,
                 temporal_stride: int = 4, spatial_res: int = 64, 
                 max_cases: int = 5000, window_stride: int = 5):
        super().__init__()
        self.seq_len = seq_len
        self.spatial_res = spatial_res
        
        # Determine the 2D reshape dimensions
        side = int(math.isqrt(spatial_res))
        assert side * side == spatial_res, f"spatial_res={spatial_res} must be a perfect square for 2D reshape"
        self.side = side  # e.g., 64 -> 8x8
        
        self.data_cache = []
        self.cond_cache = []
        self.index_map = []
        
        # HDF5 key
        h5_key = 'train' if mode == 'train' else 'test'
        
        print(f"\n[KS-{mode}] Loading {h5_path} (key='{h5_key}')")
        with h5py.File(h5_path, 'r') as f:
            raw = f[h5_key]  # (N_cases, T, Nx)
            n_total = raw.shape[0]
            n_load = min(max_cases, n_total) if max_cases else n_total
            
            # For val mode, use last 10% of loaded cases
            if mode == 'train':
                start_idx, end_idx = 0, int(n_load * 0.9)
            else:
                start_idx, end_idx = int(n_load * 0.9), n_load
            
            print(f"  Loading cases [{start_idx}:{end_idx}] of {n_total} total")
            
            # Load the chunk into memory
            chunk = raw[start_idx:end_idx]  # (N, T, Nx) 
            chunk = np.array(chunk, dtype=np.float32)
        
        # Compute global normalization stats
        g_min = chunk.min()
        g_max = chunk.max()
        self.g_min = g_min
        self.g_max = g_max
        
        print(f"  Global stats: min={g_min:.4f}, max={g_max:.4f}")
        print(f"  Temporal stride: {temporal_stride}, Spatial: {chunk.shape[2]} -> {spatial_res} ({side}x{side})")
        
        n_cases = chunk.shape[0]
        for i in range(n_cases):
            traj = chunk[i]  # (T, Nx)
            
            # Temporal subsampling
            traj = traj[::temporal_stride]  # (T', Nx)
            
            # Normalize
            traj = (traj - g_min) / (g_max - g_min + 1e-8)
            
            # Spatial downsampling: (T', Nx) -> (T', 1, 1, Nx) -> interpolate -> (T', 1, 1, spatial_res) -> reshape
            traj_t = torch.from_numpy(traj).float().unsqueeze(1).unsqueeze(1)  # (T', 1, 1, Nx)
            traj_down = F.interpolate(traj_t, size=(1, spatial_res), mode='bilinear', align_corners=False)
            # traj_down: (T', 1, 1, spatial_res) -> reshape to (T', 1, side, side)
            traj_2d = traj_down.view(-1, 1, self.side, self.side)  # (T', 1, H, W)
            
            self.data_cache.append(traj_2d)
            
            # Condition vector: simple statistical fingerprint of the trajectory
            cond = torch.tensor([
                traj.mean(), traj.std(), 
                np.abs(np.fft.fft(traj[0])[:5]).mean(),  # spectral energy hint
                float(i) / n_cases  # trajectory index (diversity proxy)
            ], dtype=torch.float32)
            self.cond_cache.append(cond)
            
            # Sliding window
            n_steps = traj_2d.shape[0]
            stride_w = window_stride if mode == 'train' else max(window_stride * 4, 20)
            if n_steps > seq_len:
                for t in range(0, n_steps - seq_len, stride_w):
                    self.index_map.append((len(self.data_cache) - 1, t))
        
        print(f"  [KS-{mode}] Cached {len(self.data_cache)} trajectories, {len(self.index_map)} samples")
        if len(self.index_map) == 0:
            raise ValueError(f"KS Dataset [{mode}] is empty after processing!")
    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        case_idx, t0 = self.index_map[idx]
        full = self.data_cache[case_idx]
        cond = self.cond_cache[case_idx]
        
        window = full[t0 : t0 + self.seq_len + 1]
        x = window[0]       # (1, H, W)
        y = window[1:]       # (seq_len, 1, H, W)
        return x, y, cond


# =============================================================================
# Ablation Runner
# =============================================================================
def run_ablation(model_name, ModelClass, base_config, train_loader, val_loader):
    print(f"\n{'='*60}")
    print(f"  KS ABLATION: {model_name}")
    print(f"{'='*60}\n")
    
    config = base_config.copy()
    config["experiment_name"] = f"KS_Ablation_{model_name}"
    
    seed_everything(config.get("seed", 42))
    paths = get_paths(config)
    device = config.get("device", "cpu")
    
    # Determine spatial_size from the 2D reshape
    spatial_res = config.get("ks_spatial_res", 64)
    side = int(math.isqrt(spatial_res))
    
    model = ModelClass(
        latent_dim=config["latent_dim"], 
        num_bases=config["num_bases"], 
        cond_dim=config["cond_dim"],
        width=config["width"],
        spatial_size=side,
        in_channels=config.get("in_channels", 1)
    ).to(device)
    
    trainer = Trainer(model, train_loader, val_loader, config, paths)
    trainer.train()


def main():
    base_config = get_args_and_config()
    
    h5_path = base_config["data_root"]
    seq_len = base_config.get("seq_len", 20)
    temporal_stride = base_config.get("ks_temporal_stride", 4)
    spatial_res = base_config.get("ks_spatial_res", 64)
    max_cases = base_config.get("ks_train_cases", 5000)
    batch_size = base_config.get("batch_size", 64)
    num_workers = base_config.get("num_workers", 8)
    
    # Build datasets ONCE, reuse across all ablations
    print("=" * 60)
    print("  LOADING KS DATASET (shared across all ablations)")
    print("=" * 60)
    
    train_ds = KSDataset(h5_path, mode='train', seq_len=seq_len,
                         temporal_stride=temporal_stride, spatial_res=spatial_res,
                         max_cases=max_cases)
    val_ds = KSDataset(h5_path, mode='val', seq_len=seq_len,
                       temporal_stride=temporal_stride, spatial_res=spatial_res,
                       max_cases=max_cases)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, persistent_workers=True)
    
    # --- Ablation Suite ---
    models_to_test = [
        ("Baseline",        GlobalTurboNIGO),
        ("UnscaledGenerator", Ablation5_UnscaledTurboNIGO),
        ("NoSkew",          Ablation1_NoSkewTurboNIGO),
        ("NoDissipative",   Ablation2_NoDissipativeTurboNIGO),
        ("DenseGenerator",  Ablation3_DenseGeneratorTurboNIGO),
        ("NoRefiner",       Ablation4_NoRefinerTurboNIGO),
    ]
    
    for name, cls in models_to_test:
        run_ablation(name, cls, base_config, train_loader, val_loader)
    
    print("\n" + "=" * 60)
    print("  ALL KS ABLATIONS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

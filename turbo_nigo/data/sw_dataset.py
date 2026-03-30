import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class ShallowWaterDataset(Dataset):
    """
    In-memory Shallow Water Equation dataset from PDEBench.

    Loads from an HDF5 file formatted with '0000', '0001', ... groups.
    Each group has 'data' tensor of shape (T, X, Y, 1).
    """

    def __init__(
        self,
        h5_path: str,
        seq_len: int = 20,
        mode: str = "train",
        temporal_stride: int = 1,
        spatial_size: int = 128,
        max_trajectories: int = 1000,
        cond_dim: int = 4,
        g_min: Optional[float] = None,
        g_max: Optional[float] = None,
    ):
        super().__init__()
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for ShallowWaterDataset: pip install h5py")

        self.seq_len = seq_len
        self.spatial_size = spatial_size
        self.cond_dim = cond_dim
        
        # Default PDEBench Shallow Water dataset contains 1000 trajectories
        # We perform an implicit 90/10 split
        if mode == "train":
            start_idx = 0
            # Limit strictly to 900 max for training
            end_idx = min(900, max_trajectories)
        else:
            # Val takes from the last 100
            start_idx = 900
            end_idx = min(1000, 900 + max_trajectories)

        print(f"[{mode}] Loading Shallow Water data from {h5_path} [Traj {start_idx} to {end_idx-1}] ...")

        self.data_cache = []
        self.index_map = []
        
        global_min = float('inf') if g_min is None else g_min
        global_max = float('-inf') if g_max is None else g_max
        
        with h5py.File(h5_path, "r") as f:
            for i in range(start_idx, end_idx):
                key = f"{i:04d}"
                if key not in f:
                    print(f"[{mode}] Warning: Key {key} not found. Stopping load at index {i}.")
                    break
                
                # Shape: (T, X, Y, 1)
                traj = f[key]['data'][:]
                
                if g_min is None and g_max is None:
                    t_min = float(traj.min())
                    t_max = float(traj.max())
                    if t_min < global_min: global_min = t_min
                    if t_max > global_max: global_max = t_max
                
                self.data_cache.append(traj)
                
        self.g_min = global_min
        self.g_max = global_max

        processed_cache = []
        stride_win = 5 if mode == "train" else min(20, seq_len)
        total_samples = 0
        
        self.cond = torch.zeros(cond_dim, dtype=torch.float32)

        for ci, traj in enumerate(self.data_cache):
            eps = 1e-8
            # Normalize to [0, 1]
            traj = (traj - self.g_min) / (self.g_max - self.g_min + eps)
            
            # (T, X, Y, C=1) -> (T, C, X, Y)
            traj_th = torch.from_numpy(traj).float().permute(0, 3, 1, 2)
            
            # Subsample in time
            traj_th = traj_th[::temporal_stride]
            T = traj_th.shape[0]
            
            # Subsample or interpolate in space
            if traj_th.shape[-1] != spatial_size or traj_th.shape[-2] != spatial_size:
                traj_th = F.interpolate(
                    traj_th, size=(spatial_size, spatial_size),
                    mode="bilinear", align_corners=False
                )
                
            processed_cache.append(traj_th)
            
            # Create sliding windows
            for t0 in range(0, T - seq_len, stride_win):
                self.index_map.append((ci, t0))
                total_samples += 1

        self.data_cache = processed_cache
        
        print(f"[{mode}] ShallowWaterDataset: {total_samples} samples | "
              f"{len(self.data_cache)} trajectories | T={processed_cache[0].shape[0]} (stride {temporal_stride}) | "
              f"spatial={spatial_size}x{spatial_size} | "
              f"g_min={self.g_min:.4f}, g_max={self.g_max:.4f}")

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ci, t0 = self.index_map[idx]
        window = self.data_cache[ci][t0 : t0 + self.seq_len + 1]
        return window[0], window[1:], self.cond

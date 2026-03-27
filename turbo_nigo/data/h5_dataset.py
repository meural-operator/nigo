"""
H5FlowDataset — HDF5-based flow dataset for high-resolution simulations.

Supports the Navier-Stokes incompressible inhomogeneous 2D benchmark
(ns_incom_inhom_2d_512-102.h5 and similar). Provides bilinear downsampling,
force-field conditioning, and temporal stride matching.

Registered as 'h5_flow' in the dataset registry.
"""
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import h5py
except ImportError:
    h5py = None  # Gracefully degrade if h5py not installed


class H5FlowDataset(Dataset):
    """
    In-memory HDF5 flow dataset with bilinear downsampling.

    Expects an HDF5 file with:
      - 'velocity': (N_batches, T, H, W, 2) — velocity fields
      - 'force':    (N_batches, H, W, 2) — forcing fields

    Args:
        h5_path: Path to the HDF5 file.
        target_res: Target spatial resolution (bilinear downsampling).
        seq_len: Number of time steps per training sample.
        mode: 'train' or 'val' — determines which batches to use.
        train_batches: List of batch indices for training.
        val_batches: List of batch indices for validation.
        temporal_stride: Stride between consecutive time steps (for dt matching).
        window_stride: Stride for sliding the sample window over time.
        g_min: Global min for normalization.
        g_max: Global max for normalization.
    """

    def __init__(
        self,
        h5_path: str,
        target_res: int = 256,
        seq_len: int = 20,
        mode: str = "train",
        train_batches: Optional[list] = None,
        val_batches: Optional[list] = None,
        temporal_stride: int = 20,
        window_stride: int = 5,
        g_min: float = -3.0,
        g_max: float = 3.0,
    ):
        if h5py is None:
            raise ImportError(
                "h5py is required for H5FlowDataset. "
                "Install it: pip install h5py"
            )

        self.seq_len = seq_len
        self.g_min = g_min
        self.g_max = g_max
        self.data_cache = []
        self.cond_cache = []
        self.index_map = []

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"{h5_path} does not exist.")

        # Defaults: batches 0-2 for train, batch 3 for val
        if train_batches is None:
            train_batches = [0, 1, 2]
        if val_batches is None:
            val_batches = [3]

        target_batches = train_batches if mode == "train" else val_batches

        print(f"[{mode}] Loading HDF5: {h5_path} @ {target_res}×{target_res}")
        with h5py.File(h5_path, "r") as f:
            velocity = f["velocity"]  # (N, T, H, W, 2)
            force = f["force"]        # (N, H, W, 2)

            for b in target_batches:
                print(f"  Processing batch {b}...")
                raw_vel = velocity[b]  # (T, H, W, 2)

                # To torch: (T, 2, H, W) → downsample
                raw_t = torch.from_numpy(raw_vel).float().permute(0, 3, 1, 2)
                down_t = F.interpolate(
                    raw_t,
                    size=(target_res, target_res),
                    mode="bilinear",
                    align_corners=False,
                )

                # Normalize
                down_t = (down_t - g_min) / (g_max - g_min + 1e-8)
                self.data_cache.append(down_t)

                # Condition: force-field fingerprint (mean/std per channel)
                raw_force = force[b]  # (H, W, 2)
                f_u, f_v = raw_force[:, :, 0], raw_force[:, :, 1]
                cond_vec = torch.tensor(
                    [np.mean(f_u), np.std(f_u), np.mean(f_v), np.std(f_v)],
                    dtype=torch.float32,
                ) / 2.0
                self.cond_cache.append(cond_vec)

                # Sliding window indices
                n_steps = down_t.shape[0]
                for t in range(0, n_steps - seq_len * temporal_stride, window_stride):
                    self.index_map.append(
                        (len(self.data_cache) - 1, t, temporal_stride)
                    )

        print(f"[{mode}] Total samples: {len(self.index_map)}")
        if len(self.index_map) == 0:
            raise ValueError(
                f"Dataset [{mode}] is empty. Check HDF5 file structure "
                f"and temporal_stride={temporal_stride}."
            )

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        cache_idx, t0, step = self.index_map[idx]
        data = self.data_cache[cache_idx]
        cond = self.cond_cache[cache_idx]

        indices = [t0 + i * step for i in range(self.seq_len + 1)]
        window = data[indices]

        # x: initial frame, y: target sequence
        return window[0], window[1:], cond

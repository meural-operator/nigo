"""
KSDataset — Kuramoto-Sivashinsky equation data loader for TurboNIGO.

Loads from an HDF5 file with structure:
    - 'train': (N_train, T, Nx)   e.g. (40000, 768, 512)
    - 'test':  (N_test,  T, Nx)   e.g. (10000, 768, 512)

Processing pipeline:
    1. Temporal stride  — subsample every `temporal_stride` steps
    2. Spatial downsample — interpolate Nx → spatial_res points
    3. 1D → 2D reshape   — (spatial_res,) → (1, S, S)  where S²=spatial_res
    4. Sliding-window indexing for training samples

Conditioning: KS has no physical parameters to condition on; a zero vector
of dimension cond_dim is used for API compatibility.

Registered as 'ks' in the dataset registry.
"""
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class KSDataset(Dataset):
    """
    In-memory Kuramoto-Sivashinsky dataset.

    Args:
        h5_path:          Path to HDF5 file.
        seq_len:          Number of future time steps per sample.
        mode:             'train' or 'val' (maps to 'train'/'test' HDF5 groups).
        temporal_stride:  Subsample every N-th time step.
        spatial_res:      Target 1D spatial resolution after downsampling.
        max_trajectories: Cap on number of trajectories to load (memory control).
        cond_dim:         Conditioning vector dimension (dummy zeros).
        g_min / g_max:    Pre-computed normalisation bounds (optional).
    """

    def __init__(
        self,
        h5_path: str,
        seq_len: int = 20,
        mode: str = "train",
        temporal_stride: int = 4,
        spatial_res: int = 64,
        max_trajectories: int = 5000,
        cond_dim: int = 4,
        g_min: Optional[float] = None,
        g_max: Optional[float] = None,
    ):
        super().__init__()
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for KSDataset: pip install h5py")

        self.seq_len = seq_len
        self.side = int(math.isqrt(spatial_res))
        assert self.side * self.side == spatial_res, (
            f"spatial_res={spatial_res} must be a perfect square for 2D reshape"
        )

        h5_key = "train" if mode == "train" else "test"
        print(f"[{mode}] Loading KS data from {h5_path} [{h5_key}] ...")

        with __import__("h5py").File(h5_path, "r") as f:
            raw = f[h5_key][:max_trajectories]  # (N, T, Nx)

        N, T_full, Nx = raw.shape

        # Temporal stride
        raw = raw[:, ::temporal_stride, :]
        T = raw.shape[1]

        # Normalise
        if g_min is None:
            g_min, g_max = float(raw.min()), float(raw.max())
        self.g_min, self.g_max = g_min, g_max
        raw = (raw - g_min) / (g_max - g_min + 1e-8)

        # Spatial downsample + 2D reshape → cache
        self.data_cache: list = []
        self.index_map: list = []

        cond = torch.zeros(cond_dim, dtype=torch.float32)
        self.cond = cond

        stride_win = 5 if mode == "train" else 20

        for i in range(N):
            traj_1d = torch.from_numpy(raw[i]).float()  # (T, Nx)
            # Downsample spatially: (T, Nx) → (T, spatial_res)
            traj_1d = F.interpolate(
                traj_1d.unsqueeze(1), size=spatial_res,
                mode="linear", align_corners=False,
            ).squeeze(1)  # (T, spatial_res)

            # Reshape to 2D frames: (T, 1, side, side)
            frames = traj_1d.view(T, 1, self.side, self.side)
            self.data_cache.append(frames)

            for t0 in range(0, T - seq_len, stride_win):
                self.index_map.append((len(self.data_cache) - 1, t0))

        print(
            f"[{mode}] KSDataset: {len(self.index_map)} samples | "
            f"{N} trajectories | T={T} (stride {temporal_stride}) | "
            f"spatial {Nx}→{spatial_res} ({self.side}×{self.side})"
        )

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ci, t0 = self.index_map[idx]
        window = self.data_cache[ci][t0: t0 + self.seq_len + 1]
        return window[0], window[1:], self.cond

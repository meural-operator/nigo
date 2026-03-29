"""
BurgersDataset — Burgers equation data loader for TurboNIGO.

Supports:
  - .mat files (FNO benchmark):  keys 'a' (N, Nx) + 'u' (N, Nx)  [single-step]
  - .mat files (temporal):       key  'u' (N, T, Nx)               [trajectory]
  - .npy / .h5 with same shapes

1D spatial data (Nx,) is bilinearly interpolated to a target_res² grid
and reshaped to (1, target_res, target_res) for 2D-conv compatibility.

Registered as 'burgers' in the dataset registry.
"""
import os
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class BurgersDataset(Dataset):
    """
    In-memory Burgers equation dataset.

    Args:
        data_path:          Path to .mat / .npy / .h5 file.
        seq_len:            Number of future frames per sample.
        mode:               'train' or 'val'.
        target_spatial_res: 2D grid side length after reshaping.
        train_split:        Fraction of trajectories used for training.
        viscosity:          ν parameter — used as conditioning scalar.
        g_min / g_max:      Pre-computed normalisation bounds (optional).
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int = 20,
        mode: str = "train",
        target_spatial_res: int = 64,
        train_split: float = 0.8,
        viscosity: float = 0.1,
        g_min: Optional[float] = None,
        g_max: Optional[float] = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.mode = mode
        self.target_res = target_spatial_res
        self.index_map: list = []
        self.data_cache: list = []
        self.cond_cache: list = []
        self.is_temporal = False

        # ------ load raw data ------
        raw, raw_out = self._load(data_path)

        # ------ train / val split ------
        N = raw.shape[0]
        split = int(train_split * N)
        idx = slice(None, split) if mode == "train" else slice(split, None)
        raw = raw[idx]
        if raw_out is not None:
            raw_out = raw_out[idx]

        # ------ normalise ------
        if g_min is None:
            vals = raw if raw_out is None else np.concatenate([raw.ravel(), raw_out.ravel()])
            g_min = float(vals.min())
            g_max = float(vals.max())
        self.g_min, self.g_max = g_min, g_max
        raw = (raw - g_min) / (g_max - g_min + 1e-8)
        if raw_out is not None:
            raw_out = (raw_out - g_min) / (g_max - g_min + 1e-8)

        # ------ build cache ------
        cond = torch.tensor([viscosity, 0.0, 0.0, 0.0], dtype=torch.float32)

        if raw.ndim == 3:                       # (N, T, Nx) — temporal
            self.is_temporal = True
            self._cache_temporal(raw, cond)
        elif raw_out is not None:               # (N, Nx) paired input / output
            self._cache_single_step(raw, raw_out, cond)
        else:
            raise ValueError(
                f"Cannot determine data layout. shape={raw.shape}, "
                "expected (N,T,Nx) or paired (N,Nx)+(N,Nx)."
            )

        print(
            f"[{mode}] BurgersDataset: {len(self.index_map)} samples | "
            f"temporal={self.is_temporal} | res={target_spatial_res}²"
        )

    # ------------------------------------------------------------------ IO
    def _load(self, path: str):
        """Returns (input_array, output_array_or_None)."""
        if path.endswith(".mat"):
            try:
                from scipy.io import loadmat
            except ImportError:
                raise ImportError("scipy is required for .mat files")
            mat = loadmat(path)
            if "a" in mat and "u" in mat:
                return mat["a"].astype(np.float32), mat["u"].astype(np.float32)
            for k in ("u", "data"):
                if k in mat and mat[k].ndim == 3:
                    return mat[k].astype(np.float32), None
            key = [k for k in mat if not k.startswith("_")][0]
            return mat[key].astype(np.float32), None

        if path.endswith((".h5", ".hdf5")):
            import h5py
            with h5py.File(path, "r") as f:
                keys = list(f.keys())
                return np.array(f[keys[0]], dtype=np.float32), None

        if path.endswith(".npy"):
            return np.load(path).astype(np.float32), None

        raise ValueError(f"Unsupported format: {path}")

    # ------------------------------------------------------------------ reshape
    def _to_2d(self, x_1d: torch.Tensor) -> torch.Tensor:
        """(Nx,) → (1, S, S) via linear interpolation."""
        Nx = x_1d.shape[-1]
        x = x_1d.view(1, 1, Nx)
        S2 = self.target_res * self.target_res
        x = F.interpolate(x, size=S2, mode="linear", align_corners=False)
        return x.view(1, self.target_res, self.target_res)

    # ------------------------------------------------------------------ cache
    def _cache_temporal(self, data: np.ndarray, cond: torch.Tensor):
        N, T, Nx = data.shape
        stride = 5 if self.mode == "train" else 20
        for i in range(N):
            frames = torch.stack(
                [self._to_2d(torch.from_numpy(data[i, t])) for t in range(T)],
                dim=0,
            )  # (T, 1, S, S)
            self.data_cache.append(frames)
            self.cond_cache.append(cond)
            for t0 in range(0, T - self.seq_len, stride):
                self.index_map.append((len(self.data_cache) - 1, t0))

    def _cache_single_step(self, a: np.ndarray, u: np.ndarray, cond: torch.Tensor):
        N = a.shape[0]
        for i in range(N):
            inp = self._to_2d(torch.from_numpy(a[i]))
            out = self._to_2d(torch.from_numpy(u[i]))
            self.data_cache.append(torch.stack([inp, out], dim=0))  # (2,1,S,S)
            self.cond_cache.append(cond)
            self.index_map.append((len(self.data_cache) - 1, 0))

    # ------------------------------------------------------------------ API
    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ci, t0 = self.index_map[idx]
        data = self.data_cache[ci]
        cond = self.cond_cache[ci]
        if self.is_temporal:
            window = data[t0: t0 + self.seq_len + 1]
            return window[0], window[1:], cond
        else:
            return data[0], data[1:], cond

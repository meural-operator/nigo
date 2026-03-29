"""
DarcyFlowDataset — Darcy flow (elliptic PDE) data loader for TurboNIGO.

Darcy flow is a steady-state (elliptic) problem: coefficient field a(x,y)
maps to solution field u(x,y).  No temporal evolution.

Adaptation strategy:
  The TurboNIGO latent evolution z(t)=exp(At)·z₀ is evaluated at a single
  pseudo-time step (seq_len=1), making the generator act as a learned
  nonlinear operator: u ≈ D(exp(A·Δt)·E(a)).

Data formats:
  - .mat  keys 'coeff'/'a' (N,H,W) + 'sol'/'u' (N,H,W)
  - .npy  directory with input.npy + output.npy
  - .h5   any layout with two main datasets

Spatial resolution is bilinearly downsampled to target_res × target_res.
Conditioning vector is derived from coefficient-field statistics:
  [mean(a), std(a), max(a), energy(a)]

Registered as 'darcy' in the dataset registry.
"""
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class DarcyFlowDataset(Dataset):
    """
    In-memory Darcy flow dataset.

    Args:
        data_path:          Path to .mat / .h5 file or directory with .npy files.
        seq_len:            Must be 1 for Darcy (single-step mapping).
        mode:               'train' or 'val'.
        target_res:         Target spatial resolution after downsampling.
        train_split:        Fraction of samples for training.
        g_min / g_max:      Pre-computed normalisation (optional).
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int = 1,
        mode: str = "train",
        target_res: int = 64,
        train_split: float = 0.8,
        g_min: Optional[float] = None,
        g_max: Optional[float] = None,
    ):
        super().__init__()
        assert seq_len == 1, "Darcy flow is single-step; seq_len must be 1."
        self.target_res = target_res

        # ------ load ------
        coeffs, sols = self._load(data_path)  # (N,H,W) each

        # ------ split ------
        N = coeffs.shape[0]
        split = int(train_split * N)
        idx = slice(None, split) if mode == "train" else slice(split, None)
        coeffs, sols = coeffs[idx], sols[idx]

        # ------ normalise (joint min/max over both fields) ------
        if g_min is None:
            g_min = float(min(coeffs.min(), sols.min()))
            g_max = float(max(coeffs.max(), sols.max()))
        self.g_min, self.g_max = g_min, g_max
        coeffs = (coeffs - g_min) / (g_max - g_min + 1e-8)
        sols = (sols - g_min) / (g_max - g_min + 1e-8)

        # ------ downsample + cache ------
        self.inputs: list = []
        self.targets: list = []
        self.conds: list = []

        for i in range(coeffs.shape[0]):
            a_2d = self._downsample(torch.from_numpy(coeffs[i]))  # (1,S,S)
            u_2d = self._downsample(torch.from_numpy(sols[i]))    # (1,S,S)
            self.inputs.append(a_2d)
            self.targets.append(u_2d.unsqueeze(0))  # (1,1,S,S) — seq_len=1

            # Conditioning from coefficient-field statistics
            cond = torch.tensor(
                [coeffs[i].mean(), coeffs[i].std(), coeffs[i].max(),
                 (coeffs[i] ** 2).mean()],
                dtype=torch.float32,
            )
            self.conds.append(cond)

        print(
            f"[{mode}] DarcyFlowDataset: {len(self.inputs)} samples | "
            f"res={target_res}² | single-step"
        )

    # ------------------------------------------------------------------ IO
    @staticmethod
    def _load(path: str):
        if path.endswith(".mat"):
            try:
                from scipy.io import loadmat
            except ImportError:
                raise ImportError("scipy required for .mat: pip install scipy")
            mat = loadmat(path)
            a_key = next(k for k in ("coeff", "a") if k in mat)
            u_key = next(k for k in ("sol", "u") if k in mat)
            return mat[a_key].astype(np.float32), mat[u_key].astype(np.float32)

        if path.endswith((".h5", ".hdf5")):
            import h5py
            with h5py.File(path, "r") as f:
                keys = sorted(f.keys())
                return (np.array(f[keys[0]], dtype=np.float32),
                        np.array(f[keys[1]], dtype=np.float32))

        if os.path.isdir(path):
            a = np.load(os.path.join(path, "input.npy")).astype(np.float32)
            u = np.load(os.path.join(path, "output.npy")).astype(np.float32)
            return a, u

        raise ValueError(f"Unsupported Darcy data path: {path}")

    def _downsample(self, field: torch.Tensor) -> torch.Tensor:
        """(H,W) → (1, target_res, target_res) via bilinear interpolation."""
        x = field.unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)
        x = F.interpolate(
            x, size=(self.target_res, self.target_res),
            mode="bilinear", align_corners=False,
        )
        return x.squeeze(0)  # (1, S, S)

    # ------------------------------------------------------------------ API
    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx], self.conds[idx]

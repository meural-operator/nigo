import os
import numpy as np
import torch
import h5py
from typing import Tuple

from .base_dataset import Base3DDataset

class NS3DDataset(Base3DDataset):
    """
    3D Navier-Stokes Dataset loader for high-resolution 3D turbulence.
    
    Expects an HDF5 file with structured 5D arrays under keys: 
    'Vx', 'Vy', 'Vz', 'density', 'pressure'
    with shape (N_samples, T, Z, Y, X).
    
    Transforms data into physical tensors of shape (N, T, C, D, H, W)
    for downstream NIGO inference and generalized visualization.
    """
    def __init__(self, root_dir: str, seq_len: int, mode: str = 'train', **kwargs):
        super().__init__(root_dir, seq_len, mode, **kwargs)
        self.file_path = os.path.join(root_dir, "3D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"[!] HDF5 dataset not found at {self.file_path}. Please verify the path.")
            
        # Target channels we want to pack into the model tensor
        self.channel_keys = ['Vx', 'Vy', 'Vz', 'density', 'pressure']
        self._setup_dataset()
        
    def _setup_dataset(self) -> None:
        print(f"[{self.mode.upper()}] Initializing NS3D Dataset from {self.file_path}...")
        
        # We don't load the entire 80GB structure into RAM immediately.
        # We read the meta-shapes to configure boundaries and statistics.
        with h5py.File(self.file_path, 'r') as f:
            self.total_samples = f[self.channel_keys[0]].shape[0]
            self.total_timesteps = f[self.channel_keys[0]].shape[1]
            
            # Subsample statistics for fast normalization rather than exhaustive search
            g_mins, g_maxs = [], []
            sample_ids = np.linspace(0, self.total_samples - 1, min(10, self.total_samples), dtype=int)
            
            for key in self.channel_keys:
                # Load sparse slices to estimate boundaries
                slices = f[key][sample_ids, :, :, :, :]
                g_mins.append(float(np.min(slices)))
                g_maxs.append(float(np.max(slices)))
                
        # To maintain uniform gradient conditioning, we track global dynamic ranges
        self.global_min = min(g_mins)
        self.global_max = max(g_maxs)
        
        # Default conditioning (could be refined to Mach/Reynolds number if metadata available)
        self.cond_mean = torch.zeros(1)
        self.cond_std = torch.ones(1)
        
        # Validate sequence slicing
        if self.seq_len >= self.total_timesteps:
            raise ValueError(f"Requested seq_len={self.seq_len} exceeds available timesteps={self.total_timesteps}")
            
    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dynamically loads the sample from HDF5 across all targeted physical channels,
        constructs the spatiotemporal cube, and normalizes it bounds.
        """
        channels_data = []
        with h5py.File(self.file_path, 'r') as f:
            for key in self.channel_keys:
                # Read exactly (seq_len + 1) steps to form init and trajectory targets
                raw_seq = f[key][idx, :self.seq_len + 1, :, :, :]
                channels_data.append(raw_seq)
                
        # Stack into (T, C, D, H, W)
        # channels_data is a list of arrays shaped (T, D, H, W)
        cube = np.stack(channels_data, axis=1)
        
        cube_tensor = torch.from_numpy(cube).float()
        
        # In-situ normalization
        cube_tensor = (cube_tensor - self.global_min) / (self.global_max - self.global_min + 1e-8)
        
        # Split into x=t0, y=t1..T
        x = cube_tensor[0]
        y = cube_tensor[1:]
        
        # Dummy conditioning parameter (e.g. Reynolds = 1.0)
        cond = torch.tensor([1.0], dtype=torch.float32)
        
        return x, y, cond

    def get_slice(self, sample_idx: int, channel: str, time_step: int, plane: str, slice_idx: int) -> np.ndarray:
        """Lazily read a 2D slice directly off disk."""
        if channel not in self.channel_keys:
            raise ValueError(f"Channel {channel} not found.")
        
        with h5py.File(self.file_path, 'r') as f:
            if plane == 'x':
                # Slice along X-axis -> shape (Y, Z) or (Z, Y). D is Z. (D,H,W) -> (Z,Y,X)
                # D=Z, H=Y, W=X
                sl = f[channel][sample_idx, time_step, :, :, slice_idx]
            elif plane == 'y':
                sl = f[channel][sample_idx, time_step, :, slice_idx, :]
            elif plane == 'z':
                sl = f[channel][sample_idx, time_step, slice_idx, :, :]
            else:
                raise ValueError("Plane must be 'x', 'y' or 'z'.")
        
        # In-situ normalization bounds map
        mn, mx = self.global_min, self.global_max
        sl_norm = (sl - mn) / (mx - mn + 1e-8)
        return sl_norm
    
    def get_volume(self, sample_idx: int, channel: str, time_step: int) -> np.ndarray:
        """Lazily read a 3D block directly off disk."""
        if channel not in self.channel_keys:
            raise ValueError(f"Channel {channel} not found.")
        
        with h5py.File(self.file_path, 'r') as f:
            vol = f[channel][sample_idx, time_step, :, :, :]
            
        mn, mx = self.global_min, self.global_max
        vol_norm = (vol - mn) / (mx - mn + 1e-8)
        return vol_norm

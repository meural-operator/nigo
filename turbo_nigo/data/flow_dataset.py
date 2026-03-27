import os
import glob
import torch
import numpy as np
from tqdm import tqdm
from typing import Tuple

from .base_dataset import BaseOperatorDataset
from .utils import read_meta, compute_global_stats_and_cond_stats

class InMemoryFlowDataset(BaseOperatorDataset):
    """
    Dataset for fluid flow simulations. Loads all sequences into memory.
    Yields (initial_frame, target_sequence, conditions).
    """

    def _setup_dataset(self) -> None:
        # Avoid re-computation if stats are provided externally
        if self.global_min is None or self.cond_mean is None:
            g_min, g_max, c_mean, c_std = compute_global_stats_and_cond_stats(self.root_dir)
            self.global_min = g_min
            self.global_max = g_max
            self.cond_mean = c_mean
            self.cond_std = c_std
            
        self.cond_mean_t = torch.from_numpy(self.cond_mean).float()
        self.cond_std_t = torch.from_numpy(self.cond_std).float()

        self.data_cache = []
        self.cond_cache = []
        self.index_map = []
        
        case_dirs = sorted(glob.glob(os.path.join(self.root_dir, "case*")))
        if not case_dirs:
            raise ValueError(f"No case directories found in {self.root_dir}")
            
        max_cases = self.kwargs.get("max_cases", None)
        if max_cases is not None and max_cases > 0:
            case_dirs = case_dirs[:max_cases]

        split_idx = int(0.9 * len(case_dirs))
        target = case_dirs[:split_idx] if self.mode == 'train' else case_dirs[split_idx:]
        if not target and self.mode == 'val': 
            target = [case_dirs[-1]]
            
        valid_cases = 0
        
        for c in tqdm(target, desc=f"Loading {self.mode} Data"):
            try:
                # Load Data
                u = np.load(os.path.join(c, "u.npy")) # (T, H, W)
                v = np.load(os.path.join(c, "v.npy")) # (T, H, W)
                
                data = np.stack([u, v], axis=1).astype(np.float32) 
                
                # Normalize Data
                data = (data - self.global_min) / (self.global_max - self.global_min + 1e-8)
                
                # Load & Normalize Meta
                meta = read_meta(c)
                cond_arr = np.array([
                    meta.get("Re",0), meta.get("radius",0), 
                    meta.get("inlet_velocity",0), meta.get("bc_type",0)
                ], dtype=np.float32)
                
                cond_t = torch.from_numpy(cond_arr).float()
                cond_t = (cond_t - self.cond_mean_t) / (self.cond_std_t + 1e-8)
                
                # Cache
                self.data_cache.append(torch.from_numpy(data))
                self.cond_cache.append(cond_t)
                
                # Create Indices
                n_steps = data.shape[0]
                stride = 5 if self.mode == 'train' else 20
                
                if n_steps > self.seq_len:
                    for t in range(0, n_steps - self.seq_len, stride):
                        self.index_map.append((valid_cases, t))
                    valid_cases += 1
            except Exception:
                pass
        
        if len(self.index_map) == 0:
            raise ValueError("Dataset is empty! Check if cases exist.")

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        case_idx, t0 = self.index_map[idx]
        full = self.data_cache[case_idx]
        cond = self.cond_cache[case_idx]
        
        window = full[t0 : t0 + self.seq_len + 1]
        
        # Input: Frame 0
        x = window[0]
        # Target: Frame 1 to seq_len
        y = window[1:]
        
        return x, y, cond

    @classmethod
    def create_with_stats(cls, root_dir: str, seq_len: int, mode: str, 
                          g_min: float, g_max: float, c_mean: np.ndarray, c_std: np.ndarray, **kwargs):
        """Helper to create dataset using pre-computed statistics."""
        instance = cls(root_dir, seq_len, mode, **kwargs)
        instance.global_min = g_min
        instance.global_max = g_max
        instance.cond_mean = c_mean
        instance.cond_std = c_std
        instance._setup_dataset()
        return instance

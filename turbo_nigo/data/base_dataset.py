from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import torch
import numpy as np
from torch.utils.data import Dataset

class BaseOperatorDataset(Dataset, ABC):
    """
    Abstract base class for Operator Learning datasets in the TurboNIGO framework.
    All datasets should inherit from this and implement the required methods to ensure
    compatibility with the standard training loop.
    """
    def __init__(self, root_dir: str, seq_len: int, mode: str = 'train', **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.mode = mode
        self.kwargs = kwargs
        
        # Datasets must populate these via self._setup_dataset()
        self.global_min = None
        self.global_max = None
        self.cond_mean = None
        self.cond_std = None

    @abstractmethod
    def _setup_dataset(self) -> None:
        """
        Scan dataset, compute normalization statistics, build cache/index map.
        Must populate:
        - self.global_min (float or Tensor)
        - self.global_max (float or Tensor)
        - self.cond_mean (Tensor)
        - self.cond_std (Tensor)
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve a single sample.
        
        Returns:
            Tuple containing:
            - x (Tensor): Initial condition/input frame. Shape: (Channels, H, W)
            - y (Tensor): Target sequence of frames. Shape: (seq_len, Channels, H, W)
            - cond (Tensor): Normalized conditions/parameters vector. Shape: (cond_dim,)
        """
        pass

    def get_normalization_stats(self) -> Dict[str, Any]:
        """
        Returns the dataset normalization statistics required for evaluation/rollout.
        """
        return {
            "global_min": self.global_min,
            "global_max": self.global_max,
            "cond_mean": self.cond_mean,
            "cond_std": self.cond_std
        }

class Base3DDataset(BaseOperatorDataset):
    """
    Abstract interface guaranteeing $O(1)$ memory queries for giant 3D datasets.
    Implements flexible slicing paradigms required by the advanced query module.
    """
    @abstractmethod
    def get_slice(self, sample_idx: int, channel: str, time_step: int, plane: str, slice_idx: int) -> np.ndarray:
        """
        Retrieves a strictly minimal 2D spatial slice out of the massive multi-dimensional structure.
        """
        pass

    @abstractmethod
    def get_volume(self, sample_idx: int, channel: str, time_step: int) -> np.ndarray:
        """
        Retrieves an exact pure 3D physical volume specifically mapped to a singular timestep.
        """
        pass

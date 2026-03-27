import os
import math
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Tuple, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from turbo_nigo.data.base_dataset import BaseOperatorDataset

class AbstractDatasetAnalyzer(ABC):
    """
    Abstract interface for dataset visualization and property analysis.
    Ensures any dataset in the TurboNIGO framework can be consistently analyzed.
    """
    
    @abstractmethod
    def plot_sample(self, idx: int, time_steps: Optional[List[int]] = None, save_path: Optional[str] = None) -> Figure:
        """Plot the spatial domains of the initial condition and target sequence."""
        pass
        
    @abstractmethod
    def plot_spectrum(self, idx: int, save_path: Optional[str] = None) -> Figure:
        """Plot the 2D spatial power spectrum of a sample."""
        pass
        
    @abstractmethod
    def plot_temporal_evolution(self, idx: int, save_path: Optional[str] = None) -> Figure:
        """Analyze how scalar properties (e.g. kinetic energy) evolve over the sequence."""
        pass
        
    @abstractmethod
    def compute_dataset_statistics(self, num_samples: int = 100) -> Dict[str, float]:
        """Compute rich physical statistics across a subset of the dataset."""
        pass


class DatasetAnalyzer(AbstractDatasetAnalyzer):
    """
    Concrete implementation of the dataset analyzer.
    Works universally with any dataset inheriting from BaseOperatorDataset.
    Automatically un-normalizes tensors using the dataset's built-in stats.
    """
    
    def __init__(self, dataset: BaseOperatorDataset):
        self.dataset = dataset
        self.stats = dataset.get_normalization_stats()
        
    def _unnormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """Helper to un-normalize data back to physical scale."""
        g_min = self.stats.get("global_min")
        g_max = self.stats.get("global_max")
        
        arr = tensor.detach().cpu().numpy()
        if g_min is not None and g_max is not None:
            # Reverse min-max scaling assumes original data was scaled to [0, 1] or [-1, 1].
            # Based on BaseOperatorDataset conventions, typically scaled to [0, 1].
            arr = arr * (g_max - g_min) + g_min
        return arr
        
    def plot_sample(self, idx: int, time_steps: Optional[List[int]] = None, save_path: Optional[str] = None) -> Figure:
        x, y, cond = self.dataset[idx]
        x_phys = self._unnormalize(x)  # (C, H, W)
        y_phys = self._unnormalize(y)  # (T, C, H, W)
        
        channels = x_phys.shape[0]
        seq_len = y_phys.shape[0]
        
        if time_steps is None:
            time_steps = [0, seq_len // 2, seq_len - 1]
            if seq_len < 3:
                time_steps = list(range(seq_len))
                
        num_cols = 1 + len(time_steps)
        fig, axes = plt.subplots(channels, num_cols, figsize=(4 * num_cols, 4 * channels))
        if channels == 1:
            axes = np.expand_dims(axes, axis=0)
            
        for c in range(channels):
            # Input condition
            im = axes[c, 0].imshow(x_phys[c], cmap='RdBu_r', origin='lower')
            axes[c, 0].set_title(f"Input C{c} (t=0)")
            plt.colorbar(im, ax=axes[c, 0], fraction=0.046, pad=0.04)
            axes[c, 0].axis('off')
            
            # Target sequence
            for j, t in enumerate(time_steps):
                if t < seq_len:
                    im = axes[c, j+1].imshow(y_phys[t, c], cmap='RdBu_r', origin='lower')
                    axes[c, j+1].set_title(f"Target C{c} (t={t+1})")
                    plt.colorbar(im, ax=axes[c, j+1], fraction=0.046, pad=0.04)
                axes[c, j+1].axis('off')
                
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    def plot_spectrum(self, idx: int, save_path: Optional[str] = None) -> Figure:
        x, _, _ = self.dataset[idx]
        x_phys = self._unnormalize(x)  # (C, H, W)
        channels = x_phys.shape[0]
        
        fig, axes = plt.subplots(1, channels, figsize=(5 * channels, 4))
        if channels == 1:
            axes = [axes]
            
        for c in range(channels):
            # 2D FFT
            field = x_phys[c]
            fft2 = np.fft.fft2(field)
            fft2_shifted = np.fft.fftshift(fft2)
            power_spectrum = np.log(np.clip(np.abs(fft2_shifted)**2, 1e-10, None))
            
            im = axes[c].imshow(power_spectrum, cmap='magma', origin='lower')
            axes[c].set_title(f"Log Power Spectrum C{c}")
            plt.colorbar(im, ax=axes[c], fraction=0.046, pad=0.04)
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    def plot_temporal_evolution(self, idx: int, save_path: Optional[str] = None) -> Figure:
        _, y, _ = self.dataset[idx]
        y_phys = self._unnormalize(y)  # (T, C, H, W)
        
        seq_len, channels, H, W = y_phys.shape
        times = np.arange(1, seq_len + 1)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        
        for c in range(channels):
            # Vector operations: kinetic energy ~ u^2 + v^2. For simplicity, just plot spatial variance
            mean_energy = np.mean(y_phys[:, c]**2, axis=(1, 2))
            ax.plot(times, mean_energy, marker='o', label=f'Mean Energy C{c}')
            
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Spatial Energy (mean squared)")
        ax.set_title("Temporal Evolution of Energy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    def compute_dataset_statistics(self, num_samples: int = 100) -> Dict[str, float]:
        num_samples = min(num_samples, len(self.dataset))
        total_energy = 0.0
        max_val = -float('inf')
        min_val = float('inf')
        
        for idx in range(num_samples):
            x, _, _ = self.dataset[idx]
            x_phys = self._unnormalize(x)
            
            total_energy += np.mean(x_phys**2)
            max_val = max(max_val, np.max(x_phys))
            min_val = min(min_val, np.min(x_phys))
            
        return {
            "mean_spatial_energy": total_energy / num_samples,
            "global_max": float(max_val),
            "global_min": float(min_val),
            "samples_analyzed": num_samples
        }

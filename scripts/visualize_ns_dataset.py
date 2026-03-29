"""
visualize_ns_dataset.py — ICML Publication-Quality Visualizations for Naval Stokes h5 dataset.
"""

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Windows Conda DLL workaround for Python >= 3.8
if os.name == 'nt' and 'CONDA_PREFIX' in os.environ:
    dll_path = os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'bin')
    if os.path.exists(dll_path):
        try: os.add_dll_directory(dll_path)
        except Exception: pass

import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LATEX_AVAILABLE = False
try:
    import subprocess
    result = subprocess.run(['pdflatex', '--version'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0: 
        LATEX_AVAILABLE = True
except Exception: pass

ICML_RC = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
}
plt.rcParams.update(ICML_RC)

from turbo_nigo.data.base_dataset import BaseOperatorDataset
from turbo_nigo.data.analyzer import DatasetAnalyzer
import json
import h5py
import torch
import torch.nn.functional as F

class MinimalH5Dataset(BaseOperatorDataset):
    def __init__(self, h5_path, target_res=256, seq_len=20, temporal_stride=20, num_samples=5):
        super().__init__(root_dir=h5_path, seq_len=seq_len, mode='test')
        self.h5_path = h5_path
        self.target_res = target_res
        self.temporal_stride = temporal_stride
        self.g_min = -3.0
        self.g_max = 3.0
        self.num_samples = num_samples
        self.cond_mean = None
        self.cond_std = None

    def __len__(self):
        return self.num_samples
        
    def _setup_dataset(self):
        pass

    def __getitem__(self, idx):
        # Read a slice of T steps from batch 0
        t0 = idx * 5
        T_need = self.seq_len * self.temporal_stride + 1
        with h5py.File(self.h5_path, 'r') as f:
            velocity = f['velocity']
            force = f['force']
            raw_vel = velocity[0, t0:t0+T_need:self.temporal_stride] # (seq_len+1, H, W, 2)
            raw_force = force[0] # (H, W, 2)
            
        raw_t = torch.from_numpy(raw_vel).float().permute(0, 3, 1, 2)
        down_t = F.interpolate(raw_t, size=(self.target_res, self.target_res), mode="bilinear", align_corners=False)
        down_t = (down_t - self.g_min) / (self.g_max - self.g_min + 1e-8)
        
        f_u, f_v = raw_force[:, :, 0], raw_force[:, :, 1]
        cond_vec = torch.tensor([np.mean(f_u), np.std(f_u), np.mean(f_v), np.std(f_v)], dtype=torch.float32) / 2.0
        
        return down_t[0], down_t[1:], cond_vec
        
    def get_normalization_stats(self):
        return {
            "global_min": self.g_min,
            "global_max": self.g_max,
            "cond_mean": self.cond_mean,
            "cond_std": self.cond_std
        }

def _save_fig(fig, path_stem):
    fig.savefig(f"{path_stem}.png", dpi=300, bbox_inches='tight', pad_inches=0.08)
    if LATEX_AVAILABLE:
        try:
            plt.rcParams.update({"text.usetex": True})
            fig.savefig(f"{path_stem}.pdf", bbox_inches='tight', pad_inches=0.08)
            plt.rcParams.update({"text.usetex": False})
        except Exception:
            plt.rcParams.update({"text.usetex": False})
            fig.savefig(f"{path_stem}.pdf", bbox_inches='tight', pad_inches=0.08)
    else:
        fig.savefig(f"{path_stem}.pdf", bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./datasets/ns_incom_inhom_2d_512-0.h5')
    parser.add_argument('--output_dir', type=str, default='./figures/ns_incom')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading dataset from {args.data_path}")
    
    # We load target_res=256 for analysis visualization
    dataset = MinimalH5Dataset(
        h5_path=args.data_path,
        target_res=256,
        seq_len=20,
        temporal_stride=20,
        num_samples=5
    )
    
    analyzer = DatasetAnalyzer(dataset)
    
    print("Plotting spatiotemporal evolution...")
    fig1 = analyzer.plot_sample(idx=0, time_steps=[0, 5, 10, 19])
    _save_fig(fig1, os.path.join(args.output_dir, "spatiotemporal_evolution"))
    
    print("Plotting power spectrum...")
    fig2 = analyzer.plot_spectrum(idx=0)
    _save_fig(fig2, os.path.join(args.output_dir, "power_spectrum"))
    
    print("Plotting temporal evolution...")
    fig3 = analyzer.plot_temporal_evolution(idx=0)
    _save_fig(fig3, os.path.join(args.output_dir, "temporal_evolution"))
    
    print("Computing statistics...")
    stats = analyzer.compute_dataset_statistics(num_samples=5)
    with open(os.path.join(args.output_dir, "stats.json"), "w") as f:
        json.dump({k: float(v) for k, v in stats.items()}, f, indent=4)
        
    print(f"Stats: {stats}")
    print("Done")


if __name__ == '__main__':
    main()

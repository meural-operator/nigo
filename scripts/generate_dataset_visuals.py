"""
Dataset Visualization Generator — ICML Rebuttal Pipeline.

Renders research-grade visuals for all benchmark datasets using
the Universal Visualization Framework (turbo_nigo.utils.visualization).
Outputs: .png + .tex (standalone compilable) per plot.
"""
import os
import sys
import numpy as np
import torch
import h5py

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from turbo_nigo.data.base_dataset import BaseOperatorDataset
from turbo_nigo.data.flow_dataset import InMemoryFlowDataset, compute_global_stats_and_cond_stats
from turbo_nigo.utils.visualization import (
    Visualizer1D, Visualizer2D,
    SpatiotemporalSampleVisualizer,
    PhysicalStatisticsVisualizer,
    InitialConditionDiversityVisualizer,
)

# ==============================================================================
# Lightweight Mock Wrappers
# ==============================================================================
# These adapt raw HDF5 files that lack a dedicated TurboNIGO Dataset class
# into the BaseOperatorDataset interface so the Visualizer can bind to them.

class KSGroundTruthMock(BaseOperatorDataset):
    """Wraps KS_GROUNDTRUTH.h5 (trajectory-keyed, 1D field)."""
    def __init__(self, h5_path: str, num_trajs: int = 5, subsample: int = 100):
        super().__init__("", seq_len=200)
        with h5py.File(h5_path, "r") as f:
            trajs = []
            for i in range(num_trajs):
                key = f"traj_{i}"
                raw = f[key][::subsample]  # (T_sub, X=512)
                trajs.append(raw)
        data = np.stack(trajs, axis=0)  # (N, T_sub, X)
        self.global_min = float(data.min())
        self.global_max = float(data.max())
        # Normalize to [0, 1]
        data = (data - self.global_min) / (self.global_max - self.global_min + 1e-8)
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = min(200, self.data.shape[1] - 1)
        self.cond_mean = torch.zeros(4)
        self.cond_std = torch.ones(4)

    def _setup_dataset(self): pass
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        traj = self.data[idx]  # (T, X)
        x = traj[0:1]  # (1, X) — initial condition
        y = traj[1:self.seq_len + 1].unsqueeze(1)  # (seq_len, 1, X) — channel dim
        return x, y, torch.zeros(4)


class BurgersDatasetMock(BaseOperatorDataset):
    """Wraps 1D_Burgers_Sols_Nu0.1.hdf5."""
    def __init__(self, h5_path: str, num_samples: int = 50):
        super().__init__("", seq_len=100)
        with h5py.File(h5_path, "r") as f:
            # 'tensor' key, shape varies — typically (N, X, T) or (N, T, X)
            raw = f["tensor"][:num_samples]
        # Sort axes so time is dim 1 (shorter axis)
        if raw.shape[2] > raw.shape[1]:
            raw = np.transpose(raw, (0, 2, 1))  # -> (N, T, X)
        self.global_min = float(raw.min())
        self.global_max = float(raw.max())
        raw = (raw - self.global_min) / (self.global_max - self.global_min + 1e-8)
        self.data = torch.tensor(raw, dtype=torch.float32)
        self.seq_len = min(100, self.data.shape[1] - 1)
        self.cond_mean = torch.zeros(4)
        self.cond_std = torch.ones(4)

    def _setup_dataset(self): pass
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        traj = self.data[idx]  # (T, X)
        x = traj[0:1]  # (1, X)
        y = traj[1:self.seq_len + 1].unsqueeze(1)  # (seq_len, 1, X)
        return x, y, torch.zeros(4)


class DarcyDatasetMock(BaseOperatorDataset):
    """Wraps 2D_DarcyFlow_beta1.0_Train.hdf5 (steady-state 2D)."""
    def __init__(self, h5_path: str, num_samples: int = 50):
        super().__init__("", seq_len=1)
        with h5py.File(h5_path, "r") as f:
            self.nu = torch.tensor(f["nu"][:num_samples], dtype=torch.float32)        # (N, H, W)
            raw_tensor = f["tensor"][:num_samples]  # (N, 1, H, W)
            # Squeeze if channel dim already present
            if raw_tensor.ndim == 4 and raw_tensor.shape[1] == 1:
                raw_tensor = raw_tensor[:, 0]  # -> (N, H, W)
            self.tensor = torch.tensor(raw_tensor, dtype=torch.float32)  # (N, H, W)
        self.global_min = 0.0
        self.global_max = 1.0
        self.cond_mean = torch.zeros(4)
        self.cond_std = torch.ones(4)

    def _setup_dataset(self): pass
    def __len__(self): return len(self.nu)
    def __getitem__(self, idx):
        x = self.nu[idx].unsqueeze(0)                # (1, H, W)
        y = self.tensor[idx].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W) — T=1, C=1
        return x, y, torch.zeros(4)


# ==============================================================================
# Main
# ==============================================================================
def main():
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    
    root = "datasets"
    out_root = os.path.join("results", "dataset_visualizations")
    os.makedirs(out_root, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. bc — Bluff-body Cylinder (2D Navier-Stokes)
    # ------------------------------------------------------------------
    print("\n[1/4] bc (Incompressible Navier-Stokes 2D)")
    bc_path = os.path.join(root, "bc")
    g_min, g_max, c_mean, c_std = compute_global_stats_and_cond_stats(bc_path)
    bc_ds = InMemoryFlowDataset.create_with_stats(
        bc_path, 20, "train", g_min, g_max, c_mean, c_std, max_cases=10
    )
    viz_bc = Visualizer2D(bc_ds)
    d = os.path.join(out_root, "bc"); os.makedirs(d, exist_ok=True)

    viz_bc.plot_sample(0, time_steps=[0, 10, 19], save_path=os.path.join(d, "bc_sample"))
    viz_bc.plot_vorticity_field(0, time_step=-1, save_path=os.path.join(d, "bc_vorticity"))
    viz_bc.plot_spectral_density(0, save_path=os.path.join(d, "bc_spectrum"))
    # Dataset-level visualizers
    SpatiotemporalSampleVisualizer(bc_ds).plot_evolution_grid(save_path=os.path.join(d, "bc_evolution_grid"))
    PhysicalStatisticsVisualizer(bc_ds).plot_averaged_spectrum(save_path=os.path.join(d, "bc_avg_spectrum"))
    PhysicalStatisticsVisualizer(bc_ds).plot_energy_distribution(save_path=os.path.join(d, "bc_energy_dist"))
    InitialConditionDiversityVisualizer(bc_ds).plot_manifold(save_path=os.path.join(d, "bc_ic_manifold"))
    print(f"    -> Saved to {d}")

    # ------------------------------------------------------------------
    # 2. KS — Kuramoto-Sivashinsky (1D chaotic)
    # ------------------------------------------------------------------
    print("\n[2/4] KS_dataset (Kuramoto-Sivashinsky 1D)")
    ks_path = os.path.join(root, "KS_dataset", "KS_GROUNDTRUTH.h5")
    ks_ds = KSGroundTruthMock(ks_path, num_trajs=5, subsample=100)
    viz_ks = Visualizer1D(ks_ds)
    d = os.path.join(out_root, "KS_dataset"); os.makedirs(d, exist_ok=True)

    viz_ks.plot_sample(0, save_path=os.path.join(d, "ks_sample"))
    viz_ks.plot_hovmoller(0, save_path=os.path.join(d, "ks_hovmoller"))
    viz_ks.plot_spectral_density(0, save_path=os.path.join(d, "ks_spectrum"))
    # Dataset-level visualizers
    SpatiotemporalSampleVisualizer(ks_ds).plot_evolution_grid(save_path=os.path.join(d, "ks_evolution_grid"))
    PhysicalStatisticsVisualizer(ks_ds).plot_averaged_spectrum(save_path=os.path.join(d, "ks_avg_spectrum"))
    PhysicalStatisticsVisualizer(ks_ds).plot_energy_distribution(save_path=os.path.join(d, "ks_energy_dist"))
    InitialConditionDiversityVisualizer(ks_ds).plot_manifold(save_path=os.path.join(d, "ks_ic_manifold"))
    print(f"    -> Saved to {d}")

    # ------------------------------------------------------------------
    # 3. Burgers (1D viscous shock)
    # ------------------------------------------------------------------
    print("\n[3/4] Burgers (1D Viscous PDE)")
    burgers_path = os.path.join(root, "Burgers", "1D_Burgers_Sols_Nu0.1.hdf5")
    burgers_ds = BurgersDatasetMock(burgers_path, num_samples=50)
    viz_burg = Visualizer1D(burgers_ds)
    d = os.path.join(out_root, "Burgers"); os.makedirs(d, exist_ok=True)

    viz_burg.plot_sample(0, save_path=os.path.join(d, "burgers_sample"))
    viz_burg.plot_hovmoller(0, save_path=os.path.join(d, "burgers_hovmoller"))
    viz_burg.plot_spectral_density(0, save_path=os.path.join(d, "burgers_spectrum"))
    # Dataset-level visualizers
    SpatiotemporalSampleVisualizer(burgers_ds).plot_evolution_grid(save_path=os.path.join(d, "burgers_evolution_grid"))
    PhysicalStatisticsVisualizer(burgers_ds).plot_averaged_spectrum(save_path=os.path.join(d, "burgers_avg_spectrum"))
    PhysicalStatisticsVisualizer(burgers_ds).plot_energy_distribution(save_path=os.path.join(d, "burgers_energy_dist"))
    InitialConditionDiversityVisualizer(burgers_ds).plot_manifold(save_path=os.path.join(d, "burgers_ic_manifold"))
    print(f"    -> Saved to {d}")

    # ------------------------------------------------------------------
    # 4. DarcyFlow (2D steady state)
    # ------------------------------------------------------------------
    print("\n[4/4] DarcyFlow (2D Steady-State Permeability)")
    darcy_path = os.path.join(root, "DarcyFlow", "2D_DarcyFlow_beta1.0_Train.hdf5")
    darcy_ds = DarcyDatasetMock(darcy_path)
    viz_darcy = Visualizer2D(darcy_ds)
    d = os.path.join(out_root, "DarcyFlow"); os.makedirs(d, exist_ok=True)

    viz_darcy.plot_sample(0, time_steps=[0], save_path=os.path.join(d, "darcy_sample"))
    viz_darcy.plot_spectral_density(0, save_path=os.path.join(d, "darcy_spectrum"))
    # Dataset-level visualizers (Darcy is steady-state so evolution grid uses T=1)
    PhysicalStatisticsVisualizer(darcy_ds).plot_energy_distribution(num_samples=50, save_path=os.path.join(d, "darcy_energy_dist"))
    InitialConditionDiversityVisualizer(darcy_ds).plot_manifold(num_samples=50, save_path=os.path.join(d, "darcy_ic_manifold"))
    print(f"    -> Saved to {d}")

    print("\n[+] All dataset visualizations complete.")


if __name__ == "__main__":
    main()

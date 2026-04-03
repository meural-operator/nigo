import os
import sys
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.fft import fft, fftfreq

# We do not want this script integrated into the main visualization framework.
# Running purely as a standalone diagnostic tool for the ICML rebuttal.

matplotlib.use("Agg")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from turbo_nigo.data.flow_dataset import InMemoryFlowDataset, compute_global_stats_and_cond_stats
from scripts.generate_dataset_visuals import KSGroundTruthMock, BurgersDatasetMock

def plot_attractor_evidence(dataset, dataset_name, out_dir):
    print(f"Generating attractor phase portrait & PSD for {dataset_name}...")
    os.makedirs(out_dir, exist_ok=True)
    
    # Extract future trajectory starting from initial condition
    _, y, _ = dataset[0]
    y_np = y.numpy()
    
    # Flatten all spatial and channel dimensions to (T, num_features)
    T = y_np.shape[0]
    y_flat = y_np.reshape(T, -1)
    
    # 1. State-Space Embedding (Phase Portrait) via PCA
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(y_flat)
    pc1 = pcs[:, 0]
    pc2 = pcs[:, 1]
    
    # Plot Phase Portrait
    plt.figure(figsize=(6, 5), dpi=300)
    plt.plot(pc1, pc2, '-', color='gray', alpha=0.5, linewidth=1)
    sc = plt.scatter(pc1, pc2, c=np.arange(T), cmap='plasma', s=25, edgecolor='k', linewidth=0.5, zorder=5)
    plt.colorbar(sc, label="Time Step $t$")
    plt.title(f"{dataset_name}: Temporal Phase Portrait (PCA)")
    plt.xlabel(f"Principal Component 1 (Var: {pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"Principal Component 2 (Var: {pca.explained_variance_ratio_[1]:.2%})")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{dataset_name}_phase_portrait.png"))
    plt.close()
    
    # 2. Temporal Power Spectral Density (PSD)
    yf = fft(pc1)
    xf = fftfreq(T, d=1.0)[:T//2]
    psd = (2.0/T) * np.abs(yf[0:T//2])**2  # Power spectral density
    
    plt.figure(figsize=(6, 5), dpi=300)
    plt.semilogy(xf[1:], psd[1:], '-b', linewidth=1.5)  # Skip DC offset component
    plt.title(f"{dataset_name}: Temporal Power Spectral Density")
    plt.xlabel("Temporal Frequency $f$")
    plt.ylabel("Power Spectral Density")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{dataset_name}_temporal_psd.png"))
    plt.close()

def main():
    root = "datasets"
    out_root = os.path.join("results", "rebuttal_attractor_evidence")
    os.makedirs(out_root, exist_ok=True)
    
    # 1. Navier-Stokes (Bluff-body Cylinder) - LIMIT CYCLE EXPECTED
    print("[1/3] Processing bc dataset...")
    bc_path = os.path.join(root, "bc")
    g_min, g_max, c_mean, c_std = compute_global_stats_and_cond_stats(bc_path)
    bc_ds = InMemoryFlowDataset.create_with_stats(
        bc_path, 600, "train", g_min, g_max, c_mean, c_std, max_cases=2
    )
    plot_attractor_evidence(bc_ds, "Navier-Stokes (bc)", out_root)
    
    # 2. Kuramoto-Sivashinsky - STRANGE ATTRACTOR (CHAOS) EXPECTED
    print("[2/3] Processing KS dataset...")
    ks_path = os.path.join(root, "KS_dataset", "KS_GROUNDTRUTH.h5")
    ks_ds = KSGroundTruthMock(ks_path, num_trajs=2, subsample=10) # Less sub-sampling for higher temporal freq res
    plot_attractor_evidence(ks_ds, "Kuramoto-Sivashinsky", out_root)
    
    # 3. Burgers' Equation - STEADY-STATE DECAY (FIXED POINT) EXPECTED
    print("[3/3] Processing Burgers dataset...")
    burgers_path = os.path.join(root, "Burgers", "1D_Burgers_Sols_Nu0.1.hdf5")
    burgers_ds = BurgersDatasetMock(burgers_path, num_samples=5)
    plot_attractor_evidence(burgers_ds, "Burgers Equation", out_root)

    # Note: Darcy Flow is highly decoupled steady-state, so temporal attractor does not apply physically.
    print("\n[+] Successfully generated attractor evidence plots.")

if __name__ == "__main__":
    main()

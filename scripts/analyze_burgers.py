"""
analyze_burgers.py — HDF5 Introspection and Physical Property Analysis for Burgers Dataset.

This script scans a new HDF5 dataset to:
1. Extract the schema, keys, shapes, and data types.
2. Compute global statistics (min, max, mean, std).
3. Analyze the temporal energy trace E(t) to determine if the system is 
   strictly dissipative, energy-conserving, or exhibits exponential transients.

Usage:
  python scripts/analyze_burgers.py --h5_path ./datasets/burgers/burgers_data.h5
"""

import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

def print_hdf5_structure(name, obj):
    """Recursively prints the HDF5 tree structure."""
    indent = "  " * name.count('/')
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}├── Dataset: {name.split('/')[-1]} | Shape: {obj.shape} | Type: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"{indent}├── Group: {name.split('/')[-1]}/")

def analyze_dataset(h5_path: str, output_dir: str, num_samples_to_plot: int = 5):
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Dataset not found at {h5_path}")
        
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"  BURGER'S DATASET ANALYSIS: {os.path.basename(h5_path)}")
    print("=" * 60)

    with h5py.File(h5_path, 'r') as f:
        print("\n[1] HDF5 File Structure:")
        f.visititems(print_hdf5_structure)
        
        # Heuristic to find the main data key (usually 'train', 'velocity', or 'data')
        data_key = None
        for key in ['train', 'velocity', 'data', 'u', 'tensor']:
            if key in f:
                data_key = key
                break
                
        if data_key is None:
            data_key = list(f.keys())[0] # Fallback to first key
            
        print(f"\n[2] Extracting Main Tensor: '{data_key}'")
        dataset = f[data_key]
        print(f"  Shape: {dataset.shape}")
        
        # Load a small chunk to compute global stats without blowing up RAM
        N_total = dataset.shape[0]
        chunk_size = min(N_total, 1000)
        print(f"  Loading a chunk of {chunk_size} samples for global statistics...")
        
        data_chunk = dataset[:chunk_size]
        g_min, g_max = np.min(data_chunk), np.max(data_chunk)
        g_mean, g_std = np.mean(data_chunk), np.std(data_chunk)
        
        print(f"  Global Min:  {g_min:.6f}")
        print(f"  Global Max:  {g_max:.6f}")
        print(f"  Global Mean: {g_mean:.6f}")
        print(f"  Global Std:  {g_std:.6f}")

        print("\n[3] Physical Property Analysis (Energy Trace)")
        # We compute the spatial energy over time: E(t) = mean(u(x,t)^2)
        # This tells us if the dataset is dissipative or chaotic
        fig, ax = plt.subplots(figsize=(8, 5))
        
        indices = np.linspace(0, chunk_size - 1, num_samples_to_plot, dtype=int)
        colors = plt.cm.viridis(np.linspace(0, 0.9, num_samples_to_plot))
        
        is_dissipative = True
        
        for idx, color in zip(indices, colors):
            sample = data_chunk[idx] # Shape: (T, spatial_dims...)
            
            # Collapse all spatial dimensions to compute mean energy per timestep
            spatial_dims = tuple(range(1, sample.ndim))
            energy_trace = np.mean(sample**2, axis=spatial_dims)
            
            # Check if energy strictly decays (allow small numerical bumps)
            if not np.all(np.diff(energy_trace) <= 1e-4):
                is_dissipative = False
                
            ax.plot(energy_trace, color=color, alpha=0.8, label=f'Sample {idx}')

        ax.set_title("Temporal Energy Trace $E(t) = \\langle u(x,t)^2 \\rangle$")
        ax.set_xlabel("Time Step $t$")
        ax.set_ylabel("Spatial Energy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_path = os.path.join(output_dir, "burgers_energy_trace.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        print(f"  Saved energy trace plot to: {plot_path}")
        
        print("\n[4] Physical Conclusion:")
        if is_dissipative:
            print("  ➤ Highly Dissipative: Energy decays monotonically over time.")
            print("  ➤ Architectural Implication: The model will heavily rely on the beta * R^T R (dissipative) component of the generator.")
        else:
            print("  ➤ Chaotic / Energy Conserving: Energy fluctuates or exhibits transient growths.")
            print("  ➤ Architectural Implication: The model will require a strong alpha * (K - K^T) (skew-symmetric) component to model rotations/oscillations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Burgers HDF5 Dataset")
    parser.add_argument("--h5_path", type=str, required=True, help="Path to the .h5 dataset file")
    parser.add_argument("--output_dir", type=str, default="./figures/burgers", help="Directory to save plots")
    
    args = parser.parse_args()
    analyze_dataset(args.h5_path, args.output_dir)
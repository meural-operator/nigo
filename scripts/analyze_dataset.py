import h5py
import numpy as np
import sys
import os

def analyze_h5_file(h5_path):
    """Prints a robust summary of the HDF5 dataset structure."""
    if not os.path.exists(h5_path):
        print(f"File not found: {h5_path}")
        return

    file_size_gb = os.path.getsize(h5_path) / (1024 ** 3)
    print(f"============== Dataset Analysis ==============")
    print(f"File: {h5_path}")
    print(f"Size: {file_size_gb:.2f} GB")
    print("-" * 44)

    try:
        with h5py.File(h5_path, 'r') as f:
            print(f"Root keys: {list(f.keys())}")
            
            def analyze_node(name, node):
                if isinstance(node, h5py.Dataset):
                    print(f"\n[Dataset] {name}")
                    print(f"  Shape: {node.shape}")
                    print(f"  Dtype: {node.dtype}")
                    print(f"  Chunks: {node.chunks}")
                    print(f"  Compression: {node.compression}")
                    
                    # Compute min/max for a small sample to avoid memory overflow
                    if node.size > 0:
                        try:
                            # If it's a huge dataset, sample the first trajectory/time-step
                            if len(node.shape) >= 1:
                                sample_idx = min(100, node.shape[0])
                                sample = node[:sample_idx]
                                print(f"  Sample min: {np.min(sample):.4f}")
                                print(f"  Sample max: {np.max(sample):.4f}")
                                print(f"  Sample mean: {np.mean(sample):.4f}")
                                print(f"  Sample nan counts: {np.isnan(sample).sum()}")
                        except Exception as e:
                            print(f"  Could not compute stats: {e}")
                elif isinstance(node, h5py.Group):
                    print(f"\n[Group] {name}")
                    print(f"  Keys: {list(node.keys())}")

            f.visititems(analyze_node)
            
            # Additional PDEbench specific checks based on typical structure
            print("\n-------------- PDEbench Checks --------------")
            if 't-coordinate' in f:
                print(f"Time coordinate shape: {f['t-coordinate'].shape}")
            if 'x-coordinate' in f:
                print(f"X coordinate shape: {f['x-coordinate'].shape}")
            if 'y-coordinate' in f:
                print(f"Y coordinate shape: {f['y-coordinate'].shape}")
                
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_dataset.py <path_to_h5>")
    else:
        analyze_h5_file(sys.argv[1])

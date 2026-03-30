import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from turbo_nigo.configs import get_args_and_config
from turbo_nigo.data.sw_dataset import ShallowWaterDataset

def visualize_sw(config, out_path):
    print("Loading Shallow Water Dataset...")
    
    common = dict(
        h5_path=config["data_root"],
        seq_len=config["seq_len"],
        temporal_stride=config.get("sw_temporal_stride", 1),
        spatial_size=config.get("sw_spatial_size", 128),
        cond_dim=config.get("cond_dim", 4),
    )
    
    # Just need 10 cases to visualize
    ds = ShallowWaterDataset(
        **common, mode="val", max_trajectories=10
    )
    
    print(f"Loaded validation set: {len(ds)} windows available.")
    
    if len(ds) == 0:
        print("Dataset is empty. Cannot visualize.")
        return
        
    sample_idx = len(ds) // 2
    state, next_states, cond = ds[sample_idx]
    
    # state: (1, 128, 128)
    # next_states: (seq_len, 1, 128, 128)
    full_seq = torch.cat([state.unsqueeze(0), next_states], dim=0) # (seq_len + 1, 1, 128, 128)
    print(f"Sequence shape to animate: {full_seq.shape}")
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Render first frame
    frame_data = full_seq[0, 0].numpy()
    im = ax.imshow(frame_data, cmap='viridis', origin='lower')
    ax.set_title(f"Shallow Water Height (t=0)")
    fig.colorbar(im, ax=ax)
    
    def update(frame_idx):
        im.set_array(full_seq[frame_idx, 0].numpy())
        ax.set_title(f"Shallow Water Height (t={frame_idx})")
        return [im]
        
    anim = FuncAnimation(fig, update, frames=len(full_seq), interval=200, blit=True)
    
    if out_path.endswith('.mp4'):
        print(f"Saving MP4 animation to {out_path} ...")
        anim.save(out_path, writer='ffmpeg')
    else:
        print(f"Saving GIF animation to {out_path} ...")
        anim.save(out_path, writer='pillow')
        
    print(f"Successfully saved {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to sw_config.yaml")
    parser.add_argument("--out", type=str, default="sw_animation.mp4", help="Output animation filename")
    
    # Fake sys.argv for the underlying get_args_and_config which also sets up argparse
    # We intercept --out, and let the rest pass to get_args_and_config
    args, unknown = parser.parse_known_args()
    
    # Overwrite sys.argv so get_args_and_config only sees its args
    sys.argv = [sys.argv[0]] + unknown
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", args.config])
        
    config = get_args_and_config()
    visualize_sw(config, args.out)

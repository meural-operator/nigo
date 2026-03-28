import sys
import os
import argparse
import numpy as np
import matplotlib
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from turbo_nigo.data.base_dataset import BaseOperatorDataset
import h5py
import torch
import torch.nn.functional as F

class MinimalH5Dataset(BaseOperatorDataset):
    def __init__(self, h5_path, target_res=256, seq_len=40, temporal_stride=10, num_samples=3):
        # Increased seq_len and reduced stride to get smoother, longer videos
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
        t0 = idx * 15 # offset start for diversity
        T_need = self.seq_len * self.temporal_stride + 1
        with h5py.File(self.h5_path, 'r') as f:
            velocity = f['velocity']
            raw_vel = velocity[0, t0:t0+T_need:self.temporal_stride] # (seq_len+1, H, W, 2)
            
        raw_t = torch.from_numpy(raw_vel).float().permute(0, 3, 1, 2)
        down_t = F.interpolate(raw_t, size=(self.target_res, self.target_res), mode="bilinear", align_corners=False)
        down_t = (down_t - self.g_min) / (self.g_max - self.g_min + 1e-8)
        
        # Un-normalize back to physical for plotting 
        arr = down_t.numpy()
        arr = arr * (self.g_max - self.g_min) + self.g_min
        
        return arr[0], arr[1:]

def generate_video_and_gif(y_phys, sample_idx, output_dir):
    """
    y_phys: (T, C, H, W) numpy array
    """
    T, C, H, W = y_phys.shape
    
    # We will plot all channels side-by-side
    fig, axes = plt.subplots(1, C, figsize=(5 * C + 1, 5))
    if C == 1:
        axes = [axes]
        
    ims = []
    
    vmax = np.max(y_phys)
    vmin = np.min(y_phys)

    for c in range(C):
        axes[c].set_title(f"Target Feature Fields (Channel {c})", fontsize=12)
        axes[c].axis('off')

    # Draw initial frame
    images = []
    for c in range(C):
        im = axes[c].imshow(y_phys[0, c], cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, animated=True)
        images.append(im)
    fig.colorbar(images[-1], ax=axes, fraction=0.046, pad=0.04)
    fig.suptitle(f"Navier Stokes Evolution (Sample {sample_idx}) | Initial", fontsize=14)
    
    def update(frame):
        for c in range(C):
            data = y_phys[frame, c]
            images[c].set_array(data)
            # Dynamically update the color bounds so contract doesn't wash out (similar to snapshots)
            images[c].set_clim(vmin=np.min(data), vmax=np.max(data))
        fig.suptitle(f"Navier Stokes Evolution (Sample {sample_idx}) | t={frame}", fontsize=14)
        return images
        
    ani = animation.FuncAnimation(fig, update, frames=T, blit=True)
    
    # Save GIF
    gif_path = os.path.join(output_dir, f"ns_sample_{sample_idx}.gif")
    writer = PillowWriter(fps=10)
    ani.save(gif_path, writer=writer)
    print(f"Saved {gif_path}")
    
    # AVI
    avi_path = os.path.join(output_dir, f"ns_sample_{sample_idx}.avi")
    try:
        if animation.writers.is_available('ffmpeg'):
            ani.save(avi_path, writer='ffmpeg', fps=10)
            print(f"Saved {avi_path} using ffmpeg")
        else:
            import cv2
            print("ffmpeg writer absent. Using OpenCV for AVI...")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = None
            for frame in range(T):
                update(frame)
                fig.canvas.draw()
                # convert to numpy array
                img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                
                if out is None:
                    h, w, _ = img.shape
                    out = cv2.VideoWriter(avi_path, fourcc, 10.0, (w, h))
                out.write(img)
            if out: out.release()
            print(f"Saved {avi_path} using cv2")
    except Exception as e:
        print(f"Failed to generate AVI for sample {sample_idx}: {e}")
        
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./datasets/ns_incom_inhom_2d_512-0.h5')
    parser.add_argument('--output_dir', type=str, default='./figures/ns_videos')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating video datasets from {args.data_path}...")
    
    dataset = MinimalH5Dataset(
        h5_path=args.data_path,
        target_res=256,
        seq_len=30, # 30 frames
        temporal_stride=5,
        num_samples=3
    )
    
    for idx in range(dataset.num_samples):
        print(f"Processing sequence {idx}...")
        x, y = dataset[idx]
        
        y = np.concatenate([np.expand_dims(x, axis=0), y], axis=0) # Shape: (T+1, C, H, W)
        
        generate_video_and_gif(y, idx, args.output_dir)

if __name__ == '__main__':
    main()

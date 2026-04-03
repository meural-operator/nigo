"""
Comprehensive 3D Navier-Stokes Visualization Script.

Generates publication-grade outputs for the NS3D turbulence dataset:
  1. Clean orthogonal plane slices (XY, XZ, YZ) — PNG + PDF per field
  2. Clean 3D volumetric snapshots — PNG + PDF per field
  3. Animated 2D slice evolution — GIF + AVI per field
  4. Animated 3D cube evolution — GIF + AVI per field
  5. Interactive 3D HTML (plotly) for GitHub embedding

Usage:
    conda run -n cfd python scripts/visualize_ns3d.py
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from turbo_nigo.data.ns3d_dataset import NS3DDataset

# ── Global style: eliminate all text/decoration for clean publication outputs ──
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Channel mapping: index → human-readable name
CHANNEL_MAP = {0: "Vx", 1: "Vy", 2: "Vz", 3: "Density", 4: "Pressure"}


def load_dataset(dataset_dir: str, seq_len: int) -> NS3DDataset:
    ds = NS3DDataset(root_dir=dataset_dir, seq_len=seq_len, mode='train')
    return ds


def unnormalize(tensor, stats):
    """Restore physical values from [0,1] normalization."""
    arr = tensor.detach().cpu().numpy()
    g_min = float(stats['global_min'])
    g_max = float(stats['global_max'])
    return arr * (g_max - g_min) + g_min


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CLEAN ORTHOGONAL SLICES  (no text, no legend, no colorbar)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_clean_slices(dataset, sample_idx, out_dir):
    """Generate separate XY, XZ, YZ plane slices per field — PNG + PDF."""
    print("\n[1/5] Generating clean orthogonal slices...")
    slice_dir = out_dir / "slices"
    slice_dir.mkdir(parents=True, exist_ok=True)

    _, y, _ = dataset[sample_idx]
    stats = dataset.get_normalization_stats()
    y_phys = unnormalize(y, stats)  # (T, C, D, H, W)
    vol = y_phys[0]  # t=0 snapshot → (C, D, H, W)

    for ch_idx, ch_name in CHANNEL_MAP.items():
        field = vol[ch_idx]  # (D, H, W)
        D, H, W = field.shape

        slices = {
            "XY": field[D // 2, :, :],
            "XZ": field[:, H // 2, :],
            "YZ": field[:, :, W // 2],
        }

        for plane_name, sheet in slices.items():
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(sheet, cmap='RdBu_r', origin='lower', aspect='equal')
            ax.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            base = slice_dir / f"{ch_name}_{plane_name}_slice"
            fig.savefig(str(base) + ".png", dpi=300, bbox_inches='tight', pad_inches=0)
            fig.savefig(str(base) + ".pdf", bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print(f"   ✓ {ch_name} {plane_name} → .png .pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CLEAN 3D VOLUMETRIC SNAPSHOTS  (no text, no legend, no colorbar)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_clean_3d_snapshots(dataset, sample_idx, out_dir,
                                 downsample=2, percentile=85.0):
    """Generate 3D scatter snapshot per field — PNG + PDF."""
    print("\n[2/5] Generating clean 3D volumetric snapshots...")
    snap_dir = out_dir / "snapshots_3d"
    snap_dir.mkdir(parents=True, exist_ok=True)

    _, y, _ = dataset[sample_idx]
    stats = dataset.get_normalization_stats()
    y_phys = unnormalize(y, stats)
    vol = y_phys[0]  # t=0 → (C, D, H, W)

    for ch_idx, ch_name in CHANNEL_MAP.items():
        field = vol[ch_idx]
        fd = field[::downsample, ::downsample, ::downsample]
        thresh = np.percentile(np.abs(fd), percentile)
        z, yc, x = np.where(np.abs(fd) > thresh)
        vals = fd[z, yc, x]

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, yc, z, c=vals, cmap='RdBu_r', s=12, alpha=0.55, depthshade=True)
        ax.set_xlim(0, fd.shape[2])
        ax.set_ylim(0, fd.shape[1])
        ax.set_zlim(0, fd.shape[0])
        ax.axis('off')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # Remove pane backgrounds
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')

        base = snap_dir / f"{ch_name}_3d_snapshot"
        fig.savefig(str(base) + ".png", dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
        fig.savefig(str(base) + ".pdf", bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)
        print(f"   ✓ {ch_name} 3D snapshot → .png .pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# 3a. ANIMATED 2D SLICE EVOLUTION  (GIF + AVI)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_slice_animations(dataset, sample_idx, out_dir, fps=5):
    """Animate mid-Z slice over all timesteps per field — GIF + AVI."""
    print("\n[3a/5] Generating 2D slice evolution animations...")
    anim_dir = out_dir / "animations_2d"
    anim_dir.mkdir(parents=True, exist_ok=True)

    _, y, _ = dataset[sample_idx]
    stats = dataset.get_normalization_stats()
    y_phys = unnormalize(y, stats)  # (T, C, D, H, W)
    T = y_phys.shape[0]

    for ch_idx, ch_name in CHANNEL_MAP.items():
        seq = y_phys[:, ch_idx]  # (T, D, H, W)
        mid_z = seq.shape[1] // 2
        sheets = seq[:, mid_z, :, :]  # (T, H, W)
        vmin, vmax = float(sheets.min()), float(sheets.max())

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(sheets[0], cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        def update(frame, im=im, sheets=sheets):
            im.set_array(sheets[frame])
            return [im]

        anim = animation.FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=True)

        base = anim_dir / f"{ch_name}_slice_evolution"
        # GIF
        anim.save(str(base) + ".gif", writer='pillow', fps=fps)
        # AVI via imageio
        try:
            import imageio
            frames = []
            for fr in range(T):
                im.set_array(sheets[fr])
                fig.canvas.draw()
                buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
                frames.append(buf)
            imageio.mimwrite(str(base) + ".avi", frames, fps=fps, codec='mjpeg')
        except Exception as e:
            print(f"   \u26a0 AVI failed for {ch_name}: {e}")

        plt.close(fig)
        print(f"   \u2713 {ch_name} slice evolution \u2192 .gif .avi")


# ═══════════════════════════════════════════════════════════════════════════════
# 3b. ANIMATED 3D CUBE EVOLUTION  (GIF + AVI)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_3d_cube_animations(dataset, sample_idx, out_dir,
                                 downsample=2, percentile=85.0, fps=4):
    """Animate 3D scatter cube over all timesteps per field — GIF + AVI."""
    print("\n[3b/5] Generating 3D cube evolution animations...")
    anim_dir = out_dir / "animations_3d"
    anim_dir.mkdir(parents=True, exist_ok=True)

    _, y, _ = dataset[sample_idx]
    stats = dataset.get_normalization_stats()
    y_phys = unnormalize(y, stats)  # (T, C, D, H, W)
    T = y_phys.shape[0]

    for ch_idx, ch_name in CHANNEL_MAP.items():
        gvol = y_phys[:, ch_idx, ::downsample, ::downsample, ::downsample]
        vmin, vmax = float(gvol.min()), float(gvol.max())
        thresh = np.percentile(np.abs(gvol), percentile)

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')

        def update(frame, ax=ax, gvol=gvol, thresh=thresh, vmin=vmin, vmax=vmax):
            ax.clear()
            vol_ds = gvol[frame]
            z, yc, x = np.where(np.abs(vol_ds) > thresh)
            vals = vol_ds[z, yc, x]
            ax.scatter(x, yc, z, c=vals, cmap='RdBu_r',
                       s=12, alpha=0.5, depthshade=True, vmin=vmin, vmax=vmax)
            ax.set_xlim(0, gvol.shape[3])
            ax.set_ylim(0, gvol.shape[2])
            ax.set_zlim(0, gvol.shape[1])
            ax.axis('off')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('none')
            ax.yaxis.pane.set_edgecolor('none')
            ax.zaxis.pane.set_edgecolor('none')
            return ax,

        anim = animation.FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=False)

        base = anim_dir / f"{ch_name}_3d_evolution"
        anim.save(str(base) + ".gif", writer='pillow', fps=fps)
        # AVI via imageio
        try:
            import imageio
            frames = []
            for fr in range(T):
                update(fr, ax=ax, gvol=gvol, thresh=thresh, vmin=vmin, vmax=vmax)
                fig.canvas.draw()
                buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
                frames.append(buf)
            imageio.mimwrite(str(base) + ".avi", frames, fps=fps, codec='mjpeg')
        except Exception as e:
            print(f"   \u26a0 AVI failed for {ch_name}: {e}")

        plt.close(fig)
        print(f"   \u2713 {ch_name} 3D cube evolution \u2192 .gif .avi")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. INTERACTIVE 3D HTML  (plotly)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_interactive_3d(dataset, sample_idx, out_dir,
                             downsample=2, percentile=80.0):
    """Generate interactive HTML 3D scatter plots per field using plotly."""
    print("\n[4/5] Generating interactive 3D HTML visualizations...")
    html_dir = out_dir / "interactive"
    html_dir.mkdir(parents=True, exist_ok=True)

    import plotly.graph_objects as go

    _, y, _ = dataset[sample_idx]
    stats = dataset.get_normalization_stats()
    y_phys = unnormalize(y, stats)
    vol = y_phys[0]  # t=0 → (C, D, H, W)

    for ch_idx, ch_name in CHANNEL_MAP.items():
        field = vol[ch_idx]
        fd = field[::downsample, ::downsample, ::downsample]
        thresh = np.percentile(np.abs(fd), percentile)
        zi, yi, xi = np.where(np.abs(fd) > thresh)
        vals = fd[zi, yi, xi]

        fig = go.Figure(data=[go.Scatter3d(
            x=xi.astype(float), y=yi.astype(float), z=zi.astype(float),
            mode='markers',
            marker=dict(
                size=2.5,
                color=vals,
                colorscale='RdBu_r',
                opacity=0.6,
                colorbar=dict(title=ch_name, thickness=15),
            ),
        )])

        fig.update_layout(
            title=f"3D {ch_name} Field (Interactive — Rotate & Zoom)",
            scene=dict(
                xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                xaxis=dict(range=[0, fd.shape[2]]),
                yaxis=dict(range=[0, fd.shape[1]]),
                zaxis=dict(range=[0, fd.shape[0]]),
                aspectmode='cube',
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            template='plotly_dark',
        )

        html_path = str(html_dir / f"{ch_name}_interactive_3d.html")
        fig.write_html(html_path, include_plotlyjs='cdn')
        print(f"   ✓ {ch_name} interactive 3D → .html")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="NS3D Comprehensive Visualization")
    parser.add_argument("--dataset_dir", type=str, default="./datasets/NS3D/")
    parser.add_argument("--output_dir", type=str, default="./results/dataset_visualizations/NS3D/")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--seq_len", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset_dir, args.seq_len)

    generate_clean_slices(dataset, args.sample_idx, out_dir)
    generate_clean_3d_snapshots(dataset, args.sample_idx, out_dir)
    generate_slice_animations(dataset, args.sample_idx, out_dir)
    generate_3d_cube_animations(dataset, args.sample_idx, out_dir)
    generate_interactive_3d(dataset, args.sample_idx, out_dir)

    print("\n" + "=" * 60)
    print("[+] All NS3D visualizations generated successfully!")
    print(f"    Output directory: {out_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()

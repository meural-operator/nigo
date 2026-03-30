"""
Universal Visualization Framework for TurboNIGO.

Exposes a strictly typed Object-Oriented interface for N-Dimensional PyTorch 
tensors ensuring structural adherence to Open-Closed OOP Principles. 

Incorporates top-tier conference-standard visualizations (ICML/NeurIPS/ICLR):
- Energy Spectrum Density maps.
- Hovmöller (Spatiotemporal) mappings.
- Orthogonal volume slicing.
- Consistent Colormap & Typography bindings (STIXGeneral / Computer Modern).
"""
import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
from matplotlib.figure import Figure

# Set up research-grade LaTeX-compliant typography globally.
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.bbox": "tight"
})

class BaseVisualizer(ABC):
    """
    Abstract Visualization Engine.
    Binds dynamically to any Dataset extending BaseOperatorDataset to ensure 
    physics bounds (g_min, g_max) are natively respected during Un-normalization.
    """
    def __init__(self, dataset):
        """
        Args:
            dataset: Any subclass of turbo_nigo.data.base_dataset.BaseOperatorDataset
        """
        self.dataset = dataset
        self.stats = dataset.get_normalization_stats()
        
    def _unnormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """Restores network [0, 1] scaled tensors directly back to physical boundaries."""
        g_min = self.stats.get('global_min')
        g_max = self.stats.get('global_max')
        
        arr = tensor.detach().cpu().numpy()
        if g_min is not None and g_max is not None:
            # Assuming linear scaling -> Physical = Network * (Max - Min) + Min
            arr = arr * float(g_max - g_min) + float(g_min)
        return arr

    def _save_dual_format(self, fig: Figure, base_path: str) -> None:
        """
        Intercepts Matplotlib Figures and universally splits them into:
        1. A high-resolution PNG for rapid local rendering/Markdown display.
        2. A pure mathematical PGF LaTeX sequence.
        3. A standalone compilable .tex document guaranteed preventing text overlap.
        """
        import os
        
        # Strip extension if accidentally passed
        if base_path.endswith('.png') or base_path.endswith('.pdf') or base_path.endswith('.tex'):
            base_path = os.path.splitext(base_path)[0]
            
        # 1. Standard PNG
        fig.savefig(base_path + ".png", dpi=300, bbox_inches='tight')
        
        filename = os.path.basename(base_path)
        
        # 2. Mathematical PGF (Portable Graphics Format) for Native LaTeX Font Tracking
        try:
            fig.savefig(base_path + ".pgf", backend="pgf", bbox_inches='tight')
            tex_content = rf"""\documentclass{{standalone}}
\usepackage{{pgf}}
\usepackage{{graphicx}} 
\begin{{document}}
\input{{{filename}.pgf}}
\end{{document}}
"""
            print(f"[*] Visual natively dumped: {base_path} [.png | .pgf | .tex]")
        except Exception as e:
            # MiKTeX bounding-box compilation crashed locally. Fallback to embedding the high-res PNG 
            # natively inside the LaTeX framework to ensure formatting constraints remain met.
            tex_content = rf"""\documentclass{{standalone}}
\usepackage{{graphicx}} 
\begin{{document}}
\begin{{figure}}
\centering
\includegraphics{{{filename}.png}}
\end{{figure}}
\end{{document}}
"""
            print(f"[!] MiKTeX PGF compilation aborted. Gracefully falling back to PNG-embedded LaTeX: {base_path} [.png | .tex]")

        # 3. Write a standalone minimal .TEX script for the user to compile cleanly
        with open(base_path + ".tex", "w", encoding="utf-8") as f:
            f.write(tex_content)

    @abstractmethod
    def plot_sample(self, idx: int, time_steps: Optional[List[int]] = None, save_path: Optional[str] = None) -> Figure:
        """Renders the physical layout of the coordinate predictions/targets."""
        pass

    def compute_energy_spectrum(self, field: np.ndarray, dx: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Derives the 1D Isotropic Power Spectral Density from N-Dimensional fields.
        Highly required by computational physics reviewers (e.g. validating Kolmogorov -5/3).
        """
        # Multi-dimensional FFT dropping zero-frequency component to center
        f_hat = np.fft.fftn(field)
        f_hat = np.fft.fftshift(f_hat)
        
        psd = np.abs(f_hat)**2
        
        # Build radial k-wavevector masks
        coords = [np.fft.fftshift(np.fft.fftfreq(n, d=dx)) for n in field.shape]
        grids = np.meshgrid(*coords, indexing='ij')
        
        k_radius = np.sqrt(sum(g**2 for g in grids))
        
        k_bins = np.linspace(0, np.max(k_radius), min(field.shape)//2)
        digitized = np.digitize(k_radius, k_bins)
        
        radial_psd = np.array([psd[digitized == i].mean() if len(psd[digitized == i]) > 0 else 0 
                               for i in range(1, len(k_bins))])
        
        return k_bins[1:], radial_psd
        
    def plot_spectral_density(self, idx: int, time_step: int = -1, save_path: Optional[str] = None) -> Figure:
        """
        Generates the standard Log-Log Power Spectral Density plotting curve bounding 
        energy conservation natively globally across the dataset sequence.
        """
        _, y, _ = self.dataset[idx]
        y_phys = self._unnormalize(y)[time_step] # (C, ...)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        for c in range(y_phys.shape[0]):
            k, psd = self.compute_energy_spectrum(y_phys[c])
            ax.loglog(k, psd, label=f'Channel {c} Energy', linewidth=2.0)
            
        ax.set_xlabel('Wave-Number $k$')
        ax.set_ylabel('Spectral Power Density $E(k)$')
        ax.set_title('Spatial Frequency Domain Signature')
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.legend()
        
        if save_path:
            self._save_dual_format(fig, save_path)
            
        return fig


class Visualizer1D(BaseVisualizer):
    """Handles geometries with layout (C, X). E.g., Kuramoto-Sivashinsky, Burgers"""
    
    def plot_sample(self, idx: int, time_steps: Optional[List[int]] = None, save_path: Optional[str] = None) -> Figure:
        x, y, _ = self.dataset[idx]
        x_phys = self._unnormalize(x)
        y_phys = self._unnormalize(y)
        
        seq_len = y_phys.shape[0]
        if time_steps is None:
            time_steps = [0, seq_len // 2, seq_len - 1]
            
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x_phys[0], label='t=0 (Condition)', color='black', linestyle='--')
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(time_steps)))
        for i, t in enumerate(time_steps):
            ax.plot(y_phys[t, 0], label=f't={t+1}', color=colors[i], alpha=0.8)
            
        ax.set_title("1D Spatial Boundary Evolution", pad=15)
        ax.set_xlabel("Spatial Domain $X$")
        ax.set_ylabel("Amplitude $u$")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        if save_path:
            self._save_dual_format(fig, save_path)
        return fig
        
    def plot_hovmoller(self, idx: int, channel: int = 0, save_path: Optional[str] = None) -> Figure:
        """
        Spatiotemporal representation aggregating structural waves over T axis.
        X-axis = space, Y-axis = time. A favorite for evaluating highly 
        chaotic 1D states.
        """
        _, y, _ = self.dataset[idx]
        y_phys = self._unnormalize(y)[:, channel, :] # (T, X)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(y_phys, aspect='auto', cmap='RdBu_r', origin='lower')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Amplitude")
        
        ax.set_title("Hovmöller Spatiotemporal Diagram")
        ax.set_xlabel("Spatial Domain $X$")
        ax.set_ylabel("Temporal Domain $T$")
        
        if save_path:
            self._save_dual_format(fig, save_path)
        return fig


class Visualizer2D(BaseVisualizer):
    """Handles geometries with layout (C, H, W). E.g., Navier-Stokes, Darcy Flow"""
    
    def plot_sample(self, idx: int, time_steps: Optional[List[int]] = None, save_path: Optional[str] = None) -> Figure:
        x, y, _ = self.dataset[idx]
        x_phys = self._unnormalize(x)
        y_phys = self._unnormalize(y)
        
        channels = x_phys.shape[0]
        seq_len = y_phys.shape[0]
        
        if time_steps is None:
            # Generate exactly 3 discrete slices
            time_steps = [0, seq_len // 2, seq_len - 1]
            
        num_cols = len(time_steps) + 1
        fig, axes = plt.subplots(channels, num_cols, figsize=(4 * num_cols, 3.5 * channels), squeeze=False)
        
        for c in range(channels):
            # Baseline Boundary Layer
            im = axes[c, 0].imshow(x_phys[c], cmap='RdBu_r', origin='lower')
            axes[c, 0].set_title(f"Input $C{c}$ ($t=0$)")
            plt.colorbar(im, ax=axes[c, 0], fraction=0.046, pad=0.04)
            axes[c, 0].axis('off')
            
            # Sub-Rollout Grid
            for j, t in enumerate(time_steps):
                if t < seq_len:
                    im = axes[c, j+1].imshow(y_phys[t, c], cmap='RdBu_r', origin='lower')
                    axes[c, j+1].set_title(f"Target $C{c}$ ($t={t+1}$)")
                    plt.colorbar(im, ax=axes[c, j+1], fraction=0.046, pad=0.04)
                axes[c, j+1].axis('off')
                
        plt.tight_layout()
        if save_path:
            self._save_dual_format(fig, save_path)
        return fig
        
    def plot_vorticity_field(self, idx: int, time_step: int = -1, save_path: Optional[str] = None) -> Figure:
        """
        Transforms 2D velocity boundaries [u, v] into a single dimension of 
        scalar Vorticity indicating rotational fluid curl. 
        Highly required for standard fluid dynamic visual baselines.
        """
        _, y, _ = self.dataset[idx]
        y_phys = self._unnormalize(y)[time_step] # (C, H, W)
        
        if y_phys.shape[0] < 2:
            raise ValueError("[!] Vorticity requires exactly a structured 2D Vector Field (U, V channels).")
            
        u, v = y_phys[0], y_phys[1]
        
        # Approximate curl (dv/dx - du/dy) via spatial gradients natively in numpy NumPy
        du_dy, _ = np.gradient(u)
        _, dv_dx = np.gradient(v)
        vorticity = dv_dx - du_dy
        
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(vorticity, cmap='PRGn', origin='lower')
        ax.set_title(f"Fluid Vorticity Profile (t={time_step+1})")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(r'$\omega = \nabla \times V$')
        ax.axis('off')
        
        if save_path:
            self._save_dual_format(fig, save_path)
            
        return fig


class Visualizer3D(BaseVisualizer):
    """Handles massive structural geometries with layout (C, D, H, W). E.g., Advanced Turbulence"""
    
    def plot_sample(self, idx: int, time_step: int = -1, channel: int = 0, save_path: Optional[str] = None) -> Figure:
        """
        Drops orthogonal mid-plane spatial slices (X-Y, X-Z, Y-Z) 
        natively traversing 3D volumes.
        """
        _, y, _ = self.dataset[idx]
        vol = self._unnormalize(y)[time_step, channel] # (D, H, W)
        
        depth, height, width = vol.shape
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # XY Mid-plane (Z cut)
        im0 = axes[0].imshow(vol[depth//2, :, :], cmap='RdBu_r', origin='lower')
        axes[0].set_title(f"XY Plane (Z={depth//2})")
        
        # XZ Mid-plane (Y cut)
        im1 = axes[1].imshow(vol[:, height//2, :], cmap='RdBu_r', origin='lower')
        axes[1].set_title(f"XZ Plane (Y={height//2})")
        
        # YZ Mid-plane (X cut)
        im2 = axes[2].imshow(vol[:, :, width//2], cmap='RdBu_r', origin='lower')
        axes[2].set_title(f"YZ Plane (X={width//2})")
        
        for ax in axes:
            ax.axis('off')
            
        plt.colorbar(im2, ax=axes, fraction=0.015, pad=0.04)
        
        if save_path:
            self._save_dual_format(fig, save_path)
            
        return fig


# ======================================================================
# Dataset-Level Visualizers (Population-Level Characterization)
# ======================================================================
# These visualizers operate over many samples at once, producing the
# high-level dataset characterization plots expected by ICML/NeurIPS
# reviewers to validate data diversity, physical fidelity, and coverage.

class SpatiotemporalSampleVisualizer(BaseVisualizer):
    """
    Renders a multi-sample spatiotemporal evolution grid.
    
    Rows: distinct trajectory samples (randomly selected).
    Columns: evenly spaced time steps across the sequence horizon.
    Validates that the dataset exhibits diverse dynamics across ICs.
    """
    
    def plot_sample(self, idx: int, time_steps: Optional[List[int]] = None, 
                    save_path: Optional[str] = None) -> Figure:
        """Required by ABC — delegates to plot_evolution_grid."""
        return self.plot_evolution_grid(save_path=save_path)

    def plot_evolution_grid(
        self,
        num_samples: int = 3,
        num_time_steps: int = 5,
        channel: int = 0,
        cmap: str = "coolwarm",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Renders a (num_samples × num_time_steps) grid of spatial fields.
        
        Args:
            num_samples:    Number of distinct trajectories to display.
            num_time_steps: Number of evenly spaced temporal snapshots per row.
            channel:        Which field channel to render.
            cmap:           Diverging colormap for signed fields.
            save_path:      Base path for dual-format output.
        """
        n_total = len(self.dataset)
        rng = np.random.default_rng(42)
        indices = rng.choice(n_total, size=min(num_samples, n_total), replace=False)
        
        # Probe sequence length from first sample
        _, y_probe, _ = self.dataset[indices[0]]
        seq_len = y_probe.shape[0]
        time_steps = np.linspace(0, seq_len - 1, num_time_steps, dtype=int)
        
        fig, axes = plt.subplots(
            len(indices), num_time_steps,
            figsize=(3.2 * num_time_steps, 3.0 * len(indices)),
            squeeze=False,
        )
        
        for row, idx in enumerate(indices):
            _, y, _ = self.dataset[idx]
            y_phys = self._unnormalize(y)  # (T, C, ...)
            
            # Consistent colorbar limits per sample row
            field_slice = y_phys[:, channel]
            vmin, vmax = float(field_slice.min()), float(field_slice.max())
            
            for col, t in enumerate(time_steps):
                frame = y_phys[t, channel]
                ndim = frame.ndim
                
                if ndim == 1:
                    axes[row, col].plot(frame, color="steelblue", linewidth=1.2)
                    axes[row, col].set_ylim(vmin, vmax)
                elif ndim == 2:
                    im = axes[row, col].imshow(
                        frame, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax
                    )
                    axes[row, col].axis("off")
                else:
                    # 3D: take central slice
                    mid = frame.shape[0] // 2
                    im = axes[row, col].imshow(
                        frame[mid], cmap=cmap, origin="lower", vmin=vmin, vmax=vmax
                    )
                    axes[row, col].axis("off")
                
                if row == 0:
                    axes[row, col].set_title(f"$t={t}$", fontsize=11)
                if col == 0:
                    axes[row, col].set_ylabel(f"Sample {idx}", fontsize=10)
        
        fig.suptitle(
            "Evolution of spatial fields across selected time horizons",
            fontsize=13, y=1.02,
        )
        plt.tight_layout()
        
        if save_path:
            self._save_dual_format(fig, save_path)
        return fig


class PhysicalStatisticsVisualizer(BaseVisualizer):
    """
    Dataset-level physical statistics: population-averaged energy spectrum
    and initial-state kinetic energy distribution.
    
    Validates that the dataset's spectral content follows expected physical
    decay laws and that initial conditions span a meaningful energy range.
    """

    def plot_sample(self, idx: int, time_steps: Optional[List[int]] = None, 
                    save_path: Optional[str] = None) -> Figure:
        """Required by ABC — delegates to plot_averaged_spectrum."""
        return self.plot_averaged_spectrum(save_path=save_path)

    def plot_averaged_spectrum(
        self,
        num_samples: int = 50,
        time_step: int = -1,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Log-log plot of the isotropic energy spectrum E(k) averaged across
        `num_samples` samples at a given time step.
        """
        n_total = len(self.dataset)
        sample_ids = np.random.default_rng(42).choice(
            n_total, size=min(num_samples, n_total), replace=False
        )
        
        all_spectra = []
        k_ref = None
        
        for idx in sample_ids:
            _, y, _ = self.dataset[int(idx)]
            y_phys = self._unnormalize(y)[time_step]  # (C, ...)
            # Average over channels
            for c in range(y_phys.shape[0]):
                k, psd = self.compute_energy_spectrum(y_phys[c])
                if k_ref is None:
                    k_ref = k
                if len(psd) == len(k_ref):
                    all_spectra.append(psd)
        
        spectra = np.array(all_spectra)
        mean_psd = spectra.mean(axis=0)
        std_psd = spectra.std(axis=0)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.loglog(k_ref, mean_psd, color="steelblue", linewidth=2.0, label="Mean $E(k)$")
        ax.fill_between(
            k_ref, 
            np.maximum(mean_psd - std_psd, 1e-12), 
            mean_psd + std_psd,
            alpha=0.25, color="steelblue", label=r"$\pm 1\sigma$"
        )
        
        ax.set_xlabel("Wavenumber $k$")
        ax.set_ylabel("Spectral Power Density $E(k)$")
        ax.set_title(
            f"Population-averaged spatial energy spectrum ($N={len(sample_ids)}$, $t={time_step}$)"
        )
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(framealpha=0.8)
        
        if save_path:
            self._save_dual_format(fig, save_path)
        return fig

    def plot_energy_distribution(
        self,
        num_samples: int = 200,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Violin/histogram of total kinetic energy at t=0 across the dataset,
        illustrating variance in initial conditions.
        """
        n_total = len(self.dataset)
        sample_ids = np.random.default_rng(42).choice(
            n_total, size=min(num_samples, n_total), replace=False
        )
        
        energies = []
        for idx in sample_ids:
            x, _, _ = self.dataset[int(idx)]
            x_phys = self._unnormalize(x)  # (C, ...)
            # Total kinetic energy: 0.5 * sum(u^2) over all channels and spatial dims
            ke = 0.5 * np.sum(x_phys ** 2)
            energies.append(ke)
        
        energies = np.array(energies)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [3, 1]})
        
        # Left: Histogram
        axes[0].hist(energies, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
        axes[0].set_xlabel("Total Kinetic Energy at $t=0$")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Distribution of initial-state kinetic energy")
        
        # Right: Violin
        parts = axes[1].violinplot(energies, showmeans=True, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("steelblue")
            pc.set_alpha(0.6)
        axes[1].set_ylabel("Kinetic Energy")
        axes[1].set_xticks([])
        axes[1].set_title("Violin summary")
        
        plt.tight_layout()
        if save_path:
            self._save_dual_format(fig, save_path)
        return fig


class InitialConditionDiversityVisualizer(BaseVisualizer):
    """
    Projects all initial conditions (t=0) into a 2D manifold via PCA or t-SNE,
    color-coded by a global scalar metric (total energy), to characterize the
    diversity and coverage of the training data manifold.
    """
    
    def plot_sample(self, idx: int, time_steps: Optional[List[int]] = None, 
                    save_path: Optional[str] = None) -> Figure:
        """Required by ABC — delegates to plot_manifold."""
        return self.plot_manifold(save_path=save_path)

    def plot_manifold(
        self,
        num_samples: int = 500,
        method: str = "pca",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        2D scatter plot of initial conditions projected via PCA or t-SNE.
        
        Args:
            num_samples: Number of samples to project.
            method:      'pca' or 'tsne'.
            save_path:   Base path for dual-format output.
        """
        from sklearn.decomposition import PCA
        
        n_total = len(self.dataset)
        sample_ids = np.random.default_rng(42).choice(
            n_total, size=min(num_samples, n_total), replace=False
        )
        
        flat_ics = []
        energies = []
        
        for idx in sample_ids:
            x, _, _ = self.dataset[int(idx)]
            x_phys = self._unnormalize(x)  # (C, ...)
            flat_ics.append(x_phys.ravel())
            energies.append(0.5 * np.sum(x_phys ** 2))
        
        X = np.stack(flat_ics, axis=0)  # (N, D)
        energies = np.array(energies)
        
        if method == "tsne":
            try:
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
                label = "t-SNE"
            except ImportError:
                print("[!] sklearn not available for t-SNE, falling back to PCA.")
                reducer = PCA(n_components=2, random_state=42)
                label = "PCA"
        else:
            reducer = PCA(n_components=2, random_state=42)
            label = "PCA"
        
        Z = reducer.fit_transform(X)  # (N, 2)
        
        fig, ax = plt.subplots(figsize=(7, 6))
        scatter = ax.scatter(
            Z[:, 0], Z[:, 1],
            c=energies, cmap="plasma", s=18, alpha=0.75, edgecolors="none",
        )
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Total Initial Kinetic Energy")
        
        ax.set_xlabel(f"{label} Component 1")
        ax.set_ylabel(f"{label} Component 2")
        ax.set_title(
            f"Initial condition manifold ({label} projection, $N={len(sample_ids)}$)"
        )
        ax.grid(True, ls="--", alpha=0.3)
        
        if save_path:
            self._save_dual_format(fig, save_path)
        return fig


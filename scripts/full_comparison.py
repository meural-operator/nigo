"""
=============================================================================
Full Comparison: FNO, UNet, PINN vs TurboNIGO checkpoints (ep90, ep120, best, latest)
All models use the SAME initial condition at ν=0.1 for fair comparison.
=============================================================================
"""

import sys, os, time, json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
PDEBENCH_ROOT = PROJECT_ROOT / "PDEBench"
sys.path.insert(0, str(PDEBENCH_ROOT))

from pdebench.models.fno.fno import FNO1d
from pdebench.models.unet.unet import UNet1d

TURBO_DIR = PROJECT_ROOT / "RUN_BURGERS_1D"
sys.path.insert(0, str(TURBO_DIR / "src_mirror"))
from models.turbo_nigo_1d import GlobalTurboNIGO_1D

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = PROJECT_ROOT / "full_comparison_results"
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR = PROJECT_ROOT / "models"

# ── Constants ──────────────────────────────────────────────────────────
NU = 0.1
TOTAL_STEPS = 1000
BLOCK_SIZE = 20
SPATIAL_RES = 256       # PDEBench resolution
SPATIAL_FULL = 1024     # TurboNIGO resolution
INITIAL_STEP = 10
G_MIN = -3.1445140838623047
G_MAX = 3.3384177684783936
G_RANGE = G_MAX - G_MIN

TURBO_CHECKPOINTS = {
    "TurboNIGO_ep90":   TURBO_DIR / "epoch_90_model.pth",
    "TurboNIGO_ep120":  TURBO_DIR / "epoch_120_model.pth",
    "TurboNIGO_best":   TURBO_DIR / "best_model.pth",
    "TurboNIGO_latest": TURBO_DIR / "latest_model.pth",
}

COLORS = {
    "FNO1d":             "#2196F3",
    "UNet1d":            "#FF9800",
    "PINN":              "#4CAF50",
    "TurboNIGO_ep90":    "#E91E63",
    "TurboNIGO_ep120":   "#9C27B0",
    "TurboNIGO_best":    "#F44336",
    "TurboNIGO_latest":  "#795548",
}

LINESTYLES = {
    "FNO1d": "-", "UNet1d": "-", "PINN": "-",
    "TurboNIGO_ep90": "--", "TurboNIGO_ep120": "-.",
    "TurboNIGO_best": "-", "TurboNIGO_latest": ":",
}


# ── PINN Network ───────────────────────────────────────────────────────
class PINNNet(nn.Module):
    """Feedforward NN matching deepxde.nn.FNN architecture. Uses self.linears (ModuleList)."""
    def __init__(self, layer_sizes=None):
        super().__init__()
        if layer_sizes is None:
            layer_sizes = [2, 40, 40, 40, 40, 40, 40, 1]
        self.linears = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.linears.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        for i, linear in enumerate(self.linears[:-1]):
            x = torch.tanh(linear(x))
        x = self.linears[-1](x)
        return x


# ── IC Generation (copied from autoregressive_rollout.py) ─────────────
def generate_ic(spatial_resolution, initial_steps, nu, seed=42):
    """
    Generate IC for PDEBench models.
    Returns: [1, spatial, initial_steps, 1]
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi, spatial_resolution, endpoint=False)
    dx = x[1] - x[0]
    dt_ic = 0.001

    u = np.zeros_like(x)
    for k in range(1, 6):
        amp = rng.uniform(-1.0, 1.0) / k
        phase = rng.uniform(0, 2 * np.pi)
        u += amp * np.sin(k * x + phase)
    u = u / (np.max(np.abs(u)) + 1e-8) * 1.0

    frames = [u.copy()]
    for _ in range(initial_steps - 1):
        dudx = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
        d2udx2 = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2
        u = u - dt_ic * u * dudx + dt_ic * nu * d2udx2
        frames.append(u.copy())

    ic = np.stack(frames, axis=1)  # [spatial, initial_steps]
    ic = ic[np.newaxis, :, :, np.newaxis]  # [1, spatial, initial_steps, 1]
    return torch.tensor(ic, dtype=torch.float32)


# ── Rollout: FNO (exact copy from autoregressive_rollout.py) ──────────
def rollout_fno(model, initial_state):
    """
    FNO1d forward: model(x, grid)
      x: [1, S, initial_step*channels]
      grid: [1, S, 1]
      output: [1, S, 1, 1]
    """
    S = SPATIAL_RES
    n_blocks = TOTAL_STEPS // BLOCK_SIZE
    grid = torch.linspace(0, 1, S).reshape(1, S, 1)
    xx = initial_state.clone()  # [1, S, initial_step, 1]

    predictions = []
    t_start = time.perf_counter()
    diverged = False

    for block_idx in range(n_blocks):
        with torch.no_grad():
            for t in range(BLOCK_SIZE):
                inp = xx.reshape(1, S, -1)          # [1, S, 10]
                pred = model(inp, grid)              # [1, S, 1, 1]
                predictions.append(pred.squeeze().cpu().numpy())

                if pred.abs().max().item() > 1e6 or torch.isnan(pred).any():
                    diverged = True
                    break

                xx = torch.cat([xx[..., 1:, :], pred], dim=-2)

        if diverged:
            last = predictions[-1]
            while len(predictions) < TOTAL_STEPS:
                predictions.append(last)
            break

    total_time = time.perf_counter() - t_start
    return np.array(predictions[:TOTAL_STEPS]), total_time, diverged


# ── Rollout: UNet (exact copy from autoregressive_rollout.py) ─────────
def rollout_unet(model, initial_state):
    """
    UNet1d forward: model(x)
      x: [1, initial_step*channels, S]
      output: [1, out_channels, S]
    """
    S = SPATIAL_RES
    n_blocks = TOTAL_STEPS // BLOCK_SIZE
    xx = initial_state.clone()  # [1, S, initial_step, 1]

    predictions = []
    t_start = time.perf_counter()
    diverged = False

    for block_idx in range(n_blocks):
        with torch.no_grad():
            for t in range(BLOCK_SIZE):
                inp = xx.reshape(1, S, -1)           # [1, S, 10]
                inp = inp.permute(0, 2, 1)           # [1, 10, S]
                pred = model(inp)                     # [1, 1, S]
                predictions.append(pred.squeeze().cpu().numpy())

                if pred.abs().max().item() > 1e6 or torch.isnan(pred).any():
                    diverged = True
                    break

                pred_r = pred.permute(0, 2, 1).unsqueeze(-1)  # [1, S, 1, 1]
                xx = torch.cat([xx[..., 1:, :], pred_r], dim=-2)

        if diverged:
            last = predictions[-1]
            while len(predictions) < TOTAL_STEPS:
                predictions.append(last)
            break

    total_time = time.perf_counter() - t_start
    return np.array(predictions[:TOTAL_STEPS]), total_time, diverged


# ── Rollout: PINN (exact copy from autoregressive_rollout.py) ─────────
def rollout_pinn(model):
    """PINN forward: model(xt) where xt is [S, 2] = (x, t) coordinates."""
    S = SPATIAL_RES
    n_blocks = TOTAL_STEPS // BLOCK_SIZE
    x_coords = torch.linspace(0, 1, S)
    t_start_domain, t_end_domain = 0.0, 2.0
    t_extended_end = t_end_domain + (TOTAL_STEPS / 100.0) * (t_end_domain - t_start_domain)
    t_values = torch.linspace(t_start_domain, t_extended_end, TOTAL_STEPS + 1)[1:]

    predictions = []
    t_start = time.perf_counter()
    diverged = False

    for block_idx in range(n_blocks):
        with torch.no_grad():
            for t_idx in range(BLOCK_SIZE):
                global_idx = block_idx * BLOCK_SIZE + t_idx
                t_val = t_values[global_idx]
                t_col = torch.full((S,), t_val.item())
                inp = torch.stack([x_coords, t_col], dim=1)
                pred = model(inp).squeeze().cpu().numpy()
                predictions.append(pred)

                if np.max(np.abs(pred)) > 1e6 or np.isnan(pred).any():
                    diverged = True
                    break

        if diverged:
            last = predictions[-1]
            while len(predictions) < TOTAL_STEPS:
                predictions.append(last)
            break

    total_time = time.perf_counter() - t_start
    return np.array(predictions[:TOTAL_STEPS]), total_time, diverged


# ── Rollout: TurboNIGO (exact copy from autoregressive_rollout.py) ────
def rollout_turbo(model, ic_1024_tensor):
    """
    TurboNIGO: model(u0, time_steps, cond)
      u0: [1, 1, 1024] (normalized)
      time_steps: [20]
      cond: [1, 4]
      output: u_pred [1, 20, 1, 1024] (normalized)
    """
    seq_len = BLOCK_SIZE
    n_blocks = TOTAL_STEPS // seq_len

    u0 = (ic_1024_tensor - G_MIN) / (G_RANGE + 1e-8)  # normalize
    time_steps = torch.linspace(0, 1.0, seq_len)
    cond = torch.zeros(1, 4)

    predictions = []
    t_start = time.perf_counter()
    diverged = False

    for block_idx in range(n_blocks):
        with torch.no_grad():
            u_pred, _, _, _, _, _ = model(u0, time_steps, cond)

            if u_pred.abs().max().item() > 1e6 or torch.isnan(u_pred).any():
                diverged = True
                last = predictions[-1] if predictions else np.zeros(SPATIAL_RES)
                while len(predictions) < TOTAL_STEPS:
                    predictions.append(last)
                break

            for t in range(seq_len):
                frame_norm = u_pred[0, t, 0, :].cpu().numpy()
                frame_phys = frame_norm * G_RANGE + G_MIN
                frame_ds = frame_phys[::SPATIAL_FULL // SPATIAL_RES]
                predictions.append(frame_ds)

            u0 = u_pred[:, -1, :, :]

    total_time = time.perf_counter() - t_start
    return np.array(predictions[:TOTAL_STEPS]), total_time, diverged


# ── Plotting ──────────────────────────────────────────────────────────

def plot_all(results):
    # ── 1. Diagnostics (Energy / Variance / Max|u|) ───────────────
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(
        f"1D Burgers — 1000-Step Autoregressive Rollout (ν = {NU})\n"
        f"FNO, UNet, PINN (PDEBench) vs TurboNIGO Checkpoints (Ours)",
        fontsize=15, fontweight="bold"
    )

    for name, res in results.items():
        traj = res["trajectory"]
        energy = np.sqrt(np.mean(traj**2, axis=1))
        variance = np.var(traj, axis=1)
        max_amp = np.max(np.abs(traj), axis=1)
        c, ls = COLORS[name], LINESTYLES[name]
        lw = 2.5 if "Turbo" in name else 1.8

        axes[0].semilogy(energy, label=f"{name} (E={energy[-1]:.4f})", color=c, ls=ls, lw=lw)
        axes[1].semilogy(variance, color=c, ls=ls, lw=lw, label=name)
        axes[2].semilogy(max_amp, color=c, ls=ls, lw=lw, label=name)

    axes[0].set_ylabel("RMS Energy", fontsize=13); axes[0].legend(fontsize=9, loc="upper right", ncol=2); axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("Spatial Variance", fontsize=13); axes[1].grid(True, alpha=0.3)
    axes[2].set_ylabel("Max |u|", fontsize=13); axes[2].set_xlabel("Rollout Step", fontsize=13); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "full_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ full_diagnostics.png")

    # ── 2. Space-Time Heatmaps ────────────────────────────────────
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 5))
    fig.suptitle(f"Space-Time Heatmaps — All Models (ν = {NU})", fontsize=14, fontweight="bold")

    for i, (name, res) in enumerate(results.items()):
        ax = axes[i] if n > 1 else axes
        traj = res["trajectory"]
        vabs = max(abs(traj.min()), abs(traj.max()))
        im = ax.imshow(traj.T, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-vabs, vmax=vabs, extent=[0, TOTAL_STEPS, 0, SPATIAL_RES])
        ax.set_title(f"{name}\n(E={res['final_energy']:.4f})", fontsize=8)
        ax.set_xlabel("Time Step", fontsize=8)
        if i == 0: ax.set_ylabel("Spatial Grid", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "full_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ full_heatmaps.png")

    # ── 3. Snapshots ──────────────────────────────────────────────
    snap_steps = [0, 20, 50, 100, 250, 500, 999]
    fig, axes = plt.subplots(1, len(snap_steps), figsize=(3.2 * len(snap_steps), 4.5))
    fig.suptitle(f"Solution Snapshots — All Models (ν = {NU})", fontsize=14, fontweight="bold")
    x_grid = np.linspace(0, 1, SPATIAL_RES)

    for i, step in enumerate(snap_steps):
        ax = axes[i]
        for name, res in results.items():
            c, ls = COLORS[name], LINESTYLES[name]
            lw = 2.0 if "Turbo" in name else 1.2
            ax.plot(x_grid, res["trajectory"][step], label=name, color=c, ls=ls, lw=lw)
        ax.set_title(f"t = {step}", fontsize=10)
        ax.set_xlabel("x", fontsize=9)
        if i == 0:
            ax.set_ylabel("u(x, t)", fontsize=10)
            ax.legend(fontsize=5, loc="lower left")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "full_snapshots.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ full_snapshots.png")

    # ── 4. Summary bar chart ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    names = list(results.keys())
    energies = [results[n]["final_energy"] for n in names]
    colors_list = [COLORS[n] for n in names]

    bars = ax.bar(range(len(names)), energies, color=colors_list, edgecolor="black", lw=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Final RMS Energy (Step 1000)", fontsize=12)
    ax.set_title(f"Final Energy Comparison — 1000-Step Rollout (ν = {NU})", fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, energies):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.3,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "full_energy_bars.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ full_energy_bars.png")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  FULL COMPARISON: PDEBench Models vs TurboNIGO Checkpoints")
    print(f"  ν = {NU} | {TOTAL_STEPS} steps | Block size = {BLOCK_SIZE}")
    print("=" * 70)

    # ── Generate ICs ──────────────────────────────────────────────
    ic_256 = generate_ic(SPATIAL_RES, INITIAL_STEP, NU)        # [1, 256, 10, 1]
    ic_1024 = generate_ic(SPATIAL_FULL, 1, NU)                  # [1, 1024, 1, 1]
    ic_turbo = ic_1024[:, :, 0, :].permute(0, 2, 1)            # [1, 1, 1024]
    print(f"  IC generated: PDEBench={list(ic_256.shape)}, TurboNIGO={list(ic_turbo.shape)}")

    results = {}

    # ── FNO ────────────────────────────────────────────────────────
    fno_path = MODELS_DIR / "burgers_FNO" / f"1D_Burgers_Sols_Nu{NU}_FNO.pt"
    if fno_path.exists():
        print(f"\n  [FNO1d] Loading...")
        model = FNO1d(num_channels=1, modes=12, width=20, initial_step=INITIAL_STEP)
        ckpt = torch.load(fno_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"]); model.eval()
        print(f"    params={sum(p.numel() for p in model.parameters()):,}, epoch={ckpt['epoch']}")
        print(f"  [FNO1d] Rolling out...")
        traj, t, div = rollout_fno(model, ic_256)
        energy = np.sqrt(np.mean(traj**2, axis=1))
        results["FNO1d"] = {"trajectory": traj, "time": t, "diverged": div, "final_energy": float(energy[-1])}
        print(f"  [FNO1d] ✓ {t:.2f}s | E_final={energy[-1]:.4f} | div={div}")
        del model

    # ── UNet ───────────────────────────────────────────────────────
    unet_path = MODELS_DIR / "burgers_Unet" / f"1D_Burgers_Sols_Nu{NU}_Unet-PF-20.pt"
    if unet_path.exists():
        print(f"\n  [UNet1d] Loading...")
        model = UNet1d(in_channels=INITIAL_STEP, out_channels=1, init_features=32)
        ckpt = torch.load(unet_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"]); model.eval()
        print(f"    params={sum(p.numel() for p in model.parameters()):,}, epoch={ckpt['epoch']}")
        print(f"  [UNet1d] Rolling out...")
        traj, t, div = rollout_unet(model, ic_256)
        energy = np.sqrt(np.mean(traj**2, axis=1))
        results["UNet1d"] = {"trajectory": traj, "time": t, "diverged": div, "final_energy": float(energy[-1])}
        print(f"  [UNet1d] ✓ {t:.2f}s | E_final={energy[-1]:.4f} | div={div}")
        del model

    # ── PINN ───────────────────────────────────────────────────────
    pinn_path = MODELS_DIR / "burgers_PINN" / f"1D_Burgers_Sols_Nu{NU}_PINN.pt-15000.pt"
    if pinn_path.exists():
        print(f"\n  [PINN] Loading...")
        model = PINNNet([2, 40, 40, 40, 40, 40, 40, 1])
        ckpt = torch.load(pinn_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"]); model.eval()
        print(f"    params={sum(p.numel() for p in model.parameters()):,}")
        print(f"  [PINN] Rolling out...")
        traj, t, div = rollout_pinn(model)
        energy = np.sqrt(np.mean(traj**2, axis=1))
        results["PINN"] = {"trajectory": traj, "time": t, "diverged": div, "final_energy": float(energy[-1])}
        print(f"  [PINN] ✓ {t:.2f}s | E_final={energy[-1]:.4f} | div={div}")
        del model

    # ── TurboNIGO checkpoints ─────────────────────────────────────
    for name, ckpt_path in TURBO_CHECKPOINTS.items():
        if not ckpt_path.exists():
            print(f"\n  [{name}] NOT FOUND: {ckpt_path.name}")
            continue
        print(f"\n  [{name}] Loading...")
        model = GlobalTurboNIGO_1D(
            latent_dim=64, in_channels=1, width=32,
            spatial_size=SPATIAL_FULL, num_layers=3, cond_dim=4,
            use_residual=True, norm_type='group',
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=True); model.eval()
        ep = ckpt.get('epoch', '?')
        vl = ckpt.get('best_val_loss', float('nan'))
        print(f"    epoch={ep}, val_loss={vl:.2e}, params={sum(p.numel() for p in model.parameters()):,}")
        print(f"  [{name}] Rolling out...")
        traj, t, div = rollout_turbo(model, ic_turbo.clone())
        energy = np.sqrt(np.mean(traj**2, axis=1))
        results[name] = {"trajectory": traj, "time": t, "diverged": div, "final_energy": float(energy[-1])}
        print(f"  [{name}] ✓ {t:.2f}s | E_final={energy[-1]:.4f} | div={div}")
        del model

    # ── Summary Table ──────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  {'Model':<22} {'Time(s)':<10} {'Final E':<12} {'Diverged'}")
    print(f"  {'-'*22} {'-'*10} {'-'*12} {'-'*8}")
    for name, r in results.items():
        print(f"  {name:<22} {r['time']:<10.3f} {r['final_energy']:<12.6f} {r['diverged']}")
    print(f"{'=' * 70}")

    # ── Plot ───────────────────────────────────────────────────────
    print("\n  Generating plots...")
    plot_all(results)

    # ── Save JSON ──────────────────────────────────────────────────
    summary = {n: {"time": r["time"], "final_energy": r["final_energy"], "diverged": r["diverged"]}
               for n, r in results.items()}
    with open(OUTPUT_DIR / "full_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  All results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

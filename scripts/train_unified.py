"""
Unified training entry point for TurboNIGO.

Supports all datasets (flow, ns_incom, burgers, ks, darcy) and all model
variants (V1 base, V2 attention-physics) via a single config-driven interface.

Usage:
    python scripts/train_unified.py --config turbo_nigo/configs/burgers_config.yaml
    python scripts/train_unified.py --config turbo_nigo/configs/ns_incom_config.yaml
    python scripts/train_unified.py --config turbo_nigo/configs/ks_config.yaml
    python scripts/train_unified.py --config turbo_nigo/configs/darcy_config.yaml
    python scripts/train_unified.py --config turbo_nigo/configs/default_config.yaml

    # Override from CLI:
    python scripts/train_unified.py --config ... --epochs 100 --batch_size 32

    # Resume from checkpoint:
    python scripts/train_unified.py --config ... --resume_from results/checkpoints/latest.pth
"""
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader

from turbo_nigo.configs import get_args_and_config
from turbo_nigo.utils.misc import seed_everything


# ======================================================================
# Model factory
# ======================================================================
def create_model(config: dict) -> torch.nn.Module:
    """Instantiate GlobalTurboNIGO (V1) or GlobalTurboNIGO_V2."""
    model_type = config.get("model_type", "v1").lower()

    common = dict(
        latent_dim=config["latent_dim"],
        num_bases=config["num_bases"],
        cond_dim=config["cond_dim"],
        width=config["width"],
        in_channels=config.get("in_channels", 2),
    )

    if model_type == "v2":
        from turbo_nigo.models.extensions import GlobalTurboNIGO_V2

        model = GlobalTurboNIGO_V2(
            **common,
            target_res=config.get("spatial_size", 64),
            physics_net_type=config.get("physics_net_type", "distribution"),
        )
    else:
        from turbo_nigo.models import GlobalTurboNIGO

        model = GlobalTurboNIGO(
            **common,
            spatial_size=config.get("spatial_size", 64),
        )

    return model


# ======================================================================
# Dataset factory
# ======================================================================
def create_dataloaders(config: dict):
    """Build train and val DataLoaders based on config['dataset_type']."""
    dt = config.get("dataset_type", "flow")
    bs = config["batch_size"]
    nw = config.get("num_workers", 4)

    loader_kwargs = dict(
        batch_size=bs,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=nw > 0,
    )

    if dt == "flow":
        return _loaders_flow(config, loader_kwargs)
    if dt == "ns_incom":
        return _loaders_ns_incom(config, loader_kwargs)
    if dt == "burgers":
        return _loaders_burgers(config, loader_kwargs)
    if dt == "ks":
        return _loaders_ks(config, loader_kwargs)
    if dt == "darcy":
        return _loaders_darcy(config, loader_kwargs)

    raise ValueError(f"Unknown dataset_type '{dt}'. "
                     f"Choose from: flow, ns_incom, burgers, ks, darcy")


# ------------------------------------------------------------------
# Flow (bluff-body cylinder)
# ------------------------------------------------------------------
def _loaders_flow(config, lkw):
    from turbo_nigo.data import InMemoryFlowDataset, compute_global_stats_and_cond_stats

    root = config["data_root"]
    sl = config["seq_len"]
    g_min, g_max, c_mean, c_std = compute_global_stats_and_cond_stats(root)

    train_ds = InMemoryFlowDataset.create_with_stats(
        root, sl, "train", g_min, g_max, c_mean, c_std,
        max_cases=config.get("max_cases"),
    )
    val_ds = InMemoryFlowDataset.create_with_stats(
        root, sl, "val", g_min, g_max, c_mean, c_std,
        max_cases=config.get("max_cases"),
    )

    return (
        DataLoader(train_ds, shuffle=True, **lkw),
        DataLoader(val_ds, shuffle=False, **lkw),
    )


# ------------------------------------------------------------------
# NS incompressible 2D (H5)
# ------------------------------------------------------------------
def _loaders_ns_incom(config, lkw):
    from turbo_nigo.data.h5_dataset import H5FlowDataset

    common = dict(
        h5_path=config["data_root"],
        target_res=config.get("target_res", 64),
        seq_len=config["seq_len"],
        temporal_stride=config.get("temporal_stride", 20),
        window_stride=config.get("window_stride", 5),
        g_min=config.get("ns_g_min", -3.0),
        g_max=config.get("ns_g_max", 3.0),
    )

    train_ds = H5FlowDataset(
        **common, mode="train",
        train_batches=config.get("train_batches", [0, 1, 2]),
        val_batches=config.get("val_batches", [3]),
    )
    val_ds = H5FlowDataset(
        **common, mode="val",
        train_batches=config.get("train_batches", [0, 1, 2]),
        val_batches=config.get("val_batches", [3]),
    )

    return (
        DataLoader(train_ds, shuffle=True, **lkw),
        DataLoader(val_ds, shuffle=False, **lkw),
    )


# ------------------------------------------------------------------
# Burgers equation
# ------------------------------------------------------------------
def _loaders_burgers(config, lkw):
    from turbo_nigo.data.burgers_dataset import BurgersDataset

    common = dict(
        data_path=config["data_root"],
        seq_len=config["seq_len"],
        target_spatial_res=config.get("target_spatial_res", 64),
        train_split=config.get("train_split", 0.8),
        viscosity=config.get("viscosity", 0.1),
    )

    train_ds = BurgersDataset(**common, mode="train")
    val_ds = BurgersDataset(
        **common, mode="val",
        g_min=train_ds.g_min, g_max=train_ds.g_max,  # use train stats
    )

    return (
        DataLoader(train_ds, shuffle=True, **lkw),
        DataLoader(val_ds, shuffle=False, **lkw),
    )


# ------------------------------------------------------------------
# Kuramoto-Sivashinsky
# ------------------------------------------------------------------
def _loaders_ks(config, lkw):
    from turbo_nigo.data.ks_dataset import KSDataset

    common = dict(
        h5_path=config["data_root"],
        seq_len=config["seq_len"],
        temporal_stride=config.get("ks_temporal_stride", 4),
        spatial_res=config.get("ks_spatial_res", 64),
        cond_dim=config.get("cond_dim", 4),
    )

    train_ds = KSDataset(
        **common, mode="train",
        max_trajectories=config.get("ks_train_cases", 5000),
    )
    val_ds = KSDataset(
        **common, mode="val",
        max_trajectories=config.get("ks_val_cases", 1000),
        g_min=train_ds.g_min, g_max=train_ds.g_max,
    )

    return (
        DataLoader(train_ds, shuffle=True, **lkw),
        DataLoader(val_ds, shuffle=False, **lkw),
    )


# ------------------------------------------------------------------
# Darcy flow
# ------------------------------------------------------------------
def _loaders_darcy(config, lkw):
    from turbo_nigo.data.darcy_dataset import DarcyFlowDataset

    common = dict(
        data_path=config["data_root"],
        seq_len=config["seq_len"],  # must be 1
        target_res=config.get("target_res", 64),
        train_split=config.get("train_split", 0.8),
    )

    train_ds = DarcyFlowDataset(**common, mode="train")
    val_ds = DarcyFlowDataset(
        **common, mode="val",
        g_min=train_ds.g_min, g_max=train_ds.g_max,
    )

    return (
        DataLoader(train_ds, shuffle=True, **lkw),
        DataLoader(val_ds, shuffle=False, **lkw),
    )


# ======================================================================
# Results directory
# ======================================================================
def setup_results_dir(config: dict) -> str:
    """Create a timestamped results directory."""
    base = config.get("results_dir", "./results")
    name = config.get("experiment_name", "experiment")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base, f"{name}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# ======================================================================
# Main
# ======================================================================
def main():
    config = get_args_and_config()
    seed_everything(config.get("seed", 42))
    device = config.get("device", "cuda")

    print(f"\n{'='*74}")
    print(f"  TurboNIGO Unified Training")
    print(f"  Dataset : {config.get('dataset_type', 'flow')}")
    print(f"  Model   : {config.get('model_type', 'v1').upper()}")
    print(f"  Device  : {device}")
    print(f"{'='*74}\n")

    # 1. Results directory
    results_dir = setup_results_dir(config)

    # 2. Datasets
    print("[1/3] Loading datasets ...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"      Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 3. Model
    print("[2/3] Building model ...")
    model = create_model(config).to(device)

    # Parameter summary
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      {model.__class__.__name__}: {total:,} params ({trainable:,} trainable)")
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        print(f"        {name:20s} — {n:>10,} params")

    # 4. Train
    print("[3/3] Starting training ...\n")
    from turbo_nigo.core.unified_trainer import UnifiedTrainer

    trainer = UnifiedTrainer(model, train_loader, val_loader, config, results_dir)
    trainer.train()


if __name__ == "__main__":
    main()

"""
Research-grade unified training engine for TurboNIGO.

Designed for multi-dataset, multi-model experiments with production-level
logging, checkpointing, and reproducibility guarantees.

Key properties:
  (a) Checkpointing: periodic (every N epochs, default 30), best, latest
  (b) Full resumability: model + optimizer + scaler + scheduler + epoch + RNG
  (c) Immutable metadata: frozen config, model info, system info at run start
  (d) Per-epoch CSV + TensorBoard logging of all losses and diagnostics
  (e) tqdm progress bars with MSE, LR, α, β, grad norm
  (f) Structured results directory with immutable / mutable separation
  (g) Anonymous, modular, extensible

Compatible with GlobalTurboNIGO (V1) and GlobalTurboNIGO_V2 — both share the
same forward signature: (u0, time_steps, cond) → (u_pred, z_base, k, r, α, β).
"""
import csv
import json
import os
import platform
import sys
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from turbo_nigo.core.losses import CompositeLoss
from turbo_nigo.utils.misc import count_parameters, format_time, get_gpu_memory_mb


# ======================================================================
# Results directory layout
# ======================================================================
_DIR_IMMUTABLE = "_immutable"
_DIR_CKPT = "checkpoints"
_DIR_LOGS = "logs"
_DIR_OUTPUTS = "outputs"


class UnifiedTrainer:
    """
    Manages the full training lifecycle for any TurboNIGO model variant.

    Args:
        model:        nn.Module — GlobalTurboNIGO or GlobalTurboNIGO_V2.
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader.
        config:       Experiment configuration dictionary.
        results_dir:  Root directory for this experiment's outputs.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        results_dir: str,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.get("device", "cuda")

        # ---- results directory structure ----
        self.paths = self._create_directory_tree(results_dir)

        # ---- optimizer ----
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 1e-6),
        )

        # ---- AMP scaler ----
        self.scaler = torch.amp.GradScaler(
            device=self.device,
            enabled=config.get("use_amp", True),
        )

        # ---- compute config ----
        self._apply_compute_settings(config)

        # ---- LR scheduler ----
        self.scheduler = self._build_scheduler(config)

        # ---- loss ----
        self.criterion = CompositeLoss(config)

        # ---- time steps (pre-computed, on device) ----
        self.time_steps = (
            torch.arange(1, config["seq_len"] + 1, dtype=torch.float32, device=self.device)
            * config.get("dt", 0.1)
        )

        # ---- state ----
        self.start_epoch = 1
        self.best_loss = float("inf")
        self.epochs_without_improvement = 0

        # ---- loggers ----
        self.csv_path = os.path.join(self.paths["logs"], "metrics.csv")
        self.tb_writer = None
        if config.get("use_tensorboard", True):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(
                    log_dir=os.path.join(self.paths["logs"], "tensorboard")
                )
            except ImportError:
                print("[WARN] tensorboard not installed — TB logging disabled.")

        # ---- save immutable metadata (before any resume) ----
        self._save_immutable_metadata()

        # ---- resume ----
        resume_path = config.get("resume_from")
        if resume_path and os.path.exists(str(resume_path)):
            self.load_checkpoint(str(resume_path))

    # ==================================================================
    # Directory setup
    # ==================================================================
    @staticmethod
    def _create_directory_tree(base: str) -> Dict[str, str]:
        paths = {
            "root": base,
            "immutable": os.path.join(base, _DIR_IMMUTABLE),
            "ckpt": os.path.join(base, _DIR_CKPT),
            "logs": os.path.join(base, _DIR_LOGS),
            "outputs": os.path.join(base, _DIR_OUTPUTS),
        }
        for p in paths.values():
            os.makedirs(p, exist_ok=True)
        return paths

    # ==================================================================
    # Compute settings
    # ==================================================================
    @staticmethod
    def _apply_compute_settings(config: Dict):
        if config.get("tf32", True) and torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
        if config.get("cudnn_benchmark", True) and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    # ==================================================================
    # LR scheduler
    # ==================================================================
    def _build_scheduler(self, cfg: Dict):
        name = cfg.get("scheduler", "none").lower()
        warmup = cfg.get("warmup_epochs", 0)
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(cfg["epochs"] - warmup, 1),
                eta_min=cfg.get("min_lr", 1e-6),
            )
        if name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min",
                factor=cfg.get("plateau_factor", 0.5),
                patience=cfg.get("plateau_patience", 10),
                min_lr=cfg.get("min_lr", 1e-6),
            )
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=cfg.get("step_size", 50),
                gamma=cfg.get("step_gamma", 0.5),
            )
        return None

    def _get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def _warmup_lr(self, epoch: int):
        warmup = self.config.get("warmup_epochs", 0)
        if warmup > 0 and epoch <= warmup:
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.config["learning_rate"] * (epoch / warmup)

    # ==================================================================
    # Immutable metadata
    # ==================================================================
    def _save_immutable_metadata(self):
        import yaml as _yaml

        immut = self.paths["immutable"]

        # 1. Frozen config
        with open(os.path.join(immut, "config.yaml"), "w") as f:
            _yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

        # 2. Model info
        model_info = count_parameters(self.model)
        model_info["architecture"] = self.model.__class__.__name__
        model_info["modules"] = {
            name: str(type(m).__name__)
            for name, m in self.model.named_children()
        }
        with open(os.path.join(immut, "model_info.json"), "w") as f:
            json.dump(model_info, f, indent=2, default=str)

        # 3. System info
        sys_info = {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "os": platform.system(),
            "platform": platform.platform(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(os.path.join(immut, "system_info.json"), "w") as f:
            json.dump(sys_info, f, indent=2)

    # ==================================================================
    # Checkpointing
    # ==================================================================
    def save_checkpoint(self, filepath: str, epoch: int):
        """Save full training state for perfect resumability."""
        state = {
            "epoch": epoch,
            "best_loss": self.best_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": self.config,
            "rng_torch": torch.random.get_rng_state(),
            "rng_numpy": np.random.get_state(),
        }
        if (
            self.scheduler is not None
            and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        ):
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        if torch.cuda.is_available():
            state["rng_cuda"] = torch.cuda.get_rng_state()
        torch.save(state, filepath)

    def load_checkpoint(self, filepath: str):
        """Restore full training state from a checkpoint."""
        print(f"[Resume] Loading: {filepath}")
        ckpt = torch.load(filepath, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        self.start_epoch = ckpt["epoch"] + 1
        self.best_loss = ckpt["best_loss"]
        self.epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)

        if (
            self.scheduler is not None
            and "scheduler_state_dict" in ckpt
            and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        ):
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        for key, setter in [
            ("rng_torch", torch.random.set_rng_state),
            ("rng_numpy", lambda s: np.random.set_state(s)),
        ]:
            if key in ckpt:
                setter(ckpt[key])
        if "rng_cuda" in ckpt and torch.cuda.is_available():
            torch.cuda.set_rng_state(ckpt["rng_cuda"])

        print(
            f"[Resume] Epoch {self.start_epoch} | "
            f"best_val={self.best_loss:.4e}"
        )

    # ==================================================================
    # Gradient diagnostics
    # ==================================================================
    def _compute_grad_norm(self) -> float:
        """
        Computes the total gradient norm. 
        Optimized to minimize CPU-GPU synchronization by using a single reduction.
        """
        all_norms = []
        for p in self.model.parameters():
            if p.grad is not None:
                # Square of the norm to sum them up later
                all_norms.append(p.grad.detach().data.norm(2).reshape(1) ** 2)
        
        if not all_norms:
            return 0.0
            
        # Sum on GPU, then one single .item() sync
        total_norm_sq = torch.sum(torch.cat(all_norms))
        total_norm = torch.sqrt(total_norm_sq).item()

        if np.isnan(total_norm) or np.isinf(total_norm):
            # If we hit NaN, find which one for debugging
            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    pn = p.grad.data.norm(2).item()
                    if np.isnan(pn) or np.isinf(pn):
                        print(f"\n[STABILITY] NaN gradient in: {name}")
                        break
            return float("nan")

        return total_norm

    # ==================================================================
    # Training epoch
    # ==================================================================
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        accum = {
            "loss": 0.0, "mse": 0.0, "spectral": 0.0,
            "physics_prior": 0.0, "relative_l2": 0.0,
            "alpha": 0.0, "beta": 0.0,
        }
        grad_norms = []

        pbar = tqdm(
            self.train_loader, desc=f"  Train {epoch:03d}", leave=False,
            bar_format="{l_bar}{bar:30}{r_bar}",
        )
        for i, (u0, u_seq_gt, cond) in enumerate(pbar):
            u0 = u0.to(self.device, non_blocking=True)
            u_seq_gt = u_seq_gt.to(self.device, non_blocking=True)
            cond = cond.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=self.device,
                enabled=self.config.get("use_amp", True),
            ):
                u_pred, _, k_c, r_c, alpha, beta = self.model(
                    u0, self.time_steps, cond
                )
                loss, ld = self.criterion(
                    u_pred, u_seq_gt, k_c, r_c, 
                    epoch=epoch, max_epochs=self.config.get("epochs")
                )

            self.scaler.scale(loss).backward()

            clip = self.config.get("grad_clip_norm", 0)
            if clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

            if self.config.get("log_grad_norm", True):
                g_norm = self._compute_grad_norm()
                grad_norms.append(g_norm)
                if np.isnan(g_norm) and epoch > self.config.get("warmup_epochs", 0):
                    pbar.set_description(f"  Train {epoch:03d} [!] NaN Grad")

            self.scaler.step(self.optimizer)
            self.scaler.update()

            accum["loss"] += ld["total"]
            accum["mse"] += ld["mse"]
            accum["spectral"] += ld.get("spectral", 0.0)
            accum["physics_prior"] += ld["physics_prior"]
            accum["relative_l2"] += ld.get("relative_l2", 0.0)
            accum["alpha"] += alpha.mean().detach()
            accum["beta"] += beta.mean().detach()

            # Throttle progress bar updates to reduce sync overhead
            if i % 10 == 0 or i == len(self.train_loader) - 1:
                pbar.set_postfix(OrderedDict(
                    mse=f"{ld['mse']:.3e}",
                    α=f"{alpha.mean().item():.3f}",
                    β=f"{beta.mean().item():.3f}",
                    lr=f"{self._get_lr():.1e}",
                ))

        n = max(len(self.train_loader), 1)
        return {
            "train_loss": accum["loss"] / n,
            "train_mse": accum["mse"] / n,
            "train_spectral": accum["spectral"] / n,
            "train_physics_prior": accum["physics_prior"] / n,
            "train_relative_l2": accum["relative_l2"] / n,
            "alpha_mean": accum["alpha"] / n,
            "beta_mean": accum["beta"] / n,
            "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        }

    # ==================================================================
    # Validation
    # ==================================================================
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        accum = {"loss": 0.0, "mse": 0.0, "spectral": 0.0,
                 "relative_l2": 0.0, "alpha": 0.0, "beta": 0.0}

        for u0, u_seq_gt, cond in self.val_loader:
            u0 = u0.to(self.device, non_blocking=True)
            u_seq_gt = u_seq_gt.to(self.device, non_blocking=True)
            cond = cond.to(self.device, non_blocking=True)

            with torch.amp.autocast(
                device_type=self.device,
                enabled=self.config.get("use_amp", True),
            ):
                u_pred, _, k_c, r_c, alpha, beta = self.model(
                    u0, self.time_steps, cond
                )
                _, ld = self.criterion(
                    u_pred, u_seq_gt, k_c, r_c,
                    # We pass max_epochs so validate uses the full horizon or matches train horizon
                    epoch=self.config.get("epochs") if not self.config.get("curriculum_eval") else None, 
                    max_epochs=self.config.get("epochs")
                )

            accum["loss"] += ld["total"]
            accum["mse"] += ld["mse"]
            accum["spectral"] += ld.get("spectral", 0.0)
            accum["relative_l2"] += ld.get("relative_l2", 0.0)
            accum["alpha"] += alpha.mean().item()
            accum["beta"] += beta.mean().item()

        n = max(len(self.val_loader), 1)
        return {
            "val_loss": accum["loss"] / n,
            "val_mse": accum["mse"] / n,
            "val_spectral": accum["spectral"] / n,
            "val_relative_l2": accum["relative_l2"] / n,
            "val_alpha": accum["alpha"] / n,
            "val_beta": accum["beta"] / n,
        }

    # ==================================================================
    # Logging helpers
    # ==================================================================
    def _log_csv(self, epoch: int, metrics: Dict[str, float]):
        write_header = not os.path.exists(self.csv_path)
        row = {"epoch": epoch, **metrics}
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow({k: f"{v:.6e}" if isinstance(v, float) else v for k, v in row.items()})

    def _log_tb(self, epoch: int, metrics: Dict[str, float]):
        if self.tb_writer is None:
            return
        for k, v in metrics.items():
            self.tb_writer.add_scalar(k, v, epoch)

    # ==================================================================
    # Main training loop
    # ==================================================================
    def train(self):
        total_epochs = self.config["epochs"]
        patience = self.config.get("early_stopping_patience", 0)
        ckpt_freq = self.config.get("checkpoint_freq", 30)

        # ---- torch.compile (optional, after resume) ----
        if self.config.get("compile", False) and hasattr(torch, "compile"):
            try:
                print("[compile] Wrapping model with torch.compile ...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception as e:
                print(f"[compile] Failed ({e}), using eager mode.")

        # ---- header ----
        print("=" * 74)
        print(f"  Experiment : {self.config.get('experiment_name', 'TurboNIGO')}")
        print(f"  Model      : {self.model.__class__.__name__}")
        print(f"  Dataset    : {self.config.get('dataset_type', 'flow')}")
        print(f"  Epochs     : {self.start_epoch} → {total_epochs}")
        print(f"  LR         : {self.config['learning_rate']}  ({self.config.get('scheduler', 'none')})")
        print(f"  Checkpoint : every {ckpt_freq} ep + best + latest")
        print(f"  Best val   : {self.best_loss:.4e}")
        print(f"  Results    : {self.paths['root']}")
        print("=" * 74)

        for epoch in range(self.start_epoch, total_epochs + 1):
            t0 = time.time()

            self._warmup_lr(epoch)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # ---- train + validate ----
            train_m = self.train_epoch(epoch)
            val_m = self.validate()

            dt_sec = time.time() - t0
            gpu_mb = get_gpu_memory_mb()

            metrics = {
                **train_m, **val_m,
                "lr": self._get_lr(),
                "epoch_time_sec": round(dt_sec, 2),
                "gpu_mem_mb": round(gpu_mb, 1),
            }

            # ---- log ----
            self._log_csv(epoch, metrics)
            self._log_tb(epoch, metrics)

            # ---- terminal ----
            eta = (total_epochs - epoch) * dt_sec
            print(
                f"Ep {epoch:03d}/{total_epochs} │ "
                f"T:{metrics['train_loss']:.3e} │ "
                f"V:{metrics['val_loss']:.3e} │ "
                f"MSE:{metrics['train_mse']:.3e} │ "
                f"α:{metrics['alpha_mean']:.3f} β:{metrics['beta_mean']:.3f} │ "
                f"∇:{metrics['grad_norm']:.2f} │ "
                f"LR:{metrics['lr']:.1e} │ "
                f"{format_time(dt_sec)} │ "
                f"ETA:{format_time(eta)}"
            )

            # ---- checkpoint: best ----
            val_loss = metrics["val_loss"]
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.epochs_without_improvement = 0
                best_p = os.path.join(self.paths["ckpt"], "best.pth")
                self.save_checkpoint(best_p, epoch)
                print(f"  ★ New best val_loss: {val_loss:.4e} → best.pth")
            else:
                self.epochs_without_improvement += 1

            # ---- checkpoint: periodic (every N epochs) ----
            if epoch % ckpt_freq == 0:
                ep_p = os.path.join(self.paths["ckpt"], f"epoch_{epoch:04d}.pth")
                self.save_checkpoint(ep_p, epoch)
                print(f"  ↳ Periodic checkpoint → epoch_{epoch:04d}.pth")

            # ---- checkpoint: latest (every epoch) ----
            self.save_checkpoint(
                os.path.join(self.paths["ckpt"], "latest.pth"), epoch
            )

            # ---- scheduler step ----
            warmup = self.config.get("warmup_epochs", 0)
            if self.scheduler is not None and epoch > warmup:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # ---- early stopping ----
            if patience > 0 and self.epochs_without_improvement >= patience:
                print(
                    f"\n[Early Stop] No improvement for {patience} epochs. "
                    f"Best: {self.best_loss:.4e}"
                )
                break

        # ---- final ----
        self.save_checkpoint(
            os.path.join(self.paths["ckpt"], "final.pth"), epoch
        )
        if self.tb_writer is not None:
            self.tb_writer.close()
        print(f"\n{'='*74}")
        print(f"  Training complete. Best val_loss: {self.best_loss:.4e}")
        print(f"  Results: {self.paths['root']}")
        print(f"{'='*74}")

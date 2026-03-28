"""
Research-grade training engine for TurboNIGO.

Features:
  - Full-state checkpointing (model + optimizer + scaler + epoch + best_loss + RNG states)
  - Perfect resume from any checkpoint
  - Configurable LR scheduler (cosine, plateau, step, or none)
  - Gradient clipping with norm tracking
  - Early stopping with patience
  - Rich terminal output with timing, LR, grad norm, VRAM
  - Comprehensive metrics logging via ExperimentLogger
"""
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from turbo_nigo.utils.logger import ExperimentLogger
from turbo_nigo.utils.misc import (
    count_parameters,
    format_time,
    get_gpu_memory_mb,
)


class Trainer:
    """Manages the full training lifecycle for TurboNIGO models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        paths: Dict[str, str],
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.paths = paths
        self.device = config["device"]

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=1e-6,
        )

        # AMP Scaler
        self.scaler = torch.amp.GradScaler(
            device=self.device,
            enabled=config.get("use_amp", True),
        )

        # Performance tuning (TF32, cuDNN Benchmark, torch.compile)
        if config.get("tf32", True) and torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True

        if config.get("cudnn_benchmark", True) and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        if config.get("compile", True) and hasattr(torch, "compile"):
            # Using try-except because some complex operations might not be fully supported in all torch versions
            try:
                print("Optimizing model with torch.compile (mode='reduce-overhead')...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception as e:
                print(f"Warning: torch.compile failed ({e}). Falling back to eager execution.")

        # LR Scheduler
        self.scheduler = self._build_scheduler(config)

        # Time steps (precomputed)
        self.time_steps = (
            torch.arange(1, config["seq_len"] + 1).float().to(self.device)
            * config["dt"]
        )

        # Logging
        self.logger = ExperimentLogger(
            log_dir=paths["log"],
            use_tensorboard=config.get("use_tensorboard", True),
        )

        # State
        self.start_epoch = 1
        self.best_loss = float("inf")
        self.epochs_without_improvement = 0

        # Save experiment metadata
        model_info = count_parameters(model)
        model_info["architecture"] = model.__class__.__name__
        self.logger.save_metadata(config, model_info)

        # Resume from checkpoint if specified
        resume_path = config.get("resume_from")
        if resume_path and os.path.exists(resume_path):
            self.load_checkpoint(resume_path)

    # ------------------------------------------------------------------
    # LR Scheduler
    # ------------------------------------------------------------------
    def _build_scheduler(self, config: Dict) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        sched_type = config.get("scheduler", "none").lower()
        warmup = config.get("warmup_epochs", 0)

        if sched_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(config["epochs"] - warmup, 1),
                eta_min=config.get("min_lr", 1e-6),
            )
        elif sched_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=config.get("plateau_factor", 0.5),
                patience=config.get("plateau_patience", 10),
                min_lr=config.get("min_lr", 1e-6),
            )
        elif sched_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.get("step_size", 50),
                gamma=config.get("step_gamma", 0.5),
            )
        return None

    def _get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def _warmup_lr(self, epoch: int):
        """Linear warmup over the first `warmup_epochs`."""
        warmup = self.config.get("warmup_epochs", 0)
        if warmup > 0 and epoch <= warmup:
            lr_scale = epoch / warmup
            base_lr = self.config["learning_rate"]
            for pg in self.optimizer.param_groups:
                pg["lr"] = base_lr * lr_scale

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(self, filepath: str, epoch: int):
        """Save full training state for perfect resumability."""
        state = {
            "epoch": epoch,
            "best_loss": self.best_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": self.config,
            "rng_state_torch": torch.random.get_rng_state(),
            "rng_state_numpy": np.random.get_state(),
        }
        if self.scheduler is not None and not isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        if torch.cuda.is_available():
            state["rng_state_cuda"] = torch.cuda.get_rng_state()

        torch.save(state, filepath)

    def load_checkpoint(self, filepath: str):
        """Load full training state to resume exactly where we left off."""
        print(f"[Resume] Loading checkpoint: {filepath}")
        ckpt = torch.load(filepath, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        self.start_epoch = ckpt["epoch"] + 1
        self.best_loss = ckpt["best_loss"]

        # Restore scheduler
        if (
            self.scheduler is not None
            and "scheduler_state_dict" in ckpt
            and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        ):
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        # Restore RNG states
        if "rng_state_torch" in ckpt:
            torch.random.set_rng_state(ckpt["rng_state_torch"])
        if "rng_state_numpy" in ckpt:
            np.random.set_state(ckpt["rng_state_numpy"])
        if "rng_state_cuda" in ckpt and torch.cuda.is_available():
            torch.cuda.set_rng_state(ckpt["rng_state_cuda"])

        print(
            f"[Resume] Resuming from epoch {self.start_epoch} "
            f"(best_val={self.best_loss:.4e})"
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _compute_grad_norm(self) -> float:
        """Compute total gradient L2 norm across all parameters."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch. Returns dict of metrics."""
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_prior = 0.0
        total_alpha = 0.0
        total_beta = 0.0
        grad_norms = []

        pbar = tqdm(self.train_loader, desc=f"Ep {epoch:03d}", leave=False)
        for u0, u_seq_gt, cond in pbar:
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
                mse = F.mse_loss(u_pred, u_seq_gt)
                physics_prior = (
                    k_c.pow(2).mean() + r_c.pow(2).mean()
                ) * self.config["physics_prior_weight"]
                loss = mse + physics_prior

            self.scaler.scale(loss).backward()

            # Gradient clipping
            clip_val = self.config.get("grad_clip_norm", 0)
            if clip_val > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), clip_val
                )

            # Track gradient norm
            if self.config.get("log_grad_norm", True):
                grad_norms.append(self._compute_grad_norm())

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_mse += mse.item()
            total_prior += physics_prior.item()
            total_alpha += alpha.mean().item()
            total_beta += beta.mean().item()
            pbar.set_postfix(
                {"mse": f"{mse.item():.4e}", "lr": f"{self._get_lr():.2e}"}
            )

        n = len(self.train_loader)
        return {
            "train_loss": total_loss / n,
            "train_mse": total_mse / n,
            "train_physics_prior": total_prior / n,
            "alpha_mean": total_alpha / n,
            "beta_mean": total_beta / n,
            "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        }

    def validate(self) -> Dict[str, float]:
        """Run validation. Returns dict of metrics."""
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_alpha = 0.0
        total_beta = 0.0

        with torch.no_grad():
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
                    mse = F.mse_loss(u_pred, u_seq_gt)
                    physics_prior = (
                        k_c.pow(2).mean() + r_c.pow(2).mean()
                    ) * self.config["physics_prior_weight"]
                    loss = mse + physics_prior

                total_loss += loss.item()
                total_mse += mse.item()
                total_alpha += alpha.mean().item()
                total_beta += beta.mean().item()

        n = len(self.val_loader)
        return {
            "val_loss": total_loss / n,
            "val_mse": total_mse / n,
            "val_alpha": total_alpha / n,
            "val_beta": total_beta / n,
        }

    def train(self):
        """Full training loop with logging, checkpointing, and early stopping."""
        patience = self.config.get("early_stopping_patience", 0)
        total_epochs = self.config["epochs"]

        print("=" * 70)
        print(f"  Training: {self.config.get('experiment_name', 'TurboNIGO')}")
        print(f"  Epochs: {self.start_epoch} → {total_epochs}")
        print(f"  LR: {self.config['learning_rate']}, Scheduler: {self.config.get('scheduler', 'none')}")
        print(f"  Best val loss so far: {self.best_loss:.4e}")
        print("=" * 70)

        for epoch in range(self.start_epoch, total_epochs + 1):
            epoch_start = time.time()

            # Warmup LR
            self._warmup_lr(epoch)

            # Reset peak memory tracker
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # Train + Validate
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            epoch_time = time.time() - epoch_start
            gpu_mem = get_gpu_memory_mb()

            # Merge all metrics
            metrics = {
                **train_metrics,
                **val_metrics,
                "lr": self._get_lr(),
                "epoch_time_sec": round(epoch_time, 2),
                "gpu_mem_mb": round(gpu_mem, 1),
            }

            # Log
            self.logger.log_epoch(epoch, metrics)

            # Terminal output
            remaining = (total_epochs - epoch) * epoch_time
            print(
                f"Ep {epoch:03d}/{total_epochs} │ "
                f"T: {metrics['train_loss']:.4e} │ "
                f"V: {metrics['val_loss']:.4e} │ "
                f"MSE: {metrics['train_mse']:.4e} │ "
                f"LR:{metrics['lr']:.2e} │ "
                f"∇:{metrics['grad_norm']:.2f} │ "
                f"{format_time(epoch_time)} │ "
                f"ETA:{format_time(remaining)}"
            )

            # Checkpoint: best
            val_loss = metrics["val_loss"]
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.epochs_without_improvement = 0
                best_path = os.path.join(self.paths["ckpt"], "best.pth")
                self.save_checkpoint(best_path, epoch)
                print(f"  ★ New best val_loss: {val_loss:.4e} — saved to best.pth")
            else:
                self.epochs_without_improvement += 1

            # Checkpoint: periodic
            if epoch % self.config.get("checkpoint_freq", 5) == 0:
                ckpt_path = os.path.join(
                    self.paths["ckpt"], f"ep{epoch:03d}.pth"
                )
                self.save_checkpoint(ckpt_path, epoch)

            # LR Scheduler step
            warmup = self.config.get("warmup_epochs", 0)
            if self.scheduler is not None and epoch > warmup:
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Always save the absolutely latest epoch for flawless resume capability
            latest_path = os.path.join(self.paths["ckpt"], "latest.pth")
            self.save_checkpoint(latest_path, epoch)

            # Early stopping
            if patience > 0 and self.epochs_without_improvement >= patience:
                print(
                    f"\n[Early Stop] No improvement for {patience} epochs. "
                    f"Best val_loss: {self.best_loss:.4e}"
                )
                break

        # Final checkpoint
        final_path = os.path.join(self.paths["ckpt"], "final.pth")
        self.save_checkpoint(final_path, epoch)
        self.logger.close()
        print(f"\nTraining Complete. Best val_loss: {self.best_loss:.4e}")

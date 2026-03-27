"""
Research-grade experiment logging for TurboNIGO.

Supports simultaneous CSV, JSON-lines, and TensorBoard logging.
All experiment metadata (config, environment, model info) is serialized
at experiment start for full reproducibility.
"""
import csv
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


class CSVLogger:
    """Atomic CSV logger — one row per epoch."""

    def __init__(self, filepath: str, headers: List[str]):
        self.filepath = filepath
        self.headers = headers
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if not os.path.exists(filepath):
            with open(filepath, 'w', newline='') as f:
                csv.writer(f).writerow(headers)

    def log(self, row: List[Any]):
        with open(self.filepath, 'a', newline='') as f:
            csv.writer(f).writerow(row)


class JSONLinesLogger:
    """Append-only JSON-lines logger — one JSON object per epoch for programmatic analysis."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def log(self, record: Dict[str, Any]):
        with open(self.filepath, 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')


class TensorBoardLogger:
    """Optional TensorBoard wrapper. Degrades gracefully if tensorboard is not installed."""

    def __init__(self, log_dir: str):
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            pass  # TensorBoard not available

    def log_scalar(self, tag: str, value: float, step: int):
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        if self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def flush(self):
        if self.writer is not None:
            self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()


class ExperimentLogger:
    """
    Unified experiment logger orchestrating CSV, JSON-lines, and TensorBoard.

    Usage:
        logger = ExperimentLogger(log_dir="results/logs", use_tensorboard=True)
        logger.save_metadata(config, model_info)
        logger.log_epoch(epoch, metrics_dict)
        logger.close()
    """

    CSV_HEADERS = [
        "epoch", "train_loss", "train_mse", "train_physics_prior",
        "val_loss", "val_mse",
        "alpha_mean", "beta_mean",
        "lr", "grad_norm",
        "epoch_time_sec", "gpu_mem_mb",
    ]

    def __init__(self, log_dir: str, use_tensorboard: bool = True):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.csv_logger = CSVLogger(
            os.path.join(log_dir, "metrics.csv"),
            self.CSV_HEADERS
        )
        self.jsonl_logger = JSONLinesLogger(
            os.path.join(log_dir, "metrics.jsonl")
        )
        self.tb_logger = TensorBoardLogger(
            os.path.join(log_dir, "tensorboard")
        ) if use_tensorboard else None

    def save_metadata(self, config: Dict[str, Any],
                      model_info: Optional[Dict[str, Any]] = None):
        """Save full experiment metadata at training start."""
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "config": config,
        }
        if model_info is not None:
            metadata["model"] = model_info

        # Add system info (anonymized)
        from turbo_nigo.utils.misc import get_system_info
        metadata["system"] = get_system_info()

        path = os.path.join(self.log_dir, "experiment_metadata.json")
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log one epoch of metrics to all backends."""
        # CSV row (ordered by CSV_HEADERS)
        row = [metrics.get(h, 0.0) for h in self.CSV_HEADERS]
        row[0] = epoch  # ensure epoch is int
        self.csv_logger.log(row)

        # JSONL record (full dict)
        record = {"epoch": epoch, **metrics}
        self.jsonl_logger.log(record)

        # TensorBoard scalars
        if self.tb_logger is not None:
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    self.tb_logger.log_scalar(f"train/{key}", val, epoch)
            self.tb_logger.flush()

    def close(self):
        if self.tb_logger is not None:
            self.tb_logger.close()

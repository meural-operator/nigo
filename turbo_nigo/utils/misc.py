"""
Utility functions for reproducibility, system info, and formatting.
All system-info functions are anonymized — no PII is ever emitted.
"""
import os
import random
import time
from typing import Any, Dict

import numpy as np
import torch


def seed_everything(seed: int = 42):
    """Sets the seed for reproducibility across all libraries."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_paths(config: Dict) -> Dict[str, str]:
    """Generates standard directory structure based on the root results_dir."""
    root = config.get("results_dir", "./results")
    paths = {
        "ckpt": os.path.join(root, "checkpoints"),
        "plot": os.path.join(root, "plots"),
        "log": os.path.join(root, "logs"),
        "spectrum": os.path.join(root, "spectrum"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Returns total, trainable, and non-trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
        "total_million": round(total / 1e6, 2),
    }


def get_system_info() -> Dict[str, Any]:
    """
    Returns anonymized system information for experiment metadata.
    No hostnames, usernames, or paths are included.
    """
    info: Dict[str, Any] = {
        "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_total_mb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e6, 1
        )
        info["gpu_count"] = torch.cuda.device_count()
    return info


def format_time(seconds: float) -> str:
    """Formats seconds into human-readable string (e.g. '2m 35s' or '1h 12m')."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h {m}m"


def get_gpu_memory_mb() -> float:
    """Returns peak GPU memory allocated in MB. Returns 0 if CUDA not available."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0

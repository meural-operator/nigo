from .base_dataset import BaseOperatorDataset
from .flow_dataset import InMemoryFlowDataset
from .utils import compute_global_stats_and_cond_stats, read_meta
from .analyzer import AbstractDatasetAnalyzer, DatasetAnalyzer

# Optional dataset loaders — guarded to avoid ImportError from missing deps
try:
    from .h5_dataset import H5FlowDataset
except ImportError:
    pass

try:
    from .ks_dataset import KSDataset
except ImportError:
    pass

__all__ = [
    "BaseOperatorDataset",
    "InMemoryFlowDataset",
    "compute_global_stats_and_cond_stats",
    "read_meta",
    "AbstractDatasetAnalyzer",
    "DatasetAnalyzer",
]

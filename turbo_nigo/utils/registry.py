from typing import Dict, Type, Any
from turbo_nigo.data.base_dataset import BaseOperatorDataset
from turbo_nigo.data.flow_dataset import InMemoryFlowDataset


class Registry:
    """
    Registry for datasets and models.

    Provides a central lookup for dataset classes and model constructors,
    enabling config-driven experiment setup without hardcoded imports.
    """

    # ---- Datasets ----
    _datasets: Dict[str, Type] = {
        "flow": InMemoryFlowDataset,
    }

    @classmethod
    def register_dataset(cls, name: str, dataset_cls: Type):
        cls._datasets[name] = dataset_cls

    @classmethod
    def get_dataset(cls, name: str) -> Type:
        if name not in cls._datasets:
            raise KeyError(
                f"Dataset '{name}' not found. Available: {list(cls._datasets.keys())}"
            )
        return cls._datasets[name]

    @classmethod
    def list_datasets(cls) -> list:
        return list(cls._datasets.keys())


# ---- Lazy registration of optional datasets ----
# These use try/except so that missing dependencies (h5py, scipy)
# don't break imports of the core framework.

def _register_optional_datasets():
    try:
        from turbo_nigo.data.h5_dataset import H5FlowDataset
        Registry.register_dataset("h5_flow", H5FlowDataset)
        Registry.register_dataset("ns_incom", H5FlowDataset)
    except Exception:
        pass

    try:
        from turbo_nigo.data.burgers_dataset import BurgersDataset
        Registry.register_dataset("burgers", BurgersDataset)
    except Exception:
        pass

    try:
        from turbo_nigo.data.ks_dataset import KSDataset
        Registry.register_dataset("ks", KSDataset)
    except Exception:
        pass

    try:
        from turbo_nigo.data.darcy_dataset import DarcyFlowDataset
        Registry.register_dataset("darcy", DarcyFlowDataset)
    except Exception:
        pass


_register_optional_datasets()

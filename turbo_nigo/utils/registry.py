from typing import Dict, Type, Any
from turbo_nigo.data.base_dataset import BaseOperatorDataset
from turbo_nigo.data.flow_dataset import InMemoryFlowDataset

class Registry:
    """
    Registry for datasets and custom models. Allows the framework to be easily extensible.
    """
    _datasets: Dict[str, Type[BaseOperatorDataset]] = {
        "flow": InMemoryFlowDataset
    }
    
    @classmethod
    def register_dataset(cls, name: str, dataset_cls: Type[BaseOperatorDataset]):
        cls._datasets[name] = dataset_cls
        
    @classmethod
    def get_dataset(cls, name: str) -> Type[BaseOperatorDataset]:
        if name not in cls._datasets:
            raise KeyError(f"Dataset '{name}' not found. Available: {list(cls._datasets.keys())}")
        return cls._datasets[name]

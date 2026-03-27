import pytest
import torch
import numpy as np
import os
import json
import tempfile

from turbo_nigo.data.flow_dataset import InMemoryFlowDataset
from turbo_nigo.data.utils import compute_global_stats_and_cond_stats

SPATIAL = 64

@pytest.fixture
def mock_flow_data():
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(2):
            case_dir = os.path.join(temp_dir, f"case_{i}")
            os.makedirs(case_dir)
            
            # shape (T, H, W) 
            u = np.random.rand(30, SPATIAL, SPATIAL).astype(np.float32)
            v = np.random.rand(30, SPATIAL, SPATIAL).astype(np.float32)
            np.save(os.path.join(case_dir, "u.npy"), u)
            np.save(os.path.join(case_dir, "v.npy"), v)
            
            meta = {
                "Re": 100.0 * (i + 1),
                "radius": 0.5,
                "inlet_velocity": 1.0,
                "bc_type": 1.0
            }
            with open(os.path.join(case_dir, "meta.json"), "w") as f:
                json.dump(meta, f)
                
        yield temp_dir

def test_compute_stats(mock_flow_data):
    g_min, g_max, c_mean, c_std = compute_global_stats_and_cond_stats(mock_flow_data)
    assert isinstance(g_min, float)
    assert isinstance(g_max, float)
    assert g_max >= g_min
    assert c_mean.shape == (4,)
    assert c_std.shape == (4,)

def test_in_memory_flow_dataset(mock_flow_data):
    seq_len = 5
    ds_train = InMemoryFlowDataset(mock_flow_data, seq_len=seq_len, mode='train')
    ds_train._setup_dataset()
    
    assert len(ds_train) > 0, "Dataset should have samples."
    
    x, y, cond = ds_train[0]
    
    assert x.shape == (2, SPATIAL, SPATIAL)
    assert y.shape == (seq_len, 2, SPATIAL, SPATIAL)
    assert cond.shape == (4,)
    
    # Check normalization
    assert torch.max(x) <= 1.0 + 1e-4
    assert torch.min(x) >= 0.0 - 1e-4

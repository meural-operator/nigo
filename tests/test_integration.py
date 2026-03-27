import pytest
import os
import tempfile
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

from turbo_nigo.data import InMemoryFlowDataset
from turbo_nigo.data.utils import compute_global_stats_and_cond_stats
from turbo_nigo.models import GlobalTurboNIGO
from turbo_nigo.core import Trainer
from turbo_nigo.utils import get_paths

SPATIAL = 64

@pytest.fixture
def mock_flow_data():
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(2):
            case_dir = os.path.join(temp_dir, f"case_{i}")
            os.makedirs(case_dir)
            
            # Must match SPATIAL for encoding/decoding
            u = np.random.rand(10, SPATIAL, SPATIAL).astype(np.float32)
            v = np.random.rand(10, SPATIAL, SPATIAL).astype(np.float32)
            np.save(os.path.join(case_dir, "u.npy"), u)
            np.save(os.path.join(case_dir, "v.npy"), v)
            
            meta = {
                "Re": 100.0 * (i + 1), "radius": 0.5, "inlet_velocity": 1.0, "bc_type": 1.0
            }
            with open(os.path.join(case_dir, "meta.json"), "w") as f:
                json.dump(meta, f)
                
        yield temp_dir

def test_training_pipeline_integration(mock_flow_data, tmp_path):
    config = {
        "results_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 1,
        "learning_rate": 0.001,
        "seq_len": 3,
        "dt": 0.1,
        "latent_dim": 16,
        "width": 8,
        "seed": 42,
        "num_workers": 0,
        "checkpoint_freq": 1,
        "use_amp": False,
        "device": "cpu",
        "physics_prior_weight": 0.001,
        "num_bases": 2
    }
    
    paths = get_paths(config)
    g_min, g_max, c_mean, c_std = compute_global_stats_and_cond_stats(mock_flow_data)
    
    train_ds = InMemoryFlowDataset.create_with_stats(
        mock_flow_data, config["seq_len"], 'train', g_min, g_max, c_mean, c_std)
    val_ds = InMemoryFlowDataset.create_with_stats(
        mock_flow_data, config["seq_len"], 'val', g_min, g_max, c_mean, c_std)
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"])
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"])
    
    model = GlobalTurboNIGO(
        latent_dim=config["latent_dim"], num_bases=config["num_bases"], 
        cond_dim=4, width=config["width"], spatial_size=SPATIAL
    )
    
    trainer = Trainer(model, train_loader, val_loader, config, paths)
    
    # Run 1 epoch
    metrics = trainer.train_epoch(1)
    
    assert metrics["train_loss"] > 0.0
    assert metrics["alpha_mean"] > 0.0
    assert metrics["beta_mean"] > 0.0
    
    val_metrics = trainer.validate()
    assert val_metrics["val_loss"] > 0.0

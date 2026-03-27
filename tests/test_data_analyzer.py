import pytest
import os
import tempfile
import numpy as np
import torch
import json
from unittest.mock import patch, MagicMock
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure

from turbo_nigo.data import InMemoryFlowDataset
from turbo_nigo.data.utils import compute_global_stats_and_cond_stats
from turbo_nigo.data.analyzer import DatasetAnalyzer

SPATIAL = 32

@pytest.fixture
def mock_flow_data():
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(2):
            case_dir = os.path.join(temp_dir, f"case_{i}")
            os.makedirs(case_dir)
            
            u = np.random.rand(5, SPATIAL, SPATIAL).astype(np.float32)
            v = np.random.rand(5, SPATIAL, SPATIAL).astype(np.float32)
            np.save(os.path.join(case_dir, "u.npy"), u)
            np.save(os.path.join(case_dir, "v.npy"), v)
            
            meta = {
                "Re": 100.0 * (i + 1), "radius": 0.5, "inlet_velocity": 1.0, "bc_type": 1.0
            }
            with open(os.path.join(case_dir, "meta.json"), "w") as f:
                json.dump(meta, f)
                
        yield temp_dir

@pytest.fixture
def dataset(mock_flow_data):
    g_min, g_max, c_mean, c_std = compute_global_stats_and_cond_stats(mock_flow_data)
    ds = InMemoryFlowDataset.create_with_stats(
        root_dir=mock_flow_data,
        seq_len=2,
        mode='train',
        g_min=g_min,
        g_max=g_max,
        c_mean=c_mean,
        c_std=c_std
    )
    return ds

class TestDatasetAnalyzer:
    def test_analyzer_initialization(self, dataset):
        analyzer = DatasetAnalyzer(dataset)
        assert analyzer.dataset == dataset
        assert analyzer.stats == dataset.get_normalization_stats()
        assert analyzer.stats["global_min"] is not None
        
    def test_unnormalize(self, dataset):
        analyzer = DatasetAnalyzer(dataset)
        tensor = torch.zeros(1)
        unnorm = analyzer._unnormalize(tensor)
        assert isinstance(unnorm, np.ndarray)
        assert unnorm[0] == analyzer.stats["global_min"]
        
    def test_plot_sample(self, dataset):
        analyzer = DatasetAnalyzer(dataset)
        fig = analyzer.plot_sample(idx=0)
        assert isinstance(fig, Figure)
        
    def test_plot_spectrum(self, dataset):
        analyzer = DatasetAnalyzer(dataset)
        fig = analyzer.plot_spectrum(idx=0)
        assert isinstance(fig, Figure)
        
    def test_plot_temporal_evolution(self, dataset):
        analyzer = DatasetAnalyzer(dataset)
        fig = analyzer.plot_temporal_evolution(idx=0)
        assert isinstance(fig, Figure)
        
    def test_compute_dataset_statistics(self, dataset):
        analyzer = DatasetAnalyzer(dataset)
        stats = analyzer.compute_dataset_statistics(num_samples=2)
        assert "mean_spatial_energy" in stats
        assert "global_max" in stats
        assert "global_min" in stats
        assert stats["samples_analyzed"] > 0
        assert stats["mean_spatial_energy"] > 0

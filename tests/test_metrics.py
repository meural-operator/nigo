import pytest
import torch
import numpy as np

from turbo_nigo.models import GlobalTurboNIGO
from turbo_nigo.core import compute_lyapunov_divergence, compute_physics_metrics

SPATIAL = 64
LATENT = 16
BASES = 2
COND = 4
WIDTH = 8

def test_lyapunov_divergence():
    model = GlobalTurboNIGO(latent_dim=LATENT, num_bases=BASES, cond_dim=COND, width=WIDTH, spatial_size=SPATIAL)
    u0 = torch.randn(1, 2, SPATIAL, SPATIAL)
    cond = torch.randn(1, COND)
    dt = 0.1
    steps = 5
    
    div_curve, init_d = compute_lyapunov_divergence(model, u0, steps, cond, dt, perturbation=1e-4)
    
    # 1 init step + 5 rollout steps = 6
    assert len(div_curve) == 6
    assert isinstance(init_d, float)
    assert div_curve[0] > 0

def test_physics_metrics():
    np.random.seed(42)
    t = np.linspace(0, 5, 500)
    dt = t[1] - t[0]
    gt = np.sin(2 * np.pi * 1.5 * t) # 1.5 Hz
    pred = np.sin(2 * np.pi * 1.5 * t) + 0.1 * np.random.randn(500)
    
    metrics = compute_physics_metrics(gt, pred, dt)
    
    assert "freq_gt" in metrics
    assert "correlation" in metrics
    assert abs(metrics["freq_gt"] - 1.5) < 0.2
    assert metrics["correlation"] > 0.8

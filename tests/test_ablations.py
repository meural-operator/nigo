import torch
import pytest
from turbo_nigo.models.ablations import (
    Ablation1_NoSkewTurboNIGO,
    Ablation2_NoDissipativeTurboNIGO,
    Ablation3_DenseGeneratorTurboNIGO,
    Ablation4_NoRefinerTurboNIGO,
    Ablation5_ScaledTurboNIGO
)

@pytest.fixture
def mock_inputs():
    u0 = torch.randn(2, 1, 32, 32)
    cond = torch.randn(2, 4)
    t = torch.tensor([0.1, 0.2])
    return u0, t, cond

def test_no_skew_model(mock_inputs):
    u0, t, cond = mock_inputs
    model = Ablation1_NoSkewTurboNIGO(latent_dim=16, num_bases=4, in_channels=1, width=8, spatial_size=32)
    out, z_base, kc, rc = model(u0, t, cond)
    assert out.shape == (2, 2, 1, 32, 32)
    
def test_no_dissipative_model(mock_inputs):
    u0, t, cond = mock_inputs
    model = Ablation2_NoDissipativeTurboNIGO(latent_dim=16, num_bases=4, in_channels=1, width=8, spatial_size=32)
    out, z_base, kc, rc = model(u0, t, cond)
    assert out.shape == (2, 2, 1, 32, 32)

def test_dense_model(mock_inputs):
    u0, t, cond = mock_inputs
    model = Ablation3_DenseGeneratorTurboNIGO(latent_dim=16, num_bases=4, in_channels=1, width=8, spatial_size=32)
    out, z_base, kc, rc = model(u0, t, cond)
    assert out.shape == (2, 2, 1, 32, 32)

def test_no_refiner_model(mock_inputs):
    u0, t, cond = mock_inputs
    model = Ablation4_NoRefinerTurboNIGO(latent_dim=16, num_bases=4, in_channels=1, width=8, spatial_size=32)
    out, z_base, kc, rc = model(u0, t, cond)
    assert out.shape == (2, 2, 1, 32, 32)

def test_scaled_model(mock_inputs):
    u0, t, cond = mock_inputs
    model = Ablation5_ScaledTurboNIGO(latent_dim=16, num_bases=4, in_channels=1, width=8, spatial_size=32)
    # Ablation5 ALSO returns 4 values strictly, but it computes alpha and beta under the hood.
    out, z_base, kc, rc = model(u0, t, cond)
    assert out.shape == (2, 2, 1, 32, 32)

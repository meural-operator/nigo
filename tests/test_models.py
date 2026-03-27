import pytest
import torch
from turbo_nigo.models import (
    SpectralEncoder,
    SpectralDecoder,
    PhysicsInferenceNet,
    HyperTurbulentGenerator,
    TemporalRefiner,
    GlobalTurboNIGO
)

# Use spatial_size=64 and width=32 matching the default production config.
# For speed, use latent_dim=16 and num_bases=2.
SPATIAL = 64
LATENT = 16
WIDTH = 8
BASES = 2
COND = 4

def test_spectral_encoder():
    enc = SpectralEncoder(in_channels=2, latent_dim=LATENT, width=WIDTH, cond_channels=COND, spatial_size=SPATIAL)
    x = torch.randn(2, 2, SPATIAL, SPATIAL)
    cond = torch.randn(2, COND)
    
    z = enc(x, cond)
    assert z.shape == (2, LATENT)
    assert z.dtype == torch.complex64

def test_physics_net():
    net = PhysicsInferenceNet(latent_dim=LATENT, num_bases=BASES, cond_dim=COND)
    z0 = torch.randn(2, LATENT, dtype=torch.complex64)
    cond = torch.randn(2, COND)
    
    alpha, beta, k_c, r_c = net(z0, cond)
    assert alpha.shape == (2, 1, 1)
    assert beta.shape == (2, 1, 1)
    assert alpha.min() > 0
    assert beta.min() > 0
    assert k_c.shape == (2, BASES)
    assert r_c.shape == (2, BASES)

def test_hyper_generator():
    gen = HyperTurbulentGenerator(latent_dim=LATENT, num_bases=BASES)
    z0 = torch.randn(2, LATENT, dtype=torch.complex64)
    time_steps = torch.arange(1, 6).float() * 0.1 # 5 steps
    
    alpha = torch.ones(2, 1, 1)
    beta = torch.ones(2, 1, 1)
    k_c = torch.randn(2, BASES)
    r_c = torch.randn(2, BASES)
    
    z_seq = gen(z0, time_steps, alpha, beta, k_c, r_c)
    assert z_seq.shape == (2, 5, LATENT)
    assert z_seq.dtype == torch.complex64

def test_temporal_refiner():
    refiner = TemporalRefiner(latent_dim=LATENT)
    z_seq = torch.randn(2, 5, LATENT, dtype=torch.complex64)
    z_refined = refiner(z_seq)
    
    assert z_refined.shape == (2, 5, LATENT)
    assert z_refined.dtype == torch.complex64

def test_spectral_decoder():
    initial_size = SPATIAL // 8  # matches 3 stride-2 convs
    dec = SpectralDecoder(latent_dim=LATENT, out_channels=2, width=WIDTH, initial_size=initial_size)
    z_seq = torch.randn(2, 5, LATENT, dtype=torch.complex64)
    u_pred = dec(z_seq)
    
    assert u_pred.shape == (2, 5, 2, SPATIAL, SPATIAL)
    assert u_pred.dtype == torch.float32

def test_global_turbo_nigo():
    model = GlobalTurboNIGO(latent_dim=LATENT, num_bases=BASES, cond_dim=COND, width=WIDTH, spatial_size=SPATIAL)
    u0 = torch.randn(2, 2, SPATIAL, SPATIAL)
    cond = torch.randn(2, COND)
    time_steps = torch.arange(1, 6).float() * 0.1
    
    u_pred, z_base, alpha, beta, k_c, r_c = model(u0, time_steps, cond)
    
    assert u_pred.shape == (2, 5, 2, SPATIAL, SPATIAL)
    assert z_base.shape == (2, 5, LATENT)
    assert alpha.shape == (2, 1, 1)

def test_global_turbo_nigo_backward():
    """Verify gradients flow through the entire model."""
    model = GlobalTurboNIGO(latent_dim=LATENT, num_bases=BASES, cond_dim=COND, width=WIDTH, spatial_size=SPATIAL)
    u0 = torch.randn(1, 2, SPATIAL, SPATIAL, requires_grad=False)
    cond = torch.randn(1, COND)
    time_steps = torch.arange(1, 4).float() * 0.1
    
    u_pred, _, _, _, _, _ = model(u0, time_steps, cond)
    loss = u_pred.sum()
    loss.backward()
    
    # Check that generator bases have gradients
    assert model.generator.K_bases.grad is not None
    assert model.generator.R_bases.grad is not None

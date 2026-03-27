"""
Tests for extension modules: dynamic encoder/decoder, attention physics nets, V2 wrapper.
"""
import pytest
import torch

from turbo_nigo.models.extensions import (
    DynamicSpectralEncoder,
    DynamicSpectralDecoder,
    DistributionAwareAttentionPhysics,
    SpatialPhysicsAttention,
    GlobalTurboNIGO_V2,
)


class TestDynamicEncoder:
    """Tests for resolution-agnostic encoder."""

    @pytest.mark.parametrize("res", [32, 64, 128, 256])
    def test_output_shape(self, res):
        enc = DynamicSpectralEncoder(2, latent_dim=32, width=16, cond_channels=4)
        x = torch.randn(2, 2, res, res)
        cond = torch.randn(2, 4)
        z0 = enc(x, cond)
        assert z0.shape == (2, 32)
        assert z0.is_complex()

    def test_return_feature_map(self):
        enc = DynamicSpectralEncoder(
            2, latent_dim=32, width=16, cond_channels=4, return_feature_map=True
        )
        x = torch.randn(2, 2, 64, 64)
        cond = torch.randn(2, 4)
        z0, feat_map = enc(x, cond)
        assert z0.shape == (2, 32)
        assert feat_map.shape == (2, 64, 8, 8)  # width*4=64, pooled to 8x8


class TestDynamicDecoder:
    """Tests for resolution-agnostic decoder."""

    @pytest.mark.parametrize("target_res", [32, 64, 128, 256])
    def test_output_shape(self, target_res):
        dec = DynamicSpectralDecoder(32, out_channels=2, width=16, target_res=target_res)
        z = torch.complex(torch.randn(2, 5, 32), torch.randn(2, 5, 32))
        out = dec(z)
        assert out.shape == (2, 5, 2, target_res, target_res)


class TestAttentionPhysicsNets:
    """Tests for attention-based physics inference."""

    def test_distribution_aware_shapes(self):
        net = DistributionAwareAttentionPhysics(latent_dim=32, num_bases=4, embed_dim=64)
        z0 = torch.complex(torch.randn(3, 32), torch.randn(3, 32))
        u0 = torch.randn(3, 2, 64, 64)
        cond = torch.randn(3, 4)
        alpha, beta, k_c, r_c = net(z0, u0, cond)
        assert alpha.shape == (3, 1, 1)
        assert beta.shape == (3, 1, 1)
        assert k_c.shape == (3, 4)
        assert r_c.shape == (3, 4)

    def test_distribution_aware_positivity(self):
        net = DistributionAwareAttentionPhysics(latent_dim=32, num_bases=4)
        for _ in range(10):
            z0 = torch.complex(torch.randn(4, 32), torch.randn(4, 32))
            u0 = torch.randn(4, 2, 32, 32)
            cond = torch.randn(4, 4)
            alpha, beta, _, _ = net(z0, u0, cond)
            assert torch.all(alpha > 0), f"α not positive: {alpha.min()}"
            assert torch.all(beta > 0), f"β not positive: {beta.min()}"

    def test_spatial_attention_shapes(self):
        net = SpatialPhysicsAttention(
            latent_dim=32, spatial_channels=64, num_bases=4, embed_dim=64
        )
        feat_map = torch.randn(3, 64, 8, 8)
        u0 = torch.randn(3, 2, 64, 64)
        alpha, beta, k_c, r_c = net(feat_map, u0)
        assert alpha.shape == (3, 1, 1)
        assert k_c.shape == (3, 4)

    def test_spatial_attention_positivity(self):
        net = SpatialPhysicsAttention(latent_dim=32, spatial_channels=64, num_bases=4)
        for _ in range(10):
            feat_map = torch.randn(4, 64, 8, 8)
            u0 = torch.randn(4, 2, 32, 32)
            alpha, beta, _, _ = net(feat_map, u0)
            assert torch.all(alpha > 0)
            assert torch.all(beta > 0)


class TestGlobalTurboNIGO_V2:
    """End-to-end tests for the V2 wrapper."""

    @pytest.mark.parametrize("physics_type", ["distribution", "spatial"])
    def test_forward_pass(self, physics_type):
        model = GlobalTurboNIGO_V2(
            latent_dim=16, num_bases=4, cond_dim=4, width=8,
            in_channels=2, target_res=32, physics_net_type=physics_type,
        )
        x = torch.randn(2, 2, 32, 32)
        cond = torch.randn(2, 4)
        time_steps = torch.arange(1, 6).float() * 0.1

        u_pred, z_base, k_c, r_c, alpha, beta = model(x, time_steps, cond)
        assert u_pred.shape == (2, 5, 2, 32, 32)
        assert z_base.shape == (2, 5, 16)
        assert torch.all(alpha > 0)
        assert torch.all(beta > 0)

    def test_backward_pass(self):
        model = GlobalTurboNIGO_V2(
            latent_dim=16, num_bases=4, cond_dim=4, width=8,
            in_channels=2, target_res=32, physics_net_type="distribution",
        )
        x = torch.randn(2, 2, 32, 32)
        cond = torch.randn(2, 4)
        target = torch.randn(2, 5, 2, 32, 32)
        time_steps = torch.arange(1, 6).float() * 0.1

        u_pred, *_ = model(x, time_steps, cond)
        loss = torch.nn.functional.mse_loss(u_pred, target)
        loss.backward()

        grads_ok = all(
            p.grad is not None for p in model.parameters() if p.requires_grad
        )
        assert grads_ok, "Some parameters did not receive gradients."

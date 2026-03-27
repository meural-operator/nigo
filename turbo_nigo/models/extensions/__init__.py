"""Extension modules for TurboNIGO — enhanced components from post-submission experiments."""

from turbo_nigo.models.extensions.dynamic_encoder import DynamicSpectralEncoder
from turbo_nigo.models.extensions.dynamic_decoder import DynamicSpectralDecoder
from turbo_nigo.models.extensions.attention_physics import (
    DistributionAwareAttentionPhysics,
    SpatialPhysicsAttention,
)
from turbo_nigo.models.extensions.turbo_nigo_v2 import GlobalTurboNIGO_V2

__all__ = [
    "DynamicSpectralEncoder",
    "DynamicSpectralDecoder",
    "DistributionAwareAttentionPhysics",
    "SpatialPhysicsAttention",
    "GlobalTurboNIGO_V2",
]

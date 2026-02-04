import torch
from omegaconf import DictConfig

from tensor_beasts.features.feature import Feature
from tensor_beasts.registry import register_feature


@register_feature
class IdFeature(Feature):
    name = "id"
    dtype = torch.int32


@register_feature
class OffspringCount(Feature):
    name = "offspring_count"
    dtype = torch.int32


@register_feature
class Biomass(Feature):
    """Biomass storage - what animals gain from eating."""
    name = "biomass"
    dtype = torch.uint8
    default_tags = {"observable"}
    default_config = DictConfig({})
    depends_on = {}


@register_feature
class GradientEMA(Feature):
    """EMA of scent gradient strength - used to modulate metabolism."""
    name = "gradient_ema"
    dtype = torch.float32
    default_tags = {"observable"}
    default_config = DictConfig({})
    depends_on = {}

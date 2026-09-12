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


@register_feature
class SlotId(Feature):
    """
    Genetic slot identifier for each individual.

    Used by the genetic algorithm system to track which genome/lineage
    each individual belongs to. Slot 0 is the default slot.
    """
    name = "slot_id"
    dtype = torch.uint8
    default_tags = set()
    default_config = DictConfig({
        "default_slot": 0,  # Default slot for all individuals
    })
    depends_on = {}

    def initialize_data(self):
        """Initialize all individuals to default slot."""
        super().initialize_data()
        default_slot = self.config.get("default_slot", 0)
        self.data.fill_(default_slot)

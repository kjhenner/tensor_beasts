from typing import Optional

import torch

from tensor_beasts.entities.entity import Entity
from tensor_beasts.registry import register_entity

from tensor_beasts.features.terrain_features import (
    Elevation, AquiferElevation, SoilVolume, SurfaceWaterVolume, SoilWaterVolume,
    SimpleWater, Nutrients
)
from tensor_beasts.features.shared_features import Oscillator, Carrion, CarrionScent


@register_entity
class Terrain(Entity):
    """
    Terrain entity with complex hydrology (soil, surface water, aquifer).

    Use with HydrodynamicPlant for full water cycle simulation.
    """

    elevation: Elevation
    aquifer_elevation: AquiferElevation
    soil_volume: SoilVolume
    soil_water_volume: SoilWaterVolume
    surface_water_volume: SurfaceWaterVolume
    nutrients: Nutrients

    def initialize(self):
        # Use automatic dependency-ordered initialization from base class
        super().initialize()

    def update(self, action: Optional[torch.Tensor] = None):
        self.surface_water_volume.update(self.world.step)
        self.soil_volume.update(self.world.step)
        self.soil_water_volume.update(self.world.step)
        self.nutrients.update(self.world.step)


@register_entity
class SimpleTerrain(Entity):
    """
    Simplified terrain entity with basic water feature.

    Uses Perlin noise for water with optional oscillator modulation.
    Use with SimplePlant for basic water-dependent growth simulation.
    Includes carrion layer for dead animal biomass with scent.
    """

    oscillator: Oscillator
    simple_water: SimpleWater
    carrion: Carrion
    carrion_scent: CarrionScent

    def initialize(self):
        super().initialize()

    def update(self, action: Optional[torch.Tensor] = None):
        self.oscillator.update(self.world.step)
        self.simple_water.update(self.world.step)
        self.carrion.update(self.world.step)
        self.carrion_scent.emit(self.world.step)  # Emit only; diffusion handled by World

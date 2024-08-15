from typing import Optional

import torch

from tensor_beasts.entities.entity import Entity

from tensor_beasts.features.terrain_features import (
    Elevation, AquiferElevation, SoilVolume, SurfaceWaterVolume, SoilWaterVolume
)


class Terrain(Entity):

    elevation: Elevation
    aquifer_elevation: AquiferElevation
    soil_volume: SoilVolume
    soil_water_volume: SoilWaterVolume
    surface_water_volume: SurfaceWaterVolume

    def initialize(self):
        # TODO: can do this automatically later, just need to ensure it's in the right order!
        self.elevation.initialize_data()
        self.aquifer_elevation.initialize_data()
        self.soil_volume.initialize_data()
        self.surface_water_volume.initialize_data()
        self.soil_water_volume.initialize_data()

    def update(self, action: Optional[torch.Tensor] = None):
        self.surface_water_volume.update(0)
        self.soil_volume.update(0)
        self.soil_water_volume.update(0)

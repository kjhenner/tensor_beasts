from typing import Optional

import torch
from omegaconf import DictConfig

from tensor_beasts.entities.entity import Entity

from tensor_beasts.features.terrain_features import (
    Elevation, AquiferElevation, SoilDepth, SurfaceWaterDepth, D8Slopes,
    D8SurfaceOutflowFlow, D8SubsurfaceFlow, D8SoilSaturationGradient, SoilWaterVolume, NeighborDistances
)


class Terrain(Entity):

    elevation: Elevation
    aquifer_elevation: AquiferElevation
    soil_depth: SoilDepth
    soil_water_volume: SoilWaterVolume
    soil_saturation_gradient: D8SoilSaturationGradient
    surface_water_depth: SurfaceWaterDepth
    slopes: D8Slopes
    surface_flow: D8SurfaceOutflowFlow
    subsurface_flow: D8SubsurfaceFlow
    neighbor_distances: NeighborDistances

    def initialize(self):
        # TODO: can do this automatically later, just need to ensure it's in the right order!
        self.elevation.initialize_data()
        self.aquifer_elevation.initialize_data()
        self.soil_depth.initialize_data()
        self.soil_water_volume.initialize_data()
        self.soil_saturation_gradient.initialize_data()
        self.surface_water_depth.initialize_data()
        self.slopes.initialize_data()
        self.surface_flow.initialize_data()
        self.subsurface_flow.initialize_data()
        self.neighbor_distances.initialize_data()

    def update(self, action: Optional[torch.Tensor] = None):
        self.slopes.update(0)
        self.soil_saturation_gradient.update(0)
        self.surface_flow.update(0)
        self.subsurface_flow.update(0)

from typing import Optional

import torch

from tensor_beasts.entities import Entity
from tensor_beasts.features.terrain_features import Elevation
from tensor_beasts.features.toy_features import FluidDensity


class DiffusionToy(Entity):

    elevation: Elevation
    fluid_density: FluidDensity

    def initialize(self):
        self.elevation.initialize_data()
        self.fluid_density.initialize_data()

    def update(self, action: Optional[torch.Tensor] = None):
        self.fluid_density.update(0)

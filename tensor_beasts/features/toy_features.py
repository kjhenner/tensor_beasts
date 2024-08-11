import random

import torch
from omegaconf import DictConfig

from tensor_beasts.features.feature import Feature
from tensor_beasts.util import flow_gradient, flow


class FluidDensity(Feature):
    name = "fluid_density"
    dtype = torch.float32
    default_config = DictConfig({
        "flow_rate": 0.05,
        "elevation_key": "${key:diffusiontoy,elevation}",
        "elevation_scale": 100,
        "rainfall_rate": 0.1,
    })

    def initialize_data(self):
        self.zero_init()
        self.data[self.shape[0]//2, self.shape[1]//2] = 1.0

    def render(self) -> torch.Tensor:
        return self.data.unsqueeze(-1).expand(-1, -1, 3) * 255

    def update(self, step: int):
        elevation = self.td.get(self.config.elevation_key) * self.config.elevation_scale
        gradient = flow_gradient(self.data + elevation)
        self.data, _, _ = flow(self.data, gradient, self.config.flow_rate)
        self.data += self.config.rainfall_rate

from typing import Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from tensor_beasts.entities.entity import Entity
from tensor_beasts.util import (
    perlin_noise, generate_direction_kernels, safe_add, safe_sub, lru_distance,
    pyramid_elevation
)
from tensor_beasts.world import FeatureDefinition


class Terrain(Entity):
    features = [
        FeatureDefinition("elevation", torch.float32, observable=True),
        FeatureDefinition("water_level", torch.float32, observable=True),
        FeatureDefinition("flow", torch.float32, shape=(8,)),
        FeatureDefinition("substrate", torch.uint8),
        FeatureDefinition("rainfall", torch.uint8),
    ]

    def __init__(self, world: 'World', config: DictConfig):
        super().__init__(world, config)
        # Additional initialization parameters can be added here

    def initialize(self):
        # Initialize elevation using Perlin noise
        if self.config.elevation.mode == "perlin":
            elevation = perlin_noise(self.world.size, self.config.elevation.scale)
        elif self.config.elevation.mode == "pyramid":
            elevation = pyramid_elevation(self.world.size, inverted=True)
        elif self.config.elevation.mode == "ramp":
            elevation = torch.linspace(
                0,
                1, self.world.size[0]
            ).unsqueeze(1).expand(*self.world.size)
        else:
            raise ValueError(f"Invalid elevation mode: {self.config.elevation.mode}")

        self.set_feature("elevation", elevation)

        # Initialize other features
        self.set_feature("water_level", torch.zeros_like(elevation))
        self.set_feature("substrate", torch.randint(0, 5, self.world.size, dtype=torch.uint8))
        self.set_feature("rainfall", torch.zeros_like(elevation, dtype=torch.uint8))
        self.set_feature("flow", torch.zeros((*self.world.size, 8), dtype=torch.float32))

    def d8_flow_direction(
        self,
        epsilon: float = 1e-8
    ):
        """
        Calculate simplified D8 flow amounts into adjacent lower-elevation cells,
        scaled by slopes.

        Args:
            flow_rate_factor (float, optional): Factor to control the overall flow rate. Defaults to 0.1.
            epsilon (float, optional): Small value to prevent division by zero. Defaults to 1e-7.

        Returns:
            torch.Tensor: Flow amounts in 8 directions, shape (H, W, 8).
        """
        elevation = self.get_feature("elevation") * self.config.elevation_max
        water_level = self.get_feature("water_level")
        total_height = elevation + water_level

        kernels = generate_direction_kernels()

        slopes = torch.zeros((*total_height.shape, 8), dtype=torch.float32)
        for i, (dy, dx) in enumerate(kernels):
            neighbor_height = torch.roll(total_height, shifts=(int(dy), int(dx)), dims=(0, 1))
            distance = lru_distance(dx, dy) * 10  # Assuming 10m between cell centers
            slopes[..., i] = (neighbor_height - total_height) / distance

        slopes += (torch.rand_like(slopes) - 0.5) * epsilon
        slopes = slopes.clamp(min=0)

        # Calculate flow rates based on slopes
        flow = self.config.flow_rate * torch.sqrt(slopes)

        # Ensure total outflow doesn't exceed available water
        total_outflow = flow.sum(dim=-1, keepdim=True)
        scaling_factor = torch.where(
            total_outflow > water_level.unsqueeze(-1),
            water_level.unsqueeze(-1) / (total_outflow + epsilon),
            torch.ones_like(total_outflow)
        )

        flow *= scaling_factor

        return self.set_feature("flow", flow)

    def update(self, action: Optional[torch.Tensor] = None):
        self.update_rainfall()
        self.d8_flow_direction()
        self.update_water_level()
        # Add more update steps as needed

    def update_water_level(self):
        water_level = self.get_feature("water_level")
        flow = self.get_feature("flow")

        # Calculate total outflow
        water_out = torch.sum(flow, dim=-1)

        water_in = torch.zeros_like(water_out)

        # Update neighboring cells
        shifts = generate_direction_kernels()
        for i, (dy, dx) in enumerate(shifts):
            water_in += torch.roll(flow[..., i], shifts=(int(dy), int(dx)), dims=(0, 1))
        self.set_feature("water_level", (water_level + water_in - water_out).clamp(0, 20))


    def update_rainfall(self) -> None:
        """
        Update water level based on rainfall.

        Returns:
            torch.Tensor: Updated water level (uint8 tensor).

        Note:
            - Water level uses a quadratic scale where 0-255 (uint8) maps to 0-10 meters.
            - The function ensures that rainfall doesn't exceed the maximum possible water level.
        """
        # Ensure inputs are valid
        water_level = self.get_feature("water_level")
        print(f"max water level: {water_level.max()}")

        self.set_feature("water_level", water_level + self.config.rainfall_rate)

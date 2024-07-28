from typing import Optional

import torch
from omegaconf import DictConfig

from tensor_beasts.entities.entity import Entity
from tensor_beasts.entities.feature import Feature
from tensor_beasts.util import (
    perlin_noise, generate_direction_kernels, lru_distance,
    pyramid_elevation, range_elevation
)


class Elevation(Feature):
    name = "elevation"
    dtype = torch.float32
    observable = True

    @staticmethod
    def render(tensor_data: torch.Tensor):
        return tensor_data.unsqueeze(-1).expand(-1, -1, 3) * 255


class AquiferElevation(Feature):
    name = "aquifer_elevation"
    dtype = torch.float32

    @staticmethod
    def render(tensor_data: torch.Tensor):
        return tensor_data.unsqueeze(-1).expand(-1, -1, 3) * 255


class SoilDepth(Feature):
    name = "soil"
    dtype = torch.float32
    observable = True

    @staticmethod
    def render(tensor_data: torch.Tensor):
        return tensor_data.unsqueeze(-1).expand(-1, -1, 3) * 255


class SoilSaturation(Feature):
    name = "soil_saturation"
    dtype = torch.float32
    observable = True

    @staticmethod
    def render(tensor_data: torch.Tensor):
        return tensor_data.unsqueeze(-1).expand(-1, -1, 3) * 255


class SurfaceWaterDepth(Feature):
    name = "surface_water_depth"
    dtype = torch.float32
    observable = True

    @staticmethod
    def render(tensor_data: torch.Tensor):
        return tensor_data.unsqueeze(-1).expand(-1, -1, 3)


class D8Slopes(Feature):
    name = "slopes"
    dtype = torch.float32
    shape = (8,)


class D8SurfaceFlow(Feature):
    name = "surface_flow"
    dtype = torch.float32
    shape = (8,)


class D8SubsurfaceFlow(Feature):
    name = "surface_flow"
    dtype = torch.float32
    shape = (8,)


class SurfaceInflow(Feature):
    name = "surface_inflow"
    dtype = torch.float32


class SurfaceOutflow(Feature):
    name = "surface_outflow"
    dtype = torch.float32


class SubsurfaceInflow(Feature):
    name = "subsurface_inflow"
    dtype = torch.float32


class SubsurfaceOutflow(Feature):
    name = "subsurface_outflow"
    dtype = torch.float32


class Terrain(Entity):
    features = {
        Elevation.name: Elevation(),
        AquiferElevation.name: AquiferElevation(),
        SoilDepth.name: SoilDepth(),
        SoilSaturation.name: SoilSaturation(),
        SurfaceWaterDepth.name: SurfaceWaterDepth(),
        D8Slopes.name: D8Slopes(),
        D8SurfaceFlow.name: D8SurfaceFlow(),
        D8SubsurfaceFlow.name: D8SubsurfaceFlow(),
        SurfaceInflow.name: SurfaceInflow(),
        SurfaceOutflow.name: SurfaceOutflow(),
        SubsurfaceInflow.name: SubsurfaceInflow(),
        SubsurfaceOutflow.name: SubsurfaceOutflow(),
    }

    def __init__(self, world: 'World', config: DictConfig):
        super().__init__(world, config)
        # Additional initialization parameters can be added here

    def initialize(self):
        elevation = torch.ones(self.world.size, dtype=torch.float32)
        for key, config in self.config.elevation.items():
            # Initialize elevation using Perlin noise
            if key == "perlin":
                elevation *= perlin_noise(self.world.size, self.config.elevation.perlin.scale)
            elif key == "pyramid":
                elevation *= pyramid_elevation(self.world.size, inverted=True)
            elif key == "ramp":
                elevation *= torch.linspace(
                    0,
                    1, self.world.size[0]
                ).unsqueeze(1).expand(*self.world.size)
            elif key == "range":
                elevation *= range_elevation(self.world.size)
            else:
                raise ValueError(f"Invalid elevation mode key: {key}")

        self.set_feature("elevation", elevation)

        # Initialize other features
        self.set_feature("water_level", torch.zeros_like(elevation))
        self.set_feature("inflow", torch.zeros_like(elevation))
        self.set_feature("outflow", torch.zeros_like(elevation))
        self.set_feature("slopes", torch.zeros((*self.world.size, 8), dtype=torch.float32))
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
        water_level = self.get_feature("water_level") / 255 * self.config.water_level_max
        total_height = elevation + water_level

        kernels = generate_direction_kernels()

        slopes = torch.zeros((*total_height.shape, 8), dtype=torch.float32)
        for i, (dy, dx) in enumerate(kernels):
            neighbor_height = torch.roll(total_height, shifts=(int(-dy), int(-dx)), dims=(0, 1))
            distance = lru_distance(dx, dy) * 10  # Assuming 10m between cell centers
            slopes[..., i] = (total_height - neighbor_height) / distance

        slopes -= torch.rand_like(slopes) * epsilon
        slopes = slopes.clamp(min=0)

        self.set_feature("slopes", slopes)

        # Calculate flow rates based on slopes
        flow = self.config.flow_rate * torch.sqrt(slopes)
        self.set_feature("flow", flow)

        # Ensure total outflow doesn't exceed available water
        total_outflow = flow.sum(dim=-1, keepdim=True)
        scaling_factor = torch.where(
            total_outflow > water_level.unsqueeze(-1),
            water_level.unsqueeze(-1) / (total_outflow + epsilon),
            torch.ones_like(total_outflow)
        )

        flow *= scaling_factor

        self.set_feature("flow", flow)

    def update(self, action: Optional[torch.Tensor] = None):
        self.update_rainfall()
        self.d8_flow_direction()
        self.update_water_level()

    def update_water_level(self):
        water_level = self.get_feature("water_level")
        flow = self.get_feature("flow")

        # Calculate total outflow
        water_out = torch.sum(flow, dim=-1)
        self.set_feature("outflow", water_out)

        water_in = torch.zeros_like(water_out)

        # Update neighboring cells
        shifts = generate_direction_kernels()
        for i, (dy, dx) in enumerate(shifts):
            water_in += torch.roll(flow[..., i], shifts=(int(dy), int(dx)), dims=(0, 1))
        self.set_feature("inflow", water_in)

        water_delta = (water_in - water_out) / self.config.water_level_max * 255

        self.set_feature(
            "water_level",
            (water_level + water_delta).clamp(0, self.config.water_level_max)
        )

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

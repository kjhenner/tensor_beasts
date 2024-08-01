import torch
from omegaconf import DictConfig

from tensor_beasts.features.feature import Feature
from tensor_beasts.util import (
    perlin_noise, pyramid_elevation, range_elevation, generate_direction_kernels,
    lru_distance, neighbors
)


class Elevation(Feature):
    name = "elevation"
    dtype = torch.float32
    default_tags = {"observable"}
    default_config = DictConfig({
        "perlin": {
            "scale": 0.1
        }
    })

    def render(self) -> torch.Tensor:
        return self.data.unsqueeze(-1).expand(-1, -1, 3) * 255

    def initialize_data(self):
        self.data = torch.ones(self.shape, dtype=self.dtype)
        for key, config in self.config.items():
            # Initialize elevation using Perlin noise
            if key == "perlin":
                self.data *= perlin_noise(self.shape, self.config.perlin.scale)
            elif key == "pyramid":
                self.data *= pyramid_elevation(self.shape, inverted=True)
            elif key == "ramp":
                self.data *= torch.linspace(
                    0,
                    1, self.shape[0]
                ).unsqueeze(1).expand(*self.shape)
            elif key == "range":
                self.data *= range_elevation(self.shape)
            else:
                raise ValueError(f"Invalid elevation mode key: {key}")


class AquiferElevation(Feature):
    name = "aquifer_elevation"
    dtype = torch.float32
    default_config = DictConfig({
        "elevation_key": "${key:terrain,elevation}",
        "scale": 0.9,
    })

    def render(self) -> torch.Tensor:
        return self.data.unsqueeze(-1).expand(-1, -1, 3) * 255

    def initialize_data(self):
        elevation = self.td.get(self.config.elevation_key)
        self.data = (elevation * self.config.scale).type(self.dtype)


class SoilDepth(Feature):
    name = "soil_depth"
    dtype = torch.float32
    default_tags = {"observable"}
    default_config = DictConfig({
        "elevation_key": "${key:terrain,elevation}",
    })

    def render(self) -> torch.Tensor:
        return self.data.unsqueeze(-1).expand(-1, -1, 3) * 255

    def initialize_data(self):
        elevation = self.td.get(self.config.elevation_key)
        self.data = (1 - elevation * self.config.get('scale', 1.0)).type(self.dtype)


class SoilWaterVolume(Feature):
    name = "soil_water_volume"
    dtype = torch.float32
    default_tags = {"observable"}

    def render(self) -> torch.Tensor:
        return self.data.unsqueeze(-1).expand(-1, -1, 3) * 255


class SurfaceWaterDepth(Feature):
    name = "surface_water_depth"
    dtype = torch.float32
    default_tags = {"observable"}
    default_config = DictConfig({
        "surface_outflow_key": "${key:terrain,surface_outflow}",
    })

    def update(self, step: int):
        pass


class NeighborDistances(Feature):
    name = "neighbor_distances"
    dtype = torch.float32
    shape = (8,)
    default_config = DictConfig({
        "scale_xy": 10
    })

    def initialize_data(self):
        self.data = torch.ones(self.shape, dtype=self.dtype)
        self.data *= torch.tensor([
            lru_distance(dx, dy, scale=self.config.scale_xy)
            for dy, dx in generate_direction_kernels(eight_directions=True)
        ]).unsqueeze(0).unsqueeze(0)


class D8Slopes(Feature):
    name = "slopes"
    dtype = torch.float32
    shape = (8,)
    default_config = DictConfig({
        "elevation_key": "${key:terrain,elevation}",
        "elevation_scale": 500,
        "soil_depth_key": "${key:terrain,soil_depth}",
        "soil_depth_scale": 1,
        "surface_water_depth_key": "${key:terrain,surface_water_depth}",
        "surface_water_depth_scale": 1,
        "neighbor_distances_key": "${key:terrain,neighbor_distances}",
        "epsilon": 1e-8,
        "padding_mode": "constant",
        "padding_value": 1.0
    })

    def update(self, step: int):
        elevation = self.td.get(self.config.elevation_key) * self.config.elevation_scale
        soil_depth = self.td.get(self.config.soil_depth_key) * self.config.soil_depth_scale
        neighbor_distances = self.td.get(self.config.neighbor_distances_key)
        surface_water_depth = self.td.get(self.config.surface_water_depth_key) * self.config.surface_water_depth_scale
        total_height = elevation + soil_depth + surface_water_depth

        neighbor_heights = neighbors(
            total_height,
            padding_value=self.config.padding_value,
            padding_mode=self.config.padding_mode,
            reverse=True,
            eight_direction=True
        )

        print(f"total_height: {total_height.shape}")
        print(f"neighbor_heights: {neighbor_heights.shape}")
        print(f"neighbor_distances: {neighbor_distances.shape}")

        self.data = (total_height.unsqueeze(-1).expand(-1, -1, 8) - neighbor_heights) / neighbor_distances
        self.data -= torch.rand_like(self.data) * self.config.epsilon
        self.data = self.data.clamp(min=0)


class D8SoilSaturationGradient(Feature):
    name = "soil_saturation_gradient"
    dtype = torch.float32
    shape = (8,)
    default_config = DictConfig({
        "soil_depth_key": "${key:terrain,soil_depth}",
        "soil_water_volume_key": "${key:terrain,soil_water_volume}",
        "neighbor_distances_key": "${key:terrain,neighbor_distances}",
        "epsilon": 1e-8,
        "soil_porosity": 0.4
    })

    def update(self, step: int):
        soil_depth = self.td.get(self.config.soil_depth_key)
        soil_water_volume = self.td.get(self.config.soil_water_volume_key)
        soil_saturation = soil_water_volume / (soil_depth * self.config.soil_porosity + self.config.epsilon)
        neighbor_distances = self.td.get(self.config.neighbor_distances_key)

        neighbor_saturation = neighbors(
            soil_saturation,
            padding_value=0,
            padding_mode="constant",
            eight_direction=True
        )
        self.data = (soil_saturation.unsqueeze(-1).expand(-1, -1, 8) - neighbor_saturation) / neighbor_distances

        self.data -= torch.rand_like(self.data) * self.config.epsilon
        self.data = self.data.clamp(min=0)


class D8SurfaceOutflowFlow(Feature):
    name = "surface_outflow"
    dtype = torch.float32
    shape = (8,)

    default_config = DictConfig({
        "slopes_key": "${key:terrain,slopes}",
        "surface_water_depth_key": "${key:terrain,surface_water_depth}",
        "flow_rate": 0.005
    })

    def update(self, step: int):
        slopes = self.td.get(self.config.slopes_key)
        surface_water_depth = self.td.get(self.config.surface_water_depth_key)
        self.data = self.config.flow_rate * torch.sqrt(slopes)

        # Ensure total outflow doesn't exceed available water
        total_outflow = self.data.sum(dim=-1, keepdim=True)

        scaling_factor = torch.where(
            total_outflow > surface_water_depth.unsqueeze(-1),
            surface_water_depth.unsqueeze(-1) / total_outflow,
            torch.ones_like(total_outflow)
        )

        self.data *= scaling_factor


class D8SubsurfaceFlow(Feature):
    name = "surface_outflow"
    dtype = torch.float32
    shape = (8,)

    default_config = DictConfig({
        "slopes_key": "${key:terrain,slopes}",
        "soil_depth_key": "${key:terrain,soil_depth}",
        "soil_saturation_key": "${key:terrain,soil_saturation}",
        "soil_saturation_gradient_key": "${key:terrain,soil_saturation_gradient}",
        "soil_water_volume_key": "${key:terrain,soil_water_volume}",
        "flow_rate": 0.0001,
        "soil_porosity": 0.4
    })

    def update(self, step: int):
        slopes = self.td.get(self.config.slopes_key)
        saturation_gradient = self.td.get(self.config.soil_saturation_gradient_key)
        soil_depth = self.td.get(self.config.soil_depth_key)
        soil_water_volume = self.td.get(self.config.soil_water_volume_key)
        soil_water_max = soil_depth * self.config.soil_porosity

        neighbor_water_capacity = neighbors(
            soil_water_max - soil_water_volume,
            padding_value=0,
            padding_mode="constant",
            eight_direction=True
        )

        self.data = torch.min(
            self.config.flow_rate * torch.sqrt(slopes) * saturation_gradient,
            neighbor_water_capacity
        )

        # Ensure total outflow doesn't exceed available water
        total_outflow = self.data.sum(dim=-1, keepdim=True)

        scaling_factor = torch.where(
            total_outflow > soil_water_volume.unsqueeze(-1),
            soil_water_volume.unsqueeze(-1) / total_outflow,
            torch.ones_like(total_outflow)
        )

        self.data *= scaling_factor

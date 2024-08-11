import torch
from omegaconf import DictConfig

from tensor_beasts.features.feature import Feature
from tensor_beasts.util import (
    perlin_noise, pyramid_elevation, range_elevation,
    unfold_neighbors, flow_gradient, flow
)


class Elevation(Feature):
    name = "elevation"
    dtype = torch.float32
    default_tags = {"observable"}
    default_config = DictConfig({})

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
        unfold_neighbors(self.td, self.key, (3, 3))


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


class SoilVolume(Feature):
    name = "soil_volume"
    dtype = torch.float32
    default_tags = {"observable"}
    default_config = DictConfig({
        "elevation_key": "${key:terrain,elevation}",
        "surface_outflow_key": "${key:terrain,surface_outflow}",
        "erosion_rate": 0.9,
        "scale": 2.0,
        "epsilon": 1e-8
    })

    def render(self) -> torch.Tensor:
        return self.data.unsqueeze(-1).expand(-1, -1, 3) / self.config.scale * 255

    def initialize_data(self):
        elevation = self.td.get(self.config.elevation_key)
        self.data = ((1 - elevation) * self.config.scale).type(self.dtype)

    def update(self, step: int):
        pass


class SoilWaterVolume(Feature):
    name = "soil_water_volume"
    dtype = torch.float32
    default_tags = {"observable"}
    default_config = DictConfig({
        "soil_volume_key": "${key:terrain,soil_volume}",
        "elevation_key": "${key:terrain,elevation}",
        "elevation_scale": 100,
        "soil_porosity": 0.4,
        "surface_water_volume_key": "${key:terrain,surface_water_volume}",
        "infiltation_rate": 0.001,
        "flow_rate": 0.001
    })

    def render(self) -> torch.Tensor:
        soil_capacity = self.td.get(self.config.soil_volume_key) * self.config.soil_porosity
        return self.data / soil_capacity * 255

    def update(self, step: int):
        elevation = self.td.get(self.config.elevation_key)
        surface_water_volume = self.td.get(self.config.surface_water_volume_key)
        original_total = self.data.sum() + surface_water_volume.sum()

        # Infiltration of surface water into soil
        self.data += surface_water_volume * self.config.infiltation_rate
        surface_water_volume -= surface_water_volume * self.config.infiltation_rate

        soil_capacity = self.td.get(self.config.soil_volume_key) * self.config.soil_porosity

        gradient = flow_gradient(elevation + self.data / soil_capacity)

        self.data, inflow, outflow = flow(
            self.data,
            gradient=gradient,
            flow_rate=self.config.flow_rate
        )

        # Excess water flows back to surface
        surface_water_volume += torch.clamp(self.data - soil_capacity, min=0)
        self.data = torch.relu(torch.clamp(self.data, max=soil_capacity))
        assert torch.isclose(surface_water_volume.sum() + self.data.sum(), original_total)


class SurfaceWaterVolume(Feature):
    name = "surface_water_volume"
    dtype = torch.float32
    default_tags = {"observable"}
    default_config = DictConfig({
        "elevation_key": "${key:terrain,elevation}",
        "elevation_scale": 100,
        "rainfall_rate": 5.7e-08,
        "flow_rate": 0.1,
    })

    def render(self) -> torch.Tensor:
        return (self.data.unsqueeze(-1).expand(-1, -1, 3) / 10) * 255

    def update(self, step: int):
        elevation = self.td.get(self.config.elevation_key) * self.config.elevation_scale
        gradient = flow_gradient(self.data + elevation)
        self.data, inflow, outflow = flow(
            self.data,
            gradient=gradient,
            flow_rate=self.config.flow_rate
        )
        self.data += self.config.rainfall_rate

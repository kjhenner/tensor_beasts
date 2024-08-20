import math

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
                self.data *= perlin_noise(self.shape, self.config.perlin.scale, 4)
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
        "elevation_scale": 100,
        "surface_outflow_key": "${key:terrain,surface_outflow}",
        "erosion_rate": 1e-3,
        "init_scale": 8.0,
        "epsilon": 1e-8
    })

    def render(self) -> torch.Tensor:
        elevation = self.td.get(self.config.elevation_key) * self.config.elevation_scale
        total_elevation = elevation + self.data
        total_min, total_max = total_elevation.min(), total_elevation.max()
        return total_elevation.unsqueeze(-1).expand(-1, -1, 3) / total_max * 255

    def initialize_data(self):
        elevation = self.td.get(self.config.elevation_key)
        self.data = ((1 - elevation) * self.config.init_scale).type(self.dtype)

    def update(self, step: int):
        surface_outflow = self.td.get(self.config.surface_outflow_key)
        result, inflow, outflow = flow(
            self.data,
            outflow=surface_outflow,
            flow_rate=self.config.erosion_rate
        )
        self.data = torch.clamp(result, min=self.config.epsilon)


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
        "soil_water_saturation_key": "${key:terrain,soil_water_saturation}",
        "infiltation_rate": 0.001,
        "flow_rate": 0.1,
        "saturation_gradient_coeff": 0.2,
        "evaporation_rate": 1e-5
    })

    def initialize_data(self):
        self.data = self.td.get(self.config.soil_volume_key) * self.config.soil_porosity * self.config.field_capacity * 0.5

    def render(self) -> torch.Tensor:
        soil_capacity = self.td.get(self.config.soil_volume_key) * self.config.soil_porosity
        return self.data / soil_capacity * 255

    def update(self, step: int):
        elevation = self.td.get(self.config.elevation_key)
        surface_water_volume = self.td.get(self.config.surface_water_volume_key)
        original_total = self.data.sum() + surface_water_volume.sum()

        soil_capacity = self.td.get(self.config.soil_volume_key) * self.config.soil_porosity

        field_capacity = soil_capacity * self.config.field_capacity

        # Infiltration of surface water into soil
        infiltration = torch.min(
            torch.stack([
                surface_water_volume,
                soil_capacity - self.data,
                torch.ones_like(surface_water_volume) * self.config.infiltation_rate
            ]),
            dim=0
        ).values
        self.data += infiltration
        surface_water_volume -= infiltration

        # Calculate excess water above field capacity
        excess_water = torch.clamp(self.data - field_capacity, min=0)

        # Water flows from high to low and from high availability to low
        gradient = flow_gradient(
            (1 - self.config.saturation_gradient_coeff) * elevation
            + self.config.saturation_gradient_coeff * torch.nan_to_num(excess_water / soil_capacity)
        )

        self.data, inflow, outflow = flow(
            self.data,
            gradient=gradient,
            flow_rate=self.config.flow_rate
        )

        # Excess water flows back to surface
        surface_water_volume += torch.clamp(self.data - soil_capacity, min=0)
        self.data = torch.clamp(torch.clamp(self.data, max=soil_capacity), min=0)
        soil_saturation = torch.nan_to_num(self.data / soil_capacity, 1.0, 1.0, 1.0)
        self.td.set(self.config.soil_water_saturation_key, soil_saturation, inplace=True)
        print(gradient.sum(dim=(-1, -2)).max())
        print(gradient.shape)
        # self.td.set(self.config.soil_water_saturation_key, gradient.sum(dim=(-1, -2)) * 150, inplace=True)
        assert torch.isclose(surface_water_volume.sum() + self.data.sum(), original_total)

        evaporation = self.config.evaporation_rate * torch.sigmoid(soil_saturation + 1)
        print(f"Evaporation total: {torch.sum(evaporation)}")
        self.data -= evaporation
        self.data = torch.clamp(self.data, min=0)


class SurfaceWaterVolume(Feature):
    name = "surface_water_volume"
    dtype = torch.float32
    default_tags = {"observable"}
    default_config = DictConfig({
        "elevation_key": "${key:terrain,elevation}",
        "outflow_key": "${key:terrain,surface_outflow}",
        "elevation_scale": 100,
        "rainfall_rate": 5.7e-08,
        "flow_rate": 0.1,
        "relaxation_factor": 0.5,
        "steps": 1
    })

    def render(self) -> torch.Tensor:
        return (self.data.unsqueeze(-1).expand(-1, -1, 3) / 10) * 255

    def update(self, step: int):
        elevation = self.td.get(self.config.elevation_key) * self.config.elevation_scale
        for _ in range(self.config.steps):
            gradient = flow_gradient(self.data + elevation)
            self.data, inflow, outflow = flow(
                self.data,
                gradient=gradient,
                flow_rate=self.config.flow_rate,
                relaxation_factor=self.config.relaxation_factor
            )
        self.td.set(self.config.outflow_key, outflow)
        if step % 1000 < 500:
            self.data += self.config.rainfall_rate
        print(f"Surface water min and max: {torch.min(self.data)}, {torch.max(self.data)}")
        print(f"Rainfall total: {torch.sum(torch.ones_like(self.data) * self.config.rainfall_rate)}")

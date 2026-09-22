import logging
import math

import torch
from omegaconf import DictConfig

from tensor_beasts.features.feature import Feature
from tensor_beasts.registry import register_feature
from tensor_beasts.util import (
    as_conv_batch, perlin_noise, pyramid_elevation, range_elevation,
    unfold_neighbors, flow_gradient, flow
)


@register_feature
class Elevation(Feature):
    name = "elevation"
    dtype = torch.float32
    default_tags = {"observable"}
    default_config = DictConfig({})
    depends_on = {}  # No dependencies - base feature

    def render(self) -> torch.Tensor:
        return self.data.unsqueeze(-1).expand(*self.data.shape, 3) * 255

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
                    1, self.shape[-2]
                ).unsqueeze(1).expand(*self.shape)
            elif key == "range":
                self.data *= range_elevation(self.shape)
            else:
                raise ValueError(f"Invalid elevation mode key: {key}")
        unfold_neighbors(self.td, self.key, (3, 3))


@register_feature
class AquiferElevation(Feature):
    name = "aquifer_elevation"
    dtype = torch.float32
    default_config = DictConfig({
        "elevation_key": "${key:terrain,elevation}",
        "scale": 0.9,
    })
    # Dependencies for initialization ordering
    depends_on = {
        "elevation": "elevation",  # Needs elevation data to compute aquifer level
    }

    def render(self) -> torch.Tensor:
        return self.data.unsqueeze(-1).expand(*self.data.shape, 3) * 255

    def initialize_data(self):
        elevation = self.td.get(self.config.elevation_key)
        self.data = (elevation * self.config.scale).type(self.dtype)


@register_feature
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
    # Dependencies for initialization and update ordering
    depends_on = {
        "elevation": "elevation",  # Needs elevation for initialization
        # Note: surface_outflow is a computed feature from surface_water_volume.update()
        # so update depends on surface_water_volume running first
    }

    def render(self) -> torch.Tensor:
        elevation = self.td.get(self.config.elevation_key) * self.config.elevation_scale
        total_elevation = elevation + self.data
        total_min, total_max = total_elevation.min(), total_elevation.max()
        return total_elevation.unsqueeze(-1).expand(*self.data.shape, 3) / total_max * 255

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


@register_feature
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
        "infiltration_rate": 0.001,
        "flow_rate": 0.1,
        "saturation_gradient_coeff": 0.2,
        "evaporation_rate": 1e-5,
        "field_capacity": 0.5
    })
    # Dependencies for initialization and update ordering
    depends_on = {
        "soil_volume": "soil_volume",  # Needs soil_volume for init and update
        "elevation": "elevation",  # Needs elevation for flow calculations
        "surface_water_volume": "surface_water_volume",  # Needs surface water for infiltration
    }

    def initialize_data(self):
        self.data = self.td.get(self.config.soil_volume_key) * self.config.soil_porosity * self.config.field_capacity * 0.5

    def render(self) -> torch.Tensor:
        soil_capacity = self.td.get(self.config.soil_volume_key) * self.config.soil_porosity
        return self.data / soil_capacity * 255

    def update(self, step: int):
        elevation = self.td.get(self.config.elevation_key)
        surface_water_volume = self.td.get(self.config.surface_water_volume_key)
        original_soil, original_surface = self.data.sum(), surface_water_volume.sum()

        soil_capacity = self.td.get(self.config.soil_volume_key) * self.config.soil_porosity

        field_capacity = soil_capacity * self.config.field_capacity

        # Infiltration of surface water into soil
        # dim=0 below is the leading *stacked candidate* axis created by
        # torch.stack (3 candidates), not a spatial axis: already rank-agnostic.
        infiltration = torch.min(
            torch.stack([
                surface_water_volume,
                soil_capacity - self.data,
                torch.ones_like(surface_water_volume) * self.config.infiltration_rate
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
        logger = logging.getLogger(__name__)
        logger.debug("SoilWaterVolume gradient max: %s", gradient.sum(dim=(-1, -2)).max())
        logger.debug("SoilWaterVolume gradient shape: %s", gradient.shape)
        assert torch.isclose(
            surface_water_volume.sum() + self.data.sum(),
            original_surface + original_soil,
        ), "infiltration + soil flow did not conserve water"

        evaporation = self.config.evaporation_rate * torch.sigmoid(soil_saturation + 1)
        logger.debug("SoilWaterVolume evaporation total: %s", torch.sum(evaporation))
        self.data -= evaporation
        self.data = torch.clamp(self.data, min=0)

        # Publish saturation *after* evaporation so the exported field matches
        # the water that is actually in the soil. It used to be written before
        # the evaporation subtraction, leaving every consumer a step stale.
        self.td.set(
            self.config.soil_water_saturation_key,
            torch.nan_to_num(self.data / soil_capacity, 1.0, 1.0, 1.0),
            inplace=True,
        )


@register_feature
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
    # Dependencies for update ordering
    depends_on = {
        "elevation": "elevation",  # Needs elevation for flow gradient
        # Note: this feature produces surface_outflow which soil_volume needs
    }

    def render(self) -> torch.Tensor:
        return (self.data.unsqueeze(-1).expand(*self.data.shape, 3) / 10) * 255

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
        logger = logging.getLogger(__name__)
        logger.debug("Surface water min/max: %s/%s", torch.min(self.data), torch.max(self.data))
        logger.debug(
            "Surface water rainfall total: %s",
            torch.sum(torch.ones_like(self.data) * self.config.rainfall_rate),
        )


@register_feature
class SimpleWater(Feature):
    """
    A simple water feature initialized with multi-scale Perlin noise.

    Optionally modulated by an oscillator for dynamic water level changes.
    A phase map (larger-scale Perlin noise) controls spatial correlation with the oscillator:
    - High phase values: positive correlation (water rises when oscillator rises)
    - Low phase values: negative correlation (water falls when oscillator rises)
    - Mid phase values: unaffected by oscillator

    Output is clamped to [0, 1] range.
    """
    name = "simple_water"
    dtype = torch.float32
    default_tags = {"observable"}
    default_config = DictConfig({
        "perlin_scale": [4, 4],       # Base scale for Perlin noise
        "perlin_octaves": 4,          # Number of octaves for multi-scale
        "perlin_persistence": 0.5,    # Amplitude decay per octave
        "oscillator_key": "${key:simpleterrain,oscillator}",  # Reference to oscillator feature
        "oscillator_amplitude": 0.1,  # How much oscillator affects water level
        "phase_scale_multiplier": 128,  # Phase map grid divisor (larger = coarser regions, 128 gives ~4x4 domains)
    })
    depends_on = {}

    def render(self) -> torch.Tensor:
        # Render as blue intensity
        blue = (self.data * 255).clamp(0, 255)
        return torch.stack([
            torch.zeros_like(blue),
            torch.zeros_like(blue),
            blue
        ], dim=-1)

    def initialize_data(self):
        # Initialize with multi-scale Perlin noise normalized to [0, 1]
        try:
            noise = perlin_noise(
                self.shape,
                self.config.perlin_scale,
                octaves=self.config.perlin_octaves,
                persistence=self.config.perlin_persistence
            )
            # Normalize to [0, 1]
            noise_min, noise_max = noise.min(), noise.max()
            self.data = ((noise - noise_min) / (noise_max - noise_min + 1e-8)).type(self.dtype)
        except (IndexError, RuntimeError):
            # Fallback to uniform random if perlin fails
            self.data = torch.rand(self.shape, dtype=self.dtype)
        # The base pattern and the phase map below are state: the update
        # rewrites the water from them every step. They are kept in the
        # TensorDict, under this feature's prefix, so that a snapshot of the
        # world carries them and a restored world runs on the landscape its
        # populations grew on. As plain attributes they were drawn afresh in
        # every World, and a state loaded from tensor_beasts/rl/bank.py kept
        # its banked water for exactly one step.
        self.td.set(self._base_key, self.data.clone())

        # Generate phase map at larger scale for oscillator correlation
        # Phase map determines how each location responds to the oscillator
        # Use downsampled random noise + bilinear upscale for smooth large regions
        phase_divisor = self.config.phase_scale_multiplier
        leading, spatial = self.shape[:-2], self.shape[-2:]
        small_shape = (*leading, max(1, spatial[0] // phase_divisor), max(1, spatial[1] // phase_divisor))
        small_noise = torch.rand(small_shape, dtype=self.dtype)
        # Upsample with bilinear interpolation for smooth transitions
        small_4d, _ = as_conv_batch(small_noise)
        phase_noise = torch.nn.functional.interpolate(
            small_4d,
            size=tuple(spatial),
            mode='bilinear',
            align_corners=True
        ).reshape(*leading, *spatial)
        # Normalize to [-1, 1] so mid values = 0 (no effect)
        phase_min, phase_max = phase_noise.min(), phase_noise.max()
        self.td.set(self._phase_key, ((phase_noise - phase_min) / (phase_max - phase_min + 1e-8) - 0.5) * 2.0)

    @property
    def _base_key(self):
        return (*self.key[:-1], f"{self.name}_base")

    @property
    def _phase_key(self):
        return (*self.key[:-1], f"{self.name}_phase")

    def update(self, step: int):
        if self.config.oscillator_key is not None:
            oscillator_value = self.td.get(self.config.oscillator_key)
            # Phase-modulated oscillation:
            # - Where phase_map > 0: positive correlation with oscillator
            # - Where phase_map < 0: negative correlation (anti-phase)
            # - Where phase_map = 0: no effect
            modulation = oscillator_value * self.config.oscillator_amplitude * self.td.get(self._phase_key)
            self.data = torch.clamp(self.td.get(self._base_key) + modulation, 0.0, 1.0)


@register_feature
class Nutrients(Feature):
    """
    Terrain nutrients feature for nutrient cycling.

    Receives biomass from dead animals and is consumed by plant growth.
    Uses float32 for precision in conservation calculations.
    """
    name = "nutrients"
    dtype = torch.float32
    default_tags = {"observable"}
    default_config = DictConfig({
        "initial_nutrients": 50.0,
        "diffusion_rate": 0.01,
    })
    depends_on = {}

    def render(self) -> torch.Tensor:
        # Render as brown intensity
        normalized = torch.clamp(self.data / 100.0, 0, 1)
        return (normalized * 255).unsqueeze(-1).expand(*self.data.shape, 3)

    def initialize_data(self):
        self.data = torch.full(
            self.shape,
            self.config.initial_nutrients,
            dtype=self.dtype
        )

    def update(self, step: int):
        # Optional slow diffusion to spread nutrients
        if self.config.diffusion_rate > 0:
            kernel = torch.tensor([
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0]
            ], dtype=torch.float32) * self.config.diffusion_rate / 4.0
            kernel[1, 1] = 1.0 - self.config.diffusion_rate

            data_4d, leading = as_conv_batch(self.data)
            padded = torch.nn.functional.pad(
                data_4d,
                (1, 1, 1, 1),
                mode='replicate'
            )
            out = torch.nn.functional.conv2d(
                padded, kernel.unsqueeze(0).unsqueeze(0)
            )
            self.data = out.reshape(*leading, *out.shape[-2:])

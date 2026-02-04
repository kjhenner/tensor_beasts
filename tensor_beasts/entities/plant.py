from typing import Optional

import torch
from omegaconf import DictConfig

from tensor_beasts.entities import Entity
from tensor_beasts.registry import register_entity
from tensor_beasts.features.shared_features import Energy, Scent
from tensor_beasts.features.plant_features import Seed, Crowding
from tensor_beasts.util import safe_add, safe_sub


@register_entity
class HydrodynamicPlant(Entity):
    """
    Plant entity with growth conditioned on complex soil hydrology.

    Uses soil water volume and soil saturation for growth calculations.
    Requires Terrain entity with soil_water_volume and soil_volume features.
    Growth consumes nutrients from terrain.
    """
    # t/ha/y * 2e-5 = kg/10m^2/m

    energy: Energy
    scent: Scent
    seed: Seed
    crowding: Crowding
    default_config = DictConfig({
        "energy_key": "${key:hydrodynamicplant,energy}",
        "soil_water_volume_key": "${key:terrain,soil_water_volume}",
        "soil_volume_key": "${key:terrain,soil_volume}",
        "soil_sat_coeff_key": "${key:hydrodynamicplant,soil_sat_coeff}",
        "nutrients_key": "${key:terrain,nutrients}",
        "nutrient_consumption_rate": 0.5,  # nutrients consumed per growth event
        "soil_porosity": 0.4,
        "ideal_growth_rate": 4e-4,  # ~200 t/ha/y
        "ideal_soil_saturation": 0.65,
        "soil_saturation_tolerance": 0.2,
        # Keys for plant features (Seed, Crowding)
        "seed": {
            "energy_key": "${key:hydrodynamicplant,energy}",
            "crowding_key": "${key:hydrodynamicplant,crowding}",
        },
        "crowding": {
            "energy_key": "${key:hydrodynamicplant,energy}",
        },
    })

    def __init__(
        self,
        world: 'World',
        config: DictConfig,
    ):
        super().__init__(world, config)

    def initialize(self):
        self.seed.initialize_data()
        self.scent.initialize_data()
        self.crowding.initialize_data()
        self.energy.initialize_data()
        self.energy.data = ~ torch.randint(
            0,
            int(self.config.init_prob * 255),
            self.energy.data.shape,
            dtype=torch.uint8
        ).type(torch.bool) * self.config.initial_energy

    def update(self, action: Optional[torch.Tensor] = None):
        self.crowding.update(0)
        self.seed.update(0)
        self.scent.emit(self.world.step)  # Emit only; diffusion handled by World

        energy = self.energy.data
        soil_water_volume = self.td.get(self.config.soil_water_volume_key)
        soil_volume = self.td.get(self.config.soil_volume_key)
        soil_saturation = torch.nan_to_num(soil_water_volume / (soil_volume * self.config.soil_porosity), 1.0, 1.0, 1.0)

        soil_sat_coeff = 1 - (
            torch.abs(
                soil_saturation - self.config.ideal_soil_saturation
            ) / self.config.soil_saturation_tolerance
        )
        self.td.set(self.config.soil_sat_coeff_key, soil_sat_coeff, inplace=True)

        condition = (
            self.config.ideal_growth_rate * soil_sat_coeff
        ).type(torch.float32)

        death_prob = torch.abs(torch.clamp(condition, max=0))
        growth_prob = torch.clamp(condition, min=0)

        growth = torch.rand(energy.shape) < growth_prob
        death = torch.rand(energy.shape) < death_prob

        # Nutrient-limited growth
        if self.config.nutrients_key is not None:
            try:
                nutrients = self.td.get(self.config.nutrients_key)
                # Can only grow where nutrients are available
                nutrient_available = nutrients >= self.config.nutrient_consumption_rate
                growth = growth & nutrient_available

                # Consume nutrients where growth occurs
                growth_mask = (energy > 0) & growth
                nutrients -= growth_mask.float() * self.config.nutrient_consumption_rate
                nutrients.clamp_(min=0)
            except KeyError:
                pass  # No nutrients feature, grow without nutrient constraint

        self.energy.data = safe_add(energy, (energy > 0) * growth, inplace=False)
        self.energy.data *= ~ death


@register_entity
class SimplePlant(Entity):
    """
    Plant entity with growth conditioned on simple water level.

    Uses SimpleWater feature for growth calculations with optimal range model.
    Simpler alternative to HydrodynamicPlant for basic simulations.
    Optionally consumes nutrients if nutrients_key is configured.
    """

    energy: Energy
    scent: Scent
    seed: Seed
    crowding: Crowding
    default_config = DictConfig({
        "energy_key": "${key:simpleplant,energy}",
        "water_key": "${key:simpleterrain,simple_water}",
        "growth_coeff_key": "${key:simpleplant,growth_coeff}",
        "nutrients_key": None,  # Optional - SimpleTerrain doesn't have nutrients by default
        "nutrient_consumption_rate": 0.5,
        "ideal_growth_rate": 4e-4,     # Base growth rate
        "ideal_water_level": 0.5,      # Optimal water level (0-1)
        "water_tolerance": 0.3,        # Tolerance around optimal
        "init_prob": 0.5,              # Probability of spawning per cell
        "initial_energy": 32,          # Starting energy for spawned plants
        "toy_init": False,             # If True, spawn single plant in center
        # Keys for plant features (Seed, Crowding)
        "seed": {
            "energy_key": "${key:simpleplant,energy}",
            "crowding_key": "${key:simpleplant,crowding}",
        },
        "crowding": {
            "energy_key": "${key:simpleplant,energy}",
        },
    })

    def __init__(
        self,
        world: 'World',
        config: DictConfig,
    ):
        super().__init__(world, config)

    def initialize(self):
        self.seed.initialize_data()
        self.scent.initialize_data()
        self.crowding.initialize_data()
        self.energy.initialize_data()

        if self.config.toy_init:
            # Single plant at random position
            h = torch.randint(0, self.world.size[0], (1,)).item()
            w = torch.randint(0, self.world.size[1], (1,)).item()
            self.energy.data[h, w] = self.config.initial_energy
        else:
            # Probability-based initialization
            spawn_mask = torch.rand(self.energy.data.shape) < self.config.init_prob
            self.energy.data = spawn_mask.to(torch.uint8) * self.config.initial_energy

    def update(self, action: Optional[torch.Tensor] = None):
        self.crowding.update(0)
        self.seed.update(0)
        self.scent.emit(self.world.step)  # Emit only; diffusion handled by World

        energy = self.energy.data
        water_level = self.td.get(self.config.water_key)

        # Optimal range model: growth coefficient based on distance from ideal
        growth_coeff = 1 - (
            torch.abs(water_level - self.config.ideal_water_level)
            / self.config.water_tolerance
        )
        self.td.set(self.config.growth_coeff_key, growth_coeff, inplace=True)

        condition = (self.config.ideal_growth_rate * growth_coeff).type(torch.float32)

        death_prob = torch.abs(torch.clamp(condition, max=0))
        growth_prob = torch.clamp(condition, min=0)

        growth = torch.rand(energy.shape) < growth_prob
        death = torch.rand(energy.shape) < death_prob

        # Nutrient-limited growth (optional)
        if self.config.nutrients_key is not None:
            try:
                nutrients = self.td.get(self.config.nutrients_key)
                nutrient_available = nutrients >= self.config.nutrient_consumption_rate
                growth = growth & nutrient_available

                growth_mask = (energy > 0) & growth
                nutrients -= growth_mask.float() * self.config.nutrient_consumption_rate
                nutrients.clamp_(min=0)
            except KeyError:
                pass

        self.energy.data = safe_add(energy, (energy > 0) * growth, inplace=False)
        # Death from bad conditions OR energy depleted to 0
        death = death | (self.energy.data == 0)
        self.energy.data *= ~death

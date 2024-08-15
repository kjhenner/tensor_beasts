from typing import Optional

import torch
from omegaconf import DictConfig

from tensor_beasts.entities import Entity
from tensor_beasts.features.shared_features import Energy, Scent
from tensor_beasts.features.plant_features import Seed, Crowding
from tensor_beasts.util import safe_add, safe_sub


class Plant(Entity):
    # t/ha/y * 2e-5 = kg/10m^2/m

    energy: Energy
    scent: Scent
    seed: Seed
    crowding: Crowding
    default_config = DictConfig({
        "energy_key": "${key:plant,energy}",
        "soil_water_volume_key": "${key:terrain,soil_water_volume}",
        "soil_volume_key": "${key:terrain,soil_volume}",
        "soil_sat_coeff_key": "${key:plant,soil_sat_coeff}",
        "soil_porosity": 0.4,
        "ideal_growth_rate": 4e-4,  # ~200 t/ha/y
        "ideal_soil_saturation": 0.65,
        "soil_saturation_tolerance": 0.2,
    })

    def __init__(
        self,
        world: 'World',
        config: DictConfig,
    ):
        super().__init__(world, config)

    def initialize(self):
        self.seed.initialize_data()
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

        self.energy.data = safe_add(energy, (energy > 0) * growth, inplace=False)
        self.energy.data *= ~ death

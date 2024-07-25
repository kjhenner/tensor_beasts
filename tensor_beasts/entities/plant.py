from typing import Optional

import torch
from omegaconf import DictConfig

from tensor_beasts.entities import Entity
from tensor_beasts.entities.feature import Feature, Energy, Scent
from tensor_beasts.util import perlin_noise, generate_diffusion_kernel, torch_correlate_2d, safe_add, safe_sub


class FertilityMap(Feature):
    name = "fertility_map"
    dtype = torch.float32


class Seed(Feature):
    name = "seed"
    dtype = torch.uint8


class Crowding(Feature):
    name = "crowding"
    dtype = torch.float32


class Plant(Entity):

    features = {
        Energy.name: Energy(),
        Scent.name: Scent(),
        FertilityMap.name: FertilityMap(),
        Seed.name: Seed(),
        Crowding.name: Crowding()
    }

    def __init__(
        self,
        world: 'World',
        config: DictConfig,
    ):
        super().__init__(world, config)
        self.initial_energy = config.initial_energy
        self.init_prob = config.init_prob
        self.growth_prob = config.growth_prob
        self.germination_prob = config.germination_prob
        self.seed_prob = config.seed_prob

    def initialize(self):
        energy = self.get_feature("energy")
        fertility_map = (perlin_noise(energy.shape, (5, 5)))
        self.set_feature("fertility_map", fertility_map)
        seed = self.get_feature("seed")
        self.set_feature("seed", torch.zeros(seed.shape, dtype=seed.dtype))
        self.set_feature(
            "energy",
            ~ torch.randint(
                0,
                int(self.init_prob * 255),
                energy.shape,
                dtype=torch.uint8
            ).type(torch.bool) * self.initial_energy
        )

    def update_crowding(self):
        energy = self.get_feature("energy")
        kernel = generate_diffusion_kernel(size=5)
        crowding = torch.clamp(
            torch_correlate_2d(energy.bool().type(torch.float32), kernel, mode="constant"),
            0,
            1
        )
        self.set_feature(
            "crowding",
            crowding
        )

    def grow(self):
        energy = self.get_feature("energy")
        fertility_map = self.get_feature("fertility_map")
        crowding = self.get_feature("crowding")
        rand = self.world.td.get("random")

        # Calculate growth probability based on fertility
        growth_prob = self.growth_prob * fertility_map ** 2
        # Combine growth probability with crowding factor
        combined_prob = growth_prob * (1 - crowding)

        # Generate growth mask
        growth = rand < (combined_prob * 255)

        # Apply growth to existing plants
        self.set_feature(
            "energy",
            safe_add(energy, (energy > 0) * growth, inplace=False)
        )

    def seed(self):
        crowding = self.get_feature("crowding")
        seed = self.get_feature("seed")
        rand = self.world.td.get("random")

        self.set_feature(
            "seed",
            seed | (rand < self.seed_prob * crowding ** 2 * 255).type(seed.dtype)
        )

    def germinate(self):
        crowding = self.get_feature("crowding")
        energy = self.get_feature("energy")
        seed = self.get_feature("seed")
        rand = self.world.td.get("random")

        seed_germination = (
            seed & ~(energy > 0) & (rand < ((1 - crowding) ** 2 * self.germination_prob * 255))
        ).type(torch.uint8)
        safe_add(energy, seed_germination)
        safe_sub(seed, seed_germination)

    def update(self, action: Optional[torch.Tensor] = None):
        self.update_crowding()
        self.grow()
        self.seed()
        self.germinate()

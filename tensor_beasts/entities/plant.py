from typing import Optional

import torch
from omegaconf import DictConfig

from tensor_beasts.entities import Entity
from tensor_beasts.features.shared_features import Energy, Scent
from tensor_beasts.features.plant_features import Seed, Crowding
from tensor_beasts.util import generate_diffusion_kernel, torch_correlate_2d, safe_add, safe_sub


class Plant(Entity):

    energy: Energy
    scent: Scent
    seed: Seed
    crowding: Crowding

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
        self.water_key = tuple(config.water_key)

    def initialize(self):
        self.seed.zero_init()

        self.energy.zero_init()
        self.energy.data = ~ torch.randint(
            0,
            int(self.init_prob * 255),
            self.energy.data.shape,
            dtype=torch.uint8
        ).type(torch.bool) * self.initial_energy

    def update_crowding(self):
        energy = self.energy.data
        kernel = generate_diffusion_kernel(size=5)
        crowding = torch.clamp(
            torch_correlate_2d(energy.bool().type(torch.float32), kernel, mode="constant"),
            0,
            1
        )
        self.crowding.data = crowding

    def grow(self):
        energy = self.energy.data
        fertility_map = self.world.td.get(self.water_key) / 20
        crowding = self.crowding.data
        rand = self.world.td.get("random")

        # Calculate growth probability based on fertility
        growth_prob = self.growth_prob * fertility_map ** 2
        # Combine growth probability with crowding factor
        combined_prob = growth_prob * (1 - crowding)

        # Generate growth mask
        growth = rand < (combined_prob * 255)

        # Apply growth to existing plants
        self.energy.data = safe_add(energy, (energy > 0) * growth, inplace=False)

        safe_sub(energy, (rand < 3).type(torch.uint8), inplace=True)

    def seed(self):
        crowding = self.crowding.data
        seed = self.seed.data
        rand = self.world.td.get("random")
        self.seed.data = seed | (rand < self.seed_prob * crowding ** 2 * 255).type(seed.dtype)

    def germinate(self):
        crowding = self.crowding.data
        energy = self.energy.data
        seed = self.seed.data
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

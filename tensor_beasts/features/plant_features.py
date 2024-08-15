import torch
from omegaconf import DictConfig

from tensor_beasts.features.feature import Feature
from tensor_beasts.util import generate_diffusion_kernel, safe_add, safe_sub


class Seed(Feature):
    name = "seed"
    dtype = torch.uint8
    default_config = DictConfig({
        "energy_key": "${key:plant,energy}",
        "crowding_key": "${key:plant,crowding}",
        "random_key": "${key:random}",
        "seed_prob": 0.01,
        "germination_prob": 0.01,
        "scale": 0.9,
    })

    def update(self, step: int):
        seed = self.data
        crowding = self.td.get(self.config.crowding_key)
        rand = self.td.get(self.config.random_key)
        self.data = seed | (rand < self.config.seed_prob * crowding ** 2 * 255).type(seed.dtype)

        energy = self.td.get(self.config.energy_key)

        seed_germination = (
            seed & ~(energy > 0) & (rand < ((1 - crowding) ** 2 * self.config.germination_prob * 255))
        ).type(torch.uint8)
        safe_add(energy, seed_germination)
        safe_sub(seed, seed_germination)


class Crowding(Feature):
    name = "crowding"
    dtype = torch.float32
    default_config = DictConfig({
        "energy_key": "${key:plant,energy}",
        "scale": 0.9,
    })

    def update(self, step: int):
        energy = self.td.get(self.config.energy_key)
        kernel = generate_diffusion_kernel(size=5)
        self.data = torch.conv2d(
            energy.unsqueeze(0).unsqueeze(0).type(torch.float32),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=(kernel.shape[0] // 2)
        ).squeeze(0).squeeze(0)

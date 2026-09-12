import torch
from omegaconf import DictConfig, ListConfig

from tensor_beasts.features.feature import Feature
from tensor_beasts.registry import register_feature
from tensor_beasts.util import as_conv_batch, generate_diffusion_kernel, safe_add, safe_sub


def _to_tuple(key):
    """Convert a key to tuple for TensorDict compatibility."""
    if isinstance(key, (list, ListConfig)):
        return tuple(key)
    return key


@register_feature
class Seed(Feature):
    name = "seed"
    dtype = torch.uint8
    default_config = DictConfig({
        "energy_key": None,     # Must be set by entity config
        "crowding_key": None,   # Must be set by entity config
        "random_key": "${key:random}",
        "seed_prob": 0.01,
        "germination_prob": 0.01,
        "scale": 0.9,
    })

    def update(self, step: int):
        seed = self.data
        crowding = self.td.get(_to_tuple(self.config.crowding_key))
        rand = self.td.get(_to_tuple(self.config.random_key))
        self.data = seed | (rand < self.config.seed_prob * crowding ** 2 * 255).type(seed.dtype)

        energy = self.td.get(_to_tuple(self.config.energy_key))

        seed_germination = (
            seed & ~(energy > 0) & (rand < ((1 - crowding) ** 2 * self.config.germination_prob * 255))
        ).type(torch.uint8)
        safe_add(energy, seed_germination)
        safe_sub(seed, seed_germination)


@register_feature
class Crowding(Feature):
    name = "crowding"
    dtype = torch.float32
    default_config = DictConfig({
        "energy_key": None,  # Must be set by entity config
        "scale": 0.9,
    })

    def update(self, step: int):
        energy = self.td.get(_to_tuple(self.config.energy_key))
        kernel = generate_diffusion_kernel(size=5)
        # Fold any leading dims into the conv batch instead of unsqueeze/squeeze,
        # which is only correct for an exactly-2D input.
        energy_4d, leading = as_conv_batch(energy.type(torch.float32))
        out = torch.conv2d(
            energy_4d,
            kernel.unsqueeze(0).unsqueeze(0),
            padding=(kernel.shape[-2] // 2)
        )
        self.data = out.reshape(*leading, *out.shape[-2:])

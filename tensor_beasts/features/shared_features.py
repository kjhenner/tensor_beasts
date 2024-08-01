import torch

from tensor_beasts.features.feature import SharedFeature
from tensor_beasts.util import generate_diffusion_kernel, torch_correlate_3d


class Scent(SharedFeature):
    name = "scent"
    dtype = torch.uint8
    default_tags = {"observable"}
    energy_key = ("shared_features", "energy")

    @staticmethod
    def custom_diffusion(
        scent,
        energy,
        energy_contribution=20.0,
        diffusion_steps=2,
        max_value=255
    ):
        kernel = generate_diffusion_kernel(size=9, sigma=0.99)
        scent_float = scent.type(torch.float32)

        for _ in range(diffusion_steps):
            diffused = torch_correlate_3d(scent_float, kernel)
            scent_float = diffused.clamp(0, max_value)

        # Energy contribution (non-linear)
        energy_float = energy.type(torch.float32)
        energy_effect = torch.where(
            energy_float > 0,
            torch.log1p(energy_float) / torch.log1p(torch.tensor(max_value, dtype=torch.float32)),
            torch.zeros_like(energy_float)
        )
        scent_float = scent_float + energy_effect * energy_contribution

        return scent_float.round().clamp(0, max_value).type(torch.uint8)

    def update(self, step: int):
        scent = self.data.clone()
        energy = self.td.get(self.energy_key).clone()
        scent = self.custom_diffusion(scent, energy)
        self.data = scent


class Energy(SharedFeature):
    name = "energy"
    dtype = torch.uint8
    default_tags = {"observable"}

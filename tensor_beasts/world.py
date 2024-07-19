from dataclasses import dataclass
from typing import List, Type, Tuple
import torch
from tensordict import TensorDict
from omegaconf import DictConfig

from tensor_beasts.entities.entity import Entity
from tensor_beasts.util import (
    torch_correlate_3d, generate_diffusion_kernel
)


@dataclass
class FeatureDefinition:
    name: str
    dtype: torch.dtype
    shape: Tuple[int, ...] = ()
    shared: bool = False
    observable: bool = False
    use_world_size: bool = True


class World:
    def __init__(self, entity_classes: List[Type[Entity]], config: DictConfig):
        self.size: Tuple[int, ...] = tuple(config.size)
        self.td = TensorDict({}, batch_size=[])
        self.entities = {
            cls.__name__.lower(): cls(self, config["entities"][cls.__name__.lower()]) for cls in entity_classes
        }
        self._initialize_features()
        for entity in self.entities.values():
            entity.initialize()
        self.step = 0

    def _initialize_features(self):

        def _features_match(f1: FeatureDefinition, f2: FeatureDefinition) -> bool:
            return all(
                getattr(f1, attr) == getattr(f2, attr) for attr in ["shape", "dtype", "use_world_size", "observable"]
            )

        shared_feature_counts = {}
        shared_feature_defs = {}

        self.td["shared_features"] = TensorDict({}, batch_size=[])

        for name, entity in self.entities.items():
            for f in entity.get_feature_definitions():
                if f.shared:
                    if f.name not in shared_feature_counts:
                        shared_feature_counts[f.name] = 1
                        shared_feature_defs[f.name] = f
                    else:
                        shared_feature_counts[f.name] += 1
                        if not _features_match(shared_feature_defs[f.name], f):
                            raise ValueError(f"Mismatched definitions for shared feature {f.name}")
                else:
                    shape = (*self.size, *f.shape) if f.use_world_size else f.shape
                    self.td[name, f.name] = torch.zeros(*shape, dtype=f.dtype)

        for i, (name, entity) in enumerate(self.entities.items()):
            for f in entity.get_feature_definitions():
                if f.shared:
                    if f.name not in self.td["shared_features"]:
                        count = shared_feature_counts[f.name]
                        shape = (*self.size, *f.shape) if f.use_world_size else f.shape
                        shape = shape + (count,)
                        self.td.set(("shared_features", f.name), torch.zeros(*shape, dtype=f.dtype))
                    entity.set_shared_feature_idx(f.name, i)

    def custom_diffusion(self, scent, energy, energy_contribution=20.0, diffusion_steps=2, max_value=255):
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

    def diffuse_scent(self):
        scent = self.td.get(("shared_features", "scent")).clone()
        energy = self.td.get(("shared_features", "energy")).clone()
        scent = self.custom_diffusion(scent, energy)
        self.td.set(("shared_features", "scent"), scent)

    def update(self):
        self.td.set("random", torch.randint(0, 256, self.size, dtype=torch.uint8))
        self.diffuse_scent()
        for entity in self.entities.values():
            entity.update(self.step)
        self.step += 1

    def get_feature(self, entity_name: str, feature_name: str) -> torch.Tensor:
        entity = self.entities[entity_name]
        return entity.get_feature(feature_name)

    def set_feature(self, entity_name: str, feature_name: str, value: torch.Tensor) -> None:
        entity = self.entities[entity_name]
        entity.set_feature(feature_name, value)

    @property
    def observable(self):
        pass

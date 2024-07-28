from dataclasses import dataclass
from typing import List, Type, Tuple, Optional, Union, Dict
import torch
from tensordict import TensorDict
from omegaconf import DictConfig

from tensor_beasts import entities
from tensor_beasts.entities.entity import Entity
from tensor_beasts.entities.feature import Feature
from tensor_beasts.util import (
    torch_correlate_3d, generate_diffusion_kernel
)


class World:
    def __init__(self, config: DictConfig):
        self.size: Tuple[int, ...] = tuple(config.size)
        self.shared_features: Dict[str, Feature] = {}
        print(f"world height: {self.size[0]}")
        print(f"world width: {self.size[1]}")
        self.td = TensorDict({}, batch_size=[])
        self.entities = {
            entity_name: getattr(entities, entity_name.title())(self, entity_config) for
            entity_name, entity_config in config.entities.items()
        }
        self._initialize_features()
        for entity in self.entities.values():
            entity.initialize()
        self.step = 0

    def __getattr__(self, item):
        if item in self.entities:
            return self.entities[item]
        else:
            raise AttributeError(f"Attribute {item} not found in World")

    def reset(self):
        self.td.clear()
        self._initialize_features()
        for entity in self.entities.values():
            entity.initialize()
        self.step = 0

    def _initialize_features(self):

        shared_feature_counts = {}

        self.td["shared_features"] = TensorDict({}, batch_size=[])

        for name, entity in self.entities.items():
            for f in entity.features.values():
                if f.shared:
                    if f.name not in shared_feature_counts:
                        shared_feature_counts[f.name] = 1
                        self.shared_features[f.name] = f
                    else:
                        shared_feature_counts[f.name] += 1
                else:
                    shape = (*self.size, *f.shape) if f.use_world_size else f.shape
                    self.td[name, f.name] = torch.zeros(*shape, dtype=f.dtype)

        for i, (name, entity) in enumerate(self.entities.items()):
            for f in entity.features.values():
                if f.shared:
                    count = shared_feature_counts[f.name]
                    if f.name not in self.td["shared_features"]:
                        shape = (*self.size, *f.shape) if f.use_world_size else f.shape
                        shape = shape + (count,)
                        self.td.set(("shared_features", f.name), torch.zeros(*shape, dtype=f.dtype))
                    entity.set_shared_feature_idx(f.name, count - 1)

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

    def update(self, action_td: Optional[TensorDict] = None):
        self.td.set("random", torch.randint(0, 256, self.size, dtype=torch.uint8))
        if "scent" in self.shared_features.keys():
            self.diffuse_scent()
        for name, entity in self.entities.items():
            entity.update(action=action_td.get(name, None) if action_td is not None else None)
        self.step += 1

    def get_feature(self, entity_name: str, feature_name: str) -> torch.Tensor:
        if entity_name == "shared_features":
            return self.td.get(("shared_features", feature_name))
        entity = self.entities[entity_name]
        return entity.get_feature(feature_name)

    def set_feature(self, entity_name: str, feature_name: str, value: torch.Tensor) -> None:
        entity = self.entities[entity_name]
        entity.set_feature(feature_name, value)

    def render_feature(self, entity_name: str, feature_name: str) -> torch.Tensor:
        print(f"entity_name: {entity_name}")
        print(f"feature_name: {feature_name}")
        if entity_name == "shared_features":
            feature = self.shared_features[feature_name]
            return feature.render(self.td.get(("shared_features", feature_name)))
        return self.entities[entity_name].render_feature(feature_name)

    @property
    def observable(self):
        observable_features = list(self.td["shared_features"].values())

        # Add non-shared features
        for entity in self.entities.values():
            for feature in entity.get_feature_definitions():
                if feature.observable and not feature.shared:
                    observable_features.append(entity.get_feature(feature.name))

        # Concatenate along the last dimension
        return torch.cat(observable_features, dim=-1)

    def entity_scores(self, entity: Union[Entity | str], reward_mode: str = "default"):
        if isinstance(entity, str):
            entity = self.entities[entity.lower()]
        reward = entity.get_feature("offspring_count").type(torch.float32) * 255 + entity.get_feature("energy")

        total_reward = torch.sum(reward)
        normalized_reward = torch.clamp(total_reward / (self.size[0] * self.size[1]) * 2 - 1, -1, 1)

        return normalized_reward

    def inspect(self, x: int, y: int):
        for k, v in self.td.items(include_nested=True, leaves_only=True):
            if v.shape[:2] == self.size:
                # Tensors have shape H, W
                print(k, v[y, x])

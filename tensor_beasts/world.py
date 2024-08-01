from typing import List, Type, Tuple, Optional, Union, Dict, Callable
import torch
from tensordict import TensorDict
from omegaconf import DictConfig

from tensor_beasts import entities
from tensor_beasts.entities.entity import Entity


class World:
    def __init__(self, config: DictConfig):
        self.size: Tuple[int, ...] = tuple(config.size)
        self.config = config
        self.td = TensorDict({}, batch_size=[])
        self.entity_dict: Dict[str, Entity] = {}
        for entity_name, entity_config in config.entities.items():
            self.entity_dict[entity_name] = getattr(entities, entity_name.title())(self, entity_config)
        for entity in self.entity_dict.values():
            entity.initialize()
        self.step = 0

    def __getattr__(self, item):
        try:
            return self.entity_dict[item]
        except KeyError:
            raise AttributeError(f"Entity {item} not found")

    @property
    def observable(self):
        pass

    def reset(self):
        self.td.clear()
        for entity in self.entity_dict.values():
            entity.initialize()
        self.step = 0

    def update(self, action_td: Optional[TensorDict] = None):
        self.td.set("random", torch.randint(0, 256, self.size, dtype=torch.uint8))
        for name, entity in self.entity_dict.items():
            # TODO: Actions should be set directly to tensor dict
            entity.update(action=action_td.get(name, None) if action_td is not None else None)
        self.step += 1

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

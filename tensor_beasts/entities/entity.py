import abc
from typing import Dict, Optional
import torch
from omegaconf import DictConfig

from tensor_beasts.entities.feature import Feature


class Entity(abc.ABC):
    features: Dict[str, Feature] = {}

    def __init__(self, world: 'World', config: DictConfig):
        self.shared_feature_idx_map = {}
        self.world = world
        self.config = config

    def set_shared_feature_idx(self, feature_name: str, idx: int) -> None:
        self.shared_feature_idx_map[feature_name] = idx

    def get_feature(self, feature_name: str) -> torch.Tensor:
        idx = self.shared_feature_idx_map.get(feature_name)
        if idx is not None:
            return self.world.td["shared_features", feature_name][..., idx]
        else:
            return self.world.td[self.__class__.__name__.lower(), feature_name]

    def set_feature(self, feature_name: str, value: torch.Tensor) -> None:
        idx = self.shared_feature_idx_map.get(feature_name)
        if idx is not None:
            self.world.td["shared_features", feature_name][..., idx] = value
        else:
            self.world.td[self.__class__.__name__.lower(), feature_name] = value

    @abc.abstractmethod
    def update(self, action: Optional[torch.Tensor] = None):
        pass

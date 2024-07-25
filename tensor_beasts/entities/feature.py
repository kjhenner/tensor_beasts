import abc
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class FeatureDefinition:
    name: str
    dtype: torch.dtype
    shape: Tuple[int, ...] = ()
    observable: bool = False
    use_world_size: bool = True
    shared: bool = False


class Feature(abc.ABC):
    _instances = {}
    name: str
    dtype: torch.dtype
    shape: Tuple[int, ...] = ()
    observable: bool = False
    use_world_size: bool = True
    shared: bool = False

    def __new__(cls, *args, **kwargs):
        if not cls.shared or cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]

    def render(self, tensor_data: torch.Tensor):
        pass


class Energy(Feature):
    name = "energy"
    dtype = torch.uint8
    observable = True
    shared = True


class Scent(Feature):
    name = "scent"
    dtype = torch.uint8
    observable = True
    shared = True

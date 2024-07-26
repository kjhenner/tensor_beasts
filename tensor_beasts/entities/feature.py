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

    @staticmethod
    def render(tensor_data: torch.Tensor):
        if tensor_data.ndim == 2:
            return tensor_data.unsqueeze(-1).expand(-1, -1, 3)
        else:
            return tensor_data


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

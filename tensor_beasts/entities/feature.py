import abc
from typing import Tuple, Set, Optional

import torch
from tensordict import TensorDict, NestedKey


class Feature(abc.ABC):
    name: str
    shape: Tuple[int, ...] = None
    dtype: torch.dtype = None
    default_tags: Set[str, ...] = None

    def __init__(
        self,
        td: TensorDict,
        key_prefix: NestedKey,
        shape_prefix: Optional[Tuple[int, ...]] = tuple(),
        additional_tags: Optional[Tuple[str, ...]] = None
    ):
        self.td = td
        self.shape = shape_prefix + self.shape
        self.key = key_prefix + (self.name,)
        self.tags = set(additional_tags).update(self.default_tags or set())

    def render(self):
        if self.data.ndim == 2:
            return self.data.unsqueeze(-1).expand(-1, -1, 3)
        else:
            return self.data

    @property
    def data(self):
        return self.td.get(self.key)

    @data.setter
    def data(self, value):
        self.td.set(self.key, value)


class SharedFeature(Feature):
    _count = 0

    def __init__(
        self,
        td: TensorDict,
        key_prefix: NestedKey = "shared_features",
        additional_tags: Tuple[str, ...] = None
    ):
        super().__init__(td, key_prefix, shape, additional_tags)
        self.idx = SharedFeature._count
        SharedFeature._count += 1

    @property
    def data(self):
        return self.td[self.key][self.idx]

    @data.setter
    def data(self, value):
        self.td[self.key][self.idx] = value


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

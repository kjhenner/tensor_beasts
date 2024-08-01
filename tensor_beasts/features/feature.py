import abc
from typing import Tuple, Set, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict, NestedKey

OmegaConf.register_new_resolver(
    "key",
    lambda *args: tuple(args)
)


class Feature(abc.ABC):
    name: str
    shape: Tuple[int, ...] = None
    dtype: torch.dtype = None
    default_tags: Set[str] = None
    default_config: DictConfig = None

    def __init__(
        self,
        td: TensorDict,
        key_prefix: NestedKey,
        shape_prefix: Optional[Tuple[int, ...]] = tuple(),
        additional_tags: Optional[Tuple[str, ...]] = None,
        config: Optional[DictConfig] = None
    ):
        self.config = self.default_config
        if config:
            self.config.update(config)
        self.td = td
        if key_prefix not in td:
            td[key_prefix] = TensorDict({}, batch_size=[])
        self.shape = tuple(shape_prefix + tuple(self.shape or ()))
        self.key = (*key_prefix, self.name) if isinstance(key_prefix, tuple) else (key_prefix, self.name)
        self.tags = set(additional_tags or ()).update(self.default_tags or set())

    def render(self):
        if self.data.ndim == 2:
            return self.data.unsqueeze(-1).expand(-1, -1, 3)
        else:
            return self.data

    def update(self, step: int):
        pass

    def zero_init(self):
        self.data = torch.zeros(self.shape, dtype=self.dtype)

    def initialize_data(self, *args, **kwargs):
        self.zero_init()

    @property
    def data(self):
        return self.td.get(self.key)

    @data.setter
    def data(self, value):
        self.td.set(self.key, value)


class SharedFeature(Feature, abc.ABC):
    _count = 0
    _key_prefix = None
    _is_parent = False

    def __init__(
        self,
        td: TensorDict,
        is_parent: bool = False,
        key_prefix: NestedKey = "shared_features",
        shape_prefix: Tuple[int, ...] = tuple(),
        additional_tags: Tuple[str, ...] = None,
        config: Optional[DictConfig] = None
    ):
        super().__init__(td, key_prefix, shape_prefix, additional_tags, config)
        self._is_parent = is_parent
        if not self._is_parent:
            self.idx = type(self)._count
            type(self)._count += 1
        if not self._key_prefix:
            self._key_prefix = key_prefix
        else:
            assert self._key_prefix == key_prefix, "SharedFeature key_prefix must be the same for all instances."

    def zero_init(self):
        if self.key not in self.td:
            self.td[self.key] = torch.zeros(self.shape, dtype=self.dtype)
        self.data = torch.zeros(self.shape[:-1], dtype=self.dtype)

    @property
    def data(self):
        if self._is_parent:
            return self.td[self.key]
        return self.td[self.key][self.idx]

    @data.setter
    def data(self, value):
        if self._is_parent:
            self.td[self.key] = value
        else:
            self.td[self.key][self.idx] = value

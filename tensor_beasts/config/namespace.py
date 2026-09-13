"""Plain-attribute config objects for the simulation hot path.

Config values are read on every update step. Resolving them through OmegaConf
at access time means re-running the interpolation grammar parser inside the
simulation loop, which dominated a measurable fraction of step time. Configs
are therefore resolved exactly once at construction and frozen into plain
Python objects here.
"""

from types import SimpleNamespace
from typing import Any, Dict

from omegaconf import DictConfig, ListConfig, OmegaConf


class ConfigNamespace(SimpleNamespace):
    """SimpleNamespace with the read-only parts of the mapping interface.

    Config used to be an OmegaConf ``DictConfig``, so call sites treat it both
    as an object and as a mapping. Both styles keep working here, without the
    per-access interpolation cost.
    """

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def __contains__(self, key: object) -> bool:
        return key in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self) -> int:
        return len(self.__dict__)

    def __getitem__(self, key: str) -> Any:
        try:
            return self.__dict__[key]
        except KeyError:
            raise KeyError(key) from None


def _convert(value: Any) -> Any:
    if isinstance(value, dict):
        # Dicts with non-string keys (like navigation_weights with tuple keys)
        # can't become attributes, so they stay dicts.
        if all(isinstance(k, str) for k in value):
            return to_namespace(value)
        return {k: _convert(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_convert(v) for v in value]
        # An all-string sequence is a TensorDict nested key; tuples are what
        # TensorDict expects and they're hashable, so normalize to tuple.
        if items and all(isinstance(v, str) for v in items):
            return tuple(items)
        return items
    return value


def to_namespace(d: Dict[str, Any]) -> ConfigNamespace:
    """Recursively convert a plain dict to a ConfigNamespace."""
    return ConfigNamespace(**{k: _convert(v) for k, v in d.items()})


def resolve_to_namespace(config: Any) -> ConfigNamespace:
    """Fully resolve an OmegaConf config (or plain dict) into a ConfigNamespace.

    Interpolations such as ``${key:terrain,elevation}`` are resolved here, once,
    rather than on every attribute access during the simulation loop.
    """
    if isinstance(config, (DictConfig, ListConfig)):
        config = OmegaConf.to_container(config, resolve=True)
    return to_namespace(config or {})

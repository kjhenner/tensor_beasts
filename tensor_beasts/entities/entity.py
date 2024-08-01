import abc
from typing import Optional, Dict, Type, Union, ClassVar, get_type_hints
import torch
from omegaconf import DictConfig

from tensor_beasts.features.feature import Feature, SharedFeature


class EntityMeta(abc.ABCMeta):
    def __new__(mcs, name, bases, namespace):
        features: Dict[str, Type[Feature]] = {}
        for base in bases:
            if hasattr(base, '__features__'):
                features.update(base.__features__)

        annotations = namespace.get('__annotations__', {})

        for key, value in annotations.items():
            if isinstance(value, type) and issubclass(value, Feature):
                features[key] = value

        namespace['__features__'] = features
        return super().__new__(mcs, name, bases, namespace)


class Entity(abc.ABC, metaclass=EntityMeta):
    __features__: ClassVar[Dict[str, Type[Feature]]]
    default_config = DictConfig({})

    def __init__(self, world: 'World', config: DictConfig):
        self.world = world
        self.config = self.default_config
        self.config.update(config)

        for feature_name, feature_class in self.__features__.items():
            if issubclass(feature_class, SharedFeature):
                feature = feature_class(
                    td=self.world.td,
                    shape_prefix=self.world.config.size,
                    config=self.config.get(feature_name, None)
                )
            else:
                feature = feature_class(
                    td=self.world.td,
                    key_prefix=(self.__class__.__name__.lower(),),
                    shape_prefix=self.world.config.size,
                    config=self.config.get(feature_name, None)
                )
            setattr(self, feature_name, feature)

    def initialize(self):
        pass

    def update(self, action: Optional[torch.Tensor] = None):
        pass

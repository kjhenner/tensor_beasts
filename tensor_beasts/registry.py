"""
Entity and Feature Registry

Provides automatic registration of entities and features via decorators,
eliminating the need for manual registry maintenance.

Usage:
    @register_feature
    class MyFeature(Feature):
        name = "my_feature"
        ...

    @register_entity
    class MyEntity(Entity):
        my_feature: MyFeature
        ...
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Type, TYPE_CHECKING

import torch
from omegaconf import DictConfig

if TYPE_CHECKING:
    from tensor_beasts.features.feature import Feature
    from tensor_beasts.entities.entity import Entity


class RegistryError(Exception):
    """Raised when registry operations fail."""
    pass


@dataclass
class FeatureInfo:
    """Metadata about a registered feature."""
    name: str
    cls: Type["Feature"]
    dtype: Optional[torch.dtype] = None
    default_config: Optional[DictConfig] = None
    dependencies: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_class(cls, feature_cls: Type["Feature"]) -> "FeatureInfo":
        """Extract metadata from a Feature class."""
        return cls(
            name=feature_cls.name,
            cls=feature_cls,
            dtype=getattr(feature_cls, "dtype", None),
            default_config=getattr(feature_cls, "default_config", None),
            dependencies=getattr(feature_cls, "depends_on", {}),
        )


@dataclass
class EntityInfo:
    """Metadata about a registered entity."""
    name: str
    cls: Type["Entity"]
    features: Dict[str, FeatureInfo] = field(default_factory=dict)

    @classmethod
    def from_class(cls, entity_cls: Type["Entity"]) -> "EntityInfo":
        """Extract metadata from an Entity class."""
        features = {}
        # EntityMeta populates __features__ with feature classes
        for feature_name, feature_cls in getattr(entity_cls, "__features__", {}).items():
            features[feature_name] = FeatureInfo.from_class(feature_cls)

        return cls(
            name=entity_cls.__name__,
            cls=entity_cls,
            features=features,
        )


class FeatureRegistry:
    """Registry for Feature classes."""

    def __init__(self):
        self._features: Dict[str, FeatureInfo] = {}

    def register(self, cls: Type["Feature"]) -> Type["Feature"]:
        """
        Decorator to register a Feature class.

        Usage:
            @feature_registry.register
            class MyFeature(Feature):
                name = "my_feature"
        """
        if not hasattr(cls, "name") or not cls.name:
            raise RegistryError(f"Feature class {cls.__name__} must have a 'name' attribute")

        name = cls.name
        if name in self._features:
            existing = self._features[name].cls
            # Allow re-registration of the same class (module reloads)
            if existing is not cls:
                raise RegistryError(
                    f"Feature name '{name}' already registered by {existing.__name__}. "
                    f"Cannot register {cls.__name__} with the same name."
                )

        self._features[name] = FeatureInfo.from_class(cls)
        return cls

    def get(self, name: str) -> FeatureInfo:
        """Get feature info by name."""
        if name not in self._features:
            available = ", ".join(sorted(self._features.keys()))
            raise RegistryError(
                f"Unknown feature '{name}'. Registered features: {available}"
            )
        return self._features[name]

    def exists(self, name: str) -> bool:
        """Check if a feature is registered."""
        return name in self._features

    def all(self) -> Dict[str, FeatureInfo]:
        """Get all registered features."""
        return dict(self._features)

    def names(self) -> Set[str]:
        """Get all registered feature names."""
        return set(self._features.keys())

    def clear(self):
        """Clear all registrations (for testing)."""
        self._features.clear()


class EntityRegistry:
    """Registry for Entity classes."""

    def __init__(self):
        self._entities: Dict[str, EntityInfo] = {}

    def register(self, cls: Type["Entity"]) -> Type["Entity"]:
        """
        Decorator to register an Entity class.

        Usage:
            @entity_registry.register
            class MyEntity(Entity):
                my_feature: MyFeature
        """
        name = cls.__name__
        if name in self._entities:
            existing = self._entities[name].cls
            # Allow re-registration of the same class (module reloads)
            if existing is not cls:
                raise RegistryError(
                    f"Entity '{name}' already registered. "
                    f"Cannot register another class with the same name."
                )

        self._entities[name] = EntityInfo.from_class(cls)
        return cls

    def get(self, name: str) -> EntityInfo:
        """Get entity info by name."""
        if name not in self._entities:
            available = ", ".join(sorted(self._entities.keys()))
            raise RegistryError(
                f"Unknown entity '{name}'. Registered entities: {available}"
            )
        return self._entities[name]

    def exists(self, name: str) -> bool:
        """Check if an entity is registered."""
        return name in self._entities

    def all(self) -> Dict[str, EntityInfo]:
        """Get all registered entities."""
        return dict(self._entities)

    def names(self) -> Set[str]:
        """Get all registered entity names."""
        return set(self._entities.keys())

    def get_features(self, entity_name: str) -> Dict[str, FeatureInfo]:
        """Get all features for an entity."""
        return self.get(entity_name).features

    def clear(self):
        """Clear all registrations (for testing)."""
        self._entities.clear()


# Module-level singleton instances
feature_registry = FeatureRegistry()
entity_registry = EntityRegistry()

# Convenience decorator aliases
register_feature = feature_registry.register
register_entity = entity_registry.register


def _ensure_entities_imported() -> None:
    """
    Ensure all entity modules are imported so decorators have run.

    This is called lazily when get_entity_class is first used.
    """
    # Import the entities module which imports all entity submodules
    # This triggers the @register_entity decorators to run
    from tensor_beasts import entities  # noqa: F401


def get_entity_class(name: str) -> Type["Entity"]:
    """
    Get an entity class by name.

    Backward-compatible function that delegates to entity_registry.

    Args:
        name: Entity name (e.g., "Terrain", "Plant")

    Returns:
        The Entity class

    Raises:
        ValueError: If entity name is not registered
    """
    # Ensure entities are imported (decorators have run)
    _ensure_entities_imported()

    try:
        return entity_registry.get(name).cls
    except RegistryError as e:
        # Re-raise as ValueError for backward compatibility
        raise ValueError(str(e)) from e


def entity_name_to_registry_name(name: str) -> str:
    """
    Convert lowercase entity name to PascalCase registry name.

    Used for ${key:terrain,feature} -> "Terrain" lookup.

    Args:
        name: Lowercase entity name (e.g., "terrain", "diffusion_toy", "simpleterrain")

    Returns:
        PascalCase entity name (e.g., "Terrain", "DiffusionToy", "SimpleTerrain")
    """
    # First, try exact match in registry (case-insensitive)
    _ensure_entities_imported()
    for registry_name in entity_registry.names():
        if registry_name.lower() == name.lower():
            return registry_name

    # Fall back to snake_case conversion: diffusion_toy -> DiffusionToy
    parts = name.split("_")
    return "".join(part.capitalize() for part in parts)

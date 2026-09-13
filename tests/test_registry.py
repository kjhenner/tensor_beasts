"""Tests for the entity and feature registry system."""

import pytest
import torch
from omegaconf import DictConfig

from tensor_beasts.registry import (
    FeatureRegistry,
    EntityRegistry,
    FeatureInfo,
    EntityInfo,
    RegistryError,
    feature_registry,
    entity_registry,
    get_entity_class,
    entity_name_to_registry_name,
)
from tensor_beasts.features.feature import Feature
from tensor_beasts.entities.entity import Entity


class TestFeatureRegistry:
    """Tests for FeatureRegistry class."""

    def test_register_feature_decorator(self):
        """Test that @register_feature decorator registers a feature."""
        registry = FeatureRegistry()

        @registry.register
        class TestFeature(Feature):
            name = "test_feature"
            dtype = torch.float32

        assert registry.exists("test_feature")
        info = registry.get("test_feature")
        assert info.cls is TestFeature
        assert info.dtype == torch.float32

    def test_register_feature_without_name_raises(self):
        """Test that registering a feature without a name raises an error."""
        registry = FeatureRegistry()

        with pytest.raises(RegistryError, match="must have a 'name' attribute"):
            @registry.register
            class BadFeature(Feature):
                pass  # No name attribute

    def test_register_duplicate_name_raises(self):
        """Test that registering two features with the same name raises an error."""
        registry = FeatureRegistry()

        @registry.register
        class Feature1(Feature):
            name = "duplicate_name"
            dtype = torch.float32

        with pytest.raises(RegistryError, match="already registered"):
            @registry.register
            class Feature2(Feature):
                name = "duplicate_name"
                dtype = torch.float32

    def test_get_unknown_feature_raises(self):
        """Test that getting an unknown feature raises an error."""
        registry = FeatureRegistry()

        with pytest.raises(RegistryError, match="Unknown feature"):
            registry.get("nonexistent")

    def test_feature_info_from_class(self):
        """Test FeatureInfo.from_class extracts correct metadata."""
        class TestFeature(Feature):
            name = "test_feature"
            dtype = torch.int32
            default_config = DictConfig({"key": "value"})
            depends_on = {"dep": "other_feature"}

        info = FeatureInfo.from_class(TestFeature)
        assert info.name == "test_feature"
        assert info.dtype == torch.int32
        assert info.dependencies == {"dep": "other_feature"}


class TestEntityRegistry:
    """Tests for EntityRegistry class."""

    def test_register_entity_decorator(self):
        """Test that @register_entity decorator registers an entity."""
        registry = EntityRegistry()

        @registry.register
        class TestEntity(Entity):
            pass

        assert registry.exists("TestEntity")
        info = registry.get("TestEntity")
        assert info.cls is TestEntity

    def test_register_duplicate_entity_raises(self):
        """Test that registering two entities with the same name raises an error."""
        registry = EntityRegistry()

        @registry.register
        class DuplicateEntity(Entity):
            pass

        with pytest.raises(RegistryError, match="already registered"):
            @registry.register
            class DuplicateEntity(Entity):  # noqa: F811
                pass

    def test_get_unknown_entity_raises(self):
        """Test that getting an unknown entity raises an error."""
        registry = EntityRegistry()

        with pytest.raises(RegistryError, match="Unknown entity"):
            registry.get("NonexistentEntity")


class TestGlobalRegistries:
    """Tests for the global singleton registries."""

    def test_all_entities_registered(self):
        """Test that all expected entities are registered in the global registry."""
        expected_entities = {
            "Terrain", "SimpleTerrain",
            "HydrodynamicPlant", "SimplePlant",
            "Herbivore", "Predator", "DiffusionToy", "Animal"
        }
        registered = entity_registry.names()

        for entity in expected_entities:
            assert entity in registered, f"Entity {entity} not registered"

    def test_all_features_registered(self):
        """Test that all expected features are registered in the global registry."""
        expected_features = {
            "elevation", "aquifer_elevation", "soil_volume",
            "soil_water_volume", "surface_water_volume",
            "simple_water", "oscillator",
            "energy", "scent", "seed", "crowding",
            "id", "offspring_count", "fluid_density"
        }
        registered = feature_registry.names()

        for feature in expected_features:
            assert feature in registered, f"Feature {feature} not registered"

    def test_terrain_features_discovered(self):
        """Test that Terrain entity's features are correctly discovered."""
        features = entity_registry.get_features("Terrain")
        expected = {"elevation", "aquifer_elevation", "soil_volume",
                    "soil_water_volume", "surface_water_volume", "nutrients"}

        assert set(features.keys()) == expected

    def test_get_entity_class_returns_correct_class(self):
        """Test get_entity_class returns the correct class."""
        from tensor_beasts.entities.terrain import Terrain

        cls = get_entity_class("Terrain")
        assert cls is Terrain

    def test_get_entity_class_unknown_raises_valueerror(self):
        """Test get_entity_class raises ValueError for unknown entity."""
        with pytest.raises(ValueError, match="Unknown entity"):
            get_entity_class("NonexistentEntity")


class TestEntityNameConversion:
    """Tests for entity name conversion utilities."""

    def test_simple_conversion(self):
        """Test simple lowercase to PascalCase conversion."""
        assert entity_name_to_registry_name("terrain") == "Terrain"
        assert entity_name_to_registry_name("plant") == "Plant"

    def test_snake_case_conversion(self):
        """Test snake_case to PascalCase conversion."""
        assert entity_name_to_registry_name("diffusion_toy") == "DiffusionToy"
        assert entity_name_to_registry_name("my_custom_entity") == "MyCustomEntity"

    def test_already_pascal_case(self):
        """Test that already PascalCase names are handled."""
        # Note: This capitalizes each part, so "Terrain" -> "Terrain"
        assert entity_name_to_registry_name("Terrain") == "Terrain"

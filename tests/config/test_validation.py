"""Tests for config validation, especially ${key:...} reference validation."""

import pytest
from omegaconf import OmegaConf

from tensor_beasts.config.load import (
    register_resolvers,
    _find_key_references,
    _validate_key_references,
    ConfigValidationError,
    load_config,
)


# Ensure resolvers are registered for all tests
@pytest.fixture(autouse=True)
def setup_resolvers():
    register_resolvers()


class TestFindKeyReferences:
    """Tests for _find_key_references function."""

    def test_find_simple_reference(self):
        """Test finding a simple ${key:...} reference."""
        config = OmegaConf.create({
            "feature": {
                "elevation_key": "${key:terrain,elevation}"
            }
        })

        refs = _find_key_references(config)
        assert len(refs) == 1
        path, entity, feature = refs[0]
        assert entity == "terrain"
        assert feature == "elevation"
        assert "elevation_key" in path

    def test_find_nested_references(self):
        """Test finding references in nested config structures."""
        config = OmegaConf.create({
            "world": {
                "entities": {
                    "Terrain": {
                        "soil_volume": {
                            "elevation_key": "${key:terrain,elevation}",
                            "outflow_key": "${key:terrain,surface_outflow}"
                        }
                    }
                }
            }
        })

        refs = _find_key_references(config)
        assert len(refs) == 2

        entities = [r[1] for r in refs]
        features = [r[2] for r in refs]

        assert "terrain" in entities
        assert "elevation" in features
        assert "surface_outflow" in features

    def test_find_references_in_list(self):
        """Test finding references inside lists."""
        config = OmegaConf.create({
            "targets": ["${key:plant,scent}", "${key:herbivore,scent}"]
        })

        refs = _find_key_references(config)
        assert len(refs) == 2

    def test_no_references(self):
        """Test config with no references returns empty list."""
        config = OmegaConf.create({
            "simple": {
                "value": 42,
                "string": "hello"
            }
        })

        refs = _find_key_references(config)
        assert len(refs) == 0


class TestValidateKeyReferences:
    """Tests for _validate_key_references function."""

    def test_valid_references_pass(self):
        """Test that valid references don't raise errors."""
        config = OmegaConf.create({
            "world": {
                "entities": {
                    "Terrain": {
                        "soil_volume": {
                            "elevation_key": "${key:terrain,elevation}"
                        }
                    }
                }
            }
        })

        # Should not raise
        _validate_key_references(config)

    def test_unknown_entity_raises(self):
        """Test that unknown entity in reference raises ConfigValidationError."""
        config = OmegaConf.create({
            "world": {
                "entities": {
                    "Terrain": {
                        "feature": {
                            "key": "${key:nonexistent_entity,feature}"
                        }
                    }
                }
            }
        })

        with pytest.raises(ConfigValidationError) as exc_info:
            _validate_key_references(config)

        assert "nonexistent_entity" in str(exc_info.value)
        assert "Unknown entity" in str(exc_info.value)

    def test_error_suggests_similar_entity(self):
        """Test that error message suggests similar entity names."""
        config = OmegaConf.create({
            "world": {
                "entities": {
                    "Terrain": {
                        "feature": {
                            "key": "${key:terrian,elevation}"  # Typo: terrian
                        }
                    }
                }
            }
        })

        with pytest.raises(ConfigValidationError) as exc_info:
            _validate_key_references(config)

        # Should suggest "terrain" as similar
        assert "terrain" in str(exc_info.value).lower()

    def test_special_keys_allowed(self):
        """Test that special keys like 'random' and 'shared_features' are allowed."""
        config = OmegaConf.create({
            "feature": {
                "random_key": "${key:random,data}",
                "shared_key": "${key:shared_features,energy}"
            }
        })

        # Should not raise
        _validate_key_references(config)

    def test_multiple_errors_collected(self):
        """Test that multiple errors are collected and reported together."""
        config = OmegaConf.create({
            "features": {
                "key1": "${key:bad_entity1,feature}",
                "key2": "${key:bad_entity2,feature}"
            }
        })

        with pytest.raises(ConfigValidationError) as exc_info:
            _validate_key_references(config)

        # Both errors should be in the message
        assert "bad_entity1" in str(exc_info.value)
        assert "bad_entity2" in str(exc_info.value)
        assert len(exc_info.value.errors) == 2


class TestConfigValidationError:
    """Tests for ConfigValidationError class."""

    def test_error_message_format(self):
        """Test that error message is properly formatted."""
        errors = ["Error 1", "Error 2"]
        exc = ConfigValidationError(errors)

        assert "Config validation failed" in str(exc)
        assert "Error 1" in str(exc)
        assert "Error 2" in str(exc)
        assert exc.errors == errors


class TestAutoGeneratedModels:
    """Tests for auto-generated Pydantic config models."""

    def test_generate_feature_config_model(self):
        """Test that feature config models are correctly generated."""
        from tensor_beasts.config.models import generate_feature_config_model
        from tensor_beasts.features.terrain_features import SoilWaterVolume

        model = generate_feature_config_model(SoilWaterVolume)

        # Check model has expected fields
        fields = model.model_fields
        assert "infiltration_rate" in fields
        assert "flow_rate" in fields
        assert "soil_volume_key" in fields

    def test_generated_model_validates_correctly(self):
        """Test that generated models validate config correctly."""
        from tensor_beasts.config.models import generate_feature_config_model
        from tensor_beasts.features.terrain_features import SoilWaterVolume

        model = generate_feature_config_model(SoilWaterVolume)

        # Valid config should pass
        valid = {"infiltration_rate": 0.01, "flow_rate": 0.2}
        instance = model(**valid)
        assert instance.infiltration_rate == 0.01

    def test_generated_model_rejects_unknown_keys(self):
        """Test that generated models reject unknown config keys."""
        from pydantic import ValidationError
        from tensor_beasts.config.models import generate_feature_config_model
        from tensor_beasts.features.terrain_features import SoilWaterVolume

        model = generate_feature_config_model(SoilWaterVolume)

        # Invalid config with unknown key should fail
        with pytest.raises(ValidationError):
            model(**{"unknown_key": "value"})

    def test_generate_entity_config_model(self):
        """Test that entity config models are correctly generated."""
        from tensor_beasts.config.models import generate_entity_config_model
        from tensor_beasts.entities.terrain import Terrain

        model = generate_entity_config_model(Terrain)

        # Check model has feature fields
        fields = model.model_fields
        assert "elevation" in fields
        assert "soil_volume" in fields
        assert "soil_water_volume" in fields

    def test_get_or_generate_uses_manual_when_available(self):
        """Test that manual models are preferred over auto-generated."""
        from tensor_beasts.config.models import (
            get_or_generate_entity_config_model,
            TerrainEntityConfig
        )
        from tensor_beasts.config.load import MANUAL_ENTITY_MODELS

        model = get_or_generate_entity_config_model("Terrain", MANUAL_ENTITY_MODELS)

        # Should return the manual model
        assert model is TerrainEntityConfig

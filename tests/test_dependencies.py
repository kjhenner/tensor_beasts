"""Tests for feature dependency resolution and topological sorting."""

import pytest
import torch
from omegaconf import OmegaConf, DictConfig

from tensor_beasts.config.load import register_resolvers
from tensor_beasts.features.feature import Feature
from tensor_beasts.entities.entity import Entity, DependencyCycleError
from tensor_beasts.registry import register_entity, register_feature


# Ensure resolvers are registered
@pytest.fixture(autouse=True)
def setup_resolvers():
    register_resolvers()


class TestFeatureDependencies:
    """Tests for feature dependency declarations."""

    def test_feature_default_no_dependencies(self):
        """Test that features have empty depends_on by default."""
        class TestFeature(Feature):
            name = "test_no_deps"
            dtype = torch.float32

        assert TestFeature.depends_on == {}

    def test_feature_with_dependencies(self):
        """Test feature with declared dependencies."""
        class TestFeature(Feature):
            name = "test_with_deps"
            dtype = torch.float32
            depends_on = {
                "elevation": "elevation",
                "other": "other_feature"
            }

        assert "elevation" in TestFeature.depends_on
        assert TestFeature.depends_on["elevation"] == "elevation"


class TestTopologicalSort:
    """Tests for topological sorting algorithm."""

    def test_simple_linear_chain(self):
        """Test sorting a simple A -> B -> C chain."""
        graph = {
            "A": set(),
            "B": {"A"},
            "C": {"B"}
        }

        result = Entity._topological_sort(graph)

        # A must come before B, B must come before C
        assert result.index("A") < result.index("B")
        assert result.index("B") < result.index("C")

    def test_multiple_roots(self):
        """Test sorting with multiple independent roots."""
        graph = {
            "A": set(),
            "B": set(),
            "C": {"A", "B"}
        }

        result = Entity._topological_sort(graph)

        # Both A and B must come before C
        assert result.index("A") < result.index("C")
        assert result.index("B") < result.index("C")

    def test_diamond_dependency(self):
        """Test sorting a diamond: A -> B, A -> C, B -> D, C -> D."""
        graph = {
            "A": set(),
            "B": {"A"},
            "C": {"A"},
            "D": {"B", "C"}
        }

        result = Entity._topological_sort(graph)

        assert result.index("A") < result.index("B")
        assert result.index("A") < result.index("C")
        assert result.index("B") < result.index("D")
        assert result.index("C") < result.index("D")

    def test_cycle_detection(self):
        """Test that cycles are detected and raise an error."""
        graph = {
            "A": {"B"},
            "B": {"A"}  # Cycle: A -> B -> A
        }

        with pytest.raises(DependencyCycleError, match="Circular dependency"):
            Entity._topological_sort(graph)

    def test_self_cycle_detection(self):
        """Test that self-referential cycles are detected."""
        graph = {
            "A": {"A"}  # Self-cycle
        }

        with pytest.raises(DependencyCycleError):
            Entity._topological_sort(graph)

    def test_empty_graph(self):
        """Test sorting an empty graph."""
        graph = {}
        result = Entity._topological_sort(graph)
        assert result == []

    def test_single_node_no_deps(self):
        """Test sorting a single node with no dependencies."""
        graph = {"A": set()}
        result = Entity._topological_sort(graph)
        assert result == ["A"]


class TestEntityInitializationOrder:
    """Tests for automatic entity initialization ordering."""

    def test_terrain_initialization_order(self):
        """Test that Terrain features are initialized in correct dependency order."""
        from tensor_beasts.entities.terrain import Terrain
        from tensor_beasts.world import World

        config = OmegaConf.create({
            'size': [8, 8],
            'device': 'cpu',
            'entities': {
                'Terrain': {
                    'elevation': {'ramp': {}},
                    'aquifer_elevation': {},
                    'soil_volume': {},
                    'soil_water_volume': {},
                    'surface_water_volume': {}
                }
            }
        })

        world = World(config)
        terrain = world.terrain

        order = terrain._get_initialization_order()

        # Elevation must come first (no dependencies)
        assert order[0] == "elevation"

        # Features that depend on elevation must come after it
        elev_idx = order.index("elevation")
        assert order.index("aquifer_elevation") > elev_idx
        assert order.index("soil_volume") > elev_idx
        assert order.index("surface_water_volume") > elev_idx

        # soil_water_volume depends on soil_volume and surface_water_volume
        swv_idx = order.index("soil_water_volume")
        assert swv_idx > order.index("soil_volume")
        assert swv_idx > order.index("surface_water_volume")

    def test_all_features_initialized(self):
        """Test that all features are properly initialized after world creation."""
        from tensor_beasts.world import World

        config = OmegaConf.create({
            'size': [8, 8],
            'device': 'cpu',
            'entities': {
                'Terrain': {
                    'elevation': {'ramp': {}},
                    'aquifer_elevation': {},
                    'soil_volume': {},
                    'soil_water_volume': {},
                    'surface_water_volume': {}
                }
            }
        })

        world = World(config)
        terrain = world.terrain

        # All features should have initialized data
        for feature_name in terrain.__features__:
            feature = getattr(terrain, feature_name)
            assert feature.data is not None, f"Feature {feature_name} not initialized"
            assert feature.data.shape == (8, 8), f"Feature {feature_name} has wrong shape"


class TestDependencyGraph:
    """Tests for building dependency graphs."""

    def test_build_dependency_graph(self):
        """Test that dependency graph is correctly built from feature depends_on."""
        from tensor_beasts.entities.terrain import Terrain
        from tensor_beasts.world import World

        config = OmegaConf.create({
            'size': [4, 4],
            'device': 'cpu',
            'entities': {
                'Terrain': {
                    'elevation': {'ramp': {}},
                    'aquifer_elevation': {},
                    'soil_volume': {},
                    'soil_water_volume': {},
                    'surface_water_volume': {}
                }
            }
        })

        world = World(config)
        terrain = world.terrain

        graph = terrain._build_dependency_graph()

        # Elevation should have no dependencies
        assert graph["elevation"] == set()

        # AquiferElevation depends on elevation
        assert "elevation" in graph["aquifer_elevation"]

        # SoilWaterVolume depends on multiple features
        assert "soil_volume" in graph["soil_water_volume"]
        assert "surface_water_volume" in graph["soil_water_volume"]


class TestCrossEntityDependencies:
    """Tests for cross-entity dependency handling."""

    def test_single_entity_no_cross_deps(self):
        """Test that a single entity has no cross-entity dependencies."""
        from tensor_beasts.world import World

        config = OmegaConf.create({
            'size': [4, 4],
            'device': 'cpu',
            'entities': {
                'Terrain': {
                    'elevation': {'ramp': {}},
                    'aquifer_elevation': {},
                    'soil_volume': {},
                    'soil_water_volume': {},
                    'surface_water_volume': {}
                }
            }
        })

        world = World(config)
        graph = world._build_entity_dependency_graph()

        # Single entity should have no dependencies
        assert graph == {'Terrain': set()}

    def test_entity_initialization_order_single(self):
        """Test initialization order with single entity."""
        from tensor_beasts.world import World

        config = OmegaConf.create({
            'size': [4, 4],
            'device': 'cpu',
            'entities': {
                'Terrain': {
                    'elevation': {'ramp': {}},
                    'aquifer_elevation': {},
                    'soil_volume': {},
                    'soil_water_volume': {},
                    'surface_water_volume': {}
                }
            }
        })

        world = World(config)
        order = world._get_entity_initialization_order()

        assert order == ['Terrain']

    def test_world_entity_dependency_graph_from_config_refs(self):
        """Test that ${key:entity,feature} refs create entity dependencies."""
        from tensor_beasts.world import World

        # DiffusionToy's fluid_density references diffusiontoy.elevation
        # which is the same entity, so no cross-entity dep
        config = OmegaConf.create({
            'size': [4, 4],
            'device': 'cpu',
            'entities': {
                'DiffusionToy': {
                    'elevation': {'ramp': {}},
                    'fluid_density': {}
                }
            }
        })

        world = World(config)
        graph = world._build_entity_dependency_graph()

        # DiffusionToy should have no external dependencies
        assert graph['DiffusionToy'] == set()

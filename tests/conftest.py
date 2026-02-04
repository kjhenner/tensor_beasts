import pytest
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from tensor_beasts.config.load import register_resolvers
from tensor_beasts.entities import Terrain
from tensor_beasts.world import World

# Register OmegaConf resolvers (e.g., ${key:...}) at module load time
register_resolvers()


@pytest.fixture
def config_overrides():
    """
    This fixture can be overridden in specific test files to provide custom configurations.
    """
    return {}


@pytest.fixture
def default_config(config_overrides) -> DictConfig:
    base_config = OmegaConf.create({
        "size": [3, 3],
        "device": "cpu",
        "entities": {
            "Terrain": {
                "elevation": {"ramp": {}},
                "aquifer_elevation": {},
                "soil_volume": {},
                "soil_water_volume": {
                    "field_capacity": 0.5,
                    "infiltration_rate": 0.0,
                    "flow_rate": 0.0,
                    "evaporation_rate": 0.0
                },
                "surface_water_volume": {
                    "rainfall_rate": 0.0,
                    "flow_rate": 0.0,
                    "steps": 1,
                    "relaxation_factor": 1.0
                }
            }
        }
    })
    return OmegaConf.merge(base_config, config_overrides)


@pytest.fixture
def world(default_config: DictConfig) -> World:
    return World(default_config)


@pytest.fixture
def tensor_dict(world: World) -> TensorDict:
    return world.td


@pytest.fixture
def terrain_entity(world: World) -> Terrain:
    return world.terrain

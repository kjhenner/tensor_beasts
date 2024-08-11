import pytest
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from tensor_beasts.entities import Terrain
from tensor_beasts.world import World


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
            "terrain": {
                "elevation": {},
                "aquifer_elevation": {},
                "soil_volume": {},
                "soil_water_volume": {},
                "surface_water_volume": {},
                "neighbor_distances": {},
                "slopes": {},
                "soil_saturation_gradient": {},
                "surface_outflow": {},
                "subsurface_outflow": {}
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

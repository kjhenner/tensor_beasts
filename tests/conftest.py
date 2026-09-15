import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from tensor_beasts.config.load import register_resolvers
from tensor_beasts.entities import Terrain
from tensor_beasts.world import World

# Register OmegaConf resolvers (e.g., ${key:...}) at module load time
register_resolvers()


@pytest.fixture
def default_device():
    """Run a test on the best available device and put the default back.

    ``torch.set_default_device`` is global process state. Two tests here used to
    set it to "mps" and leave it set, which passed on a Mac and, on any other
    machine, failed those two tests and then three unrelated ones in
    ``test_world_reset.py`` that ran afterwards and allocated on a device that
    was not there. Restoring in a fixture keeps a device-specific test from
    deciding what the rest of the suite runs on.
    """
    previous = torch.get_default_device()
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    torch.set_default_device(device)
    try:
        yield torch.device(device)
    finally:
        torch.set_default_device(previous)


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

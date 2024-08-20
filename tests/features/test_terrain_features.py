import torch
import pytest
from omegaconf import OmegaConf

from tensor_beasts.features.feature import Feature


@pytest.fixture
def config_overrides():
    return OmegaConf.create({
        "entities": {
            "terrain": {
                "elevation": {"ramp": {}},
                "surface_water_volume": {"rainfall_rate": 0.0},
            }
        }
    })


def test_surface_flow(terrain_entity):
    terrain_entity.initialize()
    slope = terrain_entity.slopes
    surface_water = terrain_entity.surface_water_volume
    surface_water.data[1, 1] = 10
    outflow = terrain_entity.surface_outflow
    slope.update(0)
    outflow.update(0)

    surface_water.update(0)

    assert torch.isclose(torch.sum(surface_water.data[:2, ...]), torch.tensor(10.0))
    assert surface_water.data[1, 1] < 10
    assert torch.sum(surface_water.data[2]) == 0


def test_subsurface_flow(terrain_entity):
    terrain_entity.initialize()
    slope = terrain_entity.slopes
    saturation_gradient = terrain_entity.soil_saturation_gradient
    surface_water = terrain_entity.surface_water_volume
    subsurface_water = terrain_entity.soil_water_volume
    subsurface_water.data[1, 1] = 10
    subsurface_water.data[2, 1] = 10
    saturation_gradient.update(0)
    outflow = terrain_entity.subsurface_outflow
    slope.update(0)
    outflow.update(0)

    subsurface_water.update(0)

    assert torch.isclose(torch.sum(subsurface_water.data + surface_water.data), torch.tensor(10.0))
    assert subsurface_water.data[1, 1] < 10
    assert torch.sum(subsurface_water.data[2]) == 0

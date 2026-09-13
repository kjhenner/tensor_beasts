import torch
import pytest
from omegaconf import OmegaConf


@pytest.fixture
def config_overrides():
    return OmegaConf.create({
        "entities": {
            "Terrain": {
                "elevation": {"ramp": {}},
                "surface_water_volume": {"rainfall_rate": 0.0},
            },
        }
    })


def test_surface_outflow_written(terrain_entity):
    terrain_entity.initialize()
    surface_water = terrain_entity.surface_water_volume
    surface_water.data[1, 1] = 10

    surface_water.update(0)

    assert ("terrain", "surface_outflow") in terrain_entity.td


def test_soil_water_saturation_written(terrain_entity):
    terrain_entity.initialize()
    surface_water = terrain_entity.surface_water_volume
    subsurface_water = terrain_entity.soil_water_volume
    subsurface_water.data[1, 1] = 10
    subsurface_water.data[2, 1] = 10

    subsurface_water.update(0)

    assert ("terrain", "soil_water_saturation") in terrain_entity.td
    saturation = terrain_entity.td.get(("terrain", "soil_water_saturation"))
    assert torch.all(saturation >= 0)
    assert torch.all(saturation <= 1)

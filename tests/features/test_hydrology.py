"""Hydrology invariants for the Terrain entity.

The pre-existing terrain tests (tests/features/test_terrain_features.py) run
with every rate set to zero, so they only ever assert that two keys get
written. These exercise the water cycle at rates comparable to the ones the
shipped configs use, where the interesting failures live:

* water is neither created nor destroyed except by rainfall and evaporation,
* surface water runs downhill rather than pooling where it falls,
* the published soil_water_saturation matches the water actually in the soil,
* the nutrient path from plant growth to terrain:nutrients is really wired up,
  and a nutrients_key that points at nothing is reported rather than swallowed.
"""

import logging

import pytest
import torch
from omegaconf import OmegaConf

from tensor_beasts.config import load_config
from tensor_beasts.world import World

SIZE = 24


def build(overrides=None, size=SIZE, seed=0, config_path="conf/terrain_config.yaml"):
    torch.manual_seed(seed)
    config = load_config(config_path)
    config.world.size = [size, size]
    if overrides:
        config.world.entities = OmegaConf.merge(
            config.world.entities, OmegaConf.create(overrides)
        )
    world = World(config.world)
    world.initialize()
    return world


def terrain_water(world):
    surface = world.td.get(("terrain", "surface_water_volume"))
    soil = world.td.get(("terrain", "soil_water_volume"))
    return float(surface.sum()), float(soil.sum())


def test_water_is_conserved_when_there_is_no_rain_and_no_evaporation():
    """Infiltration and both flow steps must only move water, never mint it.

    Run at the code's own default infiltration rate and a real flow rate, not
    at the zeroed rates the other terrain tests use.
    """
    world = build({
        "Terrain": {
            "surface_water_volume": {"rainfall_rate": 0.0},
            "soil_water_volume": {
                "evaporation_rate": 0.0,
                "infiltration_rate": 1e-3,
                "flow_rate": 0.1,
            },
        }
    })
    # Put some surface water on the map so infiltration has something to do.
    surface = world.td.get(("terrain", "surface_water_volume"))
    surface += torch.rand_like(surface)

    start = sum(terrain_water(world))
    for _ in range(60):
        world.update()
    end = sum(terrain_water(world))

    assert end == pytest.approx(start, rel=1e-5), (
        f"total water moved from {start} to {end} with no source or sink"
    )


def test_rainfall_is_the_only_unexplained_gain():
    """With evaporation off, the water gained must equal the rain delivered."""
    rainfall_rate = 1e-3
    steps = 60
    world = build({
        "Terrain": {
            "surface_water_volume": {"rainfall_rate": rainfall_rate},
            "soil_water_volume": {
                "evaporation_rate": 0.0,
                "infiltration_rate": 1e-3,
                "flow_rate": 0.1,
            },
        }
    })

    start = sum(terrain_water(world))
    for _ in range(steps):
        world.update()
    end = sum(terrain_water(world))

    # Rain falls on every cell for the first half of each 1000-step cycle.
    expected = rainfall_rate * SIZE * SIZE * steps
    assert end - start == pytest.approx(expected, rel=1e-4)


def test_surface_water_runs_downhill_on_a_ramp():
    """A ramp runs from elevation 0 at row 0 to 1 at the last row."""
    world = build({
        "Terrain": {"surface_water_volume": {"rainfall_rate": 0.0}},
    })
    elevation = world.td.get(("terrain", "elevation"))
    assert float(elevation[0].mean()) < float(elevation[-1].mean())

    surface = world.td.get(("terrain", "surface_water_volume"))
    surface += 1.0  # uniform sheet of water everywhere
    total = float(surface.sum())

    for _ in range(2 * SIZE):
        world.update()

    per_row = world.td.get(("terrain", "surface_water_volume")).sum(dim=-1)
    # The sheet drains into a level pond against the low edge; the exact number
    # of flooded rows is set by the depth, so assert the shape not the value.
    assert float(per_row[:SIZE // 4].sum()) > 0.95 * total, (
        "water did not collect at the low edge"
    )
    assert float(per_row[SIZE // 2:].sum()) < 0.01 * total, (
        "water is still sitting on the high half of the ramp"
    )
    assert float(per_row[0]) > float(per_row[1]) > float(per_row[2]), (
        "pond is not deepest at the lowest row"
    )


def test_published_saturation_matches_the_soil_water_after_update():
    """Regression: saturation used to be published before evaporation ran."""
    world = build({
        "Terrain": {"soil_water_volume": {"evaporation_rate": 1e-3}},
    })
    for _ in range(5):
        world.update()

    soil_water = world.td.get(("terrain", "soil_water_volume"))
    capacity = world.td.get(("terrain", "soil_volume")) * 0.4
    published = world.td.get(("terrain", "soil_water_saturation"))

    assert torch.allclose(published, soil_water / capacity, atol=1e-6)


def test_plant_growth_consumes_terrain_nutrients():
    """The nutrients_key path must actually debit terrain:nutrients.

    Growth probability is `ideal_growth_rate * (1 - |sat - ideal| / tol)`, which
    is zero unless soil saturation sits inside the tolerance window. The shipped
    configs never got there, which is why nutrients looked inert.
    """
    world = build({
        "Terrain": {"surface_water_volume": {"rainfall_rate": 0.0}},
        "HydrodynamicPlant": {"ideal_growth_rate": 0.5, "init_prob": 1.0},
    })
    # Park the soil exactly at the plant's ideal saturation.
    capacity = world.td.get(("terrain", "soil_volume")) * 0.4
    world.td.set(
        ("terrain", "soil_water_volume"),
        capacity * world.HydrodynamicPlant.config.ideal_soil_saturation,
        inplace=True,
    )

    before = float(world.td.get(("terrain", "nutrients")).sum())
    world.update()
    after = float(world.td.get(("terrain", "nutrients")).sum())

    assert after < before, "plant growth did not consume any nutrients"


def test_missing_nutrients_key_is_reported_not_swallowed(caplog):
    """Regression: a dangling nutrients_key silently disabled nutrient limits.

    The old code wrapped the whole block in `except KeyError: pass`, so a typo
    (or pointing at an entity with no nutrients feature, as
    conf/toy_zoo/single_herbivore.yaml does) left growth unlimited with no sign
    that anything was wrong.
    """
    from tensor_beasts.entities import plant as plant_module

    world = build({
        "HydrodynamicPlant": {"nutrients_key": "terrain:no_such_feature"},
    })
    plant_module._WARNED_NUTRIENT_KEYS.clear()

    with caplog.at_level(logging.WARNING, logger=plant_module.__name__):
        world.update()

    assert any("no_such_feature" in record.getMessage() for record in caplog.records), (
        caplog.text or "no warning was logged for a dangling nutrients_key"
    )

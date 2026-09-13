"""Semantics of perform_move: movement, reproduction and carried features.

These pin down behaviour that was previously wrong in ways the simulation did
not visibly complain about. In particular, carried-feature functions are
applied once and shared across all four directions, so they must be pure; when
one of them mutated its argument in place the effect landed on every cell in
the grid, on every step, whether or not anything moved there.
"""

import pytest
import torch

from tensor_beasts.entities.helpers.animal_helpers import perform_move


GRID = (5, 5)
UP, DOWN, LEFT, RIGHT = 1, 2, 3, 4


def empty_masks():
    # Direction masks are 0/1 uint8, as get_direction_masks produces them.
    return {d: torch.zeros(GRID, dtype=torch.uint8) for d in range(1, 5)}


def test_non_reproducing_move_transfers_energy_and_vacates_origin():
    energy = torch.zeros(GRID)
    energy[2, 2] = 100

    masks = empty_masks()
    masks[UP][2, 2] = 1  # pad_matrix direction 1 shifts content toward lower row index

    perform_move(entity_energy=energy, direction_masks=masks, divide_threshold=250)

    assert energy[2, 2] == 0, "origin should be vacated when not reproducing"
    assert energy.sum() == 100, "energy should be conserved by a plain move"
    assert energy[1, 2] == 100


def test_reproducing_move_splits_energy_between_origin_and_destination():
    energy = torch.zeros(GRID)
    energy[2, 2] = 200

    masks = empty_masks()
    masks[UP][2, 2] = 1

    perform_move(
        entity_energy=energy,
        direction_masks=masks,
        divide_threshold=100,  # 200 > 100, so this mover reproduces
        divide_fn_self=lambda x: x * 0.5,
        divide_fn_offspring=lambda x: x * 0.5,
    )

    assert energy[1, 2] == 100, "half the energy should move"
    assert energy[2, 2] == 100, "half should stay behind as offspring"


def test_carried_feature_increments_once_per_reproduction():
    """The regression this file exists for.

    offspring_count used to be incremented four times per step, in place, across
    the entire grid, because its function was impure and was called once per
    direction. Cells with no animal in them ended up with an offspring count.
    """
    energy = torch.zeros(GRID)
    energy[2, 2] = 200
    offspring_count = torch.zeros(GRID)

    masks = empty_masks()
    masks[UP][2, 2] = 1

    perform_move(
        entity_energy=energy,
        direction_masks=masks,
        divide_threshold=100,
        divide_fn_self=lambda x: x * 0.5,
        divide_fn_offspring=lambda x: x * 0.5,
        carried_features_self=[offspring_count],
        carried_feature_fns_self=[lambda x: x + 1],
    )

    assert offspring_count[1, 2] == 1, "the reproducing mover carries a count of exactly 1"
    assert offspring_count.sum() == 1, "no other cell should be touched"


def test_uninvolved_cells_are_untouched():
    energy = torch.zeros(GRID)
    energy[0, 0] = 77  # a stationary animal, in no direction mask
    offspring_count = torch.zeros(GRID)

    masks = empty_masks()  # nothing moves at all

    perform_move(
        entity_energy=energy,
        direction_masks=masks,
        divide_threshold=10,
        carried_features_self=[offspring_count],
        carried_feature_fns_self=[lambda x: x + 1],
    )

    assert energy[0, 0] == 77, "a cell that did not move must be unchanged"
    assert offspring_count.sum() == 0, "no move means no carried-feature change anywhere"


def test_result_does_not_depend_on_carried_feature_order():
    """divide_feature is normally biomass, which is itself a carried feature.

    Reproduction used to be re-decided per feature, after earlier features had
    already mutated biomass, so the answer depended on list order.
    """
    def run(order):
        energy = torch.zeros(GRID)
        energy[2, 2] = 200
        biomass = torch.zeros(GRID)
        biomass[2, 2] = 200
        marker = torch.zeros(GRID)
        marker[2, 2] = 10

        masks = empty_masks()
        masks[UP][2, 2] = 1

        carried = {"biomass": biomass, "marker": marker}
        fns = {
            "biomass": lambda x: x * 0.5,
            "marker": lambda x: x + 1,
        }
        perform_move(
            entity_energy=energy,
            direction_masks=masks,
            divide_threshold=100,
            divide_feature=biomass,
            divide_fn_self=lambda x: x * 0.5,
            divide_fn_offspring=lambda x: x * 0.5,
            carried_features_self=[carried[k] for k in order],
            carried_feature_fns_self=[fns[k] for k in order],
        )
        return energy.clone(), biomass.clone(), marker.clone()

    forward = run(["biomass", "marker"])
    reverse = run(["marker", "biomass"])

    for a, b in zip(forward, reverse):
        assert torch.equal(a, b), "perform_move must not depend on carried feature order"


def test_multiple_arrivals_at_one_cell_saturate_at_255():
    """Four movers converging on one cell carry 800 between them; an animal
    holds at most 255, so the total saturates there."""
    energy = torch.zeros(GRID)
    masks = empty_masks()
    # All four neighbours of (2, 2) move into it.
    energy[3, 2] = 200
    masks[UP][3, 2] = 1
    energy[1, 2] = 200
    masks[DOWN][1, 2] = 1
    energy[2, 3] = 200
    masks[LEFT][2, 3] = 1
    energy[2, 1] = 200
    masks[RIGHT][2, 1] = 1

    perform_move(entity_energy=energy, direction_masks=masks, divide_threshold=250)

    assert energy[2, 2] == 255, "the total should saturate at the ceiling"


def test_carried_feature_fn_is_applied_exactly_once():
    """The direct contract test.

    fn used to be invoked once per direction. With a pure fn that is merely
    wasteful, but callers pass fns built on helpers that default to in-place, so
    the extra calls changed the simulation. Pin the call count directly rather
    than relying on a caller happening to be impure.
    """
    calls = {"self": 0, "offspring": 0}

    def counting_self(x):
        calls["self"] += 1
        return x

    def counting_offspring(x):
        calls["offspring"] += 1
        return x

    energy = torch.zeros(GRID)
    energy[2, 2] = 200
    carried = torch.zeros(GRID)

    masks = empty_masks()
    masks[UP][2, 2] = 1

    perform_move(
        entity_energy=energy,
        direction_masks=masks,
        divide_threshold=100,
        carried_features_self=[carried],
        carried_feature_fns_self=[counting_self],
        carried_features_offspring=[carried],
        carried_feature_fns_offspring=[counting_offspring],
    )

    assert calls["self"] == 1, f"carried self fn called {calls['self']} times, expected 1"
    assert calls["offspring"] == 1, f"carried offspring fn called {calls['offspring']} times, expected 1"



def build_small_world(seed=0, size=8):
    from tensor_beasts.config import load_config
    from tensor_beasts.world import World

    torch.manual_seed(seed)
    config = load_config("conf/basic_config.yaml")
    config.world.size = [size, size]
    world = World(config.world)
    world.initialize()
    return world


def test_energy_gained_is_exactly_rate_times_efficiency():
    """No staircase. When energy was uint8, a rate of 2.05 burned int(2.05) = 2
    biomass for int(6.975) = 6 energy while the basal rate of exactly 2 got 7,
    a 14% tax on every setting except one exact value. Now 2.05 biomass buys
    2.05 * efficiency(2.05) energy, to float precision."""
    world = build_small_world()
    herbivore = world.entity_dict["Herbivore"]
    herbivore.biomass.data[:] = 200
    herbivore.energy.data[:] = 0
    rate = torch.full((8, 8), 2.05)
    efficiency = float(herbivore._compute_efficiency(rate)[0, 0])
    herbivore._execute_metabolism(rate)
    assert herbivore.energy.data.dtype == torch.float32
    assert float(herbivore.energy.data[0, 0]) == pytest.approx(2.05 * efficiency, abs=1e-5)
    assert float(herbivore.biomass.data[0, 0]) == pytest.approx(200 - 2.05, abs=1e-5)


def test_biomass_burned_at_a_fractional_rate_is_exact():
    """A metabolic rate of 2.5 costs 2.5 biomass, not the 2 that truncation charged."""
    world = build_small_world()
    herbivore = world.entity_dict["Herbivore"]
    herbivore.biomass.data[:] = 100
    herbivore.energy.data[:] = 10
    herbivore._execute_metabolism(torch.full((8, 8), 2.5))
    assert float(herbivore.biomass.data[0, 0]) == pytest.approx(97.5, abs=1e-5)


def test_energy_and_biomass_stay_within_0_255_over_many_steps():
    """Float removed the saturating arithmetic; the clamps must hold instead.
    Checks animals and the plants that share the Energy tensor, and the
    carrion layer the dead flow into, every step of a small world."""
    world = build_small_world(seed=0, size=64)
    tensors = {
        "herbivore energy": world.entity_dict["Herbivore"].energy,
        "herbivore biomass": world.entity_dict["Herbivore"].biomass,
        "predator energy": world.entity_dict["Predator"].energy,
        "predator biomass": world.entity_dict["Predator"].biomass,
        "plant energy": world.entity_dict["SimplePlant"].energy,
    }
    carrion = world.td.get(world.entity_dict["Herbivore"].config.carrion_key)
    assert bool((tensors["herbivore biomass"].data > 0).any()), "world spawned no herbivores"
    assert bool((tensors["predator biomass"].data > 0).any()), "world spawned no predators"
    for step in range(300):
        world.update()
        for name, feature in tensors.items():
            data = feature.data
            assert data.dtype == torch.float32, name
            assert float(data.min()) >= 0, f"{name} went negative at step {step}"
            assert float(data.max()) <= 255, f"{name} exceeded 255 at step {step}"
        assert float(carrion.min()) >= 0 and float(carrion.max()) <= 255, step


def test_dissipation_never_drives_energy_below_zero():
    world = build_small_world()
    herbivore = world.entity_dict["Herbivore"]
    herbivore.energy.data[:] = 0.3  # below the dissipation floor of 1
    herbivore._execute_dissipation()
    assert float(herbivore.energy.data.min()) == 0

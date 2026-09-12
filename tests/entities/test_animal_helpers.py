"""Semantics of perform_move: movement, reproduction and carried features.

These pin down behaviour that was previously wrong in ways the simulation did
not visibly complain about. In particular, carried-feature functions are
applied once and shared across all four directions, so they must be pure; when
one of them mutated its argument in place the effect landed on every cell in
the grid, on every step, whether or not anything moved there.
"""

import torch

from tensor_beasts.entities.helpers.animal_helpers import perform_move
from tensor_beasts.util import safe_add


GRID = (5, 5)
UP, DOWN, LEFT, RIGHT = 1, 2, 3, 4


def empty_masks():
    return {d: torch.zeros(GRID, dtype=torch.uint8) for d in range(1, 5)}


def test_non_reproducing_move_transfers_energy_and_vacates_origin():
    energy = torch.zeros(GRID, dtype=torch.uint8)
    energy[2, 2] = 100

    masks = empty_masks()
    masks[UP][2, 2] = 1  # pad_matrix direction 1 shifts content toward lower row index

    perform_move(entity_energy=energy, direction_masks=masks, divide_threshold=250)

    assert energy[2, 2] == 0, "origin should be vacated when not reproducing"
    assert energy.sum() == 100, "energy should be conserved by a plain move"
    assert energy[1, 2] == 100


def test_reproducing_move_splits_energy_between_origin_and_destination():
    energy = torch.zeros(GRID, dtype=torch.uint8)
    energy[2, 2] = 200

    masks = empty_masks()
    masks[UP][2, 2] = 1

    perform_move(
        entity_energy=energy,
        direction_masks=masks,
        divide_threshold=100,  # 200 > 100, so this mover reproduces
        divide_fn_self=lambda x: (x.float() * 0.5).to(x.dtype),
        divide_fn_offspring=lambda x: (x.float() * 0.5).to(x.dtype),
    )

    assert energy[1, 2] == 100, "half the energy should move"
    assert energy[2, 2] == 100, "half should stay behind as offspring"


def test_carried_feature_increments_once_per_reproduction():
    """The regression this file exists for.

    offspring_count used to be incremented four times per step, in place, across
    the entire grid, because its function was impure and was called once per
    direction. Cells with no animal in them ended up with an offspring count.
    """
    energy = torch.zeros(GRID, dtype=torch.uint8)
    energy[2, 2] = 200
    offspring_count = torch.zeros(GRID, dtype=torch.uint8)

    masks = empty_masks()
    masks[UP][2, 2] = 1

    perform_move(
        entity_energy=energy,
        direction_masks=masks,
        divide_threshold=100,
        divide_fn_self=lambda x: (x.float() * 0.5).to(x.dtype),
        divide_fn_offspring=lambda x: (x.float() * 0.5).to(x.dtype),
        carried_features_self=[offspring_count],
        carried_feature_fns_self=[lambda x: safe_add(x, 1, inplace=False)],
    )

    assert offspring_count[1, 2] == 1, "the reproducing mover carries a count of exactly 1"
    assert offspring_count.sum() == 1, "no other cell should be touched"


def test_uninvolved_cells_are_untouched():
    energy = torch.zeros(GRID, dtype=torch.uint8)
    energy[0, 0] = 77  # a stationary animal, in no direction mask
    offspring_count = torch.zeros(GRID, dtype=torch.uint8)

    masks = empty_masks()  # nothing moves at all

    perform_move(
        entity_energy=energy,
        direction_masks=masks,
        divide_threshold=10,
        carried_features_self=[offspring_count],
        carried_feature_fns_self=[lambda x: safe_add(x, 1, inplace=False)],
    )

    assert energy[0, 0] == 77, "a cell that did not move must be unchanged"
    assert offspring_count.sum() == 0, "no move means no carried-feature change anywhere"


def test_result_does_not_depend_on_carried_feature_order():
    """divide_feature is normally biomass, which is itself a carried feature.

    Reproduction used to be re-decided per feature, after earlier features had
    already mutated biomass, so the answer depended on list order.
    """
    def run(order):
        energy = torch.zeros(GRID, dtype=torch.uint8)
        energy[2, 2] = 200
        biomass = torch.zeros(GRID, dtype=torch.uint8)
        biomass[2, 2] = 200
        marker = torch.zeros(GRID, dtype=torch.uint8)
        marker[2, 2] = 10

        masks = empty_masks()
        masks[UP][2, 2] = 1

        carried = {"biomass": biomass, "marker": marker}
        fns = {
            "biomass": lambda x: (x.float() * 0.5).to(x.dtype),
            "marker": lambda x: safe_add(x, 1, inplace=False),
        }
        perform_move(
            entity_energy=energy,
            direction_masks=masks,
            divide_threshold=100,
            divide_feature=biomass,
            divide_fn_self=lambda x: (x.float() * 0.5).to(x.dtype),
            divide_fn_offspring=lambda x: (x.float() * 0.5).to(x.dtype),
            carried_features_self=[carried[k] for k in order],
            carried_feature_fns_self=[fns[k] for k in order],
        )
        return energy.clone(), biomass.clone(), marker.clone()

    forward = run(["biomass", "marker"])
    reverse = run(["marker", "biomass"])

    for a, b in zip(forward, reverse):
        assert torch.equal(a, b), "perform_move must not depend on carried feature order"


def test_multiple_arrivals_at_one_cell_clamp_rather_than_wrap():
    """Four movers converging on one cell exceed the uint8 range in total."""
    energy = torch.zeros(GRID, dtype=torch.uint8)
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

    assert energy[2, 2] == 255, "the total should saturate, not wrap around"


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

    energy = torch.zeros(GRID, dtype=torch.uint8)
    energy[2, 2] = 200
    carried = torch.zeros(GRID, dtype=torch.uint8)

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

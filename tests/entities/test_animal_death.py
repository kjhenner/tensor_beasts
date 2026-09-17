"""What a dead animal leaves behind: its biomass as carrion, and nothing else.

Two faults lived here. Energy is stored as a SharedFeature, and the death
handler skipped SharedFeatures, so a dead animal kept its energy and went on
moving, paying move costs and blocking living animals through the clearance
kernel until dissipation drained it. And the handler only cleaned cells with
biomass above zero, so prey eaten to exactly zero never died at all and its
id, gradient EMA, offspring count and memory waited on an empty cell for the
next arrival to be summed into them.

Worlds here are 32 or 64 wide for speed and are not valid ecologies; these
check bookkeeping, not dynamics.
"""

import torch

from tensor_beasts.config import load_config
from tensor_beasts.world import World


def build_world(size=32, seed=0):
    torch.manual_seed(seed)
    config = load_config("conf/basic_config.yaml")
    config.world.size = [size, size]
    world = World(config.world)
    world.initialize()
    return world


def per_animal_features(animal):
    return [feature for feature in animal.features() if feature is not animal.scent]


def some_living_cell(animal):
    alive = animal.biomass.data >= animal.config.survival_threshold
    cells = torch.nonzero(alive)
    assert len(cells) > 0
    y, x = cells[0].tolist()
    return y, x


def test_a_starved_animal_becomes_carrion_and_leaves_no_state_behind():
    world = build_world()
    herbivore = world.entity_dict["Herbivore"]
    carrion = world.entity_dict["SimpleTerrain"].carrion.data
    y, x = some_living_cell(herbivore)

    # Starving, with every other feature conspicuously set.
    reserve = herbivore.config.survival_threshold - 1.0
    herbivore.biomass.data[y, x] = reserve
    herbivore.energy.data[y, x] = 200.0
    herbivore.gradient_ema.data[y, x] = 0.7
    herbivore.offspring_count.data[y, x] = 3
    herbivore.id_feature.data[y, x] = 41
    carrion_before = float(carrion[y, x])

    # Death is the first thing an animal's update does.
    herbivore._handle_death(herbivore.biomass.data < herbivore.config.survival_threshold)

    assert float(carrion[y, x]) == carrion_before + reserve, "all remaining biomass becomes carrion, exactly"
    for feature in per_animal_features(herbivore):
        assert not bool(feature.data[y, x].any()), f"{feature.name} survived death"


def test_prey_eaten_to_exactly_zero_biomass_is_cleaned_up():
    """A bite that takes the last unit leaves biomass at exactly zero, which
    the old guard `dead & (biomass > 0)` read as an empty cell."""
    world = build_world()
    herbivore = world.entity_dict["Herbivore"]
    carrion = world.entity_dict["SimpleTerrain"].carrion.data
    y, x = some_living_cell(herbivore)

    herbivore.biomass.data[y, x] = 0.0
    herbivore.energy.data[y, x] = 150.0
    herbivore.gradient_ema.data[y, x] = 0.5
    herbivore.offspring_count.data[y, x] = 2
    herbivore.id_feature.data[y, x] = 17
    carrion_before = float(carrion[y, x])

    herbivore._handle_death(herbivore.biomass.data < herbivore.config.survival_threshold)

    assert float(carrion[y, x]) == carrion_before, "nothing left to convert"
    for feature in per_animal_features(herbivore):
        assert not bool(feature.data[y, x].any()), f"{feature.name} lingered on an eaten cell"


def test_no_cell_holds_state_without_a_living_animal_for_two_steps_running():
    """The invariant the two fixes buy. A cell can be in transition at the end
    of a step (an animal that burned below the threshold this step is cleaned
    next step), but nothing lingers: whatever is not alive at the end of one
    step is gone or replaced by the end of the next."""
    world = build_world(size=64)
    stale_before = {}
    for _ in range(60):
        world.update()
        for name in ("Herbivore", "Predator"):
            animal = world.entity_dict[name]
            alive = animal.biomass.data >= animal.config.survival_threshold
            holds_state = torch.zeros_like(alive)
            for feature in per_animal_features(animal):
                data = feature.data
                nonzero = data != 0
                if nonzero.dim() > alive.dim():
                    nonzero = nonzero.any(dim=-1)
                holds_state |= nonzero
            stale = holds_state & ~alive
            if name in stale_before:
                assert not bool((stale & stale_before[name]).any()), f"{name}: a dead cell kept its state for two steps"
            stale_before[name] = stale


def test_a_dead_animal_neither_moves_nor_blocks():
    """Energy is what lets a cell move and what the clearance kernel sees, so
    a dead cell with energy was a ghost: it wandered, and living animals could
    not pass it. After death the cell has no energy and does neither."""
    from tensor_beasts.entities.helpers.animal_helpers import get_direction_masks

    world = build_world()
    herbivore = world.entity_dict["Herbivore"]
    y, x = some_living_cell(herbivore)
    herbivore.biomass.data[y, x] = 1.0
    herbivore.energy.data[y, x] = 200.0
    herbivore._handle_death(herbivore.biomass.data < herbivore.config.survival_threshold)

    direction = torch.full(herbivore.energy.data.shape, 1, dtype=torch.long)  # everyone tries to move up
    masks = get_direction_masks(direction, herbivore.energy.data)
    assert int(masks[1][y, x]) == 0, "a dead cell cannot move"

    # A living animal two rows below, moving up toward the dead cell, is not blocked by it.
    if y + 2 < herbivore.energy.data.shape[0]:
        herbivore.energy.data[:] = 0
        herbivore.biomass.data[:] = 0
        herbivore.energy.data[y + 2, x] = 100.0
        herbivore.biomass.data[y + 2, x] = 100.0
        masks = get_direction_masks(direction, herbivore.energy.data, (herbivore.biomass.data > 0).to(torch.uint8))
        assert int(masks[1][y + 2, x]) == 1, "a cleared cell does not block the animal behind it"


def test_offspring_start_with_no_offspring_of_their_own():
    world = build_world()
    world.update()  # populates the world's random field, which offspring ids draw from
    herbivore = world.entity_dict["Herbivore"]
    size = herbivore.biomass.data.shape[0]

    # One animal, mid-grid, fat enough to divide, certain to move up.
    for feature in per_animal_features(herbivore):
        feature.data.zero_()
    y, x = size // 2, size // 2
    herbivore.biomass.data[y, x] = herbivore.config.reproduction_threshold + 40.0
    herbivore.energy.data[y, x] = 200.0
    herbivore.offspring_count.data[y, x] = 3
    herbivore.id_feature.data[y, x] = 9

    direction = torch.full((size, size), 1, dtype=torch.long)
    did_move = herbivore._execute_movement(direction=direction, move_probability=torch.ones(size, size))

    assert bool(did_move[y, x])
    assert int(herbivore.offspring_count.data[y - 1, x]) == 4, "the parent's count goes up by one"
    assert int(herbivore.offspring_count.data[y, x]) == 0, "the offspring's starts at zero"
    assert float(herbivore.biomass.data[y - 1, x]) == float(herbivore.biomass.data[y, x]), "biomass splits evenly"
    cost = herbivore.config.base_movement_cost
    assert float(herbivore.energy.data.sum()) == 200.0 - cost, "a division costs exactly one move"

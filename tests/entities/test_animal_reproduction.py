"""Whole-simulation invariants for reproduction bookkeeping.

offspring_count feeds World.entity_scores, which is the reward signal for the
reinforcement learning environment. It previously counted steps rather than
offspring, and accumulated on empty cells, so these guard the property that
matters rather than any particular number.
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


def test_offspring_count_only_grows_where_an_animal_was_present():
    world = build_world()
    herbivore = world.entity_dict["Herbivore"]

    for _ in range(25):
        occupied_before = (herbivore.biomass.data > 0).clone()
        count_before = herbivore.offspring_count.data.clone().to(torch.int32)

        world.update()

        count_after = herbivore.offspring_count.data.to(torch.int32)
        # A mover carries its count to the destination, so allow growth at a
        # cell that was occupied or is adjacent to one that was.
        reachable = occupied_before.clone()
        for shift, dim in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            reachable |= torch.roll(occupied_before, shifts=shift, dims=dim)

        grew = count_after > count_before
        assert not bool((grew & ~reachable).any()), (
            "offspring count increased on a cell no animal could have reached"
        )


def test_offspring_count_does_not_track_step_number():
    """The original failure mode: every cell gaining four per step."""
    world = build_world()
    herbivore = world.entity_dict["Herbivore"]

    for _ in range(25):
        world.update()

    counts = herbivore.offspring_count.data
    population = int((herbivore.biomass.data > 0).sum())
    cells = counts.numel()

    assert int((counts > 0).sum()) < cells // 2, (
        "most of the grid has an offspring count, which means it is counting steps"
    )
    assert int(counts.sum()) <= max(population, 1) * 25, (
        "offspring count total outruns any plausible number of births"
    )

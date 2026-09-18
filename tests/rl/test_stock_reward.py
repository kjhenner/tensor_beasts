"""The reward is the metric's own increment.

Each individual is paid its stock change, eat minus burn minus the reserve it
leaves behind if it dies, and the sum over every individual is the species'
stock change exactly, up to what newborns eat in their birth step. Pooling
over a box around the individual is the one knob. Small worlds; these are
bookkeeping tests, not ecology.
"""

import torch

from tensor_beasts.rl.multiagent import MultiAgentWorldEnv, gather_per_world, scatter_per_world

SIZE = (96, 96)


def _settled(entity="Predator", seed=0, steps=30, **kwargs):
    env = MultiAgentWorldEnv(size=SIZE, device="cpu", entity_name=entity, **kwargs)
    env.reset(seed=seed)
    for _ in range(steps):
        env.rule_based_step()
    return env


def _newborn_eating(env, transition):
    """What offspring ate in their birth step, per world: paid to nobody."""
    if transition.offspring is None:
        return torch.zeros(env.num_worlds)
    eaten = gather_per_world(transition.eaten, transition.offspring.clamp(min=0))
    eaten = torch.where(transition.reproduced, eaten, torch.zeros_like(eaten))
    return eaten.sum(dim=(-2, -1)).reshape(env.num_worlds)


def test_the_simulation_reports_where_each_offspring_went():
    """Reproduction leaves the offspring in the cell the parent is vacating, so
    the edge is the parent's own acting cell. The identity test below reads
    the newborn's eating through it. Herbivores, because they divide often
    enough in a small world to observe."""
    env = _settled(entity="Herbivore")
    flat = torch.arange(SIZE[0] * SIZE[1]).reshape(*SIZE)
    for _ in range(120):
        env.rule_based_step()
        transition = env.entity.last_transition
        if not bool(transition.reproduced.any()):
            continue
        divided = transition.reproduced
        assert torch.equal(transition.offspring[divided], flat[divided])
        assert bool((transition.offspring[~divided] == -1).all())
        return
    raise AssertionError("no reproduction observed; the test needs a longer window")


def test_the_transition_records_what_each_individual_burned():
    env = _settled()
    batch = env.rule_based_step()
    burned = env.entity.last_transition.burned
    assert burned is not None and burned.shape == SIZE
    assert torch.all(burned[batch.acted] >= env.entity.config.basal_rate - 1e-6), "everyone burns at least basal"
    assert float(burned[~batch.acted].abs().sum()) == 0.0, "nothing burns on an empty cell"


def test_rewards_sum_to_the_species_stock_change_exactly():
    """Sum r over acting individuals, add what newborns ate, and the result is
    B_{t+1} - B_t with B the biomass carried by living individuals. Every
    step, on an unbatched and on a batched world."""
    for worlds in (1, 2):
        env = MultiAgentWorldEnv(size=SIZE, device="cpu", entity_name="Predator", worlds=worlds)
        env.reset(seed=0)
        for _ in range(20):
            env.rule_based_step()
        checked = 0
        for _ in range(60):
            before = env.stock_per_world()
            batch = env.rule_based_step()
            after = env.stock_per_world()
            paid = batch.reward.sum(dim=(-2, -1)).reshape(worlds)
            newborn = _newborn_eating(env, env.entity.last_transition)
            assert torch.allclose(after - before, paid + newborn, atol=1e-2), (
                f"worlds={worlds}: stock moved {(after - before).tolist()} but rewards paid "
                f"{paid.tolist()} plus newborn eating {newborn.tolist()}"
            )
            checked += int(batch.acted.sum())
        assert checked > 0


def test_a_death_costs_the_reserve_left_behind():
    """An individual that ends the step below the survival threshold is paid
    minus what it still carries, which becomes carrion next step. Nobody
    else's reward changes."""
    env = _settled(steps=10)
    for _ in range(200):
        batch = env.rule_based_step()
        died = batch.acted & batch.done
        if not bool(died.any()):
            continue
        transition = env.entity.last_transition
        successor = transition.successor.clamp(min=0)
        reserve = gather_per_world(env.entity.biomass.data, successor)
        eaten = gather_per_world(transition.eaten, successor)
        expected = eaten - transition.burned - reserve
        assert torch.allclose(batch.reward[died], expected[died], atol=1e-4)
        assert torch.all(reserve[died] < env.entity.config.survival_threshold)
        return
    raise AssertionError("no death observed")


def test_radius_zero_is_the_individual_reward_and_a_world_radius_pays_everyone_the_total():
    env = _settled(steps=10)
    for _ in range(5):
        env.rule_based_step()
    transition_env = env
    # Score one transition three ways from the same recorded step.
    batch = transition_env.rule_based_step()
    transition = transition_env.entity.last_transition
    biomass_before = None
    own, _, successor, acted = _individual(transition_env, transition)
    assert torch.allclose(batch.reward, own), "radius 0 is the individual reward"

    transition_env.reward_radius = max(SIZE)
    pooled, *_ = transition_env._reward(transition, biomass_before)
    total = float(own.sum())
    assert torch.allclose(pooled[acted], torch.full_like(pooled[acted], total), atol=1e-3), (
        "a box covering the world pays everyone the species' stock change"
    )
    assert float(pooled[~acted].abs().sum()) == 0.0


def test_a_small_radius_sums_the_neighbourhood_at_the_new_cell():
    env = _settled(steps=10)
    env.reward_radius = 2
    for _ in range(3):
        batch = env.rule_based_step()
    transition = env.entity.last_transition
    env.reward_radius = 0
    own, _, successor, acted = _individual(env, transition)
    env.reward_radius = 2
    # By hand: scatter to successors, box-sum, read back.
    field = scatter_per_world(own, successor, acted)
    padded = torch.nn.functional.pad(field, (2, 2, 2, 2))
    expected = torch.zeros_like(own)
    height, width = SIZE
    for cell in successor[acted].tolist():
        r, c = divmod(cell, width)
        expected_value = float(padded[r:r + 5, c:c + 5].sum())
        expected.reshape(-1)[cell] = expected_value
    expected = gather_per_world(expected, successor)
    assert torch.allclose(batch.reward[acted], expected[acted], atol=1e-4)


def _individual(env, transition):
    radius = env.reward_radius
    env.reward_radius = 0
    try:
        return env._reward(transition, None)
    finally:
        env.reward_radius = radius


def test_scatter_per_world_is_the_inverse_of_gather_and_keeps_worlds_apart():
    height = width = 3
    cells = height * width
    values = torch.stack([torch.arange(cells).reshape(height, width) + 100 * b for b in range(3)]).float()
    index = torch.full((3, height, width), 4, dtype=torch.long)
    mask = torch.zeros(3, height, width, dtype=torch.bool)
    mask[:, 0, 0] = True
    mask[:, 2, 2] = True
    out = scatter_per_world(values, index, mask)
    for b in range(3):
        assert float(out[b].reshape(-1)[4]) == 100 * b + 0 + 100 * b + 8, "both masked entries land on cell 4, summed"
        assert float(out[b].abs().sum()) == float(out[b].reshape(-1)[4].abs())


def test_gather_per_world_reads_each_world_from_itself():
    """Successor and offspring indices are per world, not per batch."""
    height = width = 3
    cells = height * width
    field = torch.stack(
        [torch.arange(cells).reshape(height, width) + 100 * b for b in range(3)]
    ).float()
    index = torch.full((3, height, width), 4, dtype=torch.long)
    out = gather_per_world(field, index)
    assert out.shape == (3, height, width)
    for b in range(3):
        assert torch.all(out[b] == 100 * b + 4)


def test_gather_per_world_matches_the_unbatched_read():
    generator = torch.Generator().manual_seed(0)
    field = torch.rand(5, 5, generator=generator)
    index = torch.randint(0, 25, (5, 5), generator=generator)
    assert torch.equal(gather_per_world(field, index), field.reshape(-1)[index])

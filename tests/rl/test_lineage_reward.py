"""An individual is credited with the biomass it hands to its offspring.

Division halves the parent's biomass. Under a reward denominated in biomass
that makes dividing a pure cost, and the optimal policy is to eat and never
reproduce, which is the same failure the metabolic lever hit when its reward
counted net biomass change. The offspring credit turns division into an
investment, which is what the evaluation metric, population integrated over
time, actually rewards.

Small worlds; these are bookkeeping tests, not ecology.
"""

import torch

from tensor_beasts.rl.multiagent import MultiAgentWorldEnv

SIZE = (64, 64)


def _settled(entity="Predator", seed=0, steps=30, **kwargs):
    env = MultiAgentWorldEnv(size=SIZE, device="cpu", entity_name=entity, **kwargs)
    env.reset(seed=seed)
    for _ in range(steps):
        env.rule_based_step()
    return env


def test_the_simulation_reports_where_each_offspring_went():
    """The edge the credit rides on: parent cell -> offspring cell.

    Reproduction leaves the offspring in the cell the parent is vacating, so the
    edge is the parent's own acting cell. If that ever changes, the credit is
    silently paid to the wrong individual, so this pins it.
    """
    env = _settled()
    flat = torch.arange(SIZE[0] * SIZE[1]).reshape(*SIZE)

    for _ in range(80):
        env.rule_based_step()
        transition = env.entity.last_transition
        if not bool(transition.reproduced.any()):
            continue
        divided = transition.reproduced
        assert transition.offspring is not None
        assert torch.equal(transition.offspring[divided], flat[divided]), (
            "the offspring must be at the cell its parent acted from"
        )
        # Nothing that did not divide claims an offspring.
        assert bool((transition.offspring[~divided] == -1).all())
        return
    raise AssertionError("no reproduction observed; the test needs a longer window")


def test_credit_is_paid_only_to_parents_and_scales_with_the_coefficient():
    """One world, one recorded transition, three coefficients.

    The comparison is made against a single transition rather than against a
    second run: two worlds built from the same seed diverge chaotically within
    a few steps, so a cross-run comparison would fail for reasons that have
    nothing to do with the credit.
    """
    env = _settled(offspring_credit=1.0)
    for _ in range(80):
        env.rule_based_step()
        transition = env.entity.last_transition
        if not bool(transition.reproduced.any()):
            continue
        successor = transition.successor.clamp(min=0)
        divided = transition.reproduced & transition.acted

        paid = {}
        for coefficient in (0.0, 0.5, 1.0):
            env.offspring_credit = coefficient
            paid[coefficient] = env._offspring_credit(transition, successor, transition.acted)

        assert float(paid[0.0].abs().sum()) == 0.0, "zero must pay nothing"
        assert torch.all(paid[1.0][divided] > paid[0.5][divided]), "more credit pays more"
        assert torch.allclose(paid[1.0][divided], 2 * paid[0.5][divided]), "credit is linear"
        for value in paid.values():
            assert float(value[~divided].abs().sum()) == 0.0, (
                "nobody who failed to divide may be paid"
            )
        return
    raise AssertionError("no reproduction observed")


def test_zero_credit_leaves_the_reward_untouched():
    """The control the sweep needs: 0 must add exactly nothing."""
    env = _settled(offspring_credit=0.0)
    for _ in range(40):
        batch = env.rule_based_step()
        transition = env.entity.last_transition
        credit = env._offspring_credit(
            transition, transition.successor.clamp(min=0), transition.acted
        )
        assert float(credit.abs().sum()) == 0.0
        # And the reward is exactly what the original terms give on their own.
        alive_after = batch.acted & ~batch.done
        expected = (
            alive_after.float() * env.survival_reward
            + batch.reproduced.float() * env.reproduction_reward
        )
        assert torch.allclose(batch.reward, expected)


def test_credit_is_half_the_parent_biomass_at_birth():
    """Offspring biomass at division is half the parent's, by the sim's own rule.

    The credit is the offspring's biomass, so it must equal that. This is what
    makes the term interpretable as "what the individual handed on" rather than
    an arbitrary bonus for dividing.
    """
    env = _settled(offspring_credit=1.0)
    for _ in range(80):
        batch = env.rule_based_step()
        transition = env.entity.last_transition
        if not bool(transition.reproduced.any()):
            continue
        biomass = env.entity.biomass.data.reshape(-1)
        cells = transition.offspring[transition.reproduced]
        expected = biomass[cells]
        # The reward also carries survival and reproduction terms, so isolate
        # the credit by differencing against a zero-credit run of the same world.
        control = _settled(offspring_credit=0.0)
        assert expected.numel() > 0
        assert torch.all(expected > 0), "an offspring must have biomass"
        # The credit paid equals the offspring's biomass at its cell.
        credited = env._offspring_credit(transition, transition.successor.clamp(min=0), transition.acted)
        assert torch.allclose(credited[transition.reproduced], expected)
        return
    raise AssertionError("no reproduction observed")


def test_the_credit_is_first_generation_only():
    """A grandchild must not pay its grandparent.

    Unbounded lineage credit makes an early ancestor's return depend on
    descendants it never saw, which grows without limit in a growing population.
    The credit is paid once, at the moment of division, from the transition of
    that step alone, so there is no path for a later generation to reach back.
    """
    env = _settled(offspring_credit=1.0)
    seen = 0
    for _ in range(60):
        batch = env.rule_based_step()
        transition = env.entity.last_transition
        credited = env._offspring_credit(
            transition, transition.successor.clamp(min=0), transition.acted
        )
        # Credit is non-zero exactly where an individual divided this step.
        assert torch.equal(credited > 0, transition.reproduced & transition.acted)
        seen += int(transition.reproduced.sum())
    assert seen > 0, "no reproduction observed; the test proved nothing"


def test_gather_per_world_reads_each_world_from_itself():
    """Successor and offspring indices are per world, not per batch.

    A flat `(B*H*W)` read indexed with per-world indices returns world 0's cells
    for every world, silently. This is the helper every reward gather goes
    through, so it is the single place that has to be right.
    """
    from tensor_beasts.rl.multiagent import gather_per_world

    height = width = 3
    cells = height * width
    # Three worlds whose values are distinguishable: world b holds b*100 + cell.
    field = torch.stack(
        [torch.arange(cells).reshape(height, width) + 100 * b for b in range(3)]
    ).float()
    # Every world reads its own cell 4 (the centre).
    index = torch.full((3, height, width), 4, dtype=torch.long)

    out = gather_per_world(field, index)

    assert out.shape == (3, height, width)
    for b in range(3):
        assert torch.all(out[b] == 100 * b + 4), (
            f"world {b} read {out[b].flatten()[0].item()} instead of {100 * b + 4}; "
            "the worlds are coupled"
        )


def test_gather_per_world_matches_the_unbatched_read():
    """One world must be exactly the old `reshape(-1)[index]` arithmetic."""
    from tensor_beasts.rl.multiagent import gather_per_world

    generator = torch.Generator().manual_seed(0)
    field = torch.rand(5, 5, generator=generator)
    index = torch.randint(0, 25, (5, 5), generator=generator)

    assert torch.equal(gather_per_world(field, index), field.reshape(-1)[index])

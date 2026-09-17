"""Offline distillation of the rules: the sampler, the symmetries, the fit.

Small worlds for speed; this is bookkeeping, not ecology.
"""

import pytest
import torch

from tensor_beasts.rl.distill import (
    ACTION_NAMES, SYMMETRIES, action_permutation, augment, channel_permutation,
    collect_labelled_grids, direction_map, distill, evaluate, transform_field,
)
from tensor_beasts.rl.multiagent import MultiAgentWorldEnv
from tensor_beasts.rl.networks import build_network

SIZE = (48, 48)


def sample(worlds=2, grids=8, stride=2, warmup=6, metabolic=True, seed=0):
    env = MultiAgentWorldEnv(size=SIZE, device="cpu", worlds=worlds, metabolic=metabolic)
    env.reset(seed=seed)
    return collect_labelled_grids(env, grids=grids, stride=stride, warmup=warmup), env


def test_sampler_returns_the_requested_grids_with_labels():
    grids, env = sample()
    assert len(grids) == 8
    assert grids.observation.shape == (8, env.observation_channels, *SIZE)
    assert grids.observation.dtype == torch.float16
    assert grids.rule_scores.shape == (8, 5, *SIZE)
    assert grids.rule_unit.shape == (8, *SIZE)
    assert grids.acted.any(), "somebody was alive"
    # The label is the rule's own argmax where it acted.
    assert torch.equal(grids.rule_scores.float().argmax(1)[grids.acted], grids.rule_action[grids.acted])
    assert grids.channel_names == env.channel_names


def test_symmetries_form_a_group_that_returns_to_identity():
    """Every symmetry has an inverse among the eight, on fields, channels and
    actions alike."""
    grids, _ = sample(grids=2)
    for k in range(SYMMETRIES):
        once = augment(grids, k)
        back = next(j for j in range(SYMMETRIES) if torch.equal(
            transform_field(transform_field(torch.arange(16.0).reshape(4, 4), k), j), torch.arange(16.0).reshape(4, 4)
        ))
        twice = augment(once, back)
        assert torch.equal(twice.observation, grids.observation), k
        assert torch.equal(twice.rule_action, grids.rule_action), k
        assert torch.equal(twice.rule_scores, grids.rule_scores), k
        assert torch.equal(twice.acted, grids.acted), k


def test_a_vertical_flip_swaps_up_and_down_in_channels_and_labels():
    grids, env = sample(grids=1)
    names = env.channel_names
    k = 2  # flip the row axis
    assert direction_map(k) == {"up": "down", "down": "up", "left": "left", "right": "right"}
    flipped = augment(grids, k)
    up = names.index("herbivore:scent/up") if "herbivore:scent/up" in names else next(i for i, n in enumerate(names) if n.endswith("/up"))
    down = names.index(names[up].replace("/up", "/down"))
    here = names.index(names[up].replace("/up", "/here"))
    H = SIZE[0]
    # What the original called "down" at row r is "up" at row H-1-r afterwards.
    assert torch.equal(flipped.observation[0, up], grids.observation[0, down].flip(0))
    assert torch.equal(flipped.observation[0, here], grids.observation[0, here].flip(0))
    # And the rule's answer swaps with it.
    perm = action_permutation(k)
    assert perm.tolist() == [0, 2, 1, 3, 4]
    assert torch.equal(flipped.rule_action, perm[grids.rule_action.flip(1)])
    # The scores' action axis permutes the same way: the new "up" score is the old "down" score.
    assert torch.equal(flipped.rule_scores[0, ACTION_NAMES.index("up")], grids.rule_scores[0, ACTION_NAMES.index("down")].flip(0))


def test_channel_permutation_leaves_non_directional_channels_alone():
    grids, env = sample(grids=1)
    names = env.channel_names
    for k in range(SYMMETRIES):
        perm = channel_permutation(names, k)
        for i, name in enumerate(names):
            if name.startswith("self/") or name.startswith("memory/") or name.endswith("/here"):
                assert int(perm[i]) == i, (k, name)


def exact_rule_network(env):
    """A linear network whose policy head IS the rule."""
    network = build_network("linear", env.observation_channels)
    network.initialise_from_rule(env.channel_names, env.entity.config.navigation_weights, scale=100.0)
    return network


def test_the_exact_rule_network_reproduces_the_sampled_labels():
    """The sampler's labels are the rule's own decisions, so a network built
    from the rule's weights must reproduce them, up to the knife-edge ties
    float16 storage can flip."""
    grids, env = sample(worlds=2, grids=8, stride=3, metabolic=False)
    network = exact_rule_network(env)
    agreement = evaluate(network, grids, 0.0, torch.device("cpu"), 4)["argmax_agreement"]
    assert agreement > 0.95, f"the rule's own weights agree with its labels only {agreement:.3f}"


def test_the_rule_is_equivariant_under_every_symmetry():
    """The end-to-end check of the augmentation. The rule is equivariant under
    the grid's symmetries, so the exact rule network must agree with the
    transformed labels on the transformed grids as well as it does on the
    originals. A wrong channel or label permutation shows as a collapse."""
    grids, env = sample(worlds=2, grids=8, stride=3, metabolic=False)
    network = exact_rule_network(env)
    baseline = evaluate(network, grids, 0.0, torch.device("cpu"), 4)["argmax_agreement"]
    for k in range(1, SYMMETRIES):
        transformed = evaluate(network, augment(grids, k), 0.0, torch.device("cpu"), 4)["argmax_agreement"]
        # Up to the knife-edge ties, which the action order breaks differently once permuted.
        assert transformed == pytest.approx(baseline, abs=0.03), f"symmetry {k}: {transformed:.3f} vs {baseline:.3f}"


def test_distill_fits_both_heads_and_stops_at_a_plateau():
    grids, env = sample(worlds=2, grids=48, stride=2, metabolic=True)
    network = build_network("linear", env.observation_channels, metabolic=True)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-2)
    seen = []
    record = distill(network, optimizer, grids, epochs=200, minibatch=4, temperature=0.01, max_grad_norm=0.5,
                     device=torch.device("cpu"), on_epoch=seen.append,
                     generator=torch.Generator().manual_seed(0))
    assert record.get("converged") == 1.0 and record["pretrain_epoch"] < 200
    assert len(seen) == record["pretrain_epoch"]
    assert record["argmax_agreement"] > 0.5
    assert record["metabolic_error"] < 0.1
    assert record["validation_loss"] < seen[0]["validation_loss"] * 0.5

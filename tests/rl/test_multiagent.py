"""The per-individual environment's contract.

Sizes here are small for speed. They are NOT a valid ecology: below roughly 256
the predator population collapses and the three-species dynamic degenerates.
Nothing in this file should be read as a statement about the simulation's
behaviour, only about the environment's bookkeeping.
"""

import torch

from tensor_beasts.rl.multiagent import MultiAgentWorldEnv


SIZE = (48, 48)


def build(**kwargs):
    return MultiAgentWorldEnv(size=SIZE, **kwargs)


def test_observation_gives_the_policy_the_baseline_inputs():
    """A learned policy must see what the rule-based policy sees, or the
    comparison measures the observation rather than the policy."""
    env = build()
    observation, _ = env.reset(seed=0)

    assert observation.shape == (env.observation_channels, *SIZE)
    assert observation.dtype == torch.float32

    names = env.channel_names
    # One 'here' plus four neighbours per perceived feature, then own state.
    assert names[-4:] == ["self/energy", "self/biomass", "self/gradient_ema", "self/alive"]
    for suffix in ("here", "up", "down", "left", "right", "grad_up", "grad_down", "grad_left", "grad_right"):
        assert f"simpleplant:scent/{suffix}" in names
    assert any(n.startswith("predator:scent") for n in names), "herbivores must be able to sense predators"


def test_observation_is_finite_and_scaled():
    env = build()
    observation, _ = env.reset(seed=0)
    for _ in range(10):
        batch = env.step(torch.randint(0, 5, SIZE))
        observation = batch.observation
        assert torch.isfinite(observation).all()
        names = env.channel_names
        grad = torch.tensor(["/grad_" in n for n in names])
        assert observation[~grad].min() >= 0.0
        assert observation[~grad].max() <= 1.5, "value channels should be roughly unit-scaled"
        assert observation[grad].abs().max() <= 4.0, "gradient channels are clipped"


def test_reward_and_done_only_touch_acting_cells():
    env = build()
    env.reset(seed=0)
    for _ in range(15):
        batch = env.step(torch.randint(0, 5, SIZE))
        idle = ~batch.acted
        assert torch.all(batch.reward[idle] == 0)
        assert torch.all(~batch.done[idle])
        assert torch.all(~batch.reproduced[idle])


def test_successors_are_unique_so_individuals_do_not_merge():
    """Two individuals sharing a successor would mean one trajectory counted
    twice. The clearance rules should prevent it."""
    env = build()
    env.reset(seed=0)
    for _ in range(20):
        batch = env.step(torch.randint(0, 5, SIZE))
        successors = batch.successor[batch.acted]
        assert successors.numel() == torch.unique(successors).numel()


def test_successor_of_a_survivor_is_an_occupied_cell():
    env = build()
    env.reset(seed=0)
    for _ in range(20):
        batch = env.step(torch.randint(0, 5, SIZE))
        survived = batch.acted & ~batch.done
        if not bool(survived.any()):
            continue
        biomass = env.entity.biomass.data.reshape(-1)
        assert torch.all(biomass[batch.successor[survived]] > 0)


def test_survival_and_reproduction_rewards_are_separable():
    env = build(survival_reward=1.0, reproduction_reward=0.0)
    env.reset(seed=0)
    total = 0.0
    survivors = 0
    for _ in range(10):
        batch = env.step(torch.randint(0, 5, SIZE))
        total += float(batch.reward.sum())
        survivors += int((batch.acted & ~batch.done).sum())
    assert abs(total - survivors) < 1e-4, "with no reproduction bonus, reward counts survivors"


def test_reproduction_reward_is_added_on_top():
    env = build(survival_reward=0.0, reproduction_reward=3.0)
    env.reset(seed=0)
    total = 0.0
    reproductions = 0
    for _ in range(25):
        batch = env.step(torch.randint(0, 5, SIZE))
        total += float(batch.reward.sum())
        reproductions += int(batch.reproduced.sum())
    assert abs(total - 3.0 * reproductions) < 1e-4


def test_actions_actually_drive_the_simulation():
    """Commanding everything north must bias movement north."""
    def net_displacement(direction_code):
        torch.manual_seed(0)
        env = build()
        env.reset(seed=0)
        rows = 0
        moves = 0
        for _ in range(12):
            batch = env.step(torch.full(SIZE, direction_code, dtype=torch.long))
            acting = batch.acted
            origin = torch.arange(SIZE[0] * SIZE[1]).reshape(*SIZE)[acting]
            delta = batch.successor[acting] - origin
            moved = delta != 0
            rows += int(delta[moved].sum())
            moves += int(moved.sum())
        return rows, moves

    up_rows, up_moves = net_displacement(1)
    down_rows, down_moves = net_displacement(2)

    assert up_moves > 0 and down_moves > 0, "commanded moves should sometimes succeed"
    assert up_rows < 0, "direction 1 should move individuals to lower row indices"
    assert down_rows > 0, "direction 2 should move individuals to higher row indices"


def test_rule_based_step_uses_the_same_reward_bookkeeping():
    env = build()
    env.reset(seed=0)
    for _ in range(10):
        batch = env.rule_based_step()
        idle = ~batch.acted
        assert torch.all(batch.reward[idle] == 0)
        assert batch.observation.shape == (env.observation_channels, *SIZE)


def test_enabling_the_environment_does_not_perturb_the_simulation():
    """Transition tracking must be observation, not interference."""
    import hashlib
    from tensor_beasts.config import load_config
    from tensor_beasts.world import World

    def world_hash(world):
        digest = hashlib.sha256()
        for key in sorted(world.td.keys(True, True), key=str):
            value = world.td.get(key)
            if isinstance(value, torch.Tensor):
                digest.update(str(key).encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    torch.manual_seed(0)
    config = load_config("conf/basic_config.yaml")
    config.world.size = list(SIZE)
    plain = World(config.world)
    plain.initialize()
    for _ in range(20):
        plain.update()

    torch.manual_seed(0)
    tracked = MultiAgentWorldEnv(size=SIZE)
    for _ in range(20):
        tracked.world.update()

    assert world_hash(plain) == world_hash(tracked.world)


def test_unknown_entity_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="not in"):
        MultiAgentWorldEnv(size=SIZE, entity_name="Dragon")


def test_foraging_reward_tracks_the_individuals_own_biomass_change():
    """The dense term must follow the individual, not the cell it left.

    A cell-indexed version would credit an individual with the biomass of
    whichever animal moved in behind it, which is the same class of mistake the
    advantage recursion has to avoid.
    """
    env = build(survival_reward=0.0, reproduction_reward=0.0, foraging_reward=1.0)
    env.reset(seed=0)

    for _ in range(12):
        entity = env.entity
        before = entity.biomass.data.clone()
        batch = env.step(torch.randint(0, 5, SIZE))

        after_flat = entity.biomass.data.reshape(-1)
        survived = batch.acted & ~batch.done
        if not bool(survived.any()):
            continue

        expected = (
            after_flat[batch.successor].reshape(*SIZE).float() - before.float()
        )[survived]
        assert torch.allclose(batch.reward[survived], expected, atol=1e-4)


def test_foraging_reward_is_off_by_default():
    """It is reward shaping, so it must be opt-in."""
    env = build()
    assert env.foraging_reward == 0.0

    env.reset(seed=0)
    survivors = 0
    total = 0.0
    for _ in range(8):
        batch = env.step(torch.randint(0, 5, SIZE))
        survivors += int((batch.acted & ~batch.done).sum())
        total += float(batch.reward.sum())
    # With the default weights, reward is survival plus 10 per reproduction.
    assert total >= survivors, "default reward should not include a biomass term"



def test_batch_carries_the_rule_based_action_for_acting_cells():
    """The rule action is what the imitation term anchors to, so it must be a
    valid direction wherever an individual acted, on both step paths."""
    env = build()
    env.reset(seed=0)
    for stepper in (lambda: env.step(torch.randint(0, 5, SIZE)), env.rule_based_step):
        batch = stepper()
        assert batch.rule_action is not None
        assert batch.rule_action.shape == SIZE
        assert batch.rule_action.dtype == torch.long
        acting = batch.rule_action[batch.acted]
        assert acting.numel() > 0
        assert acting.min() >= 0 and acting.max() <= 4



def test_a_small_conv_can_fit_the_rule_action():
    """The regression behind the whole imitation episode.

    A model once stalled at 0.46 agreement with the rule action because the
    informative differences between neighbouring cells were two orders of
    magnitude smaller than the channel values. Explicit gradient channels at
    unit scale fixed that. The bar is 0.8 rather than higher because the rule's
    decisions are knife-edge and roughly 0.9 is the practical ceiling for any
    approximate model. Small world for speed; the fit is a property of the
    observation encoding, not of the ecology.
    """
    import torch.nn.functional as F
    from tensor_beasts.rl.networks import build_network

    torch.manual_seed(0)
    env = MultiAgentWorldEnv(size=(96, 96), device="cpu")
    env.reset(seed=0)
    for _ in range(60):
        env.world.update()

    observations, labels, masks = [], [], []
    for _ in range(24):
        stack, raw = env._observe()
        observations.append(stack)
        labels.append(env._rule_action(raw))
        masks.append(raw.alive_mask.clone())
        env.world.update()
    observations, labels, masks = map(torch.stack, (observations, labels, masks))
    assert int(masks.sum()) > 200, "need a population to fit against"

    network = build_network("conv", env.observation_channels, hidden_channels=32)
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-3)
    for _ in range(300):
        logits, _ = network(observations[:20])
        m = masks[:20]
        loss = F.cross_entropy(logits.permute(0, 2, 3, 1)[m], labels[:20][m])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits, _ = network(observations[20:])
        m = masks[20:]
        agreement = float((logits.argmax(1)[m] == labels[20:][m]).float().mean())
    assert agreement > 0.8, f"conv fit of the rule action reached only {agreement:.3f}"

"""The second lever: a learned metabolic rate alongside the learned direction.

Every world here is 32x32 or 48x48 for speed and is NOT a valid ecology: below
roughly 256 the predator population collapses and the three-species dynamic
degenerates. These tests check the plumbing from a network head down to the
biomass an animal actually burns, and nothing about whether learning the
throttle helps. That number comes from train_rl.py at 256 or 512.
"""

import hashlib

import pytest
import torch
from tensordict import TensorDict

from tensor_beasts.config import load_config
from tensor_beasts.policy.metabolism import clamp_metabolic_rate, effective_max_metabolic_rate
from tensor_beasts.policy.rule_based import RuleBasedPolicy
from tensor_beasts.rl.controller import LearnedController
from tensor_beasts.rl.multiagent import (
    MultiAgentWorldEnv,
    metabolic_level_rates,
    rate_to_metabolic_level,
)
from tensor_beasts.rl.networks import build_network
from tensor_beasts.rl.ppo import PPO, PPOConfig, iter_minibatches_with_value
from tensor_beasts.rl.rollout import Rollout
from tensor_beasts.rl.trainer import Trainer, TrainerConfig
from tensor_beasts.world import World

SIZE = 32  # for speed; NOT a valid ecology, see the module docstring.
LEVELS = 4


def build_world(size=SIZE, seed=0):
    torch.manual_seed(seed)
    config = load_config("conf/basic_config.yaml")
    config.world.size = [size, size]
    world = World(config.world)
    world.initialize()
    return world


def world_hash(world):
    digest = hashlib.sha256()
    for key in sorted(world.td.keys(True, True), key=str):
        value = world.td.get(key)
        if isinstance(value, torch.Tensor):
            digest.update(str(key).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


# ----------------------------------------------------------------------
# Simulation override
# ----------------------------------------------------------------------
def test_override_clamp_matches_the_rules_own_biomass_cap():
    """The cap is physics, so a learned rate and the rule's rate must hit the
    same ceiling for the same biomass. The rule with an enormous gradient is
    exactly its cap; an enormous override must land in the same place."""
    world = build_world()
    entity = world.entity_dict["Herbivore"]
    config = entity.config
    rule = RuleBasedPolicy(config)

    biomass = torch.arange(0, 256, dtype=torch.uint8).reshape(16, 16)
    huge = torch.full((16, 16), 1e6)
    rule_cap = rule._compute_metabolic_rate(huge, biomass)
    override = clamp_metabolic_rate(
        huge, biomass, config.basal_rate, config.max_metabolic_rate, config.survival_threshold
    )
    assert torch.equal(override, rule_cap)

    # The floor: a learned rate cannot dip below basal.
    floor = clamp_metabolic_rate(
        torch.zeros(16, 16), biomass, config.basal_rate, config.max_metabolic_rate, config.survival_threshold
    )
    assert torch.all(floor == float(config.basal_rate))
    # And the cap is the rule's shape: basal at the threshold, max at 255.
    cap = effective_max_metabolic_rate(biomass, config.basal_rate, config.max_metabolic_rate, config.survival_threshold)
    assert cap[biomass == config.survival_threshold].item() == pytest.approx(config.basal_rate)
    assert cap[biomass == 255].item() == pytest.approx(config.max_metabolic_rate)


def test_override_burns_the_capped_rate_in_the_simulation():
    """End to end: send a huge rate, the animal burns exactly its cap; send
    zero, it burns basal. Food is removed so biomass change is metabolism only,
    and everyone stays put so nothing moves or divides."""
    world = build_world()
    world.update()  # populates the world's random field the mover reads
    entity = world.entity_dict["Herbivore"]
    config = entity.config
    for key in config.food_keys:
        world.td.get(key).zero_()

    alive = entity.biomass.data >= config.survival_threshold
    assert int(alive.sum()) > 0
    # A spread of biomass below the reproduction threshold, so the cap varies.
    spread = (torch.arange(int(alive.sum())) * 37 % 150 + 40).float()
    entity.biomass.data[alive] = spread
    entity.energy.data[alive] = 100

    for desired, expected_rate in (
        (1e6, lambda b: effective_max_metabolic_rate(b, config.basal_rate, config.max_metabolic_rate, config.survival_threshold)),
        (0.0, lambda b: torch.full_like(b.float(), float(config.basal_rate))),
    ):
        before = entity.biomass.data.clone()
        entity.update({
            "direction": torch.zeros(SIZE, SIZE, dtype=torch.long),
            "metabolic_rate": torch.full((SIZE, SIZE), desired),
        })
        burned = (before - entity.biomass.data)[alive]
        expected = torch.min(expected_rate(before), before)[alive]
        # Exact: the cap is a float and the burn is no longer truncated.
        assert torch.allclose(burned, expected, atol=1e-4), f"desired={desired}"


def test_bare_direction_override_is_unchanged_and_mapping_without_rate_matches():
    """The direction-only path must not move, and a mapping that omits the
    rate must be the same thing as the bare tensor."""
    torch.manual_seed(1)
    direction = torch.randint(0, 5, (SIZE, SIZE))

    bare = build_world()
    for _ in range(6):
        bare.update(TensorDict({"Herbivore": direction}, batch_size=[]))

    mapping = build_world()
    for _ in range(6):
        mapping.update(TensorDict(
            {"Herbivore": TensorDict({"direction": direction}, batch_size=[])}, batch_size=[]
        ))

    assert world_hash(bare) == world_hash(mapping)

    # And with no override at all the world is the rule-based one; the two
    # override paths above must differ from it, or the override is not applied.
    plain = build_world()
    for _ in range(6):
        plain.update()
    assert world_hash(plain) != world_hash(bare)


def test_rate_override_changes_the_simulation():
    torch.manual_seed(0)
    direction = torch.randint(0, 5, (SIZE, SIZE))
    hashes = []
    for rate in (0.0, 1e6):
        world = build_world()
        for _ in range(6):
            world.update(TensorDict({"Herbivore": TensorDict(
                {"direction": direction, "metabolic_rate": torch.full((SIZE, SIZE), rate)}, batch_size=[]
            )}, batch_size=[]))
        hashes.append(world_hash(world))
    assert hashes[0] != hashes[1], "resting and sprinting must lead to different worlds"


# ----------------------------------------------------------------------
# Levels
# ----------------------------------------------------------------------
def test_level_rates_span_basal_to_max_evenly():
    rates = metabolic_level_rates(2, 6, 5)
    assert torch.allclose(rates, torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]))
    with pytest.raises(ValueError):
        metabolic_level_rates(2, 6, 1)


def test_rate_to_level_is_nearest_and_ties_go_low():
    rates = metabolic_level_rates(2, 6, 5)
    rate = torch.tensor([[1.0, 2.4, 2.6, 3.5, 5.9, 100.0]])
    level = rate_to_metabolic_level(rate, rates)
    assert level.dtype == torch.long
    assert level.tolist() == [[0, 0, 1, 1, 4, 4]]


def test_env_batches_carry_metabolic_fields_in_range():
    env = MultiAgentWorldEnv(size=(SIZE, SIZE), num_metabolic_levels=LEVELS)
    env.reset(seed=0)
    assert env.metabolic_level_rates.shape == (LEVELS,)

    for _ in range(4):
        level = torch.randint(0, LEVELS, (SIZE, SIZE))
        batch = env.step(torch.randint(0, 5, (SIZE, SIZE)), level)
        assert torch.equal(batch.metabolic_action, level)
        assert batch.rule_metabolic_level is not None
        assert batch.rule_metabolic_level.shape == (SIZE, SIZE)
        assert batch.rule_metabolic_level.dtype == torch.long
        acting = batch.rule_metabolic_level[batch.acted]
        assert acting.numel() > 0
        assert acting.min() >= 0 and acting.max() < LEVELS

    baseline = env.rule_based_step()
    assert baseline.metabolic_action is None, "the rules set their own rate"
    assert baseline.rule_metabolic_level is not None

    off = MultiAgentWorldEnv(size=(SIZE, SIZE))
    off.reset(seed=0)
    batch = off.step(torch.randint(0, 5, (SIZE, SIZE)))
    assert batch.metabolic_action is None and batch.rule_metabolic_level is None
    assert off.metabolic_level_rates is None


def test_rule_level_is_the_nearest_level_of_the_rules_rate():
    env = MultiAgentWorldEnv(size=(48, 48), num_metabolic_levels=LEVELS)
    env.reset(seed=0)
    for _ in range(5):
        env.world.update()
    _, raw = env._observe()
    decision = env.entity.policy(raw)
    rate = decision.metabolic_rate
    rates = env.metabolic_level_rates

    # Explicit nearest-level search, independently of the implementation.
    expected = torch.stack([(rate - r).abs() for r in rates]).argmin(dim=0)
    level = env.metabolic_rate_to_level(rate)
    assert torch.equal(level, expected)
    # And the round trip never moves a rate by more than half a level.
    spacing = float(rates[1] - rates[0])
    alive = raw.alive_mask
    assert float((env.metabolic_level_to_rate(level) - rate).abs()[alive].max()) <= spacing / 2 + 1e-6
    # The rule's rate on a settled world has to touch more than one level, or
    # the target is a constant and the anchor is meaningless.
    assert level[alive].unique().numel() > 1


# ----------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------
def test_forward_is_unchanged_and_forward_all_adds_the_head():
    channels = 6
    observation = torch.randn(2, channels, 8, 8)
    plain = build_network("conv", channels, hidden_channels=8, depth=1)
    assert not plain.has_metabolic_head
    out = plain.forward_all(observation)
    assert set(out) == {"logits", "value"}

    headed = build_network("conv", channels, hidden_channels=8, depth=1, num_metabolic_levels=LEVELS)
    assert headed.has_metabolic_head
    logits, value = headed(observation)
    out = headed.forward_all(observation)
    assert torch.equal(out["logits"], logits) and torch.equal(out["value"], value)
    assert out["metabolic_logits"].shape == (2, LEVELS, 8, 8)
    assert out["metabolic_logits"].softmax(dim=1).max() < 0.5, "the metabolic head starts near uniform"
    assert headed.num_parameters() > plain.num_parameters()


# ----------------------------------------------------------------------
# PPO: factored policy and the second anchor
# ----------------------------------------------------------------------
def test_joint_log_prob_and_entropy_are_the_sum_of_the_parts():
    torch.manual_seed(0)
    channels = 6
    network = build_network("conv", channels, hidden_channels=8, depth=1, num_metabolic_levels=LEVELS)
    observation = torch.randn(3, channels, 8, 8)
    action = torch.randint(0, 5, (3, 8, 8))
    level = torch.randint(0, LEVELS, (3, 8, 8))

    with torch.no_grad():
        log_prob, entropy, value = PPO.evaluate_actions(network, observation, action, level)
        out = network.forward_all(observation)
        direction_lp = torch.log_softmax(out["logits"], 1).gather(1, action.unsqueeze(1)).squeeze(1)
        metabolic_lp = torch.log_softmax(out["metabolic_logits"], 1).gather(1, level.unsqueeze(1)).squeeze(1)

        def ent(logits):
            lp = torch.log_softmax(logits, 1)
            return -(lp.exp() * lp).sum(1)

    assert torch.allclose(log_prob, direction_lp + metabolic_lp, atol=1e-6)
    assert torch.allclose(entropy, ent(out["logits"]) + ent(out["metabolic_logits"]), atol=1e-6)
    assert torch.equal(value, out["value"])

    with pytest.raises(ValueError, match="metabolic_action"):
        PPO.evaluate_actions(network, observation, action)


def _rollout_with_rule_levels(seed=0, steps=6, size=8, channels=10):
    """Synthetic rollout: the rule's direction is the argmax of channels 0..4
    and its metabolic level the argmax of channels 5..8, both learnable by a
    1x1 conv."""
    g = torch.Generator().manual_seed(seed)
    observation = torch.rand(steps, channels, size, size, generator=g)
    acted = torch.rand(steps, size, size, generator=g) < 0.5
    flat = torch.arange(size * size).reshape(size, size)
    return Rollout(
        observation=observation.half(),
        acted=acted,
        action=torch.randint(0, 5, (steps, size, size), generator=g),
        log_prob=torch.full((steps, size, size), -1.6 - 1.386),
        value=torch.zeros(steps, size, size),
        reward=torch.zeros(steps, size, size),
        done=torch.zeros(steps, size, size, dtype=torch.bool),
        successor=flat.expand(steps, size, size).clone(),
        advantage=torch.zeros(steps, size, size),
        ret=torch.zeros(steps, size, size),
        rule_action=observation[:, :5].argmax(dim=1),
        metabolic_action=torch.randint(0, LEVELS, (steps, size, size), generator=g),
        rule_metabolic_level=observation[:, 5:5 + LEVELS].argmax(dim=1),
    )


def test_metabolic_imitation_pulls_the_level_toward_the_rule():
    """Zero advantages, so imitation is the only signal: metabolic agreement
    must climb and the chosen-level diagnostic must be reported."""
    torch.manual_seed(0)
    rollout = _rollout_with_rule_levels()
    network = build_network("linear", 10, num_metabolic_levels=LEVELS)
    ppo = PPO(PPOConfig(imitation_coef=1.0, imitation_temperature=0.0, imitation_target_conformance=0.99,
                        epochs=1, minibatch_steps=6, entropy_coef=0.0, value_coef=0.0))
    optimizer = torch.optim.Adam(network.parameters(), lr=0.1)

    first = ppo.update(network, rollout, optimizer)
    for _ in range(40):
        last = ppo.update(network, rollout, optimizer)

    assert first["metabolic_agreement"] < 0.5
    assert last["metabolic_agreement"] > first["metabolic_agreement"] + 0.3
    assert last["metabolic_imitation_loss"] < first["metabolic_imitation_loss"]
    assert 0.0 <= last["metabolic_level_mean"] <= LEVELS - 1
    assert last["metabolic_entropy"] < first["metabolic_entropy"]


def test_metabolic_imitation_is_masked_to_acting_cells():
    rollout = _rollout_with_rule_levels()
    network = build_network("linear", 10, num_metabolic_levels=LEVELS)
    ppo = PPO(PPOConfig(imitation_coef=1.0, imitation_temperature=0.0))
    batch = list(next(iter_minibatches_with_value(rollout, rollout.steps, shuffle=False)))
    before = float(ppo.minibatch_loss(network, tuple(batch)))
    _, clean = ppo._losses(network, *batch)

    idle = ~batch[1]
    levels = batch[10].clone()
    levels[idle] = (levels[idle] + 1) % LEVELS
    batch[10] = levels
    chosen = batch[9].clone()
    chosen[idle] = (chosen[idle] + 2) % LEVELS
    batch[9] = chosen
    after = float(ppo.minibatch_loss(network, tuple(batch)))
    _, dirty = ppo._losses(network, *batch)

    assert before == pytest.approx(after)
    for key in ("metabolic_agreement", "metabolic_level_mean", "metabolic_imitation_loss"):
        assert clean[key] == pytest.approx(dirty[key], abs=1e-6), key


def test_both_levers_share_one_cross_fade_weight():
    """One anchor, two levers: the metabolic term is scaled by the same weight
    as the direction term, and both vanish together when the anchor releases."""
    rollout = _rollout_with_rule_levels()
    network = build_network("linear", 10, num_metabolic_levels=LEVELS)
    batch = next(iter_minibatches_with_value(rollout, rollout.steps, shuffle=False))

    ppo = PPO(PPOConfig(imitation_coef=1.0, imitation_temperature=0.0, entropy_coef=0.0, value_coef=0.0))
    ppo.conformance = 0.0
    _, engaged = ppo._losses(network, *batch)
    ppo.conformance = 1.0
    _, released = ppo._losses(network, *batch)

    assert engaged["imitation_weight"] == 1.0 and released["imitation_weight"] == 0.0
    # With zero advantages the policy loss is zero, so the engaged loss is
    # exactly the two imitation terms and the released loss is nothing.
    assert engaged["loss"] == pytest.approx(engaged["imitation_loss"] + engaged["metabolic_imitation_loss"], abs=1e-5)
    assert released["loss"] == pytest.approx(0.0, abs=1e-6)


# ----------------------------------------------------------------------
# Trainer, checkpoint, controller
# ----------------------------------------------------------------------
def make_trainer(tmp_path, ppo=None, **overrides):
    ppo = ppo or PPOConfig(epochs=1, minibatch_steps=2, imitation_coef=1.0)
    defaults = dict(
        size=SIZE, arch="conv", arch_kwargs={"hidden_channels": 8, "depth": 1},
        metabolic_levels=LEVELS, device="cpu", seed=0,
        total_world_steps=8, segment_steps=4, warmup_steps=2,
        eval_interval=0, eval_steps=3, eval_seeds=1, checkpoint_interval=4,
        output_dir=str(tmp_path),
    )
    defaults.update(overrides)
    return Trainer(TrainerConfig(**defaults), ppo)


def test_ratio_is_one_on_a_fresh_two_lever_segment(tmp_path):
    """The stored joint log-prob must be exactly what the update recomputes.

    One minibatch covering the whole segment, as in test_trainer: a second
    minibatch would see a network that had already taken a step."""
    trainer = make_trainer(tmp_path, ppo=PPOConfig(epochs=1, minibatch_steps=4, imitation_coef=1.0))
    trainer.env.reset(seed=0)
    rollout, _ = trainer.collect(4)
    assert rollout.metabolic_action is not None and rollout.rule_metabolic_level is not None
    diagnostics = trainer.algorithm.update(trainer.network, rollout, trainer.optimizer)
    assert diagnostics["ratio_max_deviation"] == 0.0
    assert diagnostics["metabolic_agreement"] == diagnostics["metabolic_agreement"], "not NaN"


def test_two_lever_training_runs_evaluates_and_the_controller_drives_it(tmp_path):
    trainer = make_trainer(tmp_path)
    record = trainer.train(verbose=False)
    for key in ("metabolic_agreement", "metabolic_level_mean", "metabolic_imitation_loss"):
        assert key in record and record[key] == record[key], key
    summary = trainer.evaluate()
    assert "learned_over_rule_based" in summary

    checkpoint = tmp_path / "checkpoint.pt"
    assert checkpoint.exists()
    payload = torch.load(checkpoint, weights_only=False)
    assert payload["num_metabolic_levels"] == LEVELS

    world = build_world()
    controller = LearnedController(world, checkpoint, device=torch.device("cpu"))
    assert "metabolism" in controller.describe()
    for _ in range(3):
        action = controller.action()
        entity_action = action["Herbivore"]
        assert isinstance(entity_action, TensorDict)
        assert entity_action["direction"].shape == (SIZE, SIZE)
        assert entity_action["direction"].dtype == torch.long
        rate = entity_action["metabolic_rate"]
        assert rate.shape == (SIZE, SIZE)
        rates = controller.env.metabolic_level_rates
        assert rate.min() >= rates.min() and rate.max() <= rates.max()
        world.update(action)


def test_checkpoint_with_a_head_is_refused_by_a_headless_trainer(tmp_path):
    trainer = make_trainer(tmp_path)
    path = trainer.save_checkpoint(tmp_path / "headed.pt")
    headless = make_trainer(tmp_path / "other", metabolic_levels=0)
    with pytest.raises(ValueError, match="metabolic levels"):
        headless.load_checkpoint(path)


def test_eval_only_takes_the_head_from_the_checkpoint(tmp_path):
    from train_rl import apply_overrides, build_parser

    trainer = make_trainer(tmp_path)
    path = trainer.save_checkpoint(tmp_path / "headed.pt")
    args = build_parser().parse_args(["--eval-only", str(path)])
    merged = apply_overrides(args)
    assert merged["trainer"]["metabolic_levels"] == LEVELS

    args = build_parser().parse_args([])
    assert apply_overrides(args)["trainer"]["metabolic_levels"] == 0, "off by default"

"""End-to-end checks on the training loop.

Every world here is 32x32 or 48x48. **That is for speed and is not a valid
ecology**: below roughly 256 the predator population collapses and the
three-species dynamic degenerates into herbivores grazing an empty map. These
tests say that the plumbing works, nothing whatsoever about whether learning
works. The number that answers that question comes from ``train_rl.py`` at 256
or 512, not from here.
"""

import json

import pytest
import torch

from tensor_beasts.rl.multiagent import AgentBatch
from tensor_beasts.rl.ppo import PPOConfig
from tensor_beasts.rl.trainer import (
    EpisodeTracker,
    Trainer,
    TrainerConfig,
    policy_input,
    resolve_device,
)

SIZE = 32  # for speed, NOT a valid ecology; see the module docstring.


def make_config(tmp_path, **overrides) -> TrainerConfig:
    defaults = dict(
        size=SIZE,
        arch="linear",
        device="cpu",
        seed=0,
        total_world_steps=8,
        segment_steps=4,
        warmup_steps=2,
        eval_interval=0,
        eval_steps=3,
        eval_seeds=1,
        checkpoint_interval=0,
        output_dir=str(tmp_path),
    )
    defaults.update(overrides)
    return TrainerConfig(**defaults)


def make_trainer(tmp_path, ppo_config=None, **overrides) -> Trainer:
    ppo_config = ppo_config or PPOConfig(epochs=1, minibatch_steps=2)
    return Trainer(make_config(tmp_path, **overrides), ppo_config)


# ----------------------------------------------------------------------
# Episode bookkeeping
# ----------------------------------------------------------------------
def grid_batch(height, width, entries):
    """Build an AgentBatch from (cell, reward, done, successor) tuples."""
    acted = torch.zeros(height, width, dtype=torch.bool)
    reward = torch.zeros(height, width)
    done = torch.zeros(height, width, dtype=torch.bool)
    successor = torch.zeros(height, width, dtype=torch.long)
    for (y, x), value, finished, (sy, sx) in entries:
        acted[y, x] = True
        reward[y, x] = value
        done[y, x] = finished
        successor[y, x] = sy * width + sx
    return AgentBatch(
        observation=torch.zeros(1, height, width),
        acted=acted,
        action=torch.zeros(height, width, dtype=torch.long),
        reward=reward,
        done=done,
        successor=successor,
        reproduced=torch.zeros(height, width, dtype=torch.bool),
    )


def test_episode_tracker_follows_an_individual_as_it_moves():
    """The classic failure: crediting the cell instead of the animal.

    One agent walks (0,0) -> (0,1) -> (0,2) collecting 1, 2 and 4, then dies.
    A second agent walks into the cell the first one left. If the tracker keyed
    on cells, the second agent would inherit the first one's return.
    """
    tracker = EpisodeTracker((3, 3), torch.device("cpu"))
    tracker.update(grid_batch(3, 3, [(((0, 0)), 1.0, False, (0, 1))]))
    tracker.update(
        grid_batch(3, 3, [((0, 1), 2.0, False, (0, 2)), ((1, 0), 100.0, False, (0, 0))])
    )
    tracker.update(
        grid_batch(3, 3, [((0, 2), 4.0, True, (0, 2)), ((0, 0), 5.0, True, (0, 0))])
    )

    summary = tracker.summary()
    assert summary["episodes_finished"] == 2
    assert sorted(tracker.finished_return) == [7.0, 105.0]
    assert sorted(tracker.finished_length) == [2.0, 3.0]


def test_episode_tracker_starts_newborns_at_zero():
    tracker = EpisodeTracker((3, 3), torch.device("cpu"))
    tracker.update(grid_batch(3, 3, [((0, 0), 9.0, False, (0, 0))]))
    # A newborn appears at (2, 2) and dies immediately.
    tracker.update(grid_batch(3, 3, [((2, 2), 1.0, True, (2, 2))]))
    assert tracker.finished_return == [1.0]


# ----------------------------------------------------------------------
# Collection
# ----------------------------------------------------------------------
def test_collect_produces_a_usable_rollout(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.env.reset(seed=0)
    rollout, stats = trainer.collect(4)

    assert rollout.steps == 4
    assert rollout.observation.shape[1] == trainer.observation_channels
    assert rollout.advantage is not None and rollout.ret is not None
    assert torch.isfinite(rollout.advantage).all()
    assert torch.isfinite(rollout.ret).all()
    # Advantages exist only where somebody acted.
    assert float(rollout.advantage[~rollout.acted].abs().max()) == 0.0
    assert stats["population"] > 0
    assert "reproduction_rate" in stats


def test_stored_observation_is_exactly_what_the_network_saw(tmp_path):
    """The premise of the ratio-is-one check, at the level of the real loop.

    RolloutBuffer stores float16. If collection fed the network full precision
    the two would differ in the low bits and the PPO ratio would no longer be
    exactly 1 on a fresh rollout.
    """
    trainer = make_trainer(tmp_path)
    trainer.env.reset(seed=0)
    rollout, _ = trainer.collect(2)

    for step in range(rollout.steps):
        stored = rollout.observation[step].float()
        assert torch.equal(stored, policy_input(stored))


def test_ratio_is_one_on_a_freshly_collected_segment(tmp_path):
    """Same check as in test_ppo, but on a rollout from the real environment."""
    trainer = make_trainer(
        tmp_path, ppo_config=PPOConfig(epochs=1, minibatch_steps=4)
    )
    trainer.env.reset(seed=0)
    rollout, _ = trainer.collect(4)
    diagnostics = trainer.algorithm.update(trainer.network, rollout, trainer.optimizer)
    assert diagnostics["ratio_max_deviation"] == 0.0


# ----------------------------------------------------------------------
# The loop
# ----------------------------------------------------------------------
def test_train_runs_logs_and_checkpoints(tmp_path):
    trainer = make_trainer(
        tmp_path, total_world_steps=8, segment_steps=4, checkpoint_interval=4
    )
    record = trainer.train(verbose=False)

    assert trainer.world_steps == 8
    assert trainer.updates == 2
    assert (tmp_path / "checkpoint.pt").exists()

    lines = [json.loads(line) for line in trainer.log_path.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[-1]["world_steps"] == 8
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
                "explained_variance", "world_steps_per_sec", "agent_steps_per_sec",
                "population", "reproduction_rate", "episode_return", "episode_length"):
        assert key in record, key
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl"):
        assert record[key] == record[key], f"{key} is NaN"


def test_evaluation_scores_both_policies(tmp_path):
    trainer = make_trainer(tmp_path, eval_steps=5, eval_seeds=1)
    summary = trainer.evaluate()
    for policy in ("learned", "rule_based"):
        assert summary[f"{policy}_total_reward"] >= 0
        assert summary[f"{policy}_mean_population"] > 0
        assert summary[f"{policy}_survived_agent_steps"] >= 0
    assert "learned_over_rule_based" in summary


def test_evaluation_does_not_disturb_the_training_world(tmp_path):
    trainer = make_trainer(tmp_path, eval_steps=3, eval_seeds=1)
    trainer.env.reset(seed=0)
    step_before = trainer.env.world.step
    trainer.evaluate()
    assert trainer.env.world.step == step_before


def test_evaluation_leaves_the_rng_stream_alone(tmp_path):
    """Otherwise the training trajectory depends on the evaluation schedule."""
    trainer = make_trainer(tmp_path, eval_steps=3, eval_seeds=1)
    state = torch.get_rng_state()
    trainer.evaluate()
    assert torch.equal(state, torch.get_rng_state())


# ----------------------------------------------------------------------
# Checkpointing
# ----------------------------------------------------------------------
def test_resume_restores_weights_counters_and_optimizer(tmp_path):
    trainer = make_trainer(
        tmp_path, total_world_steps=8, segment_steps=4, checkpoint_interval=4
    )
    trainer.train(verbose=False)
    path = tmp_path / "checkpoint.pt"
    weights = {k: v.clone() for k, v in trainer.network.state_dict().items()}

    resumed = make_trainer(
        tmp_path / "resumed",
        total_world_steps=12,
        segment_steps=4,
        checkpoint_interval=4,
    )
    # A different seed, so matching weights can only come from the checkpoint.
    resumed.config.seed = 1
    resumed.load_checkpoint(path)

    for key, value in resumed.network.state_dict().items():
        assert torch.equal(value, weights[key]), key
    assert resumed.world_steps == 8
    assert resumed.updates == 2
    assert resumed.optimizer.state_dict()["state"]

    resumed.train(verbose=False)
    assert resumed.world_steps == 12


def test_resume_rejects_a_checkpoint_from_a_different_observation_space(tmp_path):
    trainer = make_trainer(tmp_path)
    payload = trainer.checkpoint_payload()
    payload["observation_channels"] = trainer.observation_channels + 1
    path = tmp_path / "mismatched.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="observation"):
        trainer.load_checkpoint(path)


# ----------------------------------------------------------------------
# Odds and ends
# ----------------------------------------------------------------------
def test_resolve_device_honours_an_explicit_choice():
    assert resolve_device("cpu") == torch.device("cpu")


def test_every_architecture_constructs_and_collects(tmp_path):
    # 48x48 rather than 32x32 so the dilated trunk's 31x31 receptive field is
    # not larger than the world. Still not a valid ecology.
    for arch in ("linear", "conv", "residual", "dilated"):
        trainer = make_trainer(tmp_path / arch, arch=arch, size=48)
        trainer.env.reset(seed=0)
        rollout, _ = trainer.collect(2)
        diagnostics = trainer.algorithm.update(
            trainer.network, rollout, trainer.optimizer
        )
        assert diagnostics["loss"] == diagnostics["loss"], arch


def test_headline_ratio_is_survival_not_shaped_reward():
    """A sweep once reported a ratio of -0.72. That is impossible for a ratio of
    survival counts and was the sign that the ratio was dividing shaped
    rewards, which foraging_reward makes negative. What is optimized may change;
    what is judged must not."""
    from tensor_beasts.rl.trainer import Trainer, TrainerConfig
    from tensor_beasts.rl.ppo import PPOConfig

    # Tiny world for speed; not a valid ecology, only a bookkeeping check.
    config = TrainerConfig(
        size=32, arch="linear", warmup_steps=5, total_world_steps=0,
        eval_interval=0, eval_steps=8, eval_seeds=1, checkpoint_interval=0,
        foraging_reward=5.0, survival_reward=0.0, reproduction_reward=0.0,
        device="cpu",
    )
    trainer = Trainer(config, PPOConfig())
    summary = trainer.evaluate()

    baseline = summary["rule_based_survived_agent_steps"]
    expected = summary["learned_survived_agent_steps"] / baseline
    assert summary["learned_over_rule_based"] == pytest.approx(expected)
    assert summary["learned_over_rule_based"] >= 0.0



def test_pretraining_moves_the_policy_toward_the_rules_and_seeds_the_anchor(tmp_path):
    """A near-random start starved a learned predator population to zero within
    200 steps. Pretraining on rule-based rollouts must raise agreement with the
    rule well above chance and hand that agreement to the anchor's cross-fade.
    Tiny world for speed; this checks plumbing, not the ecology."""
    from tensor_beasts.rl.ppo import PPOConfig
    from tensor_beasts.rl.trainer import Trainer, TrainerConfig

    torch.manual_seed(0)
    trainer = Trainer(
        TrainerConfig(size=96, arch="conv", arch_kwargs={"hidden_channels": 16}, metabolic_levels=4,
                      warmup_steps=5, segment_steps=16, pretrain_updates=6, total_world_steps=0,
                      eval_interval=0, checkpoint_interval=0, device="cpu", output_dir=str(tmp_path)),
        PPOConfig(epochs=2, minibatch_steps=4, learning_rate=3e-3, imitation_coef=1.0),
    )
    trainer.env.reset(seed=0)
    trainer.warmup()
    result = trainer.pretrain(verbose=False)
    assert result["argmax_agreement"] > 0.5, f"agreement after pretraining {result['argmax_agreement']:.3f}"
    assert result["metabolic_agreement"] > 0.5
    assert trainer.algorithm.conformance == pytest.approx(result["argmax_agreement"])
    assert trainer.world_steps == 6 * 16


def test_pretraining_off_does_nothing(tmp_path):
    from tensor_beasts.rl.ppo import PPOConfig
    from tensor_beasts.rl.trainer import Trainer, TrainerConfig

    trainer = Trainer(
        TrainerConfig(size=32, arch="conv", arch_kwargs={"hidden_channels": 8}, warmup_steps=2,
                      total_world_steps=0, eval_interval=0, checkpoint_interval=0, device="cpu",
                      output_dir=str(tmp_path)),
        PPOConfig(),
    )
    before = {k: v.clone() for k, v in trainer.network.state_dict().items()}
    assert trainer.pretrain(verbose=False) == {}
    after = trainer.network.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_training_stops_when_the_population_goes_extinct(tmp_path):
    """An extinct population produces no gradient, so the run must end.

    A predator run once trained for 6,000 world steps after its last predator
    died, reporting NaN for every diagnostic, because nothing checked. The
    simulation has no immigration: once a controlled entity is gone it cannot
    come back, so continuing is pure waste.
    """
    import torch

    from tensor_beasts.rl.ppo import PPOConfig
    from tensor_beasts.rl.trainer import Trainer, TrainerConfig

    trainer = Trainer(
        TrainerConfig(
            size=32, entity="Predator", arch="conv", arch_kwargs={"hidden_channels": 4},
            warmup_steps=0, total_world_steps=400, segment_steps=8, eval_interval=0,
            checkpoint_interval=0, device="cpu", output_dir=str(tmp_path),
            extinction_patience=2,
        ),
        PPOConfig(epochs=1, minibatch_steps=2),
    )
    trainer.env.reset(seed=0)
    # Kill every predator, which is what a collapsing policy eventually does.
    trainer.env.entity.biomass.data.zero_()

    record = trainer.train(verbose=False)

    assert record.get("extinct") is True
    assert trainer.world_steps < trainer.config.total_world_steps, (
        "the loop should stop early rather than run to completion on an empty world"
    )


def test_extinction_guard_can_be_disabled(tmp_path):
    from tensor_beasts.rl.ppo import PPOConfig
    from tensor_beasts.rl.trainer import Trainer, TrainerConfig

    trainer = Trainer(
        TrainerConfig(
            size=32, entity="Predator", arch="conv", arch_kwargs={"hidden_channels": 4},
            warmup_steps=0, total_world_steps=24, segment_steps=8, eval_interval=0,
            checkpoint_interval=0, device="cpu", output_dir=str(tmp_path),
            extinction_patience=0,
        ),
        PPOConfig(epochs=1, minibatch_steps=2),
    )
    trainer.env.reset(seed=0)
    trainer.env.entity.biomass.data.zero_()

    record = trainer.train(verbose=False)

    assert "extinct" not in record
    assert trainer.world_steps == 24

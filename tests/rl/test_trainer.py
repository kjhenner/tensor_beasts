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



def test_pretraining_moves_the_policy_toward_the_rules(tmp_path):
    """A near-random start starved a learned predator population to zero within
    200 steps. Distilling the rules must raise held-out agreement well above
    chance, fit the throttle, and leave the training world and the RNG stream
    exactly where they were. Tiny world for speed; plumbing, not ecology."""
    from tensor_beasts.rl.ppo import PPOConfig
    from tensor_beasts.rl.trainer import Trainer, TrainerConfig

    torch.manual_seed(0)
    trainer = Trainer(
        TrainerConfig(size=96, arch="conv", arch_kwargs={"hidden_channels": 16}, metabolic=True,
                      warmup_steps=5, segment_steps=16, pretrain_epochs=60, pretrain_grids=32, pretrain_stride=3,
                      total_world_steps=0, eval_interval=0, checkpoint_interval=0, device="cpu",
                      output_dir=str(tmp_path)),
        PPOConfig(epochs=2, minibatch_steps=4, learning_rate=3e-3),
    )
    trainer.env.reset(seed=0)
    trainer.warmup()
    step_before = trainer.env.world.step
    rng_before = torch.get_rng_state()
    result = trainer.pretrain(verbose=False)
    assert result["argmax_agreement"] > 0.5, f"agreement after pretraining {result['argmax_agreement']:.3f}"
    # The throttle is continuous, so it is scored by how far the policy's
    # mean sits from the rule's own throttle. A fifth of the basal-to-max
    # range is loose, and deliberately: this checks that the fit pulls.
    assert result["metabolic_error"] < 0.2
    assert result["pretrain_epoch"] <= 60
    assert trainer.env.world.step == step_before, "the training world is not stepped by pretraining"
    assert trainer.world_steps == 0
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_pretraining_stops_once_agreement_plateaus(tmp_path):
    """pretrain_epochs is a ceiling. A linear network fits the rule exactly
    and plateaus quickly, so it must stop well short of a generous ceiling."""
    from tensor_beasts.rl.distill import WINDOW
    from tensor_beasts.rl.ppo import PPOConfig
    from tensor_beasts.rl.trainer import Trainer, TrainerConfig

    torch.manual_seed(0)
    trainer = Trainer(
        TrainerConfig(size=64, arch="linear", warmup_steps=5, segment_steps=8, pretrain_epochs=200,
                      pretrain_grids=32, pretrain_stride=2, total_world_steps=0, eval_interval=0, checkpoint_interval=0,
                      device="cpu", output_dir=str(tmp_path)),
        PPOConfig(epochs=2, minibatch_steps=4, learning_rate=1e-2),
    )
    trainer.env.reset(seed=0)
    trainer.warmup()
    result = trainer.pretrain(verbose=False)
    assert result.get("converged") == 1.0
    assert result["pretrain_epoch"] < 200
    assert result["pretrain_epoch"] >= 2 * WINDOW


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


def test_a_dead_batch_is_reset_and_the_run_reaches_its_budget(tmp_path):
    """An extinct world produces no gradient. The run used to stop there,
    which treated one world's fate as the run's verdict and punished exactly
    the runs that met a bust early. The world is reset instead: a predator run
    once trained for 6,000 world steps after its last predator died, reporting
    NaN for every diagnostic, and now it trains on a fresh ecology."""
    import torch

    from tensor_beasts.rl.ppo import PPOConfig
    from tensor_beasts.rl.trainer import Trainer, TrainerConfig

    trainer = Trainer(
        TrainerConfig(
            size=32, entity="Predator", arch="conv", arch_kwargs={"hidden_channels": 4},
            warmup_steps=0, total_world_steps=64, segment_steps=8, eval_interval=0,
            checkpoint_interval=0, device="cpu", output_dir=str(tmp_path),
            extinction_patience=2,
        ),
        PPOConfig(epochs=1, minibatch_steps=2),
    )
    # Kill every predator after the first segment, which is what a collapsing
    # policy eventually does. train() resets the world itself, so the kill has
    # to land inside the loop. This is the unbatched path; the batched one is
    # covered below.
    original_collect = trainer.collect

    def collect_then_kill(steps):
        rollout, stats = original_collect(steps)
        trainer.env.entity.biomass.data.zero_()
        trainer.env.entity.energy.data.zero_()
        return rollout, stats

    trainer.collect = collect_then_kill
    record = trainer.train(verbose=False)

    assert trainer.world_steps == trainer.config.total_world_steps
    assert record["world_resets"] >= 1
    assert "extinct" not in record


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


def test_wandb_host_defers_to_the_user_s_own_server(tmp_path, monkeypatch):
    """The default must not override the host the user's API key is stored for.

    wandb looks its credential up by exact host string, so a key stored for
    "0.0.0.0:8080" is not found when the base URL says "localhost:8080" even
    though both reach the same server; it then fails with "No API key
    configured", which does not mention the host at all. This project defaulted
    to localhost while the local server was set up as 0.0.0.0, so --wandb could
    not log to it.
    """
    from tensor_beasts.rl.trainer import DEFAULT_WANDB_HOST, resolve_wandb_host

    home = tmp_path / "home"
    (home / ".config" / "wandb").mkdir(parents=True)
    (home / ".config" / "wandb" / "settings").write_text(
        "[default]\nbase_url = http://0.0.0.0:8080\n"
    )
    monkeypatch.setenv("HOME", str(home))

    # The project default gives way to what the user configured.
    assert resolve_wandb_host(DEFAULT_WANDB_HOST) == "http://0.0.0.0:8080"
    # An explicit choice still wins.
    assert resolve_wandb_host("http://elsewhere:9999") == "http://elsewhere:9999"


def test_wandb_host_survives_a_missing_settings_file(tmp_path, monkeypatch):
    from tensor_beasts.rl.trainer import DEFAULT_WANDB_HOST, resolve_wandb_host

    monkeypatch.setenv("HOME", str(tmp_path / "empty"))
    assert resolve_wandb_host(DEFAULT_WANDB_HOST) == DEFAULT_WANDB_HOST
    assert resolve_wandb_host(None) is None


def test_a_metric_keeps_one_name_across_phases():
    """Pretraining and training must report shared metrics under one name.

    Prefixing one phase and not the other produces two half-empty charts for a
    single quantity: one that stops when pretraining ends and one that starts
    there, which reads as a broken run rather than as two phases.
    """
    from tensor_beasts.rl.trainer import wandb_record

    pretrain = wandb_record(
        {"phase": "pretrain", "world_steps": 960, "population": 1186.0, "argmax_agreement": 0.93}
    )
    training = wandb_record({"world_steps": 992, "population": 1025.0, "argmax_agreement": 0.77})

    assert "population" in pretrain and "population" in training
    assert "argmax_agreement" in pretrain and "argmax_agreement" in training
    assert not any(k.startswith("pretrain/") for k in pretrain)
    # The phase survives as a metric, so a chart can still be split on it.
    assert pretrain["phase"] == 0 and "phase" not in training


def test_sparse_evaluation_metrics_are_kept_out_of_the_dense_series():
    """Evaluation runs every few thousand steps; per-segment metrics every 32.

    Mixed together, a metric with two values across 150 rows is drawn as a line
    with enormous gaps, and the space between two distant points is filled in
    as though the value held there.
    """
    from tensor_beasts.rl.trainer import wandb_record

    out = wandb_record(
        {
            "world_steps": 992,
            "population": 1025.0,
            "learned_over_rule_based": 0.859,
            "learned_survived_agent_steps": 32273.0,
            "rule_based_survived_agent_steps": 37572.0,
            "film_typical_steps": 65,
        }
    )

    assert out["eval/learned_over_rule_based"] == 0.859
    assert "eval/learned_survived_agent_steps" in out
    assert "eval/rule_based_survived_agent_steps" in out
    assert out["film/typical_steps"] == 65
    # The dense metric stays where it was.
    assert out["population"] == 1025.0


def test_unmeasured_and_non_numeric_values_are_not_logged_as_metrics():
    """A direction-only run reports the metabolic metrics as NaN on every row.

    Logged, they are three charts that are empty for the whole run. The
    checkpoint path is a string and belongs nowhere on a chart.
    """
    from tensor_beasts.rl.trainer import wandb_record

    out = wandb_record(
        {
            "world_steps": 1024,
            "entropy": 0.58,
            "metabolic_error": float("nan"),
            "metabolic_unit_mean": float("nan"),
            "checkpoint": "outputs/rl/checkpoint.pt",
        }
    )

    assert out == {"world_steps": 1024, "entropy": 0.58}


def test_auto_and_bare_cuda_pick_the_device_with_the_most_free_memory(monkeypatch):
    """A CUDA index is not a stable name for a physical card.

    "cuda" without an index means cuda:0, and which card that is depends on
    CUDA_DEVICE_ORDER: PCI_BUS_ID, which is what nvidia-smi prints, orders by
    bus address, while CUDA's own default of FASTEST_FIRST puts the fastest
    card first. On a mixed machine the two disagree, so a run that asks for
    "cuda" can land on whichever card happens to be index zero. A sweep agent
    inheriting no CUDA_VISIBLE_DEVICES did exactly that, landed on an 11 GB card
    that other processes had already filled, and died in pretraining.
    """
    import torch

    from tensor_beasts.rl import trainer as trainer_module

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    # Device 1 is the roomy one, as on the machine this was found on.
    free = {0: 100 * 2**20, 1: 20 * 2**30}
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (free[i], 24 * 2**30))

    assert trainer_module.resolve_device("auto") == torch.device("cuda:1")
    assert trainer_module.resolve_device("cuda") == torch.device("cuda:1")
    # An explicit index is the caller naming a card, and is obeyed.
    assert trainer_module.resolve_device("cuda:0") == torch.device("cuda:0")
    assert trainer_module.resolve_device("cpu") == torch.device("cpu")


def test_device_selection_survives_a_card_that_cannot_be_queried(monkeypatch):
    import torch

    from tensor_beasts.rl import trainer as trainer_module

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    def flaky(index):
        if index == 0:
            raise RuntimeError("device 0 is not responding")
        return (8 * 2**30, 24 * 2**30)

    monkeypatch.setattr(torch.cuda, "mem_get_info", flaky)
    assert trainer_module.resolve_device("auto") == torch.device("cuda:1")


# ----------------------------------------------------------------------
# Extinction is a world event, not the run's verdict
# ----------------------------------------------------------------------
def _slice_hashes(world):
    """One hash per world of every batched leaf, so untouched worlds can be
    shown untouched to the bit."""
    import hashlib
    worlds = world.num_worlds
    digests = [hashlib.sha256() for _ in range(worlds)]
    for key in sorted(world.td.keys(True, True), key=str):
        value = world.td.get(key)
        if value.dim() == 0 or value.shape[0] != worlds:
            continue
        for index in range(worlds):
            digests[index].update(str(key).encode())
            digests[index].update(value[index].detach().cpu().contiguous().numpy().tobytes())
    return [d.hexdigest() for d in digests]


def test_an_extinct_world_is_reset_in_place_and_the_others_are_untouched(tmp_path):
    trainer = make_trainer(tmp_path, worlds=3, warmup_steps=3, extinction_patience=2)
    trainer.env.reset(seed=0)
    trainer.warmup()
    entity = trainer.env.entity

    # Kill world 1 outright: no biomass, no energy, nothing left to act.
    entity.biomass.data[1].zero_()
    entity.energy.data[1].zero_()
    before = _slice_hashes(trainer.env.world)
    rng_before = torch.get_rng_state()
    assert float(trainer.env.population_per_world()[1]) == 0

    assert trainer.reset_extinct_worlds() == 0, "one empty segment is inside the patience"
    assert trainer.reset_extinct_worlds() == 1, "the second is not"
    assert trainer.world_resets == 1

    after = _slice_hashes(trainer.env.world)
    assert after[0] == before[0] and after[2] == before[2], "the living worlds are bit-identical"
    assert after[1] != before[1]
    assert float(trainer.env.population_per_world()[1]) > 0, "the reset world is populated"
    assert torch.equal(torch.get_rng_state(), rng_before), "the training RNG stream is untouched"
    assert trainer.reset_extinct_worlds() == 0, "a repopulated world is not reset again"

    # And the batch keeps stepping: a collect over the reset world must work.
    rollout, stats = trainer.collect(2)
    assert stats["population_min_world"] > 0
    assert rollout.observation.shape[1] == 3


def test_an_extinct_run_now_spends_its_whole_budget(tmp_path):
    """The old guard stopped the run after `extinction_patience` empty
    segments. A world that dies is reset instead and the run continues to
    the last world step."""
    trainer = make_trainer(tmp_path, worlds=2, warmup_steps=2, extinction_patience=1,
                           total_world_steps=16, segment_steps=4)
    trainer._init_wandb = lambda: None
    original_collect = trainer.collect
    killed = {"done": False}

    def collect_then_kill(steps):
        rollout, stats = original_collect(steps)
        if not killed["done"]:
            trainer.env.entity.biomass.data[0].zero_()
            trainer.env.entity.energy.data[0].zero_()
            killed["done"] = True
        return rollout, stats

    trainer.collect = collect_then_kill
    record = trainer.train(verbose=False)
    assert trainer.world_steps == 16, "the run reached its budget"
    assert record["world_resets"] >= 1
    assert "extinct" not in record


def test_evaluation_warms_the_worlds_up_under_the_rules_before_scoring(tmp_path):
    trainer = make_trainer(tmp_path, eval_seeds=2, eval_steps=3, eval_warmup_steps=5)
    trainer.env.reset(seed=0)
    trainer.evaluate()
    env = trainer._eval_env(2)
    assert env.world.step == 5 + 3, "warmup steps precede the scored steps"

    plain = make_trainer(tmp_path / "plain", eval_seeds=2, eval_steps=3, eval_warmup_steps=0)
    plain.env.reset(seed=0)
    plain.evaluate()
    assert plain._eval_env(2).world.step == 3

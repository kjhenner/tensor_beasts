"""The rule with free values must start as the rule and stay readable.

Small worlds for speed; the checks are about the actor's arithmetic against
the simulation's own policy, not about the ecology.
"""

import pytest
import torch

from tensor_beasts.rl.multiagent import MultiAgentWorldEnv
from tensor_beasts.rl.networks import RulePolicy, architecture_names, build_network
from tensor_beasts.rl.trainer import policy_input

SIZE = (96, 96)


def settled(entity="Predator", steps=30, metabolic=True):
    env = MultiAgentWorldEnv(size=SIZE, device="cpu", entity_name=entity, metabolic=metabolic)
    env.reset(seed=0)
    for _ in range(steps):
        env.rule_based_step()
    return env


@pytest.mark.parametrize("entity", ["Predator", "Herbivore"])
def test_the_actor_starts_as_the_rule(entity):
    env = settled(entity)
    network = build_network("rule", env.observation_channels, metabolic=True, rule=env.rule_spec(), temperature=0.01)
    agreement, error, count = 0.0, 0.0, 0
    for _ in range(8):
        batch = env.rule_based_step()
        out = network.forward_all(policy_input(batch.observation).unsqueeze(0))
        mask = batch.acted
        n = int(mask.sum())
        if n == 0:
            continue
        agreement += float(((out["logits"].argmax(1)[0] == batch.rule_action) & mask).sum())
        error += float(((out["metabolic_mean"][0] - batch.rule_metabolic_unit).abs() * mask).sum())
        count += n
    assert count > 0
    # The rule breaks its knife-edge ties at random, so exact agreement is
    # not reachable; the linear control's analytic start reads the same.
    assert agreement / count > 0.9, f"{entity}: agreement {agreement / count:.3f}"
    assert error / count < 1e-2, f"{entity}: throttle error {error / count:.4f}"


def test_the_values_are_the_configs_and_read_as_a_sentence():
    env = settled()
    network = build_network("rule", env.observation_channels, metabolic=True, rule=env.rule_spec(), temperature=0.01)
    values = network.values()
    config = env.entity.config
    for key, weight in dict(config.navigation_weights).items():
        label = ":".join(key) if isinstance(key, (tuple, list)) else str(key)
        assert values[f"w:{label}"] == pytest.approx(float(weight))
    assert values["sharpness"] == pytest.approx(100.0)
    assert values["sensitivity"] == pytest.approx(float(config.metabolic_sensitivity))
    assert "sharpness=" in network.describe()

    network.set_values({"w:herbivore:scent": 2.0, "sharpness": 50.0, "sensitivity": 1.0})
    again = network.values()
    assert again["w:herbivore:scent"] == pytest.approx(2.0)
    assert again["sharpness"] == pytest.approx(50.0)
    assert again["sensitivity"] == pytest.approx(1.0)
    with pytest.raises(KeyError):
        network.set_values({"w:dragon:scent": 1.0})


def test_five_actor_parameters_and_a_separate_critic():
    env = settled()
    conv = build_network("rule", env.observation_channels, metabolic=True, rule=env.rule_spec(), critic="conv", hidden_channels=8, depth=1)
    linear = build_network("rule", env.observation_channels, metabolic=True, rule=env.rule_spec(), critic="linear")
    actor = {"weight", "log_sharpness", "sensitivity", "metabolic_log_std"}
    for network in (conv, linear):
        names = {name for name, _ in network.named_parameters() if not name.startswith("critic")}
        assert names == actor
        assert network.weight.numel() == 3
    assert linear.num_parameters() == 3 + 1 + 1 + 1 + (env.observation_channels + 1)
    assert conv.num_parameters() > linear.num_parameters()
    headless = build_network("rule", env.observation_channels, metabolic=False, rule=env.rule_spec())
    assert headless.sensitivity is None and not headless.has_metabolic_head
    assert "sensitivity" not in headless.values()
    with pytest.raises(ValueError, match="memory"):
        build_network("rule", env.observation_channels, rule=env.rule_spec(), memory_size=2)
    with pytest.raises(ValueError, match="rule spec"):
        build_network("rule", env.observation_channels)
    assert "rule" in architecture_names()


def test_sharpness_sets_how_stochastic_the_policy_is():
    env = settled()
    batch = env.rule_based_step()
    observation = policy_input(batch.observation).unsqueeze(0)
    network = build_network("rule", env.observation_channels, rule=env.rule_spec(), critic="linear")
    mask = batch.acted

    def entropy(sharpness):
        network.set_values({"sharpness": sharpness})
        log_probs = torch.log_softmax(network(observation)[0], dim=1)[0]
        return float((-(log_probs.exp() * log_probs).sum(0))[mask].mean())

    assert entropy(1.0) > entropy(100.0) > entropy(10_000.0)


def test_gradients_reach_the_five_values_through_ppo(tmp_path):
    """One PPO update on a real rollout must move the actor's values by a
    small, finite amount and leave the ratio exactly one on the first epoch."""
    from tensor_beasts.rl.ppo import PPOConfig
    from tensor_beasts.rl.trainer import Trainer, TrainerConfig

    trainer = Trainer(
        TrainerConfig(size=64, entity="Predator", arch="rule", arch_kwargs={"critic": "linear"}, metabolic=True,
                      bank_worlds=1, bank_steps=8, bank_warmup=5, bank_stride=1, total_world_steps=0,
                      eval_interval=0, checkpoint_interval=0, device="cpu", output_dir=str(tmp_path)),
        PPOConfig(epochs=1, minibatch_steps=4, learning_rate=1e-2),
    )
    assert isinstance(trainer.network, RulePolicy)
    trainer.start_worlds()
    before = trainer.network.values()
    rollout, _ = trainer.collect(4)
    diagnostics = trainer.algorithm.update(trainer.network, rollout, trainer.optimizer)
    assert diagnostics["ratio_max_deviation"] == 0.0
    after = trainer.network.values()
    assert any(abs(after[k] - before[k]) > 0 for k in before), "nothing moved"
    # Adam's first step moves each value by about the learning rate, 1e-2
    # here; the sharpness is log-parametrised so it moves by that fraction.
    assert all(abs(after[k] - before[k]) < 0.05 * max(abs(before[k]), 1.0) for k in before), (
        f"a value jumped: {before} -> {after}"
    )


def test_checkpoint_round_trips_through_the_trainer_and_the_controller(tmp_path):
    from tensor_beasts.config import load_config
    from tensor_beasts.rl.controller import LearnedController
    from tensor_beasts.rl.ppo import PPOConfig
    from tensor_beasts.rl.trainer import Trainer, TrainerConfig
    from tensor_beasts.world import World

    config = dict(size=32, entity="Predator", arch="rule", arch_kwargs={"critic": "linear"}, metabolic=True,
                  bank_worlds=1, bank_steps=4, bank_warmup=2, bank_stride=1, total_world_steps=0,
                  eval_interval=0, checkpoint_interval=0, device="cpu", output_dir=str(tmp_path))
    trainer = Trainer(TrainerConfig(**config), PPOConfig())
    trainer.network.set_values({"w:predator:scent": -0.25, "sharpness": 30.0})
    path = trainer.save_checkpoint(tmp_path / "rule.pt")
    payload = torch.load(path, weights_only=False)
    assert payload["rule_values"]["sharpness"] == pytest.approx(30.0)

    again = Trainer(TrainerConfig(**config), PPOConfig())
    again.load_checkpoint(path, load_optimizer=False)
    assert again.network.values() == pytest.approx(trainer.network.values())

    world_config = load_config("conf/basic_config.yaml")
    world_config.world.size = [32, 32]
    world = World(world_config.world)
    world.initialize()
    controller = LearnedController(world, path, device=torch.device("cpu"))
    assert controller.network.values() == pytest.approx(trainer.network.values())
    for _ in range(2):
        world.update(controller.action())


def test_eval_only_takes_the_architecture_and_entity_from_the_checkpoint(tmp_path):
    """Scoring a checkpoint must need no flags beyond its path: the YAML's
    default is conv on the herbivore, and a rule checkpoint on the predator
    loaded into that raised on the state dict."""
    from train_rl import apply_overrides, build_parser
    from tensor_beasts.rl.ppo import PPOConfig
    from tensor_beasts.rl.trainer import Trainer, TrainerConfig

    trainer = Trainer(
        TrainerConfig(size=32, entity="Predator", arch="rule", arch_kwargs={"critic": "linear"}, metabolic=True,
                      bank_worlds=1, bank_steps=4, bank_warmup=2, bank_stride=1, total_world_steps=0,
                      eval_interval=0, checkpoint_interval=0, device="cpu", output_dir=str(tmp_path)),
        PPOConfig(),
    )
    path = trainer.save_checkpoint(tmp_path / "rule.pt")
    merged = apply_overrides(build_parser().parse_args(["--eval-only", str(path)]))["trainer"]
    assert merged["arch"] == "rule" and merged["arch_kwargs"] == {"critic": "linear"}
    assert merged["entity"] == "Predator" and merged["metabolic"] is True and merged["memory_size"] == 0
    # An explicit flag still wins.
    merged = apply_overrides(build_parser().parse_args(["--eval-only", str(path), "--entity", "Herbivore"]))["trainer"]
    assert merged["entity"] == "Herbivore"

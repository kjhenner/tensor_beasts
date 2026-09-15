"""The learning stack must work for the predator, not only the herbivore.

Small world for speed; not a valid ecology, only plumbing. The predator's
perception is three scents, so its observation is wider, its food is herbivore
biomass and carrion, and its rule-based anchor is its own navigation weights.
"""

import torch

from tensor_beasts.rl.multiagent import MultiAgentWorldEnv
from tensor_beasts.rl.ppo import PPOConfig
from tensor_beasts.rl.trainer import Trainer, TrainerConfig

SIZE = (64, 64)


def test_predator_environment_exposes_the_full_contract():
    env = MultiAgentWorldEnv(size=SIZE, device="cpu", entity_name="Predator", num_metabolic_levels=4)
    _, acted = env.reset(seed=0)
    assert int(acted.sum()) > 0
    names = env.channel_names
    assert env.observation_channels == 3 * 5 + 3 * 4 + 4, "three scents: here+4 neighbours, 4 gradients, own state"
    assert any(n.startswith("herbivore:scent") for n in names), "predators must be able to smell prey"
    for _ in range(3):
        env.world.update()
    batch = env.step(torch.randint(0, 5, SIZE), torch.zeros(SIZE, dtype=torch.long))
    assert batch.rule_action is not None and batch.rule_scores.shape == (5, *SIZE)
    assert batch.rule_metabolic_level is not None
    assert torch.all(batch.reward[~batch.acted] == 0)


def test_predator_trains_evaluates_and_checkpoints(tmp_path):
    trainer = Trainer(
        TrainerConfig(size=SIZE[0], entity="Predator", arch="conv", arch_kwargs={"hidden_channels": 8},
                      metabolic_levels=4, warmup_steps=3, total_world_steps=8, segment_steps=4, eval_interval=0,
                      eval_steps=4, eval_seeds=1, checkpoint_interval=0, device="cpu",
                      output_dir=str(tmp_path)),
        PPOConfig(epochs=1, minibatch_steps=2, imitation_coef=1.0),
    )
    trainer.train(verbose=False)
    summary = trainer.evaluate()
    assert summary["learned_over_rule_based"] >= 0.0
    path = trainer.save_checkpoint(tmp_path / "predator.pt")
    payload = torch.load(path, weights_only=False)
    assert payload["trainer_config"]["entity"] == "Predator"
    assert payload["num_metabolic_levels"] == 4


def test_the_learner_observes_the_prey_field_before_the_prey_moves():
    """Documents a known fairness bug in the predator comparison.

    ``World.update`` runs Herbivore before Predator, and each entity builds its
    observation inside its own update. So the rule-based predator sees the prey
    field *after* the herbivores have moved this step, while
    ``MultiAgentWorldEnv.step`` builds the learner's observation *before*
    ``World.update`` is called at all. The learned predator therefore aims at
    where the prey was.

    Measured at 512 this halves hunting success, 1.46% of steps against 2.95%,
    and drives the population extinct where the rules recover. See
    planning/04-reinforcement-learning.md.

    ``step`` keeps this behaviour, because existing checkpoints and the viewer
    were trained and run against it. ``step_with_policy`` is the fixed path and
    is what the trainer uses; the test below pins that it agrees.
    """
    import torch

    from tensor_beasts.rl.multiagent import MultiAgentWorldEnv

    env = MultiAgentWorldEnv(size=(128, 128), device="cpu", entity_name="Predator")
    env.reset(seed=0)
    for _ in range(30):
        env.rule_based_step()

    entity = env.world.entity_dict["Predator"]
    assert env.world._entity_order.index("Herbivore") < env.world._entity_order.index("Predator"), (
        "the bug depends on prey updating before the predator"
    )

    _, raw = env._observe()
    external = env._rule_decision(raw).move_direction.to(torch.long)

    # Record what the entity's own policy chooses when it runs inside update(),
    # after the herbivores have already moved.
    original = entity.policy
    seen = {}

    class Spy:
        def __call__(self, observation):
            action = original(observation)
            seen["direction"] = action.move_direction.clone()
            return action

    entity.policy = Spy()
    try:
        env.step(external)
    finally:
        entity.policy = original

    internal = seen["direction"].to(torch.long)
    agreement = float((external == internal).float().mean())
    assert agreement < 1.0, (
        "The external action now matches what the entity computes for itself. "
        "If the observation timing was fixed, replace this test with its opposite."
    )


def test_step_with_policy_decides_from_the_same_world_the_entity_sees():
    """The fix for the stale prey field: decide when the entity would decide.

    ``step_with_policy`` asks for the action at the instant the controlled
    entity updates, after the entities before it in dependency order have
    already moved. Handed the rule-based policy, it must therefore choose
    exactly what the entity chooses for itself, which ``step`` does not.

    At 512 this is the difference between eating on 1.46% of steps and 2.81%,
    against the baseline's 2.95%, and between going extinct by step 400 and
    tracking the baseline. See planning/04-reinforcement-learning.md.
    """
    import torch

    from tensor_beasts.rl.multiagent import MultiAgentWorldEnv

    env = MultiAgentWorldEnv(size=(128, 128), device="cpu", entity_name="Predator")
    env.reset(seed=0)
    for _ in range(30):
        env.rule_based_step()

    entity = env.world.entity_dict["Predator"]
    original = entity.policy
    seen = {}

    class Spy:
        def __call__(self, observation):
            action = original(observation)
            seen["direction"] = action.move_direction.clone()
            return action

    def decide(observation):
        _, raw = env._observe()
        return env._rule_decision(raw).move_direction.to(torch.long), None, None

    entity.policy = Spy()
    try:
        batch = env.step_with_policy(decide)
    finally:
        entity.policy = original

    chosen = batch.action
    internal = seen["direction"].to(torch.long)
    assert torch.equal(chosen, internal), (
        "step_with_policy must decide from the same world state the entity's own "
        "policy sees, or a learned policy is compared against a baseline that saw "
        "a fresher world."
    )

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

"""Per-individual memory: carried, inherited, zeroed on death, read and written.

Small worlds for speed; these test bookkeeping, not the ecology. The golden
harness separately guarantees that with memory off nothing changes.
"""

import pytest
import torch

from tensor_beasts.config import load_config
from tensor_beasts.entities.helpers.animal_helpers import perform_move
from tensor_beasts.rl.multiagent import MultiAgentWorldEnv
from tensor_beasts.rl.networks import build_network
from tensor_beasts.world import World

SIZE = (48, 48)
K = 3


def build_env(**kwargs):
    return MultiAgentWorldEnv(size=SIZE, device="cpu", memory_size=K, **kwargs)


def test_memory_is_off_by_default_and_has_no_channels():
    env = MultiAgentWorldEnv(size=SIZE, device="cpu")
    assert env.memory_size == 0
    assert env.entity.memory.data.shape == (*SIZE, 0)
    assert not any(n.startswith("memory/") for n in env.channel_names)


def test_memory_adds_channels_and_the_feature_has_the_right_shape():
    env = build_env()
    assert env.memory_size == K
    assert env.entity.memory.data.shape == (*SIZE, K)
    assert [n for n in env.channel_names if n.startswith("memory/")] == [f"memory/{k}" for k in range(K)]
    observation, _ = env.reset(seed=0)
    assert observation.shape[0] == env.observation_channels == 22 + K


def test_a_write_is_readable_next_step_at_the_individuals_new_cell():
    """The whole point: what an individual writes travels with it."""
    env = build_env()
    env.reset(seed=0)
    for _ in range(5):
        env.world.update()
    torch.manual_seed(1)
    written = torch.rand(K, *SIZE) * 2 - 1
    batch = env.step(torch.randint(0, 5, SIZE), memory=written)
    survived = batch.acted & ~batch.done
    assert bool(survived.any())
    carried = env.entity.memory.data.reshape(-1, K)[batch.successor[survived]]
    expected = written.permute(1, 2, 0)[survived]
    assert torch.allclose(carried, expected, atol=1e-6)
    # And it shows up in the next observation at the new cell.
    observation = env._build_observation()
    for k in range(K):
        channel = observation[env.channel_names.index(f"memory/{k}")].reshape(-1)
        assert torch.allclose(channel[batch.successor[survived]], expected[:, k], atol=1e-6)


def test_offspring_inherit_a_copy_of_memory():
    grid = (5, 5)
    energy = torch.zeros(grid)
    energy[2, 2] = 200
    memory = torch.zeros(*grid, K)
    memory[2, 2] = torch.tensor([0.5, -0.25, 0.9])
    masks = {d: torch.zeros(grid, dtype=torch.uint8) for d in range(1, 5)}
    masks[1][2, 2] = 1  # moves up, reproducing
    slices = [memory[..., k] for k in range(K)]
    perform_move(
        entity_energy=energy, direction_masks=masks, divide_threshold=100,
        divide_fn_self=lambda x: x * 0.5,
        divide_fn_offspring=lambda x: x * 0.5,
        carried_features_self=slices, carried_feature_fns_self=[lambda x: x] * K,
        carried_features_offspring=slices, carried_feature_fns_offspring=[lambda x: x] * K,
    )
    assert torch.allclose(memory[1, 2], torch.tensor([0.5, -0.25, 0.9])), "parent carried it"
    assert torch.allclose(memory[2, 2], torch.tensor([0.5, -0.25, 0.9])), "offspring inherited a copy"


def test_death_zeroes_memory():
    env = build_env()
    env.reset(seed=0)
    entity = env.entity
    alive = entity.biomass.data > 0
    assert bool(alive.any())
    entity.memory.data[alive] = 0.7
    victim = alive.nonzero()[0]
    dead = torch.zeros(SIZE, dtype=torch.bool)
    dead[victim[0], victim[1]] = True
    entity._handle_death(dead)
    assert torch.all(entity.memory.data[victim[0], victim[1]] == 0)
    survivors = alive & ~dead
    assert torch.all(entity.memory.data[survivors] == 0.7)


def test_negative_memory_survives_a_move_unclamped():
    """Memory lives in [-1, 1]. Arrivals in perform_move are summed onto the
    destination without the [0, 255] saturation that energy gets, so a
    negative value must come through exactly and not be rewritten."""
    grid = (5, 5)
    energy = torch.zeros(grid)
    energy[2, 2] = 100
    memory = torch.zeros(*grid, K)
    memory[2, 2] = torch.tensor([-0.5, 0.25, -1.0])
    masks = {d: torch.zeros(grid, dtype=torch.uint8) for d in range(1, 5)}
    masks[1][2, 2] = 1  # moves up without reproducing
    slices = [memory[..., k] for k in range(K)]
    perform_move(
        entity_energy=energy, direction_masks=masks, divide_threshold=250,
        carried_features_self=slices, carried_feature_fns_self=[lambda x: x] * K,
    )
    assert torch.allclose(memory[1, 2], torch.tensor([-0.5, 0.25, -1.0]))
    assert torch.all(memory[2, 2] == 0), "the vacated cell holds no memory"


def test_network_memory_head_is_bounded_and_optional():
    plain = build_network("conv", 22)
    assert "memory" not in plain.forward_all(torch.randn(1, 22, 8, 8))
    with_memory = build_network("conv", 22 + K, memory_size=K)
    out = with_memory.forward_all(torch.randn(2, 22 + K, 8, 8))
    assert out["memory"].shape == (2, K, 8, 8)
    assert out["memory"].abs().max() <= 1.0


def test_trainer_and_controller_round_trip_with_memory(tmp_path):
    from tensor_beasts.rl.controller import LearnedController, apply_checkpoint_requirements
    from tensor_beasts.rl.ppo import PPOConfig
    from tensor_beasts.rl.trainer import Trainer, TrainerConfig

    trainer = Trainer(
        TrainerConfig(size=32, arch="conv", arch_kwargs={"hidden_channels": 8}, memory_size=2,
                      warmup_steps=2, total_world_steps=0, eval_interval=0, checkpoint_interval=0,
                      device="cpu"),
        PPOConfig(),
    )
    action, log_prob, value, metabolic, memory = trainer.act(trainer.env._build_observation())
    assert memory is not None and memory.shape == (2, 32, 32)
    path = tmp_path / "memory.pt"
    trainer.save_checkpoint(path)

    config = load_config("conf/basic_config.yaml")
    config.world.size = [32, 32]
    apply_checkpoint_requirements(config, path)
    assert config.world.entities.Herbivore.memory.size == 2
    world = World(config.world)
    world.initialize()
    controller = LearnedController(world, path, device=torch.device("cpu"))
    for _ in range(3):
        td = controller.action()
        assert td["Herbivore"]["memory"].shape == (32, 32, 2)
        world.update(td)



def test_evaluation_writes_memory_like_training_does():
    """The evaluator once dropped the memory write, so every memory checkpoint
    was scored with its memory stuck at zero. After scoring, living cells must
    hold non-zero memory."""
    from tensor_beasts.rl.ppo import PPOConfig
    from tensor_beasts.rl.trainer import Trainer, TrainerConfig

    trainer = Trainer(
        TrainerConfig(size=32, arch="conv", arch_kwargs={"hidden_channels": 8}, memory_size=2,
                      warmup_steps=2, total_world_steps=0, eval_interval=0, eval_steps=6, eval_seeds=1,
                      checkpoint_interval=0, device="cpu"),
        PPOConfig(),
    )
    env = trainer._eval_env(0)
    env.reset(seed=123)
    trainer._score(env, 6, "learned")
    alive = env.entity.biomass.data > 0
    assert bool(alive.any())
    assert float(env.entity.memory.data[alive].abs().sum()) > 0.0, "evaluation never wrote memory"


def test_pinned_metabolic_level_holds_the_throttle_at_basal():
    from tensor_beasts.rl.ppo import PPOConfig
    from tensor_beasts.rl.trainer import Trainer, TrainerConfig

    trainer = Trainer(
        TrainerConfig(size=32, arch="conv", arch_kwargs={"hidden_channels": 8}, warmup_steps=2,
                      total_world_steps=0, eval_interval=0, eval_steps=4, eval_seeds=1, checkpoint_interval=0,
                      device="cpu", eval_pin_metabolic_level=0),
        PPOConfig(),
    )
    env = trainer._eval_env(0)
    assert env.num_metabolic_levels >= 2, "pinning needs a level-to-rate mapping"
    seen = []
    original = env.step

    def spy(action, metabolic_action=None, memory=None):
        seen.append(metabolic_action)
        return original(action, metabolic_action, memory)

    env.step = spy
    env.reset(seed=1)
    trainer._score(env, 4, "learned")
    assert all(m is not None and bool((m == 0).all()) for m in seen), "throttle was not pinned to level 0"

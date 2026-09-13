"""The bridge from a checkpoint to a live world.

Tiny world for speed; this checks plumbing, not the ecology.
"""

import pytest
import torch
from tensordict import TensorDict

from tensor_beasts.config import load_config
from tensor_beasts.rl.controller import LearnedController
from tensor_beasts.rl.ppo import PPOConfig
from tensor_beasts.rl.trainer import Trainer, TrainerConfig
from tensor_beasts.world import World

SIZE = 32


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    trainer = Trainer(
        TrainerConfig(size=SIZE, arch="conv", arch_kwargs={"hidden_channels": 8}, warmup_steps=2,
                      total_world_steps=0, eval_interval=0, checkpoint_interval=0, device="cpu"),
        PPOConfig(),
    )
    path = tmp_path_factory.mktemp("ckpt") / "checkpoint.pt"
    trainer.save_checkpoint(path)
    return path


def build_world():
    torch.manual_seed(0)
    config = load_config("conf/basic_config.yaml")
    config.world.size = [SIZE, SIZE]
    world = World(config.world)
    world.initialize()
    return world


def test_controller_drives_a_live_world(checkpoint):
    world = build_world()
    controller = LearnedController(world, checkpoint, device=torch.device("cpu"))
    assert "conv" in controller.describe()

    for _ in range(3):
        action = controller.action()
        assert isinstance(action, TensorDict)
        direction = action["Herbivore"]
        assert direction.shape == (SIZE, SIZE)
        assert direction.dtype == torch.long
        assert direction.min() >= 0 and direction.max() <= 4
        world.update(action)


def test_deterministic_controller_is_repeatable(checkpoint):
    world = build_world()
    controller = LearnedController(world, checkpoint, deterministic=True, device=torch.device("cpu"))
    first = controller.action()["Herbivore"]
    second = controller.action()["Herbivore"]
    assert torch.equal(first, second)


def test_channel_mismatch_is_refused(checkpoint):
    """A checkpoint trained against a different perception must not silently
    read garbage; the failure has to name the cause."""
    world = build_world()
    payload = torch.load(checkpoint, weights_only=False)
    payload["observation_channels"] = 99
    bad = checkpoint.parent / "bad.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="observation channels"):
        LearnedController(world, bad, device=torch.device("cpu"))

from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig

from tensor_beasts.config import load_config

# Small world so tests stay fast; the config is otherwise the real one used by
# the simulation, so a config/entity refactor will surface here.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_CONFIG_PATH = str(PROJECT_ROOT / "conf" / "base" / "simulation.yaml")
ENV_SIZE = (16, 16)
ENV_MAX_STEPS = 8


@pytest.fixture(autouse=True)
def cpu_default_device():
    """Keep RL env tests on CPU regardless of the machine's accelerators.

    Restored afterwards: the default device is global, so leaving it set makes
    the result of a later test depend on whether these ran first.
    """
    previous = torch.get_default_device()
    torch.set_default_device("cpu")
    try:
        yield
    finally:
        torch.set_default_device(previous)


@pytest.fixture
def env_world_config() -> DictConfig:
    config = load_config(ENV_CONFIG_PATH)
    config.world.size = list(ENV_SIZE)
    config.world.device = "cpu"
    return config.world


@pytest.fixture
def env(env_world_config):
    from tensor_beasts.rl.envs import TensorBeastsEnv

    environment = TensorBeastsEnv(env_world_config, max_steps=ENV_MAX_STEPS)
    yield environment
    environment.close()

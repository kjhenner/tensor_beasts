"""
Tests for the Gymnasium wrapper around World.

These exist to catch rot: the env talks to the World/Entity API through a
handful of narrow seams (construction signature, entity lookup, feature access,
action routing), and nothing else in the test suite exercises them.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("gymnasium")

from gymnasium import spaces  # noqa: E402
from gymnasium.utils.env_checker import check_env  # noqa: E402

from tensor_beasts.rl.envs import TensorBeastsEnv, make_env  # noqa: E402
from tensor_beasts.rl.envs.world_environment import NUM_ACTIONS  # noqa: E402

from tests.rl.conftest import ENV_CONFIG_PATH, ENV_MAX_STEPS, ENV_SIZE  # noqa: E402


def test_env_constructs_and_declares_spaces(env):
    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.dtype == np.float32
    # (H, W, C): the channel count comes from World.observable.
    assert env.observation_space.shape[:2] == ENV_SIZE
    assert env.observation_space.shape[2] > 0

    assert isinstance(env.action_space, spaces.MultiDiscrete)
    assert env.action_space.shape == ENV_SIZE
    assert np.all(env.action_space.nvec == NUM_ACTIONS)


def test_unknown_entity_raises(env_world_config):
    with pytest.raises(ValueError, match="not present in the world config"):
        TensorBeastsEnv(env_world_config, entity_name="Wombat")


def test_reset_returns_observation_in_space(env):
    obs, info = env.reset(seed=0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert info["step"] == 0
    assert info["population"] >= 0


def test_step_accepts_sampled_action_and_returns_documented_tuple(env):
    env.reset(seed=0)
    action = env.action_space.sample()
    assert env.action_space.contains(action)

    result = env.step(action)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result

    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["step"] == 1


def test_action_reaches_the_simulation(env):
    """A per-cell direction must actually drive herbivore movement."""
    env.reset(seed=0)
    entity = env.entity

    # Place a single herbivore with plenty of energy/biomass in a known cell.
    for feature in (entity.energy, entity.biomass):
        feature.data.zero_()
    entity.energy.data[8, 8] = 200
    entity.biomass.data[8, 8] = 200

    # 2 == "down" in the simulation's direction encoding.
    action = np.full(ENV_SIZE, 2, dtype=np.int64)
    # Force a move attempt regardless of the policy's stochastic move probability.
    torch.manual_seed(0)
    for _ in range(10):
        env.step(action)
        rows = torch.nonzero(entity.biomass.data > 0)
        if rows.numel() and int(rows[:, 0].max()) > 8:
            break
    else:
        pytest.fail("herbivore never moved in the commanded direction")


def test_reward_is_living_population(env):
    env.reset(seed=0)
    entity = env.entity
    entity.biomass.data.zero_()
    entity.energy.data.zero_()
    threshold = entity.config.survival_threshold
    entity.biomass.data[0, 0] = threshold + 50
    entity.energy.data[0, 0] = 200
    entity.biomass.data[0, 2] = threshold + 50
    entity.energy.data[0, 2] = 200

    _, reward, _, _, info = env.step(np.zeros(ENV_SIZE, dtype=np.int64))
    assert reward == float(info["population"])
    assert reward > 0


def test_termination_on_extinction(env):
    env.reset(seed=0)
    entity = env.entity
    entity.biomass.data.zero_()
    entity.energy.data.zero_()

    obs, reward, terminated, truncated, info = env.step(
        np.zeros(ENV_SIZE, dtype=np.int64)
    )
    assert terminated is True
    assert truncated is False
    assert reward == 0.0
    assert info["population"] == 0


def test_truncation_at_max_steps(env):
    env.reset(seed=0)
    entity = env.entity
    # Keep a herbivore alive so truncation, not termination, ends the episode.
    entity.biomass.data[0, 0] = 255
    entity.energy.data[0, 0] = 255

    truncated = False
    for _ in range(ENV_MAX_STEPS):
        entity.biomass.data[0, 0] = 255
        entity.energy.data[0, 0] = 255
        _, _, terminated, truncated, _ = env.step(
            np.zeros(ENV_SIZE, dtype=np.int64)
        )
        assert not terminated
    assert truncated is True


def test_short_rollout_runs(env):
    obs, _ = env.reset(seed=0)
    total_reward = 0.0
    for _ in range(ENV_MAX_STEPS):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        total_reward += reward
        if terminated or truncated:
            break
    assert total_reward >= 0.0


def test_reset_is_seed_reproducible(env):
    obs_a, _ = env.reset(seed=1234)
    obs_b, _ = env.reset(seed=1234)
    np.testing.assert_array_equal(obs_a, obs_b)


def test_reset_does_not_corrupt_observation_shape(env):
    """World.reset() grows shared-feature tensors; the env must not inherit that."""
    obs_a, _ = env.reset(seed=0)
    for _ in range(3):
        obs_b, _ = env.reset(seed=0)
        assert obs_b.shape == obs_a.shape == env.observation_space.shape


def test_passes_gymnasium_env_checker(env_world_config):
    env = TensorBeastsEnv(env_world_config, max_steps=ENV_MAX_STEPS)
    check_env(env, skip_render_check=True)
    env.close()


def test_make_env_from_config_path():
    env = make_env(
        config_path=ENV_CONFIG_PATH,
        size=ENV_SIZE,
        max_steps=ENV_MAX_STEPS,
        device="cpu",
    )
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    env.close()


def test_gymnasium_registration(env_world_config):
    """`tensor-beasts-v0` is registered by importing tensor_beasts.rl."""
    import gymnasium

    import tensor_beasts.rl  # noqa: F401  (registers the env id)

    env = gymnasium.make(
        "tensor-beasts-v0",
        world_config=env_world_config,
        max_steps=ENV_MAX_STEPS,
    )
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    env.close()

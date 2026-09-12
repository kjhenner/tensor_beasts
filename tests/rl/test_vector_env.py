"""Vectorized environments must behave like the single environment, in parallel.

The async path is the one that matters for throughput and is also the one that
can break in ways a single-process test never sees: macOS spawns workers, so
everything crossing the process boundary has to pickle.
"""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from tensor_beasts.rl.envs import make_vector_env


SIZE = (16, 16)
NUM_ENVS = 2


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def vector_env(request):
    env = make_vector_env(
        NUM_ENVS,
        size=SIZE,
        max_steps=50,
        asynchronous=request.param,
        threads_per_env=1,
    )
    yield env
    env.close()


def test_reset_returns_stacked_observations(vector_env):
    obs, info = vector_env.reset(seed=0)
    assert obs.shape[0] == NUM_ENVS
    assert obs.shape[1:3] == SIZE
    assert obs.dtype == np.float32


def test_step_returns_per_env_arrays(vector_env):
    vector_env.reset(seed=0)
    action = vector_env.action_space.sample()
    obs, reward, terminated, truncated, info = vector_env.step(action)

    assert obs.shape[0] == NUM_ENVS
    assert reward.shape == (NUM_ENVS,)
    assert terminated.shape == (NUM_ENVS,)
    assert truncated.shape == (NUM_ENVS,)
    assert np.all(reward >= 0), "reward is a population count and cannot be negative"


def test_rollout_runs(vector_env):
    vector_env.reset(seed=0)
    for _ in range(10):
        obs, reward, terminated, truncated, info = vector_env.step(
            vector_env.action_space.sample()
        )
        assert np.isfinite(obs).all()


def test_action_space_matches_single_env_convention(vector_env):
    action = vector_env.action_space.sample()
    assert action.shape == (NUM_ENVS, *SIZE)
    assert action.min() >= 0 and action.max() <= 4, "five directions, 0 through 4"


def test_thread_cap_divides_cores():
    from tensor_beasts.rl.envs.vector import _thread_cap

    assert _thread_cap(1) >= 1
    assert _thread_cap(1000) == 1, "more workers than cores must not ask for zero threads"
    assert _thread_cap(2) <= _thread_cap(1), "more workers means fewer threads each"

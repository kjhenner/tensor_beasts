"""Vectorized tensor-beasts environments.

Reinforcement learning cares about total environment steps per second, not
about how fast one world runs. Independent worlds are embarrassingly parallel,
so the cheapest large win is to run several of them in separate processes.

Measured on this machine (12 cores, 128x128 worlds, conf/basic_config.yaml):

    processes   torch threads each   per process   total env-steps/s
        1               8               103.6            103.6
        2               4                92.3            184.6
        4               2                71.6            286.3
        8               1                49.1            392.7
       12               1                34.9            418.7

Note the thread split. Torch defaults to using every core in one process, so
without capping threads per worker the processes fight each other and scaling
is much worse than this. :func:`make_vector_env` sets the cap for you.

Why processes rather than one batched world of shape ``(B, H, W)``: batching
the simulation into a leading dimension was measured at roughly 1.7x on CPU,
because at these sizes the work is memory-bandwidth bound rather than
dispatch bound, and it costs an invasive refactor. Processes give about 4x for
a few dozen lines. Batching remains the right answer for GPU, where extra
processes do not help and a single world cannot fill the device; see
planning/03-performance-and-rl-foundation.md.
"""

import os
from typing import Optional, Tuple

from tensor_beasts.rl.envs.world_environment import (
    DEFAULT_ENTITY,
    DEFAULT_MAX_STEPS,
    make_env,
)

DEFAULT_CONFIG = "conf/base/simulation.yaml"


def _thread_cap(num_envs: int) -> int:
    """How many torch threads each worker should get.

    Workers that each grab every core spend their time contending rather than
    simulating, so divide the cores between them.
    """
    cores = os.cpu_count() or 1
    return max(1, cores // max(num_envs, 1))


class _EnvFactory:
    """Builds one environment inside a worker process.

    This is a class rather than a closure because macOS starts workers with
    spawn, so the factory has to survive pickling, and nested functions do not.
    """

    def __init__(
        self,
        config_path: str,
        size: Optional[Tuple[int, int]],
        entity_name: str,
        max_steps: int,
        threads: int,
    ):
        self.config_path = config_path
        self.size = size
        self.entity_name = entity_name
        self.max_steps = max_steps
        self.threads = threads

    def __call__(self):
        import torch

        torch.set_num_threads(self.threads)
        return make_env(
            self.config_path,
            size=self.size,
            entity_name=self.entity_name,
            max_steps=self.max_steps,
        )


def make_vector_env(
    num_envs: int,
    config_path: str = DEFAULT_CONFIG,
    size: Optional[Tuple[int, int]] = None,
    entity_name: str = DEFAULT_ENTITY,
    max_steps: int = DEFAULT_MAX_STEPS,
    asynchronous: bool = True,
    threads_per_env: Optional[int] = None,
):
    """Create ``num_envs`` tensor-beasts environments as one vector environment.

    Args:
        num_envs: How many worlds to run.
        config_path: Simulation config, as understood by ``load_config``.
        size: Optional ``(height, width)`` override.
        entity_name: Registry name of the controlled entity.
        max_steps: Episode length before truncation.
        asynchronous: True runs each world in its own process, which is the
            point of this module. False runs them in one process in sequence,
            which is useful for debugging because tracebacks stay readable.
        threads_per_env: Torch threads per worker. Defaults to cores divided by
            ``num_envs``, which is what the scaling above depends on.

    Returns:
        A ``gymnasium.vector.VectorEnv``.
    """
    import gymnasium as gym

    threads = threads_per_env if threads_per_env is not None else _thread_cap(num_envs)
    factories = [
        _EnvFactory(config_path, size, entity_name, max_steps, threads)
        for _ in range(num_envs)
    ]

    if asynchronous:
        return gym.vector.AsyncVectorEnv(factories)
    return gym.vector.SyncVectorEnv(factories)

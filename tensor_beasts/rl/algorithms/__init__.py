"""Learners, selectable by name.

Mirrors :data:`tensor_beasts.rl.networks.ARCHITECTURES`: a dict from name to
class, plus a constructor that validates the name. Every entry satisfies the
:class:`~tensor_beasts.rl.ppo.Algorithm` protocol,

    diagnostics = algorithm.update(network, rollout, optimizer)

so the trainer can hold any of them without knowing which.

The three differ in how much they are allowed to reuse a segment:

    ppo      one segment, a few epochs, trust region enforced by clipping.
    vtrace   one segment, a few epochs, staleness corrected by truncated
             importance ratios rather than prevented by clipping.
    awr      many segments, drawn from a replay buffer, staleness tolerated by
             regressing onto advantage-weighted behaviour.

They exist as a set because the simulation is the bottleneck --- roughly nine
world steps per second at 512x512 --- so the question "how many gradient steps
can one world step support" is the one that decides throughput, and these three
answer it differently.
"""

from typing import Dict, Type

from tensor_beasts.rl.ppo import PPO, PPOConfig
from tensor_beasts.rl.algorithms.replay import AWR, AWRConfig, SegmentReplayBuffer
from tensor_beasts.rl.algorithms.vtrace import VTrace, VTraceConfig, compute_vtrace

ALGORITHMS: Dict[str, Type] = {
    "ppo": PPO,
    "vtrace": VTrace,
    "awr": AWR,
}

CONFIGS: Dict[str, Type] = {
    "ppo": PPOConfig,
    "vtrace": VTraceConfig,
    "awr": AWRConfig,
}


def build_algorithm(name: str, **kwargs):
    """Construct a learner by name, with hyperparameters as keyword arguments.

    The keyword arguments are the fields of that algorithm's config dataclass,
    so an unknown hyperparameter fails here rather than being silently ignored::

        build_algorithm("vtrace", rho_bar=1.0, epochs=8)

    Raises:
        ValueError: If ``name`` is not a known algorithm.
        TypeError: If a keyword argument is not a field of its config.
    """
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {name!r}. Options: {sorted(ALGORITHMS)}")
    config = CONFIGS[name](**kwargs)
    return ALGORITHMS[name](config)


__all__ = [
    "ALGORITHMS",
    "CONFIGS",
    "build_algorithm",
    "AWR",
    "AWRConfig",
    "SegmentReplayBuffer",
    "VTrace",
    "VTraceConfig",
    "compute_vtrace",
]

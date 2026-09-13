"""World.reset() must produce a usable world, repeatably.

reset() used to be unusable: SharedFeature.zero_init appended the slice count
to its own shape attribute on every call, so shapes grew each time it ran and
the shared scent and energy tensors gained a dimension per reset. The
reinforcement learning environment worked around it by throwing the world away
and building a new one.
"""

import hashlib

import torch

from tensor_beasts.config import load_config
from tensor_beasts.world import World


def state_hash(world: World) -> str:
    digest = hashlib.sha256()
    for key in sorted(world.td.keys(True, True), key=str):
        value = world.td.get(key)
        if isinstance(value, torch.Tensor):
            digest.update(str(key).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build(size=16):
    config = load_config("conf/base/simulation.yaml")
    config.world.size = [size, size]
    world = World(config.world)
    world.initialize()
    return world


def test_repeated_reset_preserves_tensor_shapes():
    world = build()
    shapes_before = {str(k): tuple(world.td.get(k).shape) for k in world.td.keys(True, True)}

    for _ in range(5):
        world.reset()
        shapes_after = {str(k): tuple(world.td.get(k).shape) for k in world.td.keys(True, True)}
        assert shapes_after == shapes_before, "reset changed the shape of world state"


def test_world_runs_after_reset():
    world = build()
    for _ in range(3):
        world.update()
    world.reset()

    assert world.step == 0
    for _ in range(5):
        world.update()
    assert world.observable.ndim == 3


def test_seeded_reset_is_reproducible():
    world = build(size=32)

    torch.manual_seed(7)
    world.reset()
    for _ in range(10):
        world.update()
    first = state_hash(world)

    torch.manual_seed(7)
    world.reset()
    for _ in range(10):
        world.update()
    assert state_hash(world) == first, "same seed must replay the same episode"

    torch.manual_seed(8)
    world.reset()
    for _ in range(10):
        world.update()
    assert state_hash(world) != first, "a different seed must give a different episode"

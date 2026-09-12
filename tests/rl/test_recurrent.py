"""The differentiable memory carry must match the simulation's own carry.

Small worlds for speed; this is bookkeeping, not ecology.
"""

import torch

from tensor_beasts.rl.multiagent import MultiAgentWorldEnv
from tensor_beasts.rl.recurrent import propagate_memory

SIZE = (48, 48)
K = 3


def test_propagated_write_equals_what_the_simulation_carried():
    """Write a random memory, step the world, and check that routing that
    write through the successor map reproduces the memory channels the
    environment actually observes next step, at every cell."""
    env = MultiAgentWorldEnv(size=SIZE, device="cpu", memory_size=K)
    env.reset(seed=0)
    for _ in range(8):
        env.world.update()

    torch.manual_seed(2)
    for _ in range(6):
        write = torch.rand(K, *SIZE) * 2 - 1
        batch = env.step(torch.randint(0, 5, SIZE), memory=write)
        predicted = propagate_memory(write, batch.successor, batch.acted, batch.reproduced, batch.done)

        observation = env._build_observation()
        actual = torch.stack([observation[env.channel_names.index(f"memory/{k}")] for k in range(K)])
        assert torch.allclose(predicted, actual, atol=1e-6), "recomputed carry disagrees with the simulation"


def test_dead_individuals_write_goes_nowhere():
    write = torch.ones(K, 4, 4)
    acted = torch.zeros(4, 4, dtype=torch.bool)
    acted[1, 1] = True
    successor = torch.arange(16).reshape(4, 4)
    successor[1, 1] = 6  # would move to (1, 2)
    done = acted.clone()  # it died
    out = propagate_memory(write, successor, acted, torch.zeros(4, 4, dtype=torch.bool), done)
    assert torch.all(out == 0)


def test_offspring_get_a_copy_and_parent_moves_on():
    write = torch.zeros(K, 4, 4)
    write[:, 1, 1] = torch.tensor([0.2, -0.4, 0.6])
    acted = torch.zeros(4, 4, dtype=torch.bool)
    acted[1, 1] = True
    successor = torch.arange(16).reshape(4, 4)
    successor[1, 1] = 5  # moved to (1, 0)... flat 5 is (1, 1)? use (0, 1) = 1
    successor[1, 1] = 1
    reproduced = acted.clone()
    out = propagate_memory(write, successor, acted, reproduced)
    assert torch.allclose(out[:, 0, 1], write[:, 1, 1]), "parent carried its memory to the new cell"
    assert torch.allclose(out[:, 1, 1], write[:, 1, 1]), "offspring inherited a copy at the origin"
    assert float(out.abs().sum()) == float(write[:, 1, 1].abs().sum()) * 2


def test_gradient_flows_from_read_back_to_write():
    """The reason this function exists: a loss on the next step's read must
    produce a gradient on this step's write."""
    write = (torch.rand(K, 4, 4) * 2 - 1).requires_grad_(True)
    acted = torch.zeros(4, 4, dtype=torch.bool)
    acted[2, 2] = True
    successor = torch.arange(16).reshape(4, 4)
    successor[2, 2] = 10  # moves to (2, 2)->(2, 2)? flat 10 = (2, 2); move to flat 9 = (2, 1)
    successor[2, 2] = 9
    read = propagate_memory(write, successor, acted, torch.zeros(4, 4, dtype=torch.bool))
    read[:, 2, 1].sum().backward()
    assert torch.all(write.grad[:, 2, 2] == 1.0), "gradient reached the cell that wrote"
    assert float(write.grad.abs().sum()) == float(K), "and nowhere else"

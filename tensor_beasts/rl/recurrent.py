"""Following a memory write to the individual that reads it next step.

This is the forward-direction twin of what ``compute_gae`` does backward for
advantages. An individual writes its memory at the cell it acts from at step
``t``; by step ``t + 1`` it is standing at its successor cell, and if it
reproduced, a copy of it is standing where it was. Recurrent training has to
route each step's recomputed write to exactly those cells so that the gradient
at step ``t + 1``'s decision can flow back into step ``t``'s write. The
simulation performs the same carry with its own tensors; the point of doing it
again here, differentiably, is that the simulation's copy is a number and this
one is a function of the network.
"""

from typing import Optional

import torch


def propagate_memory(
    write: torch.Tensor,
    successor: torch.Tensor,
    acted: torch.Tensor,
    reproduced: torch.Tensor,
    done: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Where each step's memory write is read at the next step.

    Args:
        write: (K, H, W), the memory written at each cell this step. Only
            cells where ``acted`` is true carry an individual; the rest is
            ignored.
        successor: (H, W) int64, flat index of the cell each acting individual
            occupies next step.
        acted: (H, W) bool.
        reproduced: (H, W) bool, individuals that divided this step. Their
            offspring stands at the origin cell and inherits a copy.
        done: (H, W) bool, individuals that died during this step. Their write
            goes nowhere. Optional; without it a dead individual's write lands
            at its successor cell, which the simulation will have zeroed.

    Returns:
        (K, H, W), the memory field as read at the next step, zero everywhere
        nobody wrote. Differentiable with respect to ``write``.
    """
    channels, height, width = write.shape
    flat_write = write.reshape(channels, -1)
    read = torch.zeros_like(flat_write)

    carried = acted if done is None else (acted & ~done)
    origin = carried.reshape(-1).nonzero(as_tuple=True)[0]
    if origin.numel():
        destination = successor.reshape(-1)[origin]
        read.index_copy_(1, destination, flat_write[:, origin])

    # Offspring inherit a copy at the cell the parent left. A parent that
    # reproduced also moved, so origin and destination never collide here.
    inherited = (acted & reproduced).reshape(-1).nonzero(as_tuple=True)[0]
    if inherited.numel():
        read.index_copy_(1, inherited, flat_write[:, inherited])

    return read.reshape(channels, height, width)

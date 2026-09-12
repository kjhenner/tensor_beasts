"""Estimating training memory before it kills the machine.

A sweep of three parallel trials on a 256x256 world was killed by the operating
system for running out of memory, and the cause was not where I expected. The
stored rollout is modest; the backward pass is not. A fully convolutional policy
keeps one activation tensor per convolution at full grid resolution, and the
minibatch dimension multiplies all of them:

    bytes ~ minibatch_steps * hidden_channels * height * width * 4 * conv_layers

Measured at 256x256:

    architecture  hidden  minibatch_steps   activations   stored rollout
    conv              64                4        0.34 GB          0.12 GB
    conv              64               16        1.34 GB          0.12 GB
    dilated           48               16        1.21 GB          0.12 GB
    residual          64               16        2.95 GB          0.12 GB

Three residual workers at the default minibatch therefore ask for about 9 GB of
activations alone, which is what happened.

The lever is ``minibatch_steps``, and it is a cheap one: minibatching here is
over whole timesteps rather than over individuals, because the policy is
convolutional and one pass covers every agent on the grid. Halving
``minibatch_steps`` halves peak memory and costs only more, smaller passes over
the same data.
"""

from typing import Optional

import torch
import torch.nn as nn

BYTES_PER_FLOAT32 = 4


def count_conv_layers(network: nn.Module) -> int:
    return sum(1 for module in network.modules() if isinstance(module, nn.Conv2d))


def widest_channel_count(network: nn.Module, default: int = 64) -> int:
    widths = [
        module.out_channels for module in network.modules() if isinstance(module, nn.Conv2d)
    ]
    return max(widths) if widths else default


def estimate_activation_bytes(
    network: nn.Module,
    minibatch_steps: int,
    height: int,
    width: int,
    safety_factor: float = 2.0,
) -> int:
    """Peak bytes the backward pass will hold for one minibatch.

    The safety factor covers what this deliberately does not model: the gradient
    buffers, Adam's two moment estimates, the softmax and gather temporaries, and
    the allocator's fragmentation. It is a planning number, not a guarantee.
    """
    layers = count_conv_layers(network)
    channels = widest_channel_count(network)
    per_layer = minibatch_steps * channels * height * width * BYTES_PER_FLOAT32
    return int(per_layer * layers * safety_factor)


def estimate_rollout_bytes(
    segment_steps: int,
    observation_channels: int,
    height: int,
    width: int,
) -> int:
    """Bytes held by one stored segment.

    Observations are float16, which is why they are the smaller half of this;
    the remaining per-cell fields are two int64, three float32 and two bool.
    """
    observations = segment_steps * observation_channels * height * width * 2
    per_cell_fields = segment_steps * height * width * (16 + 12 + 2)
    return int(observations + per_cell_fields)


def estimate_training_bytes(
    network: nn.Module,
    minibatch_steps: int,
    segment_steps: int,
    observation_channels: int,
    height: int,
    width: int,
) -> int:
    """Rough peak resident bytes for one training process."""
    return estimate_activation_bytes(
        network, minibatch_steps, height, width
    ) + estimate_rollout_bytes(segment_steps, observation_channels, height, width)


def total_system_bytes() -> Optional[int]:
    """Physical RAM, or None if it cannot be determined on this platform."""
    try:
        import subprocess

        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0 and out.stdout.strip().isdigit():
            return int(out.stdout.strip())
    except Exception:  # noqa: BLE001 - best effort only
        pass
    try:
        import os

        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


def available_system_bytes() -> Optional[int]:
    """Memory actually free right now, or None if unknown.

    Total physical RAM is the wrong budget. A sweep sized against it drove a
    machine that already had other things resident six gigabytes into swap,
    throughput fell 20 to 100 fold, and the main process was killed. On macOS
    this reads vm_stat and counts free plus inactive pages.
    """
    try:
        import re
        import subprocess

        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            page = re.search(r"page size of (\d+) bytes", out.stdout)
            free = re.search(r"Pages free:\s+(\d+)", out.stdout)
            inactive = re.search(r"Pages inactive:\s+(\d+)", out.stdout)
            if page and free and inactive:
                return (int(free.group(1)) + int(inactive.group(1))) * int(page.group(1))
    except Exception:  # noqa: BLE001 - best effort only
        pass
    try:
        import os

        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


def format_bytes(value: int) -> str:
    gigabytes = value / 1e9
    if gigabytes >= 1.0:
        return f"{gigabytes:.2f} GB"
    return f"{value / 1e6:.0f} MB"


def workers_that_fit(
    per_worker_bytes: int,
    requested: int,
    budget_fraction: float = 0.5,
) -> int:
    """How many parallel trials fit, given a fraction of memory free right now.

    Budgeted against available memory, not physical RAM, and half of that by
    default. Overshooting does not fail loudly: it pushes the machine into swap,
    every worker slows by an order of magnitude or more, and eventually the
    parent is killed and the workers are orphaned. Finishing slowly is not the
    failure mode; not finishing is.
    """
    total = available_system_bytes() or total_system_bytes()
    if total is None or per_worker_bytes <= 0:
        return requested
    affordable = max(1, int((total * budget_fraction) // per_worker_bytes))
    return max(1, min(requested, affordable))

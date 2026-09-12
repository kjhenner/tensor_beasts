"""Shared policy networks for per-individual agents.

Every network here is fully convolutional and produces one action distribution
and one value per cell. That is not an implementation convenience, it is the
framing: all individuals share weights, so a single forward pass over the grid
evaluates every agent at once, and translation equivariance encodes the fact
that an animal's situation does not depend on where in the world it is standing.

There is no global pooling anywhere, deliberately. Pooling would let an
individual's decision depend on the state of the far side of the map, which it
cannot perceive. A network's receptive field is therefore its perception range,
and is reported by ``receptive_field`` so it can be compared against the
simulation's own perception radius.

The interface is::

    logits, value = network(observation)

with ``observation`` of shape ``(B, C, H, W)``, ``logits`` of shape
``(B, 5, H, W)`` over [stay, up, down, left, right], and ``value`` of shape
``(B, H, W)``.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ACTIONS = 5


def orthogonal_init(module: nn.Module, gain: float = 1.0) -> nn.Module:
    """Orthogonal weight init, the usual default for on-policy actor-critics."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    return module


class ActorCritic(nn.Module):
    """Base class fixing the contract and the head initialization.

    Subclasses build ``self.trunk`` and set ``self.trunk_channels``; this class
    supplies the policy and value heads as 1x1 convolutions, which keeps the
    per-cell structure intact.

    The policy head uses a small gain so the initial distribution over the five
    directions is close to uniform. Starting near-deterministic is a common and
    hard-to-diagnose cause of on-policy collapse.
    """

    def __init__(self, in_channels: int, trunk_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.trunk_channels = trunk_channels
        self.policy_head = orthogonal_init(nn.Conv2d(trunk_channels, NUM_ACTIONS, 1), gain=0.01)
        self.value_head = orthogonal_init(nn.Conv2d(trunk_channels, 1, 1), gain=1.0)

    def features(self, observation: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features(observation)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(1)
        return logits, value

    @property
    def receptive_field(self) -> int:
        """Side length in cells that one output position can see."""
        field = 1
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                field += (module.kernel_size[0] - 1) * module.dilation[0]
        return field

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LinearPolicy(ActorCritic):
    """A single 1x1 convolution: no hidden layer, no spatial extent.

    This is the sanity check, not a contender. The rule-based policy is a linear
    function of exactly these observation channels, so this network can
    represent it exactly. If training cannot get this to match the baseline,
    the problem is in the learning setup rather than in model capacity, and
    everything below is a waste of time until that is fixed.
    """

    def __init__(self, in_channels: int):
        super().__init__(in_channels, in_channels)
        self.trunk = nn.Identity()

    def features(self, observation: torch.Tensor) -> torch.Tensor:
        return observation


class ConvActorCritic(ActorCritic):
    """Small convolutional trunk. The default workhorse.

    Three 3x3 layers give a 7x7 receptive field, so an individual sees roughly
    three cells in each direction, modestly further than the simulation's own
    one-cell perception. That headroom is intentional: the interesting question
    is whether a learner can exploit structure the hand-written rules cannot.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64, depth: int = 3):
        super().__init__(in_channels, hidden_channels)
        layers = []
        channels = in_channels
        for _ in range(depth):
            layers.append(orthogonal_init(nn.Conv2d(channels, hidden_channels, 3, padding=1), gain=2.0**0.5))
            layers.append(nn.ReLU(inplace=True))
            channels = hidden_channels
        self.trunk = nn.Sequential(*layers)

    def features(self, observation: torch.Tensor) -> torch.Tensor:
        return self.trunk(observation)


class ChannelNorm(nn.Module):
    """Normalize over channels at each cell independently.

    Every standard normalization layer is wrong here, and wrong in a way that is
    easy to miss. Batch norm mixes statistics across timesteps whose population
    is booming or busting. Group norm and layer norm as usually applied to
    images pool over height and width, which means a change anywhere on the map
    shifts every cell's activations. That silently breaks the property this
    whole framing rests on: an individual may only condition on what it can
    perceive. A test in tests/rl/test_networks.py perturbs one corner and
    asserts the opposite corner does not move, and group norm fails it.

    Normalizing across channels at a single position leaves locality intact.
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class ResidualBlock(nn.Module):
    """Pre-activation residual block using per-cell channel normalization."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = ChannelNorm(channels)
        self.conv1 = orthogonal_init(nn.Conv2d(channels, channels, 3, padding=1), gain=2.0**0.5)
        self.norm2 = ChannelNorm(channels)
        self.conv2 = orthogonal_init(nn.Conv2d(channels, channels, 3, padding=1), gain=2.0**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.relu(self.norm1(x)))
        h = self.conv2(F.relu(self.norm2(h)))
        return x + h


class ResidualActorCritic(ActorCritic):
    """Deeper residual trunk, for when the small network saturates.

    Follows the IMPALA convolutional stack in spirit but without any
    downsampling, since every cell needs its own output at full resolution.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64, blocks: int = 4):
        super().__init__(in_channels, hidden_channels)
        self.stem = orthogonal_init(nn.Conv2d(in_channels, hidden_channels, 3, padding=1), gain=2.0**0.5)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_channels) for _ in range(blocks)])
        self.out_norm = ChannelNorm(hidden_channels)

    def features(self, observation: torch.Tensor) -> torch.Tensor:
        h = self.stem(observation)
        h = self.blocks(h)
        return F.relu(self.out_norm(h))


class DilatedActorCritic(ActorCritic):
    """Dilated stack: a wide receptive field for few parameters and little compute.

    Scent is a diffused field, so the informative structure is a smooth gradient
    over many cells rather than fine local detail. Dilation buys range cheaply,
    which is the right trade for that kind of signal. Dilations 1, 2, 4, 8 give a
    31x31 field from four layers.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 48, dilations: Tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__(in_channels, hidden_channels)
        layers = []
        channels = in_channels
        for dilation in dilations:
            layers.append(
                orthogonal_init(
                    nn.Conv2d(channels, hidden_channels, 3, padding=dilation, dilation=dilation),
                    gain=2.0**0.5,
                )
            )
            layers.append(nn.ReLU(inplace=True))
            channels = hidden_channels
        self.trunk = nn.Sequential(*layers)

    def features(self, observation: torch.Tensor) -> torch.Tensor:
        return self.trunk(observation)


ARCHITECTURES = {
    "linear": LinearPolicy,
    "conv": ConvActorCritic,
    "residual": ResidualActorCritic,
    "dilated": DilatedActorCritic,
}


def build_network(name: str, in_channels: int, **kwargs) -> ActorCritic:
    """Construct a network by name. See ARCHITECTURES for the options.

    Always built on the CPU, whatever the default device is, and moved by the
    caller. Orthogonal initialization needs a QR decomposition, which Metal does
    not implement; the interactive viewer sets Metal as the default device and
    crashed here the first time it loaded a checkpoint. Building on the CPU
    costs one small transfer and removes the trap for every caller.
    """
    if name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {name!r}. Options: {sorted(ARCHITECTURES)}")
    with torch.device("cpu"):
        return ARCHITECTURES[name](in_channels, **kwargs)

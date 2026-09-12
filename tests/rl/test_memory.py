"""Memory estimation, added after a sweep was killed by the operating system.

The estimate does not need to be accurate to the byte. It needs to get the
*scaling* right, because the mistake it exists to prevent is choosing a worker
count from the core count while the real constraint is the backward pass.
"""

import pytest
import torch

from tensor_beasts.rl.memory import (
    count_conv_layers,
    estimate_activation_bytes,
    estimate_rollout_bytes,
    estimate_training_bytes,
    format_bytes,
    widest_channel_count,
    workers_that_fit,
)
from tensor_beasts.rl.networks import build_network


CHANNELS = 14


def test_activation_memory_scales_with_minibatch_and_area():
    network = build_network("conv", CHANNELS)
    base = estimate_activation_bytes(network, minibatch_steps=4, height=128, width=128)

    doubled_batch = estimate_activation_bytes(network, minibatch_steps=8, height=128, width=128)
    assert doubled_batch == pytest.approx(2 * base, rel=1e-6)

    doubled_side = estimate_activation_bytes(network, minibatch_steps=4, height=256, width=256)
    assert doubled_side == pytest.approx(4 * base, rel=1e-6), "area, not side length"


def test_deeper_networks_cost_more():
    minibatch, size = 8, 128
    costs = {
        name: estimate_activation_bytes(build_network(name, CHANNELS), minibatch, size, size)
        for name in ("linear", "conv", "residual")
    }
    assert costs["linear"] < costs["conv"] < costs["residual"]


def test_the_backward_pass_dominates_the_stored_rollout():
    """The finding that motivated this module.

    Sizing a sweep from the rollout alone is what ran the machine out of memory.
    """
    network = build_network("residual", CHANNELS)
    activations = estimate_activation_bytes(network, minibatch_steps=16, height=256, width=256)
    rollout = estimate_rollout_bytes(
        segment_steps=64, observation_channels=CHANNELS, height=256, width=256
    )
    assert activations > 10 * rollout


def test_conv_layer_and_width_introspection():
    network = build_network("conv", CHANNELS, hidden_channels=32, depth=3)
    # three trunk convolutions plus the policy and value heads
    assert count_conv_layers(network) == 5
    assert widest_channel_count(network) == 32


def test_linear_policy_has_no_hidden_activations_to_speak_of():
    linear = build_network("linear", CHANNELS)
    conv = build_network("conv", CHANNELS)
    assert estimate_training_bytes(linear, 8, 64, CHANNELS, 128, 128) < estimate_training_bytes(
        conv, 8, 64, CHANNELS, 128, 128
    )


def test_workers_that_fit_never_returns_zero_or_exceeds_the_request():
    assert workers_that_fit(per_worker_bytes=10**15, requested=8) == 1, "always allow one"
    assert workers_that_fit(per_worker_bytes=1, requested=4) == 4, "never exceed the request"
    assert workers_that_fit(per_worker_bytes=0, requested=4) == 4, "degrade gracefully"


def test_workers_that_fit_respects_the_budget_fraction():
    generous = workers_that_fit(per_worker_bytes=10**9, requested=64, budget_fraction=0.8)
    stingy = workers_that_fit(per_worker_bytes=10**9, requested=64, budget_fraction=0.1)
    assert generous >= stingy


def test_format_bytes_reads_sensibly():
    assert format_bytes(2_500_000_000) == "2.50 GB"
    assert format_bytes(250_000_000) == "250 MB"


def test_estimate_is_within_an_order_of_magnitude_of_reality():
    """Sanity-check the model against an actual forward and backward pass.

    Exactness is not the goal and is not achievable; being wrong by a factor of
    ten would make the guard useless in either direction.
    """
    size, minibatch = 64, 4
    network = build_network("conv", CHANNELS, hidden_channels=32, depth=3)
    predicted = estimate_activation_bytes(network, minibatch, size, size)

    observation = torch.randn(minibatch, CHANNELS, size, size)
    logits, value = network(observation)
    (logits.mean() + value.mean()).backward()

    # What the model claims to count: one float32 activation per convolution.
    counted = minibatch * 32 * size * size * 4 * count_conv_layers(network)
    assert 0.1 * counted < predicted < 10 * counted

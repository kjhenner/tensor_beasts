"""Policy networks must stay per-cell and translation equivariant.

Both properties are load-bearing rather than aesthetic. Per-cell output is what
lets one forward pass serve every individual. Translation equivariance encodes
that an animal's situation does not depend on where in the world it stands, and
it is also the property that would silently break if anyone added global pooling
or a positional input, which would let an individual condition on parts of the
map it cannot perceive.
"""

import pytest
import torch

from tensor_beasts.rl.networks import ARCHITECTURES, NUM_ACTIONS, build_network


CHANNELS = 14
SIZE = 24


@pytest.mark.parametrize("name", sorted(ARCHITECTURES))
def test_shapes(name):
    network = build_network(name, CHANNELS)
    logits, value = network(torch.randn(3, CHANNELS, SIZE, SIZE))
    assert logits.shape == (3, NUM_ACTIONS, SIZE, SIZE)
    assert value.shape == (3, SIZE, SIZE)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()


@pytest.mark.parametrize("name", sorted(ARCHITECTURES))
def test_translation_equivariance(name):
    network = build_network(name, CHANNELS).eval()
    observation = torch.randn(1, CHANNELS, SIZE, SIZE)
    shift = 4

    with torch.no_grad():
        logits, value = network(observation)
        shifted_logits, shifted_value = network(torch.roll(observation, shift, dims=-1))

    # Compare away from the borders, where zero padding legitimately differs.
    interior = slice(shift + network.receptive_field, SIZE - network.receptive_field)
    assert torch.allclose(
        torch.roll(logits, shift, dims=-1)[..., interior, interior],
        shifted_logits[..., interior, interior],
        atol=1e-4,
    )
    assert torch.allclose(
        torch.roll(value, shift, dims=-1)[..., interior, interior],
        shifted_value[..., interior, interior],
        atol=1e-4,
    )


@pytest.mark.parametrize("name", sorted(ARCHITECTURES))
def test_a_cell_is_unaffected_by_distant_cells(name):
    """Perturbing a far-away cell must not change this cell's decision.

    This is the test that catches an accidental global pooling layer, which
    would let an individual react to things it cannot sense.
    """
    network = build_network(name, CHANNELS).eval()
    size = 64
    observation = torch.randn(1, CHANNELS, size, size)

    with torch.no_grad():
        before, _ = network(observation)
        perturbed = observation.clone()
        perturbed[..., 0, 0] += 10.0
        after, _ = network(perturbed)

    far = size - 1
    assert torch.allclose(before[..., far, far], after[..., far, far], atol=1e-6)


def test_receptive_field_is_reported_and_ordered():
    fields = {name: build_network(name, CHANNELS).receptive_field for name in ARCHITECTURES}
    assert fields["linear"] == 1, "a 1x1 policy sees only its own cell"
    assert fields["conv"] > fields["linear"]
    assert fields["dilated"] > fields["conv"], "dilation is there to buy range cheaply"


def test_linear_policy_can_express_a_direction_preference():
    """The linear policy is the sanity check: it must be able to represent the
    kind of function the rule-based policy computes, namely a weighted vote over
    directional channels. If it cannot, the sanity check is worthless."""
    network = build_network("linear", CHANNELS)
    observation = torch.zeros(1, CHANNELS, 4, 4)
    # Channels 1..4 are the four directional readings of the first feature.
    observation[:, 2] = 1.0  # strong signal in the 'down' direction

    with torch.no_grad():
        network.policy_head.weight.zero_()
        network.policy_head.bias.zero_()
        network.policy_head.weight[2, 2] = 5.0  # map that channel onto action 2
        logits, _ = network(observation)

    assert torch.all(logits.argmax(dim=1) == 2)


def test_initial_policy_is_close_to_uniform():
    """A near-deterministic start is a classic and hard-to-spot cause of
    on-policy collapse, so the policy head is initialized with a small gain."""
    network = build_network("conv", CHANNELS)
    with torch.no_grad():
        logits, _ = network(torch.randn(4, CHANNELS, SIZE, SIZE))
    probabilities = logits.softmax(dim=1)
    assert probabilities.max() < 0.5, "initial action distribution should not be peaked"


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_network("transformer", CHANNELS)



@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs a Metal device")
def test_networks_can_be_built_under_a_metal_default_device():
    """The viewer sets Metal as the default device. Orthogonal init needs a QR
    decomposition that Metal lacks, and this is the exact path that crashed."""
    previous = torch.get_default_device()
    torch.set_default_device("mps")
    try:
        for name in ARCHITECTURES:
            network = build_network(name, CHANNELS).to("mps")
            logits, value = network(torch.randn(1, CHANNELS, 16, 16, device="mps"))
            assert logits.device.type == "mps"
    finally:
        torch.set_default_device(previous)

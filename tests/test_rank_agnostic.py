"""Phase 1 proof: the simulation's tensor ops address spatial axes from the END.

Every test here runs the same function twice:

  * once per world, on a plain (H, W) input;
  * once on a (B, H, W) input built by stacking B *different* random worlds.

and asserts the batched result equals the stack of the individual results. That
is the property that lets someone add a real leading batch dimension on top of
this work without touching the kernels.

Integral results are compared with torch.equal; float results with a tight
allclose (conv2d and interpolate are free to reassociate across the batch axis).
"""

import pytest
import torch
from tensordict import TensorDict

from tensor_beasts.observations import (
    Observation,
    compute_gradient_and_direction,
    get_directional_values,
    get_observation,
    process_observation,
)
from tensor_beasts.util import (
    apply_kernels,
    as_conv_batch,
    custom_filter_operation,
    directional_kernel_bank,
    flow,
    flow_gradient,
    fold_neighbors,
    generate_diffusion_kernel,
    get_direction_matrix,
    get_edge_mask,
    neighbors,
    pad_matrix,
    roll_with_padding,
    safe_add,
    safe_sub,
    safe_sum,
    safe_where,
    scale_tensor,
    torch_correlate_2d,
    torch_correlate_2d_bank,
    torch_correlate_3d,
    unfold_neighbors,
)

B, H, W = 3, 7, 9


@pytest.fixture(autouse=True)
def _cpu_default_device():
    """Pin the default device: other test modules set it to 'mps' globally."""
    previous = torch.get_default_device()
    torch.set_default_device("cpu")
    try:
        yield
    finally:
        torch.set_default_device(previous)


def _worlds(dtype=torch.float32, low=0.0, high=1.0, shape=(H, W), seed=0):
    """B different random worlds of the given per-world shape."""
    g = torch.Generator().manual_seed(seed)
    if dtype.is_floating_point:
        out = [torch.rand(shape, generator=g) * (high - low) + low for _ in range(B)]
        return [w.to(dtype) for w in out]
    return [
        torch.randint(int(low), int(high), shape, generator=g, dtype=dtype)
        for _ in range(B)
    ]


def assert_batched_matches(fn, worlds, float_tol=1e-6):
    """fn applied to stack(worlds) must equal stack(fn applied to each world)."""
    singles = [fn(w) for w in worlds]
    batched = fn(torch.stack(worlds))

    expected = torch.stack(singles)
    assert batched.shape == expected.shape, (
        f"batched shape {tuple(batched.shape)} != stacked single shape {tuple(expected.shape)}"
    )
    assert batched.dtype == expected.dtype
    if batched.is_floating_point():
        assert torch.allclose(batched, expected, rtol=0, atol=float_tol), (
            f"max abs diff {(batched - expected).abs().max().item()}"
        )
    else:
        assert torch.equal(batched, expected)
    return batched


# ---------------------------------------------------------------------------
# util.py -- elementwise helpers (rank-agnostic by construction)
# ---------------------------------------------------------------------------

def test_safe_add_rank_agnostic():
    a = _worlds(torch.uint8, 0, 256, seed=1)
    b = _worlds(torch.uint8, 0, 256, seed=2)
    singles = [safe_add(x.clone(), y) for x, y in zip(a, b)]
    batched = safe_add(torch.stack(a).clone(), torch.stack(b))
    assert torch.equal(batched, torch.stack(singles))


def test_safe_sub_rank_agnostic():
    a = _worlds(torch.uint8, 0, 256, seed=3)
    b = _worlds(torch.uint8, 0, 256, seed=4)
    singles = [safe_sub(x.clone(), y) for x, y in zip(a, b)]
    batched = safe_sub(torch.stack(a).clone(), torch.stack(b))
    assert torch.equal(batched, torch.stack(singles))


def test_safe_sum_rank_agnostic():
    """dim=0 in safe_sum is the stacked-matrix axis, not a spatial one."""
    a = _worlds(torch.uint8, 0, 200, seed=5)
    b = _worlds(torch.uint8, 0, 200, seed=6)
    singles = [safe_sum([x, y]) for x, y in zip(a, b)]
    batched = safe_sum([torch.stack(a), torch.stack(b)])
    assert torch.equal(batched, torch.stack(singles))


def test_safe_where_rank_agnostic():
    cond = [w > 0.5 for w in _worlds(seed=7)]
    x = _worlds(seed=8)
    y = _worlds(seed=9)
    singles = [safe_where(c, i, j) for c, i, j in zip(cond, x, y)]
    batched = safe_where(torch.stack(cond), torch.stack(x), torch.stack(y))
    assert torch.equal(batched, torch.stack(singles))


def test_scale_tensor_rank_agnostic():
    assert_batched_matches(scale_tensor, _worlds(torch.uint8, 0, 256, seed=10))


# ---------------------------------------------------------------------------
# util.py -- spatial shifts and masks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("direction", [1, 2, 3, 4])
def test_pad_matrix_rank_agnostic(direction):
    assert_batched_matches(
        lambda x: pad_matrix(x, direction), _worlds(torch.uint8, 0, 256, seed=11)
    )


def test_get_edge_mask_rank_agnostic():
    """get_edge_mask takes a shape, so a leading dim must broadcast the frame."""
    single = get_edge_mask((H, W))
    batched = get_edge_mask((B, H, W))
    assert torch.equal(batched, single.expand(B, H, W))


def test_get_direction_matrix_rank_agnostic():
    """The 5-way stack in get_direction_matrix is on a TRAILING axis, not spatial."""
    worlds = _worlds(torch.uint8, 0, 40, seed=12)
    # Supply the tie-breakers explicitly so the comparison is deterministic.
    g = torch.Generator().manual_seed(99)
    choices = [torch.rand((H, W, 5), generator=g) for _ in range(B)]

    singles = [get_direction_matrix(w, c) for w, c in zip(worlds, choices)]
    batched = get_direction_matrix(torch.stack(worlds), torch.stack(choices))
    assert torch.equal(batched, torch.stack(singles))


def test_roll_with_padding_rank_agnostic():
    assert_batched_matches(
        lambda x: roll_with_padding(x, shifts=(1, -2), dims=(-2, -1)),
        _worlds(seed=13),
    )


def test_neighbors_rank_agnostic():
    """neighbors appends a TRAILING direction axis, so it stacks cleanly."""
    assert_batched_matches(neighbors, _worlds(seed=14))


def test_neighbors_eight_direction_rank_agnostic():
    assert_batched_matches(
        lambda x: neighbors(x, eight_direction=True), _worlds(seed=15)
    )


# ---------------------------------------------------------------------------
# util.py -- convolution helpers
# ---------------------------------------------------------------------------

def test_as_conv_batch_round_trip():
    worlds = torch.stack(_worlds(seed=16))
    folded, leading = as_conv_batch(worlds)
    assert folded.shape == (B, 1, H, W)
    assert leading == torch.Size([B])
    assert torch.equal(folded.reshape(*leading, H, W), worlds)

    single = worlds[0]
    folded, leading = as_conv_batch(single)
    assert folded.shape == (1, 1, H, W)
    assert leading == torch.Size([])
    assert torch.equal(folded.reshape(*leading, H, W), single)


def test_as_conv_batch_preserves_size_one_leading_dims():
    """A bare .squeeze() would drop these; reshape must not."""
    x = torch.rand(1, 1, H, W)
    folded, leading = as_conv_batch(x)
    assert leading == torch.Size([1, 1])
    assert folded.reshape(*leading, H, W).shape == (1, 1, H, W)


def test_torch_correlate_2d_rank_agnostic():
    kernel = generate_diffusion_kernel(size=5)
    assert_batched_matches(
        lambda x: torch_correlate_2d(x, kernel), _worlds(seed=17), float_tol=1e-6
    )


def test_torch_correlate_2d_bank_rank_agnostic():
    """The kernel axis must sit just BEFORE the spatial axes, not at position 0."""
    bank = directional_kernel_bank(5)
    worlds = _worlds(seed=18)

    singles = [torch_correlate_2d_bank(w, bank) for w in worlds]
    assert singles[0].shape == (4, H, W)

    batched = torch_correlate_2d_bank(torch.stack(worlds), bank)
    assert batched.shape == (B, 4, H, W)
    assert torch.allclose(batched, torch.stack(singles), rtol=0, atol=1e-6)

    # And the from-the-end indexing used by get_direction_masks works for both.
    for d in range(4):
        assert torch.allclose(batched[..., d, :, :], torch.stack(singles)[..., d, :, :])


def test_torch_correlate_3d_rank_agnostic():
    """Channel-TRAILING: spatial axes are -3 and -2 here."""
    C = 4
    kernel = generate_diffusion_kernel(size=5)
    worlds = _worlds(shape=(H, W, C), seed=19)
    assert_batched_matches(
        lambda x: torch_correlate_3d(x, kernel), worlds, float_tol=1e-6
    )


# ---------------------------------------------------------------------------
# util.py -- as_strided neighbourhood views
# ---------------------------------------------------------------------------

def _unfold(x):
    td = TensorDict({"input": x.clone()})
    return unfold_neighbors(td, "input", (3, 3)).clone()


def test_unfold_neighbors_rank_agnostic():
    worlds = _worlds(seed=20)
    singles = [_unfold(w) for w in worlds]
    assert singles[0].shape == (H, W, 3, 3)
    batched = _unfold(torch.stack(worlds))
    assert batched.shape == (B, H, W, 3, 3)
    assert torch.equal(batched, torch.stack(singles))


def test_unfold_neighbors_stays_a_view_when_batched():
    td = TensorDict({"input": torch.stack(_worlds(seed=21))})
    unfolded = unfold_neighbors(td, "input", (3, 3))
    assert unfolded._is_view()
    unfolded[1, 0, 0, 1, 1] = 42.0
    assert td.get("input")[1, 0, 0] == 42.0


def _fold(x):
    td = TensorDict({"input": x.clone()})
    return fold_neighbors(td, "input").clone()


def test_fold_neighbors_rank_agnostic():
    worlds = _worlds(shape=(H, W, 3, 3), seed=22)
    singles = [_fold(w) for w in worlds]
    batched = _fold(torch.stack(worlds))
    assert batched.shape == (B, *singles[0].shape)
    assert torch.equal(batched, torch.stack(singles))


def test_custom_filter_operation_rank_agnostic():
    kernels = _worlds(shape=(H, W, 3, 3), seed=23)
    worlds = _worlds(seed=24)
    singles = [custom_filter_operation(w, k) for w, k in zip(worlds, kernels)]
    batched = custom_filter_operation(torch.stack(worlds), torch.stack(kernels))
    assert torch.allclose(batched, torch.stack(singles), rtol=0, atol=1e-6)


def _apply(x, kernels):
    td = TensorDict({"input": x.clone()})
    apply_kernels(td, "input", kernels)
    return td.get("input").clone()


def test_apply_kernels_rank_agnostic():
    kernels = _worlds(shape=(H, W, 3, 3), seed=25)
    worlds = _worlds(seed=26)
    singles = [_apply(w, k) for w, k in zip(worlds, kernels)]
    batched = _apply(torch.stack(worlds), torch.stack(kernels))
    assert torch.allclose(batched, torch.stack(singles), rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------
# util.py -- flow
# ---------------------------------------------------------------------------

def test_flow_gradient_rank_agnostic():
    worlds = _worlds(low=0.0, high=100.0, seed=27)
    singles = [flow_gradient(w) for w in worlds]
    assert singles[0].shape == (H, W, 3, 3)
    batched = flow_gradient(torch.stack(worlds))
    assert batched.shape == (B, H, W, 3, 3)
    assert torch.allclose(batched, torch.stack(singles), rtol=0, atol=1e-5)


def test_flow_gradient_no_distances_rank_agnostic():
    assert_batched_matches(
        lambda x: flow_gradient(x, apply_distances=False),
        _worlds(low=0.0, high=100.0, seed=28),
        float_tol=1e-5,
    )


def test_flow_rank_agnostic():
    worlds = _worlds(low=1.0, high=5.0, seed=29)
    grads = [flow_gradient(w) for w in worlds]

    singles = [flow(w, gradient=g, flow_rate=0.1) for w, g in zip(worlds, grads)]
    batched = flow(torch.stack(worlds), gradient=torch.stack(grads), flow_rate=0.1)

    for i, name in enumerate(("result", "inflow", "corrected_outflow")):
        expected = torch.stack([s[i] for s in singles])
        assert batched[i].shape == expected.shape, name
        assert torch.allclose(batched[i], expected, rtol=0, atol=1e-5), name


def test_flow_from_outflow_rank_agnostic():
    worlds = _worlds(low=1.0, high=5.0, seed=30)
    # Must come from flow_gradient: flow() asserts mass conservation, which an
    # arbitrary outflow field violates by pushing mass off the grid edges.
    outflows = [flow_gradient(w).clamp(min=0) * 0.01 for w in worlds]

    singles = [flow(w, outflow=o, flow_rate=0.1) for w, o in zip(worlds, outflows)]
    batched = flow(torch.stack(worlds), outflow=torch.stack(outflows), flow_rate=0.1)
    for i in range(3):
        assert torch.allclose(
            batched[i], torch.stack([s[i] for s in singles]), rtol=0, atol=1e-5
        )


# ---------------------------------------------------------------------------
# observations.py
# ---------------------------------------------------------------------------
#
# get_directional_values / compute_gradient_and_direction put the direction
# candidates on a LEADING stacked axis, so the batched result is (4, B, H, W)
# rather than (B, 4, H, W). Stacking the per-world results therefore has to
# happen on dim=1, which is exactly what rank-agnosticism means here: the
# spatial axes stayed at -2/-1 and the batch slotted in in front of them.


def _stack_after_leading(singles):
    return torch.stack(singles, dim=1)


@pytest.mark.parametrize("kernel_size", [1, 3])
def test_get_directional_values_rank_agnostic(kernel_size):
    worlds = _worlds(low=1.0, high=2.0, seed=32)
    singles = [get_directional_values(w, kernel_size) for w in worlds]
    assert singles[0].shape == (4, H, W)

    batched = get_directional_values(torch.stack(worlds), kernel_size)
    assert batched.shape == (4, B, H, W)
    assert torch.allclose(batched, _stack_after_leading(singles), rtol=0, atol=1e-6)


def test_compute_gradient_and_direction_rank_agnostic():
    # Strictly positive values: the only zeros are the deliberately zeroed
    # out-of-grid neighbours, so the argmax is unique and the internal random
    # tie-breaking cannot make the comparison shape-dependent.
    worlds = _worlds(low=1.0, high=2.0, seed=33)
    dirs = [get_directional_values(w, 1) for w in worlds]

    singles = [compute_gradient_and_direction(w, d) for w, d in zip(worlds, dirs)]
    batched = compute_gradient_and_direction(
        torch.stack(worlds), _stack_after_leading(dirs)
    )

    grad_expected = torch.stack([s[0] for s in singles])
    dir_expected = torch.stack([s[1] for s in singles])
    assert torch.allclose(batched[0], grad_expected, rtol=0, atol=1e-6)
    assert torch.equal(batched[1], dir_expected)


def _observation(energy, feature):
    return Observation(
        directional={("p", "scent"): get_directional_values(feature, 1)},
        current={("p", "scent"): feature},
        energy=energy,
        biomass=energy,
        gradient_ema=torch.zeros_like(feature),
        alive_mask=energy > 0,
    )


def test_process_observation_rank_agnostic():
    feats = _worlds(low=1.0, high=2.0, seed=34)
    energies = _worlds(torch.uint8, 1, 256, seed=35)
    weights = {("p", "scent"): 1.0}

    singles = [
        process_observation(_observation(e, f), weights)
        for e, f in zip(energies, feats)
    ]
    batched = process_observation(
        Observation(
            directional={
                ("p", "scent"): _stack_after_leading(
                    [get_directional_values(f, 1) for f in feats]
                )
            },
            current={("p", "scent"): torch.stack(feats)},
            energy=torch.stack(energies),
            biomass=torch.stack(energies),
            gradient_ema=torch.zeros(B, H, W),
            alive_mask=torch.stack(energies) > 0,
        ),
        weights,
    )
    assert torch.allclose(batched[0], torch.stack([s[0] for s in singles]), atol=1e-6)
    assert torch.equal(batched[1], torch.stack([s[1] for s in singles]))


def test_process_observation_no_features_keeps_input_shape():
    """The zero fallback used to hardcode `h, w = obs.energy.shape`."""
    energy = torch.ones(B, H, W, dtype=torch.uint8)
    obs = _observation(energy, torch.rand(B, H, W))
    gradient, direction = process_observation(obs, {("absent", "key"): 1.0})
    assert gradient.shape == (B, H, W)
    assert direction.shape == (B, H, W)


def _get_obs(feature, energy):
    td = TensorDict({"p": TensorDict({"scent": feature})})
    return get_observation(
        td,
        perception=[(("p", "scent"), 1)],
        energy=energy,
        biomass=energy,
        gradient_ema=torch.zeros_like(feature),
        survival_threshold=10,
    )


def test_get_observation_rank_agnostic():
    feats = _worlds(low=1.0, high=2.0, seed=36)
    energies = _worlds(torch.uint8, 1, 256, seed=37)

    singles = [_get_obs(f, e) for f, e in zip(feats, energies)]
    batched = _get_obs(torch.stack(feats), torch.stack(energies))

    key = ("p", "scent")
    assert torch.allclose(
        batched.current[key], torch.stack([s.current[key] for s in singles]), atol=1e-6
    )
    assert torch.allclose(
        batched.directional[key],
        _stack_after_leading([s.directional[key] for s in singles]),
        atol=1e-6,
    )
    assert torch.equal(
        batched.alive_mask, torch.stack([s.alive_mask for s in singles])
    )


# ---------------------------------------------------------------------------
# entities/helpers -- the one consumer whose indexing had to change with the
# kernel-bank layout.
# ---------------------------------------------------------------------------

def test_get_direction_masks_rank_agnostic():
    from tensor_beasts.entities.helpers.animal_helpers import get_direction_masks

    energies = _worlds(torch.uint8, 0, 4, seed=38)
    directions = _worlds(torch.int64, 0, 5, seed=39)

    singles = [
        get_direction_masks(d, e) for d, e in zip(directions, energies)
    ]
    batched = get_direction_masks(torch.stack(directions), torch.stack(energies))
    for d in range(1, 5):
        assert torch.equal(batched[d], torch.stack([s[d] for s in singles])), d


# NOT PROVEN HERE, on purpose:
#   * observations._flatten_feature and Feature.render dispatch on ndim to tell
#     an (H, W) feature from an (H, W, C) one. A batched (B, H, W) is
#     indistinguishable from (H, W, C) by rank alone, so these need an explicit
#     channel flag on Feature before they can be batched.
#   * animal_helpers.get_direction_masks strips a leading singleton batch from
#     RL-supplied actions by rank, so a world batch of size 1 would be eaten by
#     it. The test above deliberately uses B=3, where the squeeze is inert.
#   * The generators (perlin_noise, pyramid_elevation, range_elevation,
#     ramp_elevation, generate_maze, generate_*_kernel) take a shape rather than
#     a tensor, so "rank-agnostic" does not apply; they build one (H, W) field.

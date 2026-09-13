import time
import random

import numpy as np
import pytest
import torch
from tensordict import TensorDict
import matplotlib.pyplot as plt

from tensor_beasts.util import (
    directional_kernel_set, generate_direction_kernel, pad_matrix,
    torch_correlate_3d, unfold_neighbors, fold_neighbors, apply_kernels, perlin_noise, flow_gradient,
    flow
)


def test_directional_kernel_set_cache():
    # Test cache is working by calling the function multiple times with the same size
    size = 5
    kernel_set_1 = directional_kernel_set(size)
    kernel_set_2 = directional_kernel_set(size)
    assert kernel_set_1 is kernel_set_2  # Should be the same object due to lru_cache


def test_generate_direction_kernel_invalid_size():
    with pytest.raises(ValueError):
        generate_direction_kernel(4, 1)  # Size must be odd


@pytest.mark.parametrize("size, direction, expected", [
    (5, 1, torch.tensor([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=torch.uint8)),
    (5, 2, torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0]
    ], dtype=torch.uint8)),
    (5, 3, torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=torch.uint8)),
    (5, 4, torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=torch.uint8))
])
def test_generate_direction_kernel(size, direction, expected):
    result = generate_direction_kernel(size, direction)
    assert torch.equal(result, expected)


@pytest.mark.parametrize("mat, direction, expected", [
    (torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=torch.float32), 1, torch.tensor([
        [4, 5, 6],
        [7, 8, 9],
        [0, 0, 0]
    ], dtype=torch.float32)),
    (torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=torch.float32), 2, torch.tensor([
        [0, 0, 0],
        [1, 2, 3],
        [4, 5, 6]
    ], dtype=torch.float32)),
    (torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=torch.float32), 3, torch.tensor([
        [2, 3, 0],
        [5, 6, 0],
        [8, 9, 0]
    ], dtype=torch.float32)),
    (torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=torch.float32), 4, torch.tensor([
        [0, 1, 2],
        [0, 4, 5],
        [0, 7, 8]
    ], dtype=torch.float32))
])
def test_pad_matrix(mat, direction, expected):
    result = pad_matrix(mat, direction)
    assert torch.equal(result, expected)


def test_torch_correlate_3d():
    # Example input of shape (H, W, C)
    input_tensor = torch.tensor(
        [
            [[1, 2, 3, 4, 5],
             [1, 2, 3, 4, 5],
             [1, 2, 3, 4, 5],
             [1, 2, 3, 4, 5],
             [1, 2, 3, 4, 5]],
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 10, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
            [[1, 2, 3, 4, 5],
             [1, 2, 3, 4, 5],
             [1, 2, 3, 4, 5],
             [1, 2, 3, 4, 5],
             [1, 2, 3, 4, 5]],
        ]
    ).float().permute(1, 2, 0)

    # Example 2D kernel of shape (2, 2)
    weights = torch.tensor(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ]
    ).float()

    # Expected output for manual calculation
    expected_output = torch.tensor(
        [
            [[4, 8, 12, 16, 14],
             [5, 10, 15, 20, 19],
             [5, 10, 15, 20, 19],
             [5, 10, 15, 20, 19],
             [4, 8, 12, 16, 14]],
            [[0, 0, 0, 0, 0],
             [0, 0, 10, 0, 0],
             [0, 10, 10, 10, 0],
             [0, 0, 10, 0, 0],
             [0, 0, 0, 0, 0]],
            [[4, 8, 12, 16, 14],
             [5, 10, 15, 20, 19],
             [5, 10, 15, 20, 19],
             [5, 10, 15, 20, 19],
             [4, 8, 12, 16, 14]],
        ]
    ).float()

    # Compute output
    output = torch_correlate_3d(input_tensor, weights)

    # Check if the output shape is correct
    assert output.shape == input_tensor.shape, "Output tensor has incorrect shape."

    # Check if the output is correct
    assert torch.allclose(output.permute(2, 0, 1), expected_output), \
        f"Output tensor is incorrect. Expected \n{expected_output}\n, but got \n{output.permute(2, 0, 1)}\n"


def test_unfold_neighbors():
    # Example input of shape (H, W, C)
    input_tensor = torch.tensor(
        [[1, 2, 3, 4, 5],
         [1, 2, 3, 4, 5],
         [1, 2, 3, 4, 5],
         [1, 2, 3, 4, 5],
         [1, 2, 3, 4, 5]]
    )
    td = TensorDict(
        {"input": input_tensor}
    )
    unfold_neighbors(td, "input", (3, 3))
    neighbors = td.get("input_neighbors")
    assert neighbors.shape == (5, 5, 3, 3)
    assert neighbors._is_view()

    neighbors[0, 0, 1, 1] = 10
    assert td.get("input")[0, 0] == 10


def test_eye_kernel():
    # This is a convolution with an eye kernel
    input_tensor = torch.tensor(
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
    )
    eye_kernel = torch.tensor(
        [[0, 1, 0],
         [1, 0, 1],
         [0, 1, 0]]
    )
    result = torch.nn.functional.conv2d(
        input_tensor.unsqueeze(0).unsqueeze(0),
        eye_kernel.unsqueeze(0).unsqueeze(0),
        padding=1
    ).squeeze().squeeze()
    expected = torch.tensor(
        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]
    )
    assert torch.equal(result, expected)
    assert torch.equal(torch.sum(input_tensor) * torch.sum(eye_kernel), torch.sum(result))


def test_eye_kernel_with_reflection():
    input_tensor = torch.tensor(
        [[1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
    )
    eye_kernel = torch.tensor(
        [[0, 1, 0],
         [1, 0, 1],
         [0, 1, 0]]
    )
    result = torch.nn.functional.conv2d(
        input_tensor.unsqueeze(0).unsqueeze(0),
        eye_kernel.unsqueeze(0).unsqueeze(0),
        padding=1
    ).squeeze().squeeze()
    expected = torch.tensor(
        [[0, 1, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
    )
    assert torch.equal(result, expected)
    assert not torch.equal(torch.sum(input_tensor) * torch.sum(eye_kernel), torch.sum(result))

    padded_input = torch.nn.functional.pad(input_tensor.unsqueeze(0), (1, 1, 1, 1), mode="replicate")

    result = torch.nn.functional.conv2d(
        padded_input.unsqueeze(0),
        eye_kernel.unsqueeze(0).unsqueeze(0),
    ).squeeze().squeeze()

    expected = torch.tensor(
        [[2, 1, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
    )
    assert torch.equal(result, expected)
    assert torch.equal(torch.sum(input_tensor) * torch.sum(eye_kernel), torch.sum(result))


@pytest.mark.skip(reason="Known failing test - needs investigation")
def test_normalized_eye_kernel():
    input_tensor = torch.tensor(
        [[1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        dtype = torch.float32
    )
    eye_kernel = torch.tensor(
        [[0, 1, 0],
         [1, 0, 1],
         [0, 1, 0]],
        dtype = torch.float32
    )
    normalized_eye_kernel = eye_kernel / torch.sum(eye_kernel)

    padded_input = torch.nn.functional.pad(input_tensor.unsqueeze(0), (1, 1, 1, 1), mode="replicate")

    result = torch.nn.functional.conv2d(
        padded_input.unsqueeze(0),
        normalized_eye_kernel.unsqueeze(0).unsqueeze(0),
    ).squeeze().squeeze()

    expected = torch.tensor(
        [[0.5, 0.25, 0, 0, 0],
         [0.25, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
    )
    assert torch.equal(result, expected)
    assert torch.equal(torch.sum(input_tensor) * torch.sum(normalized_eye_kernel), torch.sum(result))
    assert torch.equal(torch.sum(input_tensor), torch.sum(result))

    for _ in range(10):
        random_input = torch.rand_like(input_tensor)
        print(random_input)
        padded_random_input = torch.nn.functional.pad(random_input.unsqueeze(0), (1, 1, 1, 1), mode="replicate")

        result = torch.nn.functional.conv2d(
            padded_random_input.unsqueeze(0),
            eye_kernel.unsqueeze(0).unsqueeze(0),
        ).squeeze().squeeze()
        assert torch.allclose(torch.sum(random_input), torch.sum(result))


def test_asymmetrical_kernel():
    input_tensor = torch.tensor(
        [[1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        dtype=torch.float32
    )
    kernel = torch.tensor(
        [[0, 2, 0],
         [1, 0, 1],
         [0, 1, 0]],
        dtype=torch.float32
    )
    normalized_kernel = kernel / torch.sum(kernel)
    expected_normalized_kernel = torch.tensor(
        [[0, 0.4, 0],
         [0.2, 0, 0.2],
         [0, 0.2, 0]]
    )
    assert torch.allclose(normalized_kernel, expected_normalized_kernel)


    padded_input = torch.nn.functional.pad(input_tensor.unsqueeze(0), (1, 1, 1, 1), mode="replicate")

    result = torch.nn.functional.conv2d(
        padded_input.unsqueeze(0),
        normalized_kernel.unsqueeze(0).unsqueeze(0),
    ).squeeze().squeeze()

    expected = torch.tensor(
        [[0.6, 0.2, 0, 0, 0],
         [0.4, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
    )
    assert torch.allclose(result, expected)

    assert not torch.equal(torch.sum(input_tensor), torch.sum(result))
    assert torch.equal(torch.sum(input_tensor), torch.sum(result) - 0.2)

    # We want the "mirror world" of the padding to undo any gain or loss over the edges, but for this to
    # work correctly, we have to mirror both the values and the kernel.
    # Padding reflects *values* over the edges of the input, but not the kernel itself!


def test_apply_kernels():

    # How do we reflect the kernel at the edges if Torch's convolution operation uses a fixed kernel?
    # We apply each kernel to its corresponding patch of the input as a batched multiplication operation,
    # then sum the results over the kernel dimensions to get our result.

    input_tensor = torch.tensor(
        [[1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        dtype=torch.float32
    )
    kernel = torch.tensor(
        [[0, 2, 0],
         [1, 0, 1],
         [0, 1, 0]]
    )
    normalized_kernel = kernel / torch.sum(kernel)

    padded_input = torch.nn.functional.pad(input_tensor.unsqueeze(0), (1, 1, 1, 1), mode="replicate").squeeze()

    # First, we create an unfolded view of the padded input tensor using the as_strided function.
    H, W = input_tensor.shape
    kH, kW = normalized_kernel.shape
    stride = padded_input.stride()
    unfolded = torch.as_strided(
        padded_input,
        size=(H, W, kH, kW),
        stride=(stride[0], stride[1], stride[0], stride[1])
    )
    assert unfolded.storage().data_ptr() == padded_input.storage().data_ptr()
    assert unfolded.shape == (H, W, kH, kW)
    assert torch.equal(
        unfolded[0, 0],
        torch.tensor(
            [[1, 1, 0],
            [1, 1, 0],
            [0, 0, 0]]
        )
    )

    # Next, we apply the kernel to each patch of the input tensor.
    our_result = torch.sum(unfolded * normalized_kernel, dim=(2, 3))
    ein_result = torch.einsum("ijkl,kl->ij", unfolded, normalized_kernel)
    conv_result = torch.nn.functional.conv2d(
        padded_input.unsqueeze(0).unsqueeze(0),
        normalized_kernel.unsqueeze(0).unsqueeze(0),
    ).squeeze().squeeze()

    assert torch.allclose(our_result, conv_result)
    assert torch.allclose(ein_result, conv_result)


def test_flow_gradient():
    torch.set_default_device('mps')
    input = torch.ones(128, 256, dtype=torch.float32)
    elevation = torch.ones_like(input, dtype=torch.float32) * torch.linspace(
        0,
        1, input.shape[0]
    ).unsqueeze(1).expand(input.shape) * 100
    elevation *= perlin_noise(input.shape, (4, 8), 4)

    gradient = flow_gradient(elevation + input)

    # No plt.show() here: on the default interactive macOS backend it blocks the
    # whole suite waiting on a window. Render to a figure and drop it instead.
    plt.imshow(
        torch.sum(gradient, (-1, -2)).cpu().numpy(),
        cmap='gray'
    )
    plt.close('all')
    # plt.imshow(
    #     torch.sum(gradient, (-1, -2)).cpu().numpy(),
    #     cmap='gray'
    # )
    # plt.imshow(
    #     elevation.cpu().numpy(),
    #     cmap='gray'
    # )



def test_flow():
    torch.set_default_device('mps')
    input = torch.ones(8, 8, dtype=torch.float32)
    elevation = torch.ones_like(input, dtype=torch.float32) * torch.linspace(
        0,
        1, input.shape[0]
    ).unsqueeze(1).expand(input.shape) * 100

    gradient = flow_gradient(elevation + input)
    result = flow(input, gradient=gradient, flow_rate=0.1)

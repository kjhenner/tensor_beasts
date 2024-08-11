import time
import random

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from tensor_beasts.util import (
    directional_kernel_set, safe_add, safe_sub, generate_direction_kernel, pad_matrix,
    torch_correlate_3d, unfold_neighbors, fold_neighbors, apply_kernels, perlin_noise, flow_gradient,
    flow,
)


def test_directional_kernel_set_cache():
    # Test cache is working by calling the function multiple times with the same size
    size = 5
    kernel_set_1 = directional_kernel_set(size)
    kernel_set_2 = directional_kernel_set(size)
    assert kernel_set_1 is kernel_set_2  # Should be the same object due to lru_cache


def test_safe_add():
    a = torch.tensor([250, 200], dtype=torch.uint8)
    b = torch.tensor([10, 100], dtype=torch.uint8)
    result = safe_add(a, b)
    expected = torch.tensor([255, 255], dtype=torch.uint8)
    assert torch.equal(result, expected)


def test_safe_sub():
    a = torch.tensor([250, 200], dtype=torch.uint8)
    b = torch.tensor([10, 100], dtype=torch.uint8)
    result = safe_sub(a, b)
    expected = torch.tensor([240, 100], dtype=torch.uint8)
    assert torch.equal(result, expected)

    a = torch.tensor([10, 200], dtype=torch.uint8)
    b = torch.tensor([20, 100], dtype=torch.uint8)
    result = safe_sub(a, b)
    expected = torch.tensor([0, 100], dtype=torch.uint8)
    assert torch.equal(result, expected)


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
    conv_result = torch.nn.functional.conv2d(
        padded_input.unsqueeze(0).unsqueeze(0),
        normalized_kernel.unsqueeze(0).unsqueeze(0),
    ).squeeze().squeeze()

    assert torch.allclose(our_result, conv_result)


def test_flow():
    torch.set_default_device('mps')
    torch.manual_seed(0)
    random.seed(0)
    input = torch.ones(256, 256, dtype=torch.float32)
    # print(input[0, 0])
    # elevation = torch.rand_like(input)
    elevation = torch.ones_like(input, dtype=torch.float32) * torch.linspace(
        0,
        1, input.shape[0]
    ).unsqueeze(1).expand(input.shape) * 100

    gradient = flow_gradient(elevation + input)
    # print(gradient[0, 0])
    result = flow(input, gradient, 0.1)
    #
    # mps_input = torch.ones(256, 256, dtype=torch.float32, device='mps')
    # # print(input[0, 0])
    # # elevation = torch.rand_like(input)
    # mps_elevation = torch.ones_like(mps_input, dtype=torch.float32, device='mps') * torch.linspace(
    #     0,
    #     1, mps_input.shape[0],
    #     device='mps'
    # ).unsqueeze(1).expand(mps_input.shape) * 100
    #
    # gradient_log = {}
    #
    # mps_gradient = flow_gradient(mps_elevation + mps_input, gradient_log, "mps")
    # # print(gradient[0, 0])
    # mps_result = flow(mps_input, mps_gradient, 0.1)
    #
    # torch.set_default_device('cpu')
    # cpu_input = torch.ones(256, 256, dtype=torch.float32, device='cpu')
    # # print(input[0, 0])
    # # elevation = torch.rand_like(input)
    # cpu_elevation = torch.ones_like(cpu_input, dtype=torch.float32, device='cpu') * torch.linspace(
    #     0,
    #     1, cpu_input.shape[0],
    #     device='cpu'
    # ).unsqueeze(1).expand(cpu_input.shape) * 100
    #
    # cpu_gradient = flow_gradient(cpu_elevation + cpu_input, gradient_log, "cpu")
    # # print(gradient[0, 0])
    # cpu_result = flow(cpu_input, cpu_gradient, 0.1)
    #
    # for k, v in gradient_log['cpu'].items():
    #     print(k)
    #     # for i in range(v.shape[0]):
    #     #     for j in range(v.shape[1]):
    #     #         if not torch.allclose(v[i][j], gradient_log['mps'][k][i][j]):
    #     #             print(i, j)
    #     #             print("CPU")
    #     #             print(v[i][j])
    #     #             print("MPS")
    #     #             print(gradient_log['mps'][k][i][j])
    #     #             print("Difference")
    #     #             print(v[i][j] - gradient_log['mps'][k][i][j])
    #     for tol in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
    #         all_close = torch.allclose(v, gradient_log["mps"][k], rtol=tol, atol=tol)
    #         print(f"Gradient {k} all close at tol={tol}: {all_close}")
    #
    #
    # assert torch.allclose(mps_input.cpu(), cpu_input)
    # assert torch.allclose(mps_elevation.cpu(), cpu_elevation)
    # assert torch.allclose(mps_gradient.cpu(), cpu_gradient)
    # assert torch.allclose(mps_result.cpu(), cpu_result)
    #
    # assert torch.isclose(torch.sum(cpu_result), torch.sum(cpu_input))
    # assert torch.all(cpu_result >= 0)
    #
    # assert torch.isclose(torch.sum(mps_result), torch.sum(mps_input))
    # assert torch.all(mps_result >= 0)


def test_device(input_shape=(256, 256), dtype=torch.float32):
    # Set up input tensors on both CPU and MPS
    torch.manual_seed(0)
    cpu_input = torch.rand(input_shape, dtype=dtype, device='cpu')
    mps_input = cpu_input.to('mps')

    print("Input tensors created. Proceeding with step-by-step comparison.")

    # Step 1: Padding
    cpu_padded = torch.nn.functional.pad(cpu_input.unsqueeze(0), (1, 1, 1, 1), mode="constant", value=0).squeeze()
    mps_padded = torch.nn.functional.pad(mps_input.unsqueeze(0), (1, 1, 1, 1), mode="constant", value=0).squeeze()

    assert torch.allclose(cpu_padded, mps_padded.cpu(), rtol=1e-5, atol=1e-5), "Divergence in padding step"
    print("Padding step: Passed")

    # Step 2: Unfolding
    H, W = input_shape
    cpu_stride = cpu_padded.stride()
    mps_stride = mps_padded.stride()

    cpu_unfolded = torch.as_strided(cpu_padded, size=(H, W, 3, 3), stride=(cpu_stride[0], cpu_stride[1], cpu_stride[0], cpu_stride[1]))
    mps_unfolded = torch.as_strided(mps_padded, size=(H, W, 3, 3), stride=(mps_stride[0], mps_stride[1], mps_stride[0], mps_stride[1]))

    assert torch.allclose(cpu_unfolded, mps_unfolded.cpu(), rtol=1e-5, atol=1e-5), "Divergence in unfolding step"
    print("Unfolding step: Passed")

    # Step 3: Input expansion
    cpu_expanded = cpu_input.reshape(H, W, 1, 1).expand(H, W, 3, 3)
    mps_expanded = mps_input.reshape(H, W, 1, 1).expand(H, W, 3, 3)

    assert torch.allclose(cpu_expanded, mps_expanded.cpu(), rtol=1e-5, atol=1e-5), "Divergence in input expansion step"
    print("Input expansion step: Passed")

    # Step 4: Euclidean distance matrix
    euclidean_distance_matrix = torch.tensor(
        [[1.4142, 1.0000, 1.4142],
         [1.0000, 1.0000, 1.0000],
         [1.4142, 1.0000, 1.4142]],
        dtype=dtype
    )
    cpu_euclidean = euclidean_distance_matrix.to('cpu')
    mps_euclidean = euclidean_distance_matrix.to('mps')

    assert torch.allclose(cpu_euclidean, mps_euclidean.cpu(), rtol=1e-5, atol=1e-5), "Divergence in Euclidean distance matrix"
    print("Euclidean distance matrix step: Passed")

    # Step 5: Gradient calculation
    cpu_gradient = (cpu_expanded - cpu_unfolded) / cpu_euclidean.unsqueeze(0).unsqueeze(0)
    mps_gradient = (mps_expanded - mps_unfolded) / mps_euclidean.unsqueeze(0).unsqueeze(0)

    if not torch.allclose(cpu_gradient, mps_gradient.cpu(), rtol=1e-5, atol=1e-5):
        print("Divergence in gradient calculation step")
        diff = torch.abs(cpu_gradient - mps_gradient.cpu())
        max_diff = torch.max(diff)
        max_diff_index = torch.argmax(diff)
        print(f"Maximum difference: {max_diff}")
        print(f"At index: {np.unravel_index(max_diff_index.item(), diff.shape)}")
    else:
        print("Gradient calculation step: Passed")

    # Step 6: Zeroing out edges
    cpu_gradient[0, :, 0, :] = 0
    cpu_gradient[-1, :, -1, :] = 0
    cpu_gradient[:, 0, :, 0] = 0
    cpu_gradient[:, -1, :, -1] = 0

    mps_gradient[0, :, 0, :] = 0
    mps_gradient[-1, :, -1, :] = 0
    mps_gradient[:, 0, :, 0] = 0
    mps_gradient[:, -1, :, -1] = 0

    if not torch.allclose(cpu_gradient, mps_gradient.cpu(), rtol=1e-8, atol=1e-8):
        print("Divergence after zeroing out edges")
        diff = torch.abs(cpu_gradient - mps_gradient.cpu())
        max_diff = torch.max(diff)
        max_diff_index = torch.argmax(diff)
        print(f"Maximum difference: {max_diff}")
        print(f"At index: {np.unravel_index(max_diff_index.item(), diff.shape)}")
    else:
        print("Zeroing out edges step: Passed")

    print("Detailed comparison completed.")

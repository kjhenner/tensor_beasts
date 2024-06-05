import pytest
import torch
from tensor_beasts.util import (
    directional_kernel_set, safe_add, safe_sub, generate_direction_kernel, pad_matrix,
    torch_correlate_3d
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

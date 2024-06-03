import pytest
import numpy as np
from tensor_beasts.util_numpy import directional_kernel_set, safe_add, safe_sub, generate_direction_kernel, pad_matrix


def test_directional_kernel_set_cache():
    # Test cache is working by calling the function multiple times with the same size
    size = 5
    kernel_set_1 = directional_kernel_set(size)
    kernel_set_2 = directional_kernel_set(size)
    assert kernel_set_1 is kernel_set_2  # Should be the same object due to lru_cache


def test_safe_add():
    a = np.array([250, 200], dtype=np.uint8)
    b = np.array([10, 100], dtype=np.uint8)
    result = safe_add(a, b)
    expected = np.array([255, 255], dtype=np.uint8)
    assert np.array_equal(result, expected)


def test_safe_sub():
    a = np.array([250, 200], dtype=np.uint8)
    b = np.array([10, 100], dtype=np.uint8)
    result = safe_sub(a, b)
    expected = np.array([240, 100], dtype=np.uint8)
    assert np.array_equal(result, expected)

    a = np.array([10, 200], dtype=np.uint8)
    b = np.array([20, 100], dtype=np.uint8)
    result = safe_sub(a, b)
    expected = np.array([0, 100], dtype=np.uint8)
    assert np.array_equal(result, expected)


def test_generate_direction_kernel_invalid_size():
    with pytest.raises(ValueError):
        generate_direction_kernel(4, 1)  # Size must be odd


@pytest.mark.parametrize("size, direction, expected", [
    (5, 1, np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])),
    (5, 2, np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0]
    ])),
    (5, 3, np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])),
    (5, 4, np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0]
    ]))
])
def test_generate_direction_kernel(size, direction, expected):
    result = generate_direction_kernel(size, direction)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize("mat, direction, expected", [
    (np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]), 1, np.array([
        [4, 5, 6],
        [7, 8, 9],
        [0, 0, 0]
    ])),
    (np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]), 2, np.array([
        [0, 0, 0],
        [1, 2, 3],
        [4, 5, 6]
    ])),
    (np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]), 3, np.array([
        [2, 3, 0],
        [5, 6, 0],
        [8, 9, 0]
    ])),
    (np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]), 4, np.array([
        [0, 1, 2],
        [0, 4, 5],
        [0, 7, 8]
    ]))
])
def test_pad_matrix(mat, direction, expected):
    result = pad_matrix(mat, direction)
    assert np.array_equal(result, expected)

import torch
from functools import lru_cache


@lru_cache
def directional_kernel_set(size: int):
    return {
        1: generate_direction_kernel(size, 1),
        2: generate_direction_kernel(size, 2),
        3: generate_direction_kernel(size, 3),
        4: generate_direction_kernel(size, 4)
    }


def safe_add(a, b):
    a = a + b
    a[a < b] = 255
    return a


def safe_sub(a, b):
    a = a - b
    a[a > 255 - b] = 0
    return a


def safe_mult(a, b):
    a = a * b
    a[a > 255] = 255
    return a


def generate_direction_kernel(size, direction):
    if size % 2 == 0:
        raise ValueError("Size must be an odd number for symmetrical shape.")

    kernel = torch.zeros((size, size), dtype=torch.uint8)
    center = size // 2

    if direction == 1:  # North
        for i in range(center):
            start = center - i
            end = center + i + 1
            kernel[i, start:end] = 1.0
    elif direction == 2:  # South
        for i in range(center):
            start = center - i
            end = center + i + 1
            kernel[-(i + 1), start:end] = 1.0
    elif direction == 3:  # West
        for i in range(center):
            start = center - i
            end = center + i + 1
            kernel[start:end, i] = 1.0
    elif direction == 4:  # East
        for i in range(center):
            start = center - i
            end = center + i + 1
            kernel[start:end, -(i + 1)] = 1.0
    else:
        raise ValueError("Invalid direction. Use 1 for North, 2 for South, 3 for West, 4 for East.")

    return kernel


def pad_matrix(mat, direction):
    if direction == 1:  # Up
        return torch.nn.functional.pad(mat[1:], (0, 0, 0, 1), value=0)
    elif direction == 2:  # Down
        return torch.nn.functional.pad(mat[:-1], (0, 0, 1, 0), value=0)
    elif direction == 3:  # Left
        return torch.nn.functional.pad(mat[:, 1:], (0, 1, 0, 0), value=0)
    elif direction == 4:  # Right
        return torch.nn.functional.pad(mat[:, :-1], (1, 0, 0, 0), value=0)

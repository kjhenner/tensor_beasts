from functools import lru_cache

import numpy as np


@lru_cache
def directional_kernel_set(size: int):
    return {
        1: generate_direction_kernel(size, 1),
        2: generate_direction_kernel(size, 2),
        3: generate_direction_kernel(size, 3),
        4: generate_direction_kernel(size, 4)
    }


def safe_add(a, b):
    a += b
    a[a < b] = 255
    return a


def safe_sub(a, b):
    a -= b
    a[a > 255 - b] = 0
    return a


def generate_direction_kernel(size, direction):
    if size % 2 == 0:
        raise ValueError("Size must be an odd number for symmetrical shape.")

    kernel = np.zeros((size, size))
    center = size // 2

    if direction == 1:  # North
        for i in range(center):
            start = center - i
            end = center + i + 1
            kernel[i, start:end] = 1
    elif direction == 2:  # South
        for i in range(center):
            start = center - i
            end = center + i + 1
            kernel[-(i + 1), start:end] = 1
    elif direction == 3:  # West
        for i in range(center):
            start = center - i
            end = center + i + 1
            kernel[start:end, i] = 1
    elif direction == 4:  # East
        for i in range(center):
            start = center - i
            end = center + i + 1
            kernel[start:end, -(i + 1)] = 1
    else:
        raise ValueError("Invalid direction. Use 1 for North, 2 for South, 3 for West, 4 for East.")

    return kernel


def pad_matrix(mat, direction):
    if direction == 1:  # Up
        return np.pad(mat[1:], ((0, 1), (0, 0)), constant_values=0)
    elif direction == 2:  # Down
        return np.pad(mat[:-1], ((1, 0), (0, 0)), constant_values=0)
    elif direction == 3:  # Left
        return np.pad(mat[:, 1:], ((0, 0), (0, 1)), constant_values=0)
    elif direction == 4:  # Right
        return np.pad(mat[:, :-1], ((0, 0), (1, 0)), constant_values=0)


@lru_cache
def get_edge_mask(shape: tuple):
    mask = np.zeros(shape)
    mask[0, :] = 1
    mask[-1, :] = 1
    mask[:, 0] = 1
    mask[:, -1] = 1
    return mask

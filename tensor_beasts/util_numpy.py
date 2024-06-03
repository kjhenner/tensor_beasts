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


def get_direction_matrix_bak(matrix):
    # Step 1: Roll matrices and handle edges
    up = np.roll(matrix, shift=-1, axis=0)
    down = np.roll(matrix, shift=1, axis=0)
    left = np.roll(matrix, shift=-1, axis=1)
    right = np.roll(matrix, shift=1, axis=1)

    # Setting the boundaries to -inf to avoid wrapping around behavior
    up[-1, :] = 0
    down[0, :] = 0
    left[:, -1] = 0
    right[:, 0] = 0

    # Step 2: Stack matrices to work with all directions together
    stacked = np.stack([matrix, up, down, left, right], axis=-1)

    # Step 3: Determine the indices of the max values along the stacked direction axis
    max_indices = np.argmax(stacked, axis=-1)

    # Mapping index to corresponding direction:
    # 0 -> same cell, 1 -> up, 2 -> down, 3 -> left, 4 -> right
    index_to_direction = np.array([0, 1, 2, 3, 4])

    # Step 4: Apply mapping to get the final direction matrix
    direction_matrix = index_to_direction[max_indices]

    return direction_matrix


def get_direction_matrix(matrix):
    rows, cols = matrix.shape

    # Step 1: Roll matrices and handle edges
    # up = np.roll(matrix, shift=-1, axis=0)
    # down = np.roll(matrix, shift=1, axis=0)
    # left = np.roll(matrix, shift=-1, axis=1)
    # right = np.roll(matrix, shift=1, axis=1)
    down = np.roll(matrix, shift=-1, axis=0)
    up = np.roll(matrix, shift=1, axis=0)
    right = np.roll(matrix, shift=-1, axis=1)
    left = np.roll(matrix, shift=1, axis=1)

    # Setting the boundaries to -inf to avoid wrapping around behavior
    up[-1, :] = 0
    down[0, :] = 0
    left[:, -1] = 0
    right[:, 0] = 0

    # Stack matrices to work with all directions together
    stacked = np.stack([matrix, up, down, left, right], axis=-1)

    # Step 2: Compute the maximum values across the stacked axis
    max_values = np.max(stacked, axis=-1)

    # Create masks for each direction
    masks = (stacked == max_values[..., None])

    # Step 3: Randomly resolve ties:
    # Generate random tie-breaker decisions
    random_choices = np.random.random(masks.shape)
    # Use random choices to introduce random tie-breakers
    random_max_masks = masks * random_choices
    # Find the direction with the maximum random choice for those tied maxima
    direction_indices = np.argmax(random_max_masks, axis=-1)

    return direction_indices
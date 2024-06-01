import numpy as np
from scipy.ndimage import convolve


def pad_matrix(mat, direction):
    if direction == 1:  # Up
        return np.pad(mat[:-1], ((1, 0), (0, 0)), constant_values=0)
    elif direction == 2:  # Down
        return np.pad(mat[1:], ((0, 1), (0, 0)), constant_values=0)
    elif direction == 3:  # Left
        return np.pad(mat[:, :-1], ((0, 0), (1, 0)), constant_values=0)
    elif direction == 4:  # Right
        return np.pad(mat[:, 1:], ((0, 0), (0, 1)), constant_values=0)


direction_kernels = {
    1: np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]),
    2: np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0]
    ]),
    3: np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]),
    4: np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0]
    ])
}


def move_bak(A):
    # Generate a random direction matrix
    directions = np.random.randint(1, 5, size=A.shape)

    # Create boolean direction matrices
    bool_direction_masks = {d: (directions == d).astype(np.uint8) for d in range(1, 5)}

    # Calculate the clearance mask for each direction
    clearance_masks = {}
    for d in bool_direction_masks:
        clearance_convolution = convolve(A > 0, direction_kernels[d], mode='constant')
        clearance_masks[d] = ((clearance_convolution == 0) * (A > 0) * bool_direction_masks[d]).astype(np.uint8)

    # Multiply each direction matrix by this clearance mask and the input matrix, then apply directional padding
    offset_matrices = {d: pad_matrix(clearance_masks[d] * A, d) for d in bool_direction_masks}

    # Combine all the offset matrices along with the inverse clearance mask
    clearance_mask_union = sum(clearance_masks.values()).astype(bool)
    inverse_clearance_mask = np.logical_not(clearance_mask_union).astype(np.uint8)
    remaining_input = A * inverse_clearance_mask

    # Sum everything together
    return sum(offset_matrices.values()) + remaining_input


def move(A, divide_threshold):
    # Generate a random direction matrix
    directions = np.random.randint(1, 5, size=A.shape)

    # Create boolean direction matrices
    bool_direction_masks = {d: (directions == d).astype(np.uint8) for d in range(1, 5)}

    # Calculate the clearance mask for each direction
    clearance_masks = {}
    for d in bool_direction_masks:
        clearance_convolution = convolve(A > 0, direction_kernels[d], mode='constant')
        clearance_masks[d] = ((clearance_convolution == 0) * (A > 0) * bool_direction_masks[d]).astype(np.uint8)

    # Multiply each direction matrix by this clearance mask and the input matrix, then apply directional padding
    # offset_matrices = {d: pad_matrix((clearance_masks[d] * A) // 3, d)
    #                    for d in bool_direction_masks}

    offset_matrices = {
        d: pad_matrix(
            np.where(
                clearance_masks[d] * A > divide_threshold,
                (clearance_masks[d] * A) // 2,
                clearance_masks[d] * A
            ),
            d
        )
        for d in bool_direction_masks
    }

    # Combine all the offset matrices along with the inverse clearance mask
    clearance_mask_union = sum(clearance_masks.values()).astype(bool)
    inverse_clearance_mask = np.logical_not(clearance_mask_union).astype(np.uint8)

    remaining_input = np.where(A > divide_threshold, A // 8, A * inverse_clearance_mask)

    # Sum everything together
    return sum(offset_matrices.values()) + remaining_input


if __name__ == "__main__":
    # Initialize random seed for reproducibility
    np.random.seed(42)

    # Input matrix
    A = np.array([
        [0, 0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 1, 10, 1, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 4, 0, 1, 0, 0]
    ])
    result = move(A)

    # Print output
    print("Output: \n", result)

    # Validate that the sum of the result matches the sum of the input
    print("Is the final result correct?", np.sum(result) == np.sum(A))
    print("Sum of result:", np.sum(result))
    print("Sum of input:", np.sum(A))
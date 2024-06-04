import torch
import torch.nn.functional as F
from functools import lru_cache


@lru_cache
def directional_kernel_set(size: int):
    return {
        1: generate_direction_kernel(size, 1),
        2: generate_direction_kernel(size, 2),
        3: generate_direction_kernel(size, 3),
        4: generate_direction_kernel(size, 4)
    }


def safe_add(a, b, inplace=True):
    if not inplace:
        a = a.clone()
    a += b
    a[a < b] = 255
    return a


def safe_sub(a, b, inplace=True):
    if not inplace:
        a = a.clone()
    a -= b
    a[a > 255 - b] = 0
    return a


def safe_mult(a, b, inplace=True):
    if not inplace:
        a = a.clone()  # Create a copy to avoid modifying the original tensor if inplace is False

    result = a.to(torch.uint16) * b.to(torch.uint16)

    overflow_mask = result > 255
    result = torch.where(overflow_mask, torch.tensor(255, dtype=torch.uint8), result)

    a[:] = result.to(torch.uint8)
    return a


def generate_direction_kernel(size, direction):
    if size % 2 == 0:
        raise ValueError("Size must be an odd number for symmetrical shape.")

    kernel = torch.zeros((size, size), dtype=torch.float32)
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


@lru_cache
def get_edge_mask(shape: tuple):
    mask = torch.zeros(shape, dtype=torch.float32)
    mask[0, :] = 1
    mask[-1, :] = 1
    mask[:, 0] = 1
    mask[:, -1] = 1
    return mask


def get_direction_matrix(matrix, random_choices=None):
    down = torch.roll(matrix, shifts=-1, dims=0)
    up = torch.roll(matrix, shifts=1, dims=0)
    right = torch.roll(matrix, shifts=-1, dims=1)
    left = torch.roll(matrix, shifts=1, dims=1)

    # Setting the boundaries to 0 to avoid wrapping around behavior
    up[-1, :] = 0
    down[0, :] = 0
    left[:, -1] = 0
    right[:, 0] = 0

    # Stack matrices to work with all directions together
    stacked = torch.stack([matrix, up, down, left, right], dim=-1)

    # Step 2: Compute the maximum values across the stacked axis
    max_values = torch.max(stacked, dim=-1).values

    # Create masks for each direction
    masks = (stacked == max_values[..., None])

    # Generate random tie-breaker decisions
    if random_choices is None:
        random_choices = torch.rand_like(masks, dtype=torch.float32)

    # Use random choices to introduce random tie-breakers
    random_max_masks = masks * random_choices
    # Find the direction with the maximum random choice for those tied maxima
    direction_indices = torch.argmax(random_max_masks, dim=-1)

    return direction_indices


def torch_correlate(input: torch.Tensor, kernel, mode='constant', cval=0):
    """
    Mimic scipy.ndimage.correlate using PyTorch's conv2d.

    Parameters:
    - input: 2D torch tensor, the input image.
    - kernel: 2D torch tensor, the kernel for correlation.
    - mode: str, boundary mode (only 'constant' mode implemented similar to scipy.ndimage.correlate).
    - cval: float, value to fill pad when mode is 'constant'.

    Returns:
    - result: 2D torch tensor, result of correlation.
    """
    input_dtype = input.dtype
    input = input.type(torch.float32)
    kernel = kernel.type(torch.float32)
    # Ensure kernel and input are in the right format
    if input.dim() == 2:
        input = input.unsqueeze(0).unsqueeze(0)
    elif input.dim() == 3:
        input = input.unsqueeze(1)

    if kernel.dim() == 2:
        kernel = kernel.unsqueeze(0).unsqueeze(0)

    # Define padding based on mode
    pad_size = (kernel.shape[-1] // 2, kernel.shape[-2] // 2)
    pad = (pad_size[0], pad_size[0], pad_size[1], pad_size[1])

    if mode == 'constant':
        input_padded = F.pad(input, pad=pad, mode='constant', value=cval)
    else:
        raise ValueError("Only 'constant' mode is implemented.")

    result = F.conv2d(input_padded, kernel)

    # Remove the extra dimensions added earlier
    result = result.squeeze()
    return result.type(input_dtype)

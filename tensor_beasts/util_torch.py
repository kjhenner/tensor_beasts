import torch
import torch.nn.functional as F
from functools import lru_cache

import time
import statistics

# Initialize a dictionary to store function execution times
execution_times = {}


def timing(func):
    def wrapper(*args, **kwargs):
        global execution_times
        start_time = time.time()          # Record start time
        result = func(*args, **kwargs)    # Call the original function
        end_time = time.time()            # Record end time
        elapsed_time = end_time - start_time  # Calculate elapsed time

        # Record the execution time in the dictionary
        if func.__name__ not in execution_times:
            execution_times[func.__name__] = []
        execution_times[func.__name__].append(elapsed_time)

        return result

    return wrapper


def get_mean_execution_times():
    return {
        "function_timing": {k: statistics.mean(v) for k, v in execution_times.items()}
    }


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


def torch_correlate_2d(input: torch.Tensor, kernel, mode='constant', cval=0):
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


def torch_correlate_3d(input_tensor, weights):
    """
    Apply a batched 2D convolution to a (H, W, C) tensor using (H, W) weights and return (H, W, C) tensor.
    Each channel in the input tensor is treated as a separate batch for 2D convolution.

    Parameters:
    - input_tensor: torch.Tensor of shape (H, W, C)
    - weights: torch.Tensor of shape (K, K)

    Returns:
    - output_tensor: torch.Tensor of shape (H, W, C)
    """
    # Ensure input_tensor is of shape (H, W, C)
    assert len(input_tensor.shape) == 3, "input_tensor must be of shape (H, W, C)"

    # Ensure weights is of shape (K, K)
    assert len(weights.shape) == 2, "weights must be 2D"

    # Convert input_tensor to shape (C, 1, H, W) to treat channels as separate batches
    input_4d = input_tensor.permute(2, 0, 1).unsqueeze(1)

    # Convert weights to shape (1, 1, K, K) to apply the same kernel on all channels
    weight_4d = weights.unsqueeze(0).unsqueeze(0)

    # Apply 2D convolution for each channel separately using conv2d with groups=C
    conv_output = F.conv2d(input_4d, weight_4d, stride=1, padding='same', groups=1)

    # Convert output back to (H, W, C) by inverting the initial permutation
    output_tensor = conv_output.squeeze(1).permute(1, 2, 0)

    return output_tensor
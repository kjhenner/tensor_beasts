import math
import random
from typing import List

import torch
import torch.nn.functional as F
from functools import lru_cache

import time
import statistics

# Initialize a dictionary to store function execution times
execution_times = {}


DIRECTION_NAMES = {
    0: 'hold',
    1: 'up',
    2: 'down',
    3: 'left',
    4: 'right'
}


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


def safe_sum(matrices: List[torch.Tensor]):
    original_dtype = matrices[0].dtype
    return torch.stack(matrices).type(torch.int16).sum(dim=0).clamp(0, 255).type(original_dtype)


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


def generate_maze(size: int):
    size //= 16
    maze = torch.ones((size*2, size*2), dtype=torch.bool)

    # Starting point
    x, y = (0, 0)
    maze[2*x, 2*y] = 0

    # Initialize the stack
    _stack = [(x, y)]
    while len(_stack) > 0:
        x, y = _stack[-1]

        # Define possible directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            print(nx, ny)
            if (nx >= 0) and (ny >= 0) and (nx < size) and (ny < size) and maze[2*nx, 2*ny] == 1:
                maze[2*nx, 2*ny] = 0
                maze[2*x+dx, 2*y+dy] = 0
                _stack.append((nx, ny))
                break
        else:
            _stack.pop()

    # Create an entrance and an exit
    maze[1, 0] = 0
    maze[-2, -1] = 0
    return maze.repeat_interleave(8, dim=0).repeat_interleave(8, dim=1)


@lru_cache
def _generate_diffusion_kernel():
    kernel = torch.tensor([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=torch.float32)
    return kernel / torch.sum(kernel)


@lru_cache
def generate_diffusion_kernel(size: int = 7, sigma: float = 1.0, slice_height: float = 0.1):
    """
    Generate a 2D slice of a hemispherical diffusion kernel on a flat plane.

    Args:
    size (int): The size of the kernel (must be odd)
    sigma (float): The standard deviation of the distribution
    slice_height (float): Height of the slice above the plane

    Returns:
    torch.Tensor: The diffusion kernel slice
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    center = size // 2
    kernel = torch.zeros((size, size), dtype=torch.float32)

    # Normalization constant (doubled because of hemispherical distribution)
    constant = 2 / (sigma * (2 * math.pi) ** 1.5)

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            r_squared = x*x + y*y

            # Contribution from the real source
            kernel[i, j] = constant * math.exp(-(r_squared + slice_height**2) / (2 * sigma * sigma))

            # Contribution from the image source (mirror source below the plane)
            kernel[i, j] += constant * math.exp(-(r_squared + (2-slice_height)**2) / (2 * sigma * sigma))

    return kernel

@lru_cache
def generate_plant_crowding_kernel():
    return torch.tensor([
        [0, 1, 1, 1, 0],
        [1, 1, 2, 1, 1],
        [1, 2, 0, 2, 1],
        [1, 1, 2, 1, 1],
        [0, 1, 1, 1, 0],
    ], dtype=torch.uint8)


@lru_cache
def generate_direction_kernels():
    # Create kernels for the 8 directions
    return torch.tensor([
        [-1, -1], [-1, 0], [-1, 1],
        [ 0, -1],          [ 0, 1],
        [ 1, -1], [ 1, 0], [ 1, 1]
    ], dtype=torch.float32)


@lru_cache
def lru_distance(dx, dy):
    """Cached distance function for efficient 8 direction distance calculation."""
    return torch.sqrt(dx**2 + dy**2)


def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def lerp(a, b, t):
    return a + t * (b - a)


def gradient(h, x, y):
    vectors = torch.tensor([
      [1, 1], [-1, 1], [1, -1], [-1, -1],
      [1, 0], [-1, 0], [0, 1], [0, -1]
    ], dtype=torch.float32)
    g = vectors[h % 8]
    return g[..., 0] * x + g[..., 1] * y


def perlin_noise(size, res):
    delta = (res[0] / size[0], res[1] / size[1])

    grid = torch.stack(torch.meshgrid(
        torch.arange(0, res[0], delta[0], dtype=torch.float32),
        torch.arange(0, res[1], delta[1], dtype=torch.float32)
    ), dim=-1)

    grid0 = grid.to(torch.int32)
    grid1 = grid0 + 1

    # Generate random gradients between -1 and 1
    random_grid = torch.rand((res[0] + 1, res[1] + 1, 2), dtype=torch.float32) * 2 - 1

    def gradient(hash, x, y):
        return hash[..., 0] * x + hash[..., 1] * y

    dot00 = gradient(
        random_grid[grid0[..., 0], grid0[..., 1]], grid[..., 0] - grid0[..., 0], grid[..., 1] - grid0[..., 1]
    )
    dot01 = gradient(
        random_grid[grid0[..., 0], grid1[..., 1]], grid[..., 0] - grid0[..., 0], grid[..., 1] - grid1[..., 1]
    )
    dot10 = gradient(
        random_grid[grid1[..., 0], grid0[..., 1]], grid[..., 0] - grid1[..., 0], grid[..., 1] - grid0[..., 1]
    )
    dot11 = gradient(
        random_grid[grid1[..., 0], grid1[..., 1]], grid[..., 0] - grid1[..., 0], grid[..., 1] - grid1[..., 1]
    )

    def fade(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    u = fade(grid - grid0)

    def lerp(a, b, t):
        return a + t * (b - a)

    nx0 = lerp(dot00, dot10, u[..., 0])
    nx1 = lerp(dot01, dot11, u[..., 0])
    nxy = lerp(nx0, nx1, u[..., 1])

    return torch.clamp(nxy + 0.5, 0, 1)


def pyramid_elevation(size: tuple, inverted: True, max_height: float = 1) -> torch.Tensor:
    """
    Create an inverted pyramid (cone) elevation map with the lowest point in the center.

    Args:
        size (tuple): The size of the elevation map (height, width).
        max_height (float): The maximum elevation at the edges of the map.

    Returns:
        torch.Tensor: The elevation map as a 2D tensor.
    """
    height, width = size
    center_y, center_x = height // 2, width // 2

    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')

    # Calculate distance from center
    distance = torch.maximum(
        torch.abs(y - center_y),
        torch.abs(x - center_x)
    )

    # Normalize distance to [0, 1] range
    max_distance = max(center_y, center_x)
    normalized_distance = distance.float() / max_distance

    # Create inverted pyramid
    if inverted:
        elevation = normalized_distance * max_height
    else:
        elevation = (1 - normalized_distance) * max_height

    return elevation


def range_elevation(size) -> torch.Tensor:
    tensor = torch.arange(0, size[0] * size[1])
    tensor = tensor.reshape(size)
    return tensor


def ramp_elevation(size: tuple, max_height: 255, dimension: int) -> torch.Tensor:
    """
    Create an inverted pyramid (cone) elevation map with the lowest point in the center.

    Args:
        size (tuple): The size of the elevation map (height, width).
        max_height (float): The maximum elevation at the edges of the map.

    Returns:
        torch.Tensor: The elevation map as a 2D tensor.
    """
    height, width = size
    if dimension == 0:
        return torch.arange(width).repeat(height, 1)
    elif dimension == 1:
        return torch.arange(height).repeat(width, 1).T


def scale_tensor(input_tensor, floor=64):
    # Ensure the input_tensor is of dtype uint8
    if input_tensor.dtype != torch.uint8:
        raise ValueError("Input tensor must be of type uint8")

    # Create a mask for the non-zero elements
    non_zero_mask = input_tensor != 0

    # Scale the non-zero elements
    scaled_tensor = input_tensor.clone().float()  # Create a copy and convert to float for scaling

    # Apply the scaling formula:
    scaled_tensor[non_zero_mask] = scaled_tensor[non_zero_mask] * ((floor - 1) / 255.0) + floor

    # Convert back to uint8
    scaled_tensor = scaled_tensor.to(torch.uint8)

    return scaled_tensor

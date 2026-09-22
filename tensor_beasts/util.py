import math
import random
from contextlib import nullcontext
from typing import List, Union, Tuple, SupportsAbs, Optional

import numpy as np
import torch
import torch.nn.functional as F
from functools import lru_cache, wraps

import time
import statistics

from tensordict import TensorDict, NestedKey

# Initialize a dictionary to store function execution times
execution_times = {}


def device_lru_cache(fn):
    """lru_cache for functions returning tensors, keyed by device as well as args.

    A plain lru_cache here caches a kernel built on whichever device happened to
    be default at the first call. Any later call under a different default
    device then gets a kernel on the wrong device, which surfaces as
    "Input type (MPSFloatType) and weight type (torch.FloatTensor) should be the
    same" from conv2d rather than as anything that names the cache.
    """
    cached = lru_cache(maxsize=None)(lambda _device, *args, **kwargs: fn(*args, **kwargs))

    @wraps(fn)
    def wrapper(*args, **kwargs):
        return cached(torch.get_default_device(), *args, **kwargs)

    wrapper.cache_clear = cached.cache_clear
    return wrapper



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


@device_lru_cache
def directional_kernel_set(size: int):
    return {
        1: generate_direction_kernel(size, 1),
        2: generate_direction_kernel(size, 2),
        3: generate_direction_kernel(size, 3),
        4: generate_direction_kernel(size, 4)
    }


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
    """Shift (..., H, W) by one cell in `direction`, zero-filling the vacated edge.

    F.pad counts from the last dimension backwards, so the pad tuples below are
    already rank-agnostic; only the slicing needed to move off the leading axes.
    """
    if direction == 1:  # Up
        return torch.nn.functional.pad(mat[..., 1:, :], (0, 0, 0, 1), value=0)
    elif direction == 2:  # Down
        return torch.nn.functional.pad(mat[..., :-1, :], (0, 0, 1, 0), value=0)
    elif direction == 3:  # Left
        return torch.nn.functional.pad(mat[..., :, 1:], (0, 1, 0, 0), value=0)
    elif direction == 4:  # Right
        return torch.nn.functional.pad(mat[..., :, :-1], (1, 0, 0, 0), value=0)


@device_lru_cache
def get_edge_mask(shape: tuple):
    mask = torch.zeros(shape, dtype=torch.float32)
    mask[..., 0, :] = 1
    mask[..., -1, :] = 1
    mask[..., :, 0] = 1
    mask[..., :, -1] = 1
    return mask


def get_direction_matrix(matrix, random_choices=None):
    down = torch.roll(matrix, shifts=-1, dims=-2)
    up = torch.roll(matrix, shifts=1, dims=-2)
    right = torch.roll(matrix, shifts=-1, dims=-1)
    left = torch.roll(matrix, shifts=1, dims=-1)

    # Setting the boundaries to 0 to avoid wrapping around behavior
    up[..., -1, :] = 0
    down[..., 0, :] = 0
    left[..., :, -1] = 0
    right[..., :, 0] = 0

    # Stack matrices to work with all directions together. dim=-1 here is the
    # *stacked* direction axis (5 candidates), not a spatial axis; it is trailing
    # and stays trailing at any input rank, so the reductions below are already
    # rank-agnostic.
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


def as_conv_batch(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
    """Collapse every leading dimension of a (..., H, W) tensor into conv2d's N.

    Returns the (N, 1, H, W) view and the leading shape needed to restore it.
    Using reshape rather than unsqueeze/squeeze keeps this correct at any rank,
    including the degenerate cases where a leading dimension happens to be 1 --
    a bare .squeeze() would silently drop those.
    """
    if input.dim() < 2:
        raise ValueError(f"Expected at least 2 dimensions (..., H, W), got shape {tuple(input.shape)}")
    leading = input.shape[:-2]
    return input.reshape(-1, 1, *input.shape[-2:]), leading


def torch_correlate_2d(input: torch.Tensor, kernel, mode='constant', cval=0):
    """
    Mimic scipy.ndimage.correlate using PyTorch's conv2d.

    Rank-agnostic: any leading dimensions are folded into the conv batch and
    restored afterwards.

    Parameters:
    - input: (..., H, W) torch tensor, the input.
    - kernel: 2D torch tensor, the kernel for correlation.
    - mode: str, boundary mode (only 'constant' mode implemented similar to scipy.ndimage.correlate).
    - cval: float, value to fill pad when mode is 'constant'.

    Returns:
    - result: (..., H, W) torch tensor, result of correlation.
    """
    if mode != 'constant':
        raise ValueError("Only 'constant' mode is implemented.")

    input_dtype = input.dtype
    input = input.type(torch.float32)
    kernel = kernel.type(torch.float32)

    input_4d, leading = as_conv_batch(input)

    if kernel.dim() == 2:
        kernel = kernel.unsqueeze(0).unsqueeze(0)

    # Define padding based on mode
    pad_size = (kernel.shape[-1] // 2, kernel.shape[-2] // 2)
    pad = (pad_size[0], pad_size[0], pad_size[1], pad_size[1])

    input_padded = F.pad(input_4d, pad=pad, mode='constant', value=cval)

    result = F.conv2d(input_padded, kernel)

    # Restore the original leading dimensions.
    result = result.reshape(*leading, *result.shape[-2:])
    return result.type(input_dtype)


@device_lru_cache
def directional_kernel_bank(size: int) -> torch.Tensor:
    """The four directional kernels stacked as conv2d weights of shape (4, 1, K, K).

    Index d-1 holds the kernel for direction d, matching directional_kernel_set.
    """
    kernels = directional_kernel_set(size)
    return torch.stack([kernels[d].type(torch.float32) for d in range(1, 5)]).unsqueeze(1)


def torch_correlate_2d_bank(input: torch.Tensor, kernels: torch.Tensor, cval: float = 0) -> torch.Tensor:
    """Correlate a 2D input against a bank of kernels in a single conv2d.

    Equivalent to stacking torch_correlate_2d(input, k) over each kernel in the
    bank, but pays the pad and dispatch cost once rather than once per kernel.

    Rank-agnostic: the kernel axis is inserted just before the spatial axes, so a
    (H, W) input still gives (K, H, W) while a (B, H, W) input gives (B, K, H, W).
    Index it as result[..., d, :, :] rather than result[d].

    Parameters:
    - input: (..., H, W) tensor.
    - kernels: (K, 1, kh, kw) conv2d weights.
    - cval: constant boundary fill value.

    Returns:
    - (..., K, H, W) tensor.
    """
    pad_h, pad_w = kernels.shape[-2] // 2, kernels.shape[-1] // 2
    input_4d, leading = as_conv_batch(input.type(torch.float32))
    input_padded = F.pad(
        input_4d,
        pad=(pad_w, pad_w, pad_h, pad_h),
        mode='constant',
        value=cval,
    )
    result = F.conv2d(input_padded, kernels)
    return result.reshape(*leading, *result.shape[-3:])


def torch_correlate_3d(input_tensor, weights):
    """
    Apply a batched 2D convolution to a (..., H, W, C) tensor using (K, K) weights.

    Each channel is convolved independently with the same kernel.

    NOTE: unlike most tensors in the simulation, this one is *channel-trailing* --
    the spatial axes are -3 and -2, not -2 and -1. Any leading dimensions (e.g. a
    world batch) are folded into the conv batch and restored afterwards.

    Parameters:
    - input_tensor: torch.Tensor of shape (..., H, W, C)
    - weights: torch.Tensor of shape (K, K)

    Returns:
    - output_tensor: torch.Tensor of shape (..., H, W, C)
    """
    assert input_tensor.dim() >= 3, "input_tensor must be of shape (..., H, W, C)"

    # Ensure weights is of shape (K, K)
    assert len(weights.shape) == 2, "weights must be 2D"

    leading = input_tensor.shape[:-3]
    H, W, C = input_tensor.shape[-3:]

    # Move the channel axis in front of the spatial axes and fold everything
    # before H, W into the conv batch: (..., C, H, W) -> (N, 1, H, W)
    input_4d = input_tensor.movedim(-1, -3).reshape(-1, 1, H, W)

    # Convert weights to shape (1, 1, K, K) to apply the same kernel on all channels
    weight_4d = weights.unsqueeze(0).unsqueeze(0)

    conv_output = F.conv2d(input_4d, weight_4d, stride=1, padding='same', groups=1)

    # Restore (..., C, H, W) and then move the channel axis back to the end.
    output_tensor = conv_output.reshape(*leading, C, H, W).movedim(-3, -1)

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


@device_lru_cache
def _generate_diffusion_kernel():
    kernel = torch.tensor([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=torch.float32)
    return kernel / torch.sum(kernel)


@device_lru_cache
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

@device_lru_cache
def generate_plant_crowding_kernel():
    return torch.tensor([
        [0, 1, 1, 1, 0],
        [1, 1, 2, 1, 1],
        [1, 2, 0, 2, 1],
        [1, 1, 2, 1, 1],
        [0, 1, 1, 1, 0],
    ], dtype=torch.uint8)


@device_lru_cache
def generate_direction_kernels(eight_directions: bool = False, include_center: bool = False):
    if not include_center:
        if eight_directions:
            # Create kernels for the 8 directions
            return torch.tensor([
                [-1, -1], [-1, 0], [-1, 1],
                [ 0, -1],          [ 0, 1],
                [ 1, -1], [ 1, 0], [ 1, 1]
            ], dtype=torch.float32)
        else:
            # Create kernels for the 4 directions
            return torch.tensor([
                          [-1, 0],
                [ 0, -1],          [ 0, 1],
                          [ 1, 0]
            ], dtype=torch.float32)
    else:
        if eight_directions:
            # Create kernels for the 8 directions
            return torch.tensor([
                [-1, -1], [-1, 0], [-1, 1],
                [ 0, -1], [ 0, 0], [ 0, 1],
                [ 1, -1], [ 1, 0], [ 1, 1]
            ], dtype=torch.float32)
        else:
            # Create kernels for the 4 directions
            return torch.tensor([
                [-1, 0], [ 0, -1], [ 0, 0], [ 0, 1], [ 1, 0]
            ], dtype=torch.float32)


@lru_cache
def lru_distance(dx, dy, scale: float = 1.0):
    """LRU distance function for efficient 8 direction distance calculation."""
    return torch.sqrt(dx**2 + dy**2) * scale


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


def perlin_noise(size, res, octaves=4, persistence=0.5, lacunarity=2.0):
    """Perlin noise of shape ``size``, which may carry leading batch dimensions.

    The generator itself is two-dimensional: it builds a meshgrid over
    ``size[0]`` by ``size[1]``. Given a batched ``(B, H, W)`` it used to read B
    and H as the grid and return the wrong shape, and its one caller wrapped
    that in ``except (IndexError, RuntimeError)`` and silently fell back to
    uniform random, so a batched world quietly lost its terrain. Leading
    dimensions are now peeled off and each world gets an INDEPENDENT field,
    which is the point: worlds that share their terrain are not independent
    worlds.
    """
    if len(size) > 2:
        leading, spatial = tuple(size[:-2]), tuple(size[-2:])
        count = 1
        for extent in leading:
            count *= extent
        fields = [
            _perlin_noise_2d(spatial, res, octaves, persistence, lacunarity)
            for _ in range(count)
        ]
        return torch.stack(fields).reshape(*leading, *spatial)
    return _perlin_noise_2d(tuple(size), res, octaves, persistence, lacunarity)


def _perlin_noise_2d(size, res, octaves=4, persistence=0.5, lacunarity=2.0):
    # Run on CPU to avoid MPS advanced indexing race conditions, then move to target device
    target_device = torch.get_default_device()
    run_on_cpu = target_device is not None and target_device.type == 'mps'

    if run_on_cpu:
        torch.mps.synchronize()  # Ensure all MPS ops complete first

    def generate_noise(x, y, res):
        grid0_x, grid0_y = x.to(torch.int32), y.to(torch.int32)
        grid1_x, grid1_y = grid0_x + 1, grid0_y + 1

        random_grid = torch.rand((res[0] + 1, res[1] + 1, 2), dtype=torch.float32) * 2 - 1

        def gradient(hash, x, y):
            return hash[..., 0] * x + hash[..., 1] * y

        dot00 = gradient(random_grid[grid0_x, grid0_y], x - grid0_x, y - grid0_y)
        dot01 = gradient(random_grid[grid0_x, grid1_y], x - grid0_x, y - grid1_y)
        dot10 = gradient(random_grid[grid1_x, grid0_y], x - grid1_x, y - grid0_y)
        dot11 = gradient(random_grid[grid1_x, grid1_y], x - grid1_x, y - grid1_y)

        def fade(t):
            return 6 * t**5 - 15 * t**4 + 10 * t**3

        u = fade(x - grid0_x)
        v = fade(y - grid0_y)

        def lerp(a, b, t):
            return a + t * (b - a)

        nx0 = lerp(dot00, dot10, u)
        nx1 = lerp(dot01, dot11, u)
        nxy = lerp(nx0, nx1, v)

        return nxy

    def domain_warp(x, y, warp_amount=0.1):
        warp_res = (max(1, res[0]//2), max(1, res[1]//2))
        # Scale coordinates to match the warp resolution
        sx = x * (warp_res[0] / res[0])
        sy = y * (warp_res[1] / res[1])
        wx = x + warp_amount * generate_noise(sx, sy, warp_res)
        wy = y + warp_amount * generate_noise(sx, sy, warp_res)
        return wx, wy

    # Generate on CPU if MPS to avoid race conditions
    with torch.device('cpu') if run_on_cpu else nullcontext():
        # Exclude endpoint so coordinates stay in [0, res) — prevents out-of-bounds grid indexing
        base_x = torch.linspace(0, res[0], size[0] + 1)[:-1]
        base_y = torch.linspace(0, res[1], size[1] + 1)[:-1]
        x, y = torch.meshgrid(base_x, base_y, indexing='ij')

        # Apply domain warping, then clamp to valid range
        x, y = domain_warp(x, y)
        x = torch.clamp(x, 0, res[0] - 1e-6)
        y = torch.clamp(y, 0, res[1] - 1e-6)

        # Generate fractal Brownian motion (fBm)
        noise = torch.zeros(size)
        frequency = 1
        amplitude = 1
        for _ in range(octaves):
            noise += amplitude * generate_noise(x * frequency, y * frequency, (int(res[0]*frequency), int(res[1]*frequency)))
            frequency *= lacunarity
            amplitude *= persistence

        # Normalize the noise
        noise = (noise - noise.min()) / (noise.max() - noise.min())

    # Move to target device if we ran on CPU
    if run_on_cpu:
        noise = noise.to(target_device)

    return noise


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


def roll_with_padding(
    input: torch.Tensor,
    shifts: Union[int, Tuple[int, ...]],
    dims: Union[int, Tuple[int, ...]],
    padding_mode: str ='constant',
    padding_value: float = 0.0
):
    if isinstance(shifts, int):
        shifts = (shifts,)
    if isinstance(dims, int):
        dims = (dims,)

    if len(shifts) != len(dims):
        raise ValueError("Length of shifts must match length of dims")

    ndim = input.dim()
    # Normalize dims so negative (from-the-end) indices work, which is what makes
    # this rank-agnostic: callers pass dims=(-2, -1) for the spatial axes.
    dims = tuple(d % ndim for d in dims)

    # F.pad counts pairs from the LAST dimension backwards, while dims are
    # counted from the front, so the pair for dim d sits at index 2*(ndim-1-d).
    paddings = [0] * (ndim * 2)
    for shift, dim in zip(shifts, dims):
        pad_left = max(0, shift)
        pad_right = max(0, -shift)
        base = 2 * (ndim - 1 - dim)
        paddings[base] = pad_left
        paddings[base + 1] = pad_right

    if padding_mode == 'constant':
        result = F.pad(input, paddings, mode='constant', value=padding_value)
    elif padding_mode in ['reflect', 'replicate', 'circular']:
        result = F.pad(input, paddings, mode=padding_mode)
    else:
        raise ValueError(f"Unsupported padding mode: {padding_mode}")

    result = torch.roll(result, shifts, dims)

    slices = [slice(None)] * input.dim()
    for shift, dim in zip(shifts, dims):
        if shift > 0:
            slices[dim] = slice(shift, None)
        elif shift < 0:
            slices[dim] = slice(None, shift)

    return result[tuple(slices)]


def neighbors(
    input: torch.Tensor,
    padding_value: float = 0,
    padding_mode: str = 'constant',
    eight_direction: bool = False,
    reverse: bool = False
) -> torch.Tensor:
    kernels = generate_direction_kernels(eight_direction)
    # If we roll the input tensor by the negative of the kernel direction, the resulting tensor will
    # contain the neighbor in the direction of the kernel.
    if not reverse:
        # NOT in place: generate_direction_kernels is device_lru_cached, so `*=`
        # mutated the cached tensor and flipped the sign on every subsequent call.
        kernels = kernels * -1
    if eight_direction:
        neighbors_tensor = torch.zeros(input.shape + (8,), dtype=input.dtype)
    else:
        neighbors_tensor = torch.zeros(input.shape + (4,), dtype=input.dtype)
    for i, (dy, dx) in enumerate(kernels):
        neighbors_tensor[..., i] = roll_with_padding(
            input,
            shifts=(int(dy), int(dx)),
            dims=(-2, -1),
            padding_mode=padding_mode,
            padding_value=padding_value
        )
    return neighbors_tensor


def pad_and_view(td: TensorDict, key: NestedKey, pad: Tuple[int, ...], value: Union[int, float] = 0):
    padded_key = f"{key}_padded" if isinstance(key, str) else key[:-1] + (f"{key[-1]}_padded",)
    if padded_key not in td:
        original = td[key]
        padded = torch.nn.functional.pad(original, pad, mode='constant', value=value)
        td[padded_key] = padded

        # F.pad consumes pairs from the LAST dimension backwards, so the pad pairs
        # map onto the trailing dims in reverse order. Leading (e.g. batch) dims
        # are left alone via the Ellipsis.
        slices = (Ellipsis,) + tuple(
            slice(pad[i * 2], -pad[i * 2 + 1] if pad[i * 2 + 1] else None)
            for i in reversed(range(len(pad) // 2))
        )
        view = td[padded_key][slices]
        td[key] = view
    assert td[key].storage().data_ptr() == td[padded_key].storage().data_ptr()


def unfold_neighbors(td: TensorDict, key: NestedKey, kernel_size: Tuple[int, ...], pad_value: float = 0.0):
    neighbors_key = f"{key}_neighbors" if isinstance(key, str) else key[:-1] + (f"{key[-1]}_neighbors",)
    if neighbors_key in td:
        return td[neighbors_key]
    # Ensure we have a padded version
    padding = (kernel_size[0] - 1) // 2
    # Only the two spatial dims are padded; any leading dims are untouched.
    pad = (padding,) * 4
    pad_and_view(td, key, pad, pad_value)

    padded_key = f"{key}_padded" if isinstance(key, str) else key[:-1] + (f"{key[-1]}_padded",)
    x_padded = td[padded_key]
    assert td[key].storage().data_ptr() == x_padded.storage().data_ptr()

    leading = td[key].shape[:-2]
    H, W = td[key].shape[-2:]
    kH, kW = kernel_size

    # Calculate strides
    stride = x_padded.stride()

    # Create the unfolded view. Leading dims keep their own size/stride; the
    # spatial strides are reused for the kernel window axes.
    unfolded = torch.as_strided(
        x_padded,
        size=(*leading, H, W, kH, kW),
        stride=(*stride[:-2], stride[-2], stride[-1], stride[-2], stride[-1]),
    )
    assert unfolded.storage().data_ptr() == x_padded.storage().data_ptr()

    # Store the result
    td[neighbors_key] = unfolded
    return unfolded


def fold_neighbors(td: TensorDict, key: NestedKey):
    inverse_neighbors_key = f"{key}_inverse" if isinstance(key, str) else key[:-1] + (f"{key[-1]}_inverse",)
    if inverse_neighbors_key in td:
        return td[inverse_neighbors_key]

    pad = (0, 0, 0, 0, 1, 1, 1, 1)
    pad_and_view(td, key, pad, 0)

    padded_key = f"{key}_padded" if isinstance(key, str) else key[:-1] + (f"{key[-1]}_padded",)
    x_padded = td[padded_key].contiguous()

    leading = x_padded.shape[:-4]
    H, W, kH, kW = x_padded.shape[-4:]
    H_orig, W_orig = H - 2, W - 2
    assert kH == 3 and kW == 3, "This function assumes a 3x3 neighborhood"

    # Calculate the correct strides (x_padded was made contiguous above)
    s0 = W * kH * kW
    s1 = kH * kW
    leading_strides = x_padded.stride()[:-4]

    inverse_neighbors = torch.as_strided(
        x_padded,
        size=(*leading, H_orig, W_orig, 3, 3),
        stride=(*leading_strides, s0, s1, W, 1),
        storage_offset=s0 + s1
    )

    td[inverse_neighbors_key] = inverse_neighbors
    return inverse_neighbors


def custom_filter_operation(input_tensor, kernels):
    leading = input_tensor.shape[:-2]
    H, W = input_tensor.shape[-2:]

    # Pad the input tensor (F.pad already counts from the last dim backwards)
    padded = F.pad(input_tensor, (1, 1, 1, 1))

    # Unfold the padded tensor to create patches. Each unfold replaces its dim
    # with the window count and appends the window, so unfolding dim -2 twice
    # walks H then W: (..., H+2, W+2) -> (..., H, W+2, 3) -> (..., H, W, 3, 3).
    patches = padded.unfold(-2, 3, 1).unfold(-2, 3, 1)

    # Reshape patches and kernels for batched matrix multiplication
    patches = patches.reshape(*leading, H, W, 9)
    kernels = kernels.reshape(*leading, H, W, 9)

    # Perform batched matrix multiplication
    result = torch.sum(patches * kernels, dim=-1)

    return result


def apply_kernels(td, input_key, kernels):
    input_patches = unfold_neighbors(td, input_key, (3, 3))

    # Apply the kernels through einstein summation
    updated = torch.einsum('...ijkl, ...ijkl -> ...ij', input_patches, kernels)
    td.set(input_key, updated, inplace=True)


def flow_gradient(input, apply_distances=True):
    """Per-cell 3x3 outward gradient of a (..., H, W) field, returned as (..., H, W, 3, 3)."""
    device = input.device
    leading = input.shape[:-2]
    H, W = input.shape[-2:]
    # The unsqueeze(0)/squeeze() round trip this used to do was a no-op for a 2D
    # input and would have dropped any size-1 leading dim for a batched one.
    padded_input = torch.nn.functional.pad(
        input,
        (1, 1, 1, 1),
        mode="constant",
        value=0
    )
    stride = padded_input.stride()
    unfolded = torch.as_strided(
        padded_input,
        size=(*leading, H, W, 3, 3),
        stride=(*stride[:-2], stride[-2], stride[-1], stride[-2], stride[-1])
    )

    expanded_input = input.reshape(*leading, H, W, 1, 1).expand(*leading, H, W, 3, 3)

    if apply_distances:
        euclidean_distance_matrix = torch.tensor(
            [[1.4142, 1.0000, 1.4142],
             [1.0000, 1.0000, 1.0000],
             [1.4142, 1.0000, 1.4142]],
            dtype=input.dtype,
            device=device
        )
        gradient = (expanded_input - unfolded) / euclidean_distance_matrix.unsqueeze(0).unsqueeze(0)
    else:
        gradient = (expanded_input - unfolded)

    # Zero the flows that would leave the grid. The first two indices are the
    # spatial axes, the last two the 3x3 window.
    gradient[..., 0, :, 0, :] = 0
    gradient[..., -1, :, -1, :] = 0
    gradient[..., :, 0, :, 0] = 0
    gradient[..., :, -1, :, -1] = 0
    assert not torch.isnan(gradient).any()
    assert not torch.isinf(gradient).any()

    return gradient


def flow(
    input: torch.tensor,
    gradient: Optional[torch.tensor] = None,
    outflow: Optional[torch.tensor] = None,
    flow_rate=0.1,
    relaxation_factor=1.0
):
    if gradient is None:
        assert outflow is not None, "Outflow must be provided if gradient is not provided."
        outflow = outflow.clone()
        outflow *= flow_rate

    if outflow is None:
        assert gradient is not None, "Gradient must be provided if outflow is not provided."
        outflow = gradient.clamp(min=0.0) * flow_rate

    total_outflow = torch.sum(outflow, dim=(-1, -2))

    condition = (total_outflow <= input).unsqueeze(-1).unsqueeze(-1).expand_as(outflow)
    adjusted_outflow = (
        outflow *
        torch.nan_to_num(input / total_outflow, 0.0).unsqueeze(-1).unsqueeze(-1).expand_as(outflow)
    )

    corrected_outflow = safe_where(
        condition,
        outflow,
        adjusted_outflow
    )

    padded_corrected_outflow = torch.nn.functional.pad(
        corrected_outflow,
        (0, 0, 0, 0, 1, 1, 1, 1),
        mode="constant", value=0
    )

    padded_input = torch.nn.functional.pad(
        input,
        (1, 1, 1, 1),
        mode="constant",
        value=0
    )
    inflow = torch.zeros_like(padded_input)
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue  # Skip the center cell
            shifted_outflow = torch.roll(
                padded_corrected_outflow[..., i, j], shifts=(i - 1, j - 1), dims=(-2, -1)
            )
            inflow.add_(shifted_outflow)

    inflow = inflow[..., 1:-1, 1:-1]

    assert torch.isclose(torch.sum(inflow), torch.sum(corrected_outflow))

    result = (input + inflow) - (torch.sum(corrected_outflow, dim=(-1, -2)))

    if relaxation_factor != 1.0:
        result = result * relaxation_factor + input * (1 - relaxation_factor)

    return result, inflow, corrected_outflow


def safe_where(condition, x, y):
    """
    Custom implementation to replace torch.where without MPS issues.

    Args:
    condition (torch.Tensor): A boolean tensor
    x (torch.Tensor): Tensor to use where condition is True
    y (torch.Tensor): Tensor to use where condition is False

    Returns:
    torch.Tensor: A tensor with values from x where condition is True, and values from y where condition is False
    """
    condition = condition.to(torch.float32)
    return condition * x + (1 - condition) * y




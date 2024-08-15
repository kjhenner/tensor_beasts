from typing import Tuple

import torch
from omegaconf import DictConfig
from tensordict import TensorDict


def default_renderer(td: TensorDict, config):
    data = td.get(config.key)
    if data.ndim == 2:
        return data.unsqueeze(-1).expand(-1, -1, 3)
    else:
        return data


def layered_renderer(td: TensorDict, config: DictConfig):
    # Initialize result with black background
    result = None

    for layer in config.layers:
        data = td.get(layer.key)
        if result is None:
            result = torch.zeros(data.shape[0], data.shape[1], 3, dtype=torch.uint8)

        # Apply threshold and create mask
        mask = (data > layer.threshold).float().unsqueeze(-1)

        # Normalize based on input_range
        input_min, input_max = layer.input_range
        normalized = torch.clamp((data - input_min) / (input_max - input_min), 0, 1)

        color_min = torch.tensor(layer.color_min, dtype=torch.float32)
        color_max = torch.tensor(layer.color_max, dtype=torch.float32)

        colored = color_min + normalized.unsqueeze(-1) * (color_max - color_min)

        # Apply this layer to the result
        result = result * (1 - mask) + colored * mask

    # Convert to uint8
    result = torch.clamp(result, 0, 255).to(torch.uint8)
    return result


def dispatch_render(td: TensorDict, config: DictConfig):
    if config.fn_name == "default":
        return default_renderer(td, config)
    if config.fn_name == "layered":
        return layered_renderer(td, config)
    else:
        raise ValueError(f"Unknown display function: {config.fn_name}")

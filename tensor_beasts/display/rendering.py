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


def interpolate_color(
    color1: torch.tensor,
    color2: torch.tensor,
    t: torch.tensor
) -> torch.tensor:
    color1 = color1.unsqueeze(0).repeat(t.shape[0], 1)
    color2 = color2.unsqueeze(0).repeat(t.shape[0], 1)
    return color1 + (color2 - color1) * t.unsqueeze(-1)


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


def cross_section(td: TensorDict, config: DictConfig):

    background_color = torch.tensor(list(config.background_color))
    screen_height = config.screen_height

    if config.section_dim == 'y':
        slice_fn = lambda x: x[config.section_idx, :]
        screen_width = td.get(config.levels[0].key).shape[1]
    elif config.section_dim == 'x':
        slice_fn = lambda x: x[:, config.section_idx]
        screen_width = td.get(config.levels[0].key).shape[0]
    else:
        raise ValueError("Invalid section_dim. Must be 'x' or 'y'.")

    output = torch.zeros(screen_height, screen_width, 3, dtype=torch.uint8)

    base_height = torch.zeros(screen_width, dtype=torch.float32)
    mask_ref = torch.linspace(screen_height, 0, screen_height).unsqueeze(-1).expand(-1, screen_width)

    for level in config.levels:
        color = torch.tensor(level.color)

        # Create a mask for this level
        mask = (mask_ref >= base_height.unsqueeze(0).expand(screen_height, -1)).bool()

        # Apply the mask to the color
        if level.get("color_mod_key") and level.get("color_mod"):
            color_mod_scale = slice_fn(td.get(level.color_mod_key))
            level_color = interpolate_color(color, torch.tensor(level.color_mod), color_mod_scale)
        else:
            level_color = color.unsqueeze(0).unsqueeze(0).repeat(screen_height, screen_width, 1)
        output = torch.where(mask.unsqueeze(-1), level_color, output)

        # Get the height data for this level
        height_data = td.get(level.key) * level.scale
        base_height += slice_fn(height_data)

    mask = (mask_ref >= base_height.unsqueeze(0).expand(screen_height, -1)).bool()
    level_color = background_color.unsqueeze(0).unsqueeze(0).repeat(screen_height, screen_width, 1)
    output = torch.where(mask.unsqueeze(-1), level_color, output)
    print(output.shape)

    return output


def dispatch_render(td: TensorDict, config: DictConfig):
    if config.fn_name == "default":
        return default_renderer(td, config)
    if config.fn_name == "layered":
        return layered_renderer(td, config)
    if config.fn_name == "cross_section":
        return cross_section(td, config)
    else:
        raise ValueError(f"Unknown display function: {config.fn_name}")

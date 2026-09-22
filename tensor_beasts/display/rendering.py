from typing import Tuple

import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from tensor_beasts.snapshot import WorldSnapshot


def default_renderer(source, config):
    data = source.get(config.key)
    input_range = getattr(config, "input_range", None)
    if input_range is not None:
        input_min, input_max = input_range
        normalized = torch.clamp((data - input_min) / (input_max - input_min), 0, 1)
        data = (normalized * 255).to(torch.uint8)
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


def layered_renderer(source, config: DictConfig):
    # Initialize result with black background
    result = None

    for layer in config.layers:
        data = source.get(layer.key).float()
        if result is None:
            result = torch.zeros(data.shape[0], data.shape[1], 3, dtype=torch.uint8)

        # Apply threshold and create mask
        mask = (data > layer.threshold).float().unsqueeze(-1)

        # Normalize based on input_range, with optional log scale
        input_min, input_max = layer.input_range
        if layer.get("log_scale", False):
            # Log scale: map log1p(data) to [0, 1]
            log_min = torch.log1p(torch.tensor(max(input_min, 0), dtype=torch.float32))
            log_max = torch.log1p(torch.tensor(input_max, dtype=torch.float32))
            normalized = torch.clamp((torch.log1p(data) - log_min) / (log_max - log_min + 1e-8), 0, 1)
        else:
            normalized = torch.clamp((data - input_min) / (input_max - input_min), 0, 1)

        color_min = torch.tensor(layer.color_min, dtype=torch.float32)
        color_max = torch.tensor(layer.color_max, dtype=torch.float32)

        colored = color_min + normalized.unsqueeze(-1) * (color_max - color_min)

        # Apply this layer to the result
        result = result * (1 - mask) + colored * mask

    # Convert to uint8
    result = torch.clamp(result, 0, 255).to(torch.uint8)
    return result


def cross_section(source, config: DictConfig):

    background_color = torch.tensor(list(config.background_color))
    screen_height = config.screen_height

    if config.section_dim == 'y':
        slice_fn = lambda x: x[config.section_idx, :]
        screen_width = source.get(config.levels[0].key).shape[1]
    elif config.section_dim == 'x':
        slice_fn = lambda x: x[:, config.section_idx]
        screen_width = source.get(config.levels[0].key).shape[0]
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
            color_mod_scale = slice_fn(source.get(level.color_mod_key))
            level_color = interpolate_color(color, torch.tensor(level.color_mod), color_mod_scale)
        else:
            level_color = color.unsqueeze(0).unsqueeze(0).repeat(screen_height, screen_width, 1)
        output = torch.where(mask.unsqueeze(-1), level_color, output)

        # Get the height data for this level
        height_data = source.get(level.key) * level.scale
        base_height += slice_fn(height_data)

    mask = (mask_ref >= base_height.unsqueeze(0).expand(screen_height, -1)).bool()
    level_color = background_color.unsqueeze(0).unsqueeze(0).repeat(screen_height, screen_width, 1)
    output = torch.where(mask.unsqueeze(-1), level_color, output)

    return output


def rgb_species_renderer(source, config: DictConfig) -> torch.Tensor:
    """
    Render species as RGB channels with scent and biomass merged additively.

    Each channel combines:
    - Scent: normalized to scent_weight (default 0.5) of the channel
    - Biomass: adds brightness on top of scent where entities exist

    Config:
        red/green/blue: {scent_key, biomass_key, biomass_range}
        scent_max: max scent value for normalization (default 255)
        scent_log_scale: use log scale for scent (default False)
        scent_weight: fraction of channel allocated to scent (default 0.5)
        biomass_weight: fraction of channel allocated to biomass (default 0.5)
    """
    shape = None
    channels = {}

    scent_max = config.get("scent_max", 255.0)
    scent_log_scale = config.get("scent_log_scale", False)
    scent_weight = config.get("scent_weight", 0.5)
    biomass_weight = config.get("biomass_weight", 0.5)

    for channel_name in ["red", "green", "blue"]:
        channel_config = config.get(channel_name)
        if channel_config is None:
            continue

        # Get scent data and normalize
        scent_data = source.get(channel_config.scent_key).float()
        if shape is None:
            shape = scent_data.shape

        if scent_log_scale:
            # Log scale: log1p maps [0, scent_max] to [0, log1p(scent_max)]
            log_max = torch.log1p(torch.tensor(scent_max, dtype=torch.float32))
            scent_normalized = (torch.log1p(scent_data) / log_max).clamp(0, 1)
        else:
            scent_normalized = (scent_data / scent_max).clamp(0, 1)

        # Get biomass data and normalize
        biomass_data = source.get(channel_config.biomass_key)
        biomass_min, biomass_max = channel_config.get("biomass_range", [0, 255])
        biomass_normalized = ((biomass_data.float() - biomass_min) / (biomass_max - biomass_min + 1e-8)).clamp(0, 1)

        # Merge: scent contributes up to scent_weight * 255, biomass adds up to biomass_weight * 255
        # Both are additive, clamped to 255
        channel_value = (scent_normalized * scent_weight * 255 + biomass_normalized * biomass_weight * 255)
        channel = channel_value.clamp(0, 255).to(torch.uint8)
        channels[channel_name] = channel

    # Build RGB output
    output = torch.zeros(shape[0], shape[1], 3, dtype=torch.uint8)
    if "red" in channels:
        output[:, :, 0] = channels["red"]
    if "green" in channels:
        output[:, :, 1] = channels["green"]
    if "blue" in channels:
        output[:, :, 2] = channels["blue"]

    return output


def histogram_renderer(source, config: DictConfig) -> torch.Tensor:
    """
    Render a histogram of feature values as an image.

    Config options:
        key: Feature key to histogram
        bins: Number of histogram bins (default: 32)
        range: [min, max] value range (default: auto from data)
        log_scale: Use log scale for counts (default: False)
        color: Bar color [R, G, B] (default: [0, 200, 255])
        background: Background color [R, G, B] (default: [20, 20, 20])
        show_stats: Show mean/std/count text (default: True)
        filter_zeros: Exclude zero values (default: True)
        screen_height: Output image height (default: 256)
        screen_width: Output image width (default: 512)
    """
    data = source.get(config.key)

    # Flatten and optionally filter zeros
    values = data.flatten().float()
    if config.get("filter_zeros", True):
        values = values[values > 0]

    # Get config values with defaults
    bins = config.get("bins", 32)
    color = torch.tensor(config.get("color", [0, 200, 255]), dtype=torch.uint8)
    background = torch.tensor(config.get("background", [20, 20, 20]), dtype=torch.uint8)
    screen_height = config.get("screen_height", 256)
    screen_width = config.get("screen_width", 512)

    # Initialize output
    output = background.unsqueeze(0).unsqueeze(0).expand(screen_height, screen_width, 3).clone()

    # Handle empty data
    if values.numel() == 0:
        return output.to(torch.uint8)

    # Determine range
    if config.get("range"):
        range_min, range_max = config.range
    else:
        range_min, range_max = values.min().item(), values.max().item()
        if range_min == range_max:
            range_max = range_min + 1

    # Compute histogram
    hist = torch.histc(values, bins=bins, min=range_min, max=range_max)

    # Apply log scale if requested
    if config.get("log_scale", False):
        hist = torch.log1p(hist)

    # Normalize to [0, 1]
    hist_max = hist.max()
    if hist_max > 0:
        hist_normalized = hist / hist_max
    else:
        hist_normalized = hist

    # Render bars
    bar_width = screen_width // bins
    margin_bottom = 20 if config.get("show_stats", True) else 5
    margin_top = 5
    bar_height_max = screen_height - margin_bottom - margin_top

    for i in range(bins):
        bar_height = int(hist_normalized[i].item() * bar_height_max)
        if bar_height > 0:
            x_start = i * bar_width
            x_end = min((i + 1) * bar_width - 1, screen_width - 1)
            y_start = screen_height - margin_bottom - bar_height
            y_end = screen_height - margin_bottom

            # Draw bar
            output[y_start:y_end, x_start:x_end, :] = color

    # Draw axis line
    axis_color = torch.tensor([100, 100, 100], dtype=torch.uint8)
    output[screen_height - margin_bottom, :, :] = axis_color

    # Add stats text area (simple indicator bars instead of text)
    if config.get("show_stats", True) and values.numel() > 0:
        mean_val = values.mean().item()
        count = values.numel()

        # Draw mean indicator as a vertical line
        if range_max > range_min:
            mean_pos = int((mean_val - range_min) / (range_max - range_min) * screen_width)
            mean_pos = max(0, min(mean_pos, screen_width - 1))
            mean_color = torch.tensor([255, 255, 0], dtype=torch.uint8)
            output[:screen_height - margin_bottom, mean_pos, :] = mean_color

        # Draw count indicator (bar at bottom showing relative fullness)
        max_count = data.numel()
        fill_ratio = count / max_count
        fill_width = int(fill_ratio * screen_width)
        count_color = torch.tensor([100, 255, 100], dtype=torch.uint8)
        output[screen_height - 10:screen_height - 5, :fill_width, :] = count_color

    return output.to(torch.uint8)


def _parse_key(key: str):
    """Parse 'entity:feature' string to tuple or return as-is if not a string."""
    if isinstance(key, str) and ':' in key:
        parts = key.split(':')
        if len(parts) == 2:
            return (parts[0].strip(), parts[1].strip())
    return key


def genetic_layered_renderer(source, config: DictConfig) -> torch.Tensor:
    """
    Render multiple layers with genetic slot coloring for animals.

    Shows a base layer (e.g., plants) with animals colored by genetic slot on top.

    Config:
        base_layers: List of base layers (rendered first, like plants)
            - key: tensor key for presence/energy
            - threshold: minimum value to show
            - color_min/color_max: color range based on value
            - input_range: [min, max] for normalization
        genetic_layers: List of genetic slot layers (rendered on top)
            - slot_id_key: Key for slot_id tensor
            - slot_colors_key: Key for slot_colors tensor
            - biomass_key: Key for biomass tensor
            - biomass_range: [min, max] for brightness normalization
            - brightness_min/brightness_max: brightness multiplier range
    """
    result = None

    # Render base layers first (e.g., plants)
    for layer in config.get("base_layers", []):
        data = source.get(_parse_key(layer.key)).float()
        if result is None:
            result = torch.zeros(data.shape[0], data.shape[1], 3, dtype=torch.float32)

        threshold = layer.get("threshold", 0)
        mask = (data > threshold).float().unsqueeze(-1)

        input_min, input_max = layer.input_range
        normalized = torch.clamp((data - input_min) / (input_max - input_min + 1e-8), 0, 1)

        color_min = torch.tensor(layer.color_min, dtype=torch.float32)
        color_max = torch.tensor(layer.color_max, dtype=torch.float32)
        colored = color_min + normalized.unsqueeze(-1) * (color_max - color_min)

        result = result * (1 - mask) + colored * mask

    # Render genetic layers on top (e.g., herbivores, predators)
    for layer in config.get("genetic_layers", []):
        slot_ids = source.get(_parse_key(layer.slot_id_key))
        slot_colors = source.get(_parse_key(layer.slot_colors_key))
        biomass = source.get(_parse_key(layer.biomass_key))

        if result is None:
            H, W = slot_ids.shape
            result = torch.zeros(H, W, 3, dtype=torch.float32)

        H, W = slot_ids.shape

        # Normalize biomass for brightness
        biomass_min, biomass_max = layer.get("biomass_range", [0, 255])
        brightness_min = layer.get("brightness_min", 0.3)
        brightness_max = layer.get("brightness_max", 1.0)

        biomass_normalized = ((biomass.float() - biomass_min) / (biomass_max - biomass_min + 1e-8)).clamp(0, 1)
        brightness = brightness_min + biomass_normalized * (brightness_max - brightness_min)

        # Map slot_ids to colors
        flat_ids = slot_ids.flatten().long()
        flat_colors = slot_colors[flat_ids]
        colors = flat_colors.view(H, W, 3).float()

        # Apply brightness
        colors = colors * brightness.unsqueeze(-1)

        # Create mask where entities exist
        alive_mask = (biomass > 0).unsqueeze(-1).float()

        # Overlay on result
        result = result * (1 - alive_mask) + colors * alive_mask

    return result.clamp(0, 255).to(torch.uint8)


def genetic_slot_renderer(source, config: DictConfig) -> torch.Tensor:
    """
    Render entities colored by their genetic slot.

    Each slot has a distinct color, and entity brightness scales with biomass.

    Config:
        slot_id_key: Key for slot_id tensor (H, W) - which slot each cell belongs to
        slot_colors_key: Key for slot_colors tensor (num_slots, 3) - RGB per slot
        biomass_key: Key for biomass/energy tensor (H, W) - controls brightness
        biomass_range: [min, max] for biomass normalization (default: [0, 255])
        brightness_min: Minimum brightness multiplier (default: 0.3)
        brightness_max: Maximum brightness multiplier (default: 1.0)
    """
    slot_ids = source.get(_parse_key(config.slot_id_key))  # (H, W)
    slot_colors = source.get(_parse_key(config.slot_colors_key))  # (num_slots, 3)
    biomass = source.get(_parse_key(config.biomass_key))  # (H, W)

    H, W = slot_ids.shape

    # Normalize biomass for brightness
    biomass_min, biomass_max = config.get("biomass_range", [0, 255])
    brightness_min = config.get("brightness_min", 0.3)
    brightness_max = config.get("brightness_max", 1.0)

    biomass_normalized = ((biomass.float() - biomass_min) / (biomass_max - biomass_min + 1e-8)).clamp(0, 1)
    brightness = brightness_min + biomass_normalized * (brightness_max - brightness_min)

    # Map slot_ids to colors: index into slot_colors
    # slot_colors is (num_slots, 3), slot_ids is (H, W)
    # Result should be (H, W, 3)
    flat_ids = slot_ids.flatten().long()  # (H*W,)
    flat_colors = slot_colors[flat_ids]  # (H*W, 3)
    colors = flat_colors.view(H, W, 3).float()

    # Apply brightness based on biomass
    colors = colors * brightness.unsqueeze(-1)

    # Where biomass is 0, show black (no entity)
    alive_mask = (biomass > 0).unsqueeze(-1).float()
    colors = colors * alive_mask

    return colors.clamp(0, 255).to(torch.uint8)


def _get_source(source):
    if isinstance(source, WorldSnapshot):
        return source
    if isinstance(source, TensorDict):
        return source
    raise ValueError("Render source must be WorldSnapshot or TensorDict")


def dispatch_render(source, config: DictConfig):
    source = _get_source(source)
    if config.fn_name == "default":
        return default_renderer(source, config)
    if config.fn_name == "layered":
        return layered_renderer(source, config)
    if config.fn_name == "cross_section":
        return cross_section(source, config)
    if config.fn_name == "histogram":
        return histogram_renderer(source, config)
    if config.fn_name == "rgb_species":
        return rgb_species_renderer(source, config)
    if config.fn_name == "genetic_slot":
        return genetic_slot_renderer(source, config)
    if config.fn_name == "genetic_layered":
        return genetic_layered_renderer(source, config)
    else:
        raise ValueError(f"Unknown display function: {config.fn_name}")

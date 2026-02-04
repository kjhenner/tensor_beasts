import math

import torch
from omegaconf import DictConfig

from tensor_beasts.features.feature import Feature, SharedFeature
from tensor_beasts.registry import register_feature
from tensor_beasts.util import generate_diffusion_kernel, torch_correlate_2d, torch_correlate_3d


@register_feature
class Oscillator(Feature):
    """
    A sinusoidal oscillator that produces a scalar value varying over time.

    Output: offset + amplitude * sin(2 * pi * frequency * step + phase)

    The value is stored as a 0-dimensional tensor for easy broadcasting.
    """
    name = "oscillator"
    dtype = torch.float32
    default_config = DictConfig({
        "frequency": 0.01,  # Oscillations per step
        "amplitude": 0.5,   # Peak deviation from offset
        "offset": 0.5,      # DC bias (center value)
        "phase": 0.0,       # Initial phase in radians
    })
    depends_on = {}

    def initialize_data(self):
        # Store as 0-d tensor; will broadcast when added to 2D tensors
        self.data = torch.tensor(self.config.offset, dtype=self.dtype)

    def update(self, step: int):
        value = (
            self.config.offset
            + self.config.amplitude * math.sin(
                2 * math.pi * self.config.frequency * step + self.config.phase
            )
        )
        self.data = torch.tensor(value, dtype=self.dtype)


@register_feature
class Scent(SharedFeature):
    name = "scent"
    dtype = torch.float16
    default_tags = {"observable"}

    # These config keys must be identical across all scents sharing a tensor
    # (diffusion dynamics are shared; emission is per-instance)
    shared_config_keys = ("diffusion_steps", "kernel_size", "kernel_sigma", "max_decay", "decay_half")

    default_config = DictConfig({
        # Per-instance: what emits this scent and how strongly
        "energy_key": "${key:shared_features,energy}",
        "emission_rate": 1.0,           # Scent emitted per unit energy per step

        # Shared across family: diffusion dynamics (first instance sets these)
        "diffusion_steps": 2,           # Blur iterations per step
        "kernel_size": 9,               # Gaussian kernel size
        "kernel_sigma": 1.5,            # Gaussian kernel sigma
        "max_decay": 0.15,              # Maximum decay rate (at high scent)
        "decay_half": 50.0,             # Scent level at which decay is half of max
    })

    def _get_shared_config(self, key: str):
        """Get a shared config value from the registry."""
        reg = self._registry_ref[self._effective_shared_name]
        shared_config = reg.get("shared_config", {})
        if shared_config and key in shared_config:
            return shared_config[key]
        # Fallback to local config
        return getattr(self.config, key, None)

    def _get_kernel(self):
        """Get normalized diffusion kernel (cached per shared family)."""
        reg = self._registry_ref[self._effective_shared_name]
        if "kernel_cache" not in reg or reg["kernel_cache"] is None:
            kernel = generate_diffusion_kernel(
                size=self._get_shared_config("kernel_size"),
                sigma=self._get_shared_config("kernel_sigma")
            )
            # Normalize to sum to 1.0 for mass conservation
            reg["kernel_cache"] = kernel / kernel.sum()
        return reg["kernel_cache"]

    def emit(self, step: int):
        """
        Emit scent based on energy source (per-entity operation).

        Each entity calls this to add scent to its slice based on its energy.
        Diffusion is handled separately by diffuse() on the parent.
        """
        if self._is_parent:
            return

        # Handle both tuple keys and OmegaConf ListConfig keys
        energy_key = self.config.energy_key
        if not isinstance(energy_key, tuple):
            energy_key = tuple(energy_key)
        energy = self.td.get(energy_key).float()

        # Emission: add scent proportional to energy
        scent = self.data.float() + energy * self.config.emission_rate
        self.data = scent.to(torch.float16)

    def diffuse(self, step: int):
        """
        Apply decay and diffusion to entire shared tensor (parent-only operation).

        This processes all scent slices in one batched operation.
        Must be called once per step by World, not by individual entities.
        """
        if not self._is_parent:
            return

        kernel = self._get_kernel()
        scent = self.data.float()  # Full 3D tensor (H, W, C)

        # Get shared diffusion params from registry
        max_decay = self._get_shared_config("max_decay")
        decay_half = self._get_shared_config("decay_half")
        diffusion_steps = self._get_shared_config("diffusion_steps")

        # Value-dependent decay (Michaelis-Menten style):
        # - High scent: decay approaches max_decay
        # - Low scent: decay approaches 0
        # This prevents saturation at top and preserves gradients at bottom
        decay_rate = max_decay * scent / (scent + decay_half)
        scent = scent * (1.0 - decay_rate)

        # Diffuse (mass-conserving Gaussian blur) - batched across all slices
        for _ in range(diffusion_steps):
            scent = torch_correlate_3d(scent, kernel)

        # Store as float16
        self.data = scent.to(torch.float16)

    def update(self, step: int):
        """
        Legacy update method - prefer emit() + diffuse() for new code.

        For non-parent instances, this just calls emit().
        For parent instances, this calls diffuse().
        """
        if self._is_parent:
            self.diffuse(step)
        else:
            self.emit(step)


@register_feature
class Carrion(Feature):
    """
    Carrion (dead animal biomass) that persists and decays over time.

    When animals die, their biomass is transferred here instead of disappearing.
    Predators and scavengers can eat carrion.

    Note: This is a regular Feature (not SharedFeature) because it's a single
    terrain layer, not something shared across multiple entities.
    """
    name = "carrion"
    dtype = torch.uint8
    default_tags = {"observable"}
    default_config = DictConfig({
        "decay_rate": 0.02,  # Fraction lost per step
    })

    def initialize_data(self):
        # Initialize carrion to zero - will be populated by animal deaths
        self.data = torch.zeros(self.shape, dtype=self.dtype)

    def update(self, step: int):
        # Decay: carrion slowly disappears over time
        if self.config.decay_rate > 0:
            decay = (self.data.float() * self.config.decay_rate).to(torch.uint8)
            decay = torch.clamp(decay, min=1)  # At least 1 if any carrion present
            # Only decay where there is carrion
            has_carrion = self.data > 0
            self.data = torch.where(
                has_carrion,
                torch.clamp(self.data.int() - decay.int(), min=0).to(torch.uint8),
                self.data
            )


@register_feature
class CarrionScent(Scent):
    """
    Scent emitted by carrion (dead animal biomass).

    Has its own feature name (carrion_scent) for entity-specific keys,
    but shares the same backing tensor as other Scent features via shared_name.

    Diffusion parameters are inherited from the Scent family (first instance sets them).
    Only emission parameters (energy_key, emission_rate) are per-instance.
    """
    name = "carrion_scent"
    shared_name = "scent"  # Share tensor with Scent family
    default_config = DictConfig({
        # Per-instance only - diffusion params inherited from Scent family
        "energy_key": "${key:simpleterrain,carrion}",
        "emission_rate": 3.0,  # Strong scent from rotting meat
    })


@register_feature
class Energy(SharedFeature):
    name = "energy"
    dtype = torch.uint8
    default_tags = {"observable"}

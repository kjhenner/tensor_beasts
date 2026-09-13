from typing import Any, Annotated, Dict, List, Optional, Tuple, Type, Union, Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, confloat, conint, conlist, create_model

# Energy, biomass and everything measured against them live on a 0..255 scale
# (float32 tensors; the scale is inherited from when they were uint8). Any
# config value written into or compared against those tensors is validated to
# that range so a threshold can never sit above what an animal can hold.
Scale255 = confloat(ge=0, le=255)


def _parse_td_key(value: Union[str, Tuple[str, str], None]) -> Optional[Tuple[str, str]]:
    """Parse key string to tuple, or pass through if already tuple/None.

    Supports two formats:
    - New format: "entity:feature"
    - Old OmegaConf format: "${key:entity,feature}"
    """
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return tuple(value)
    if not isinstance(value, str):
        raise ValueError(f"Expected string or tuple, got {type(value)}")

    # Handle old OmegaConf format: ${key:entity,feature}
    if value.startswith("${key:") and value.endswith("}"):
        inner = value[6:-1]  # Strip ${key: and }
        parts = inner.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid key format: {value}. Expected '${{key:entity,feature}}'.")
        return (parts[0].strip(), parts[1].strip())

    # Handle new format: entity:feature
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid key format: {value}. Expected 'entity:feature'.")
    return (parts[0].strip(), parts[1].strip())


def _parse_td_key_required(value: Union[str, Tuple[str, str]]) -> Tuple[str, str]:
    """Parse 'entity:feature' string to tuple (required, non-None)."""
    result = _parse_td_key(value)
    if result is None:
        raise ValueError("Key is required")
    return result


def _parse_navigation_weights(value: Dict[str, float]) -> Dict[Tuple[str, str], float]:
    """Parse navigation weights dict, converting string keys to tuple keys."""
    if value is None:
        return {}
    return {_parse_td_key_required(k): v for k, v in value.items()}


# TensorDict key type: "entity:feature" string -> (entity, feature) tuple
TensorDictKey = Annotated[Optional[Tuple[str, str]], BeforeValidator(_parse_td_key)]
TensorDictKeyRequired = Annotated[Tuple[str, str], BeforeValidator(_parse_td_key_required)]
NavigationWeights = Annotated[Dict[Tuple[str, str], float], BeforeValidator(_parse_navigation_weights)]


class ScentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    energy_key: Optional[str] = None
    emission_rate: float = 1.0       # Scent emitted per unit energy per step
    diffusion_steps: conint(ge=0) = 2
    kernel_size: conint(gt=0) = 9
    kernel_sigma: float = 1.5
    max_decay: float = 0.15          # Maximum decay rate (at high scent)
    decay_half: float = 50.0         # Scent level at which decay is half of max


class EnergyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SeedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    energy_key: Optional[str] = None  # Required: reference to entity's energy
    crowding_key: Optional[str] = None  # Required: reference to entity's crowding
    random_key: Optional[str] = None
    seed_prob: float = 0.01
    germination_prob: float = 0.01
    scale: float = 0.9


class CrowdingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    energy_key: Optional[str] = None  # Required: reference to entity's energy
    scale: float = 0.9


class ElevationPerlinConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scale: Tuple[conint(gt=0), conint(gt=0)]


class ElevationPyramidConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inverted: Optional[bool] = None


class ElevationRampConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ElevationRangeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ElevationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    perlin: Optional[ElevationPerlinConfig] = None
    pyramid: Optional[ElevationPyramidConfig] = None
    ramp: Optional[ElevationRampConfig] = None
    range: Optional[ElevationRangeConfig] = None


class AquiferElevationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    elevation_key: Optional[str] = None
    scale: float = 0.9


class SoilVolumeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    elevation_key: Optional[str] = None
    elevation_scale: float = 100
    surface_outflow_key: Optional[str] = None
    erosion_rate: float = 1e-3
    init_scale: float = 8.0
    epsilon: float = 1e-8


class SoilWaterVolumeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    soil_volume_key: Optional[str] = None
    elevation_key: Optional[str] = None
    elevation_scale: float = 100
    soil_porosity: float = 0.4
    surface_water_volume_key: Optional[str] = None
    soil_water_saturation_key: Optional[str] = None
    infiltration_rate: float = 0.001
    flow_rate: float = 0.1
    saturation_gradient_coeff: float = 0.2
    evaporation_rate: float = 1e-5
    field_capacity: float = 0.5


class SurfaceWaterVolumeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    elevation_key: Optional[str] = None
    outflow_key: Optional[str] = None
    elevation_scale: float = 100
    rainfall_rate: float = 5.7e-08
    flow_rate: float = 0.1
    relaxation_factor: float = 0.5
    steps: int = 1


class FluidDensityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    flow_rate: float = 0.05
    elevation_key: Optional[str] = None
    elevation_scale: float = 100
    rainfall_rate: float = 0.1


class HydrodynamicPlantEntityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    initial_energy: Scale255 = 100
    init_prob: float = 0.001
    # These are entity-level keys read directly via td.get(), so they must be
    # parsed into (entity, feature) tuples here - unlike nested feature configs,
    # entity-level config bypasses OmegaConf interpolation.
    energy_key: TensorDictKey = None
    soil_water_volume_key: TensorDictKey = None
    soil_volume_key: TensorDictKey = None
    soil_sat_coeff_key: TensorDictKey = None
    nutrients_key: TensorDictKey = None
    nutrient_consumption_rate: float = 0.5
    soil_porosity: float = 0.4
    ideal_growth_rate: float = 4e-4
    ideal_soil_saturation: float = 0.65
    soil_saturation_tolerance: float = 0.2
    energy: Optional[EnergyConfig] = None
    scent: Optional[ScentConfig] = None
    seed: Optional[SeedConfig] = None
    crowding: Optional[CrowdingConfig] = None


class OscillatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frequency: float = 0.01
    amplitude: float = 0.5
    offset: float = 0.5
    phase: float = 0.0


class SimpleWaterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    perlin_scale: Tuple[conint(gt=0), conint(gt=0)] = (4, 4)
    perlin_octaves: int = 4
    perlin_persistence: float = 0.5
    oscillator_key: Optional[str] = None
    oscillator_amplitude: float = 0.1


class SimplePlantEntityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    initial_energy: Scale255 = 100
    init_prob: float = 0.001
    toy_init: bool = False  # If True, spawn single plant in center
    energy_key: TensorDictKey = None
    water_key: TensorDictKey = None
    growth_coeff_key: TensorDictKey = None
    nutrients_key: TensorDictKey = None
    nutrient_consumption_rate: float = 0.5
    ideal_growth_rate: float = 4e-4
    ideal_water_level: float = 0.5
    water_tolerance: float = 0.3
    energy: Optional[EnergyConfig] = None
    scent: Optional[ScentConfig] = None
    seed: Optional[SeedConfig] = None
    crowding: Optional[CrowdingConfig] = None


class CarrionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decay_rate: float = 0.02  # Fraction lost per step


class SimpleTerrainEntityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    oscillator: Optional[OscillatorConfig] = None
    simple_water: Optional[SimpleWaterConfig] = None
    carrion: Optional[CarrionConfig] = None
    carrion_scent: Optional[ScentConfig] = None  # Scent emitted by carrion


class NutrientsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    initial_nutrients: float = 50.0
    diffusion_rate: float = 0.01


class BiomassConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GradientEMAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PerceptionConfig(BaseModel):
    """Configuration for a single perception channel."""
    model_config = ConfigDict(extra="forbid")
    key: TensorDictKeyRequired
    kernel_size: int = 1


class GeneticsConfig(BaseModel):
    """Configuration for genetic algorithm system."""
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False                  # Whether genetic system is active
    num_slots: conint(ge=1, le=255) = 8    # Number of genetic slots
    mutation_probability: float = 0.1      # Chance of slot change on reproduction
    mutation_rate: float = 0.1             # Probability of mutating each parameter
    mutation_scale: float = 0.1            # Magnitude of parameter mutations
    log_interval: int = 0                  # Log genetic status every N steps (0 = disabled)


class AnimalEntityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Initialization
    initial_energy: Scale255 = 50
    initial_biomass: Scale255 = 150
    init_prob: float = 0.001
    toy_init: bool = False

    # Eating: cap on what one animal takes per step, on the food's 0..255 scale
    # food_keys: list of food sources tried in order until satiated
    eat_max: Scale255 = 10
    food_keys: List[TensorDictKeyRequired] = Field(default_factory=list)

    # Perception: list of features to perceive with kernel sizes
    perception: List[PerceptionConfig] = Field(default_factory=lambda: [
        PerceptionConfig(key="hydrodynamicplant:scent", kernel_size=1),
        PerceptionConfig(key="herbivore:scent", kernel_size=1),
    ])

    # Navigation weights: signed weights (positive=attract, negative=repel)
    navigation_weights: NavigationWeights = Field(default_factory=lambda: {
        "hydrodynamicplant:scent": 1.0,
        "herbivore:scent": -0.1,
    })

    # Perception processing
    log_scale: float = 10.0  # Scale factor before log compression

    # Metabolism - gradient-based, in biomass units per step
    basal_rate: Scale255 = 1
    metabolic_sensitivity: float = 1.0  # biomass burn increase per unit gradient_ema
    max_metabolic_rate: Scale255 = 5
    gradient_ema_alpha: float = 0.1

    # Metabolic efficiency curve - diminishing returns at higher metabolic rates
    # At basal_rate: max_efficiency (resting, efficient)
    # At max_metabolic_rate: min_efficiency (sprinting, costly)
    # Total energy always increases with rate, but each extra biomass gives less
    min_efficiency: float = 2.5  # energy per biomass at max exertion
    max_efficiency: float = 3.5  # energy per biomass at rest

    # Dissipation
    dissipation_rate: float = 0.05
    dissipation_floor: Scale255 = 1

    # Movement cost, in energy units per move
    base_movement_cost: Scale255 = 1

    # Death & Reproduction, compared against biomass
    survival_threshold: Scale255 = 5
    carrion_key: TensorDictKey = None  # where dead biomass goes (carrion layer)
    reproduction_threshold: Scale255 = 200
    # Note: On reproduction, both parent and offspring receive 50% of biomass/energy

    # Features
    energy: Optional[EnergyConfig] = None
    biomass: Optional[BiomassConfig] = None
    gradient_ema: Optional[GradientEMAConfig] = None
    scent: Optional[ScentConfig] = None
    slot_id: Optional[Dict[str, Any]] = None  # SlotId feature config
    memory: Optional[Dict[str, Any]] = None  # Memory feature config: {size: K}

    # Genetic algorithm
    genetics: Optional[GeneticsConfig] = None

    # Debugging
    # Reinforcement learning support. Records per-individual transitions on the
    # entity each step so trajectories can be stitched together. Off by default
    # because it costs a few tensor ops per step and nothing else reads it.
    track_transitions: bool = False

    verbose: bool = False


class TerrainEntityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    elevation: Optional[ElevationConfig] = None
    aquifer_elevation: Optional[AquiferElevationConfig] = None
    soil_volume: Optional[SoilVolumeConfig] = None
    soil_water_volume: Optional[SoilWaterVolumeConfig] = None
    surface_water_volume: Optional[SurfaceWaterVolumeConfig] = None
    nutrients: Optional[NutrientsConfig] = None


class DiffusionToyEntityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    elevation: Optional[ElevationConfig] = None
    fluid_density: Optional[FluidDensityConfig] = None


class DefaultDisplay(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    fn_name: Literal["default"]
    key: str
    input_range: Optional[conlist(float, min_length=2, max_length=2)] = None


class LayerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str
    threshold: float
    color_min: conlist(conint(ge=0, le=255), min_length=3, max_length=3)
    color_max: conlist(conint(ge=0, le=255), min_length=3, max_length=3)
    input_range: conlist(float, min_length=2, max_length=2)
    log_scale: bool = False


class LayeredDisplay(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    fn_name: Literal["layered"]
    layers: List[LayerConfig]


class CrossSectionLevel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str
    color: conlist(conint(ge=0, le=255), min_length=3, max_length=3)
    scale: float
    color_mod_key: Optional[str] = None
    color_mod: Optional[conlist(conint(ge=0, le=255), min_length=3, max_length=3)] = None


class CrossSectionDisplay(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    fn_name: Literal["cross_section"]
    background_color: conlist(conint(ge=0, le=255), min_length=3, max_length=3)
    screen_height: conint(gt=0)
    section_idx: conint(ge=0)
    section_dim: Literal["x", "y"]
    levels: List[CrossSectionLevel]


class HistogramDisplay(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    fn_name: Literal["histogram"]
    key: str
    bins: conint(gt=0) = 32
    range: Optional[conlist(float, min_length=2, max_length=2)] = None
    log_scale: bool = False
    color: conlist(conint(ge=0, le=255), min_length=3, max_length=3) = [0, 200, 255]
    background: conlist(conint(ge=0, le=255), min_length=3, max_length=3) = [20, 20, 20]
    show_stats: bool = True
    filter_zeros: bool = True
    screen_height: conint(gt=0) = 256
    screen_width: conint(gt=0) = 512


class RGBChannelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scent_key: str
    biomass_key: str
    biomass_range: conlist(float, min_length=2, max_length=2) = [0, 255]


class RGBSpeciesDisplay(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    fn_name: Literal["rgb_species"]
    red: Optional[RGBChannelConfig] = None
    green: Optional[RGBChannelConfig] = None
    blue: Optional[RGBChannelConfig] = None
    scent_max: float = 255.0
    scent_log_scale: bool = False
    biomass_floor: conint(ge=0, le=255) = 128


class GeneticSlotDisplay(BaseModel):
    """Display entities colored by their genetic slot."""
    model_config = ConfigDict(extra="forbid")
    title: str
    fn_name: Literal["genetic_slot"]
    slot_id_key: str          # Key for slot_id tensor (H, W)
    slot_colors_key: str      # Key for slot_colors tensor (num_slots, 3)
    biomass_key: str          # Key for biomass/energy tensor (H, W)
    biomass_range: conlist(float, min_length=2, max_length=2) = [0, 255]
    brightness_min: float = 0.3   # Minimum brightness multiplier
    brightness_max: float = 1.0   # Maximum brightness multiplier


class GeneticLayerConfig(BaseModel):
    """Config for a genetic slot layer in layered display."""
    model_config = ConfigDict(extra="forbid")
    slot_id_key: str
    slot_colors_key: str
    biomass_key: str
    biomass_range: conlist(float, min_length=2, max_length=2) = [0, 255]
    brightness_min: float = 0.4
    brightness_max: float = 1.0


class GeneticLayeredDisplay(BaseModel):
    """Display with base layers and genetic slot-colored animals on top."""
    model_config = ConfigDict(extra="forbid")
    title: str
    fn_name: Literal["genetic_layered"]
    base_layers: List[LayerConfig] = Field(default_factory=list)
    genetic_layers: List[GeneticLayerConfig] = Field(default_factory=list)


class DisplayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    color_displays: List[Union[DefaultDisplay, LayeredDisplay, CrossSectionDisplay, HistogramDisplay, RGBSpeciesDisplay, GeneticSlotDisplay, GeneticLayeredDisplay]] = Field(
        default_factory=list
    )
    text_displays: List[DefaultDisplay] = Field(default_factory=list)


class SharedScentConfig(BaseModel):
    """Global config for shared scent diffusion parameters."""
    model_config = ConfigDict(extra="forbid")
    diffusion_steps: conint(ge=0) = 2
    kernel_size: conint(gt=0) = 9
    kernel_sigma: float = 1.5
    max_decay: float = 0.15
    decay_half: float = 50.0


class SharedFeaturesConfig(BaseModel):
    """Global config for shared feature parameters."""
    model_config = ConfigDict(extra="forbid")
    scent: Optional[SharedScentConfig] = None


class WorldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    size: Tuple[conint(gt=0), conint(gt=0)]
    device: Optional[str] = "auto"
    running: bool = True
    max_ups: Optional[float] = 0
    shared_features: Optional[SharedFeaturesConfig] = None
    entities: Dict[str, Dict[str, Any]]


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    world: WorldConfig
    display: Optional[DisplayConfig] = None


# =============================================================================
# Auto-generation of Pydantic models from Feature/Entity default_config
# =============================================================================

# Cache for generated models to avoid recreating them
_generated_feature_models: Dict[str, Type[BaseModel]] = {}
_generated_entity_models: Dict[str, Type[BaseModel]] = {}


def _infer_type_from_value(value: Any) -> Type:
    """
    Infer a Pydantic-compatible type from a config value.

    Args:
        value: The default value from config

    Returns:
        The inferred type for the Pydantic field
    """
    if value is None:
        return Optional[Any]
    elif isinstance(value, bool):
        return bool
    elif isinstance(value, int):
        return int
    elif isinstance(value, float):
        return float
    elif isinstance(value, str):
        return str
    elif isinstance(value, list):
        if len(value) == 0:
            return List[Any]
        # Infer from first element
        elem_type = _infer_type_from_value(value[0])
        return List[elem_type]
    elif isinstance(value, dict):
        return Dict[str, Any]
    else:
        return Any


def generate_feature_config_model(
    feature_class: Type,
    model_name: Optional[str] = None
) -> Type[BaseModel]:
    """
    Generate a Pydantic config model from a Feature class's default_config.

    Args:
        feature_class: The Feature class to generate a model for
        model_name: Optional custom name for the model

    Returns:
        A Pydantic BaseModel class for validating the feature's config
    """
    feature_name = getattr(feature_class, 'name', feature_class.__name__)

    # Check cache
    if feature_name in _generated_feature_models:
        return _generated_feature_models[feature_name]

    # Get default config
    default_config = getattr(feature_class, 'default_config', None)
    if default_config is None:
        # No config, return empty model
        model = create_model(
            model_name or f"{feature_class.__name__}Config",
            __config__=ConfigDict(extra="forbid"),
        )
        _generated_feature_models[feature_name] = model
        return model

    # Convert to dict if it's a DictConfig
    if isinstance(default_config, DictConfig):
        config_dict = OmegaConf.to_container(default_config, resolve=False)
    else:
        config_dict = dict(default_config)

    # Build field definitions
    fields = {}
    for key, value in config_dict.items():
        inferred_type = _infer_type_from_value(value)
        # Make all fields optional with their default value
        fields[key] = (Optional[inferred_type], value)

    # Create the model
    model = create_model(
        model_name or f"{feature_class.__name__}Config",
        __config__=ConfigDict(extra="forbid"),
        **fields
    )

    _generated_feature_models[feature_name] = model
    return model


def generate_entity_config_model(
    entity_class: Type,
    feature_models: Optional[Dict[str, Type[BaseModel]]] = None,
    model_name: Optional[str] = None
) -> Type[BaseModel]:
    """
    Generate a Pydantic config model for an Entity from its features.

    Args:
        entity_class: The Entity class to generate a model for
        feature_models: Optional dict of feature_name -> Pydantic model to use
                       (falls back to auto-generated models)
        model_name: Optional custom name for the model

    Returns:
        A Pydantic BaseModel class for validating the entity's config
    """
    entity_name = entity_class.__name__

    # Check cache
    if entity_name in _generated_entity_models:
        return _generated_entity_models[entity_name]

    feature_models = feature_models or {}

    # Get features from the entity
    features = getattr(entity_class, '__features__', {})

    # Get entity-level default config
    entity_default_config = getattr(entity_class, 'default_config', None)
    entity_fields = {}

    if entity_default_config:
        if isinstance(entity_default_config, DictConfig):
            config_dict = OmegaConf.to_container(entity_default_config, resolve=False)
        else:
            config_dict = dict(entity_default_config)

        for key, value in config_dict.items():
            # Skip feature configs (handled separately)
            if key not in features:
                inferred_type = _infer_type_from_value(value)
                entity_fields[key] = (Optional[inferred_type], value)

    # Add feature config fields
    for feature_name, feature_class in features.items():
        if feature_name in feature_models:
            feature_model = feature_models[feature_name]
        else:
            feature_model = generate_feature_config_model(feature_class)

        entity_fields[feature_name] = (Optional[feature_model], None)

    # Create the model
    model = create_model(
        model_name or f"{entity_name}EntityConfig",
        __config__=ConfigDict(extra="forbid"),
        **entity_fields
    )

    _generated_entity_models[entity_name] = model
    return model


def get_or_generate_entity_config_model(
    entity_name: str,
    manual_models: Optional[Dict[str, Type[BaseModel]]] = None
) -> Type[BaseModel]:
    """
    Get a config model for an entity, using manual model if available,
    otherwise auto-generating one.

    Args:
        entity_name: The entity class name
        manual_models: Dict of entity_name -> manually defined Pydantic model

    Returns:
        A Pydantic BaseModel class for validating the entity's config
    """
    manual_models = manual_models or {}

    if entity_name in manual_models:
        return manual_models[entity_name]

    # Auto-generate
    from tensor_beasts.registry import entity_registry

    if not entity_registry.exists(entity_name):
        raise ValueError(f"Unknown entity: {entity_name}")

    entity_info = entity_registry.get(entity_name)
    return generate_entity_config_model(entity_info.cls)

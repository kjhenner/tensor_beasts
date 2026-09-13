import os
import re
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Set

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ValidationError

from tensor_beasts.config.models import (
    AnimalEntityConfig,
    DiffusionToyEntityConfig,
    DisplayConfig,
    HydrodynamicPlantEntityConfig,
    SimplePlantEntityConfig,
    SimpleTerrainEntityConfig,
    SimulationConfig,
    TerrainEntityConfig,
    get_or_generate_entity_config_model,
)
from tensor_beasts.registry import (
    get_entity_class,
    entity_registry,
    entity_name_to_registry_name,
)


class ConfigValidationError(Exception):
    """Raised when config validation fails with detailed error information."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        message = "Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(message)


# Regex to match ${key:entity,feature} patterns
KEY_REF_PATTERN = re.compile(r'\$\{key:([^,}]+),([^}]+)\}')


def register_resolvers() -> None:
    if not OmegaConf.has_resolver("key"):
        OmegaConf.register_new_resolver("key", lambda *args: tuple(args))


def _find_key_references(
    cfg: DictConfig,
    path: str = ""
) -> List[Tuple[str, str, str]]:
    """
    Recursively find all ${key:entity,feature} references in config.

    Returns:
        List of (config_path, entity, feature) tuples
    """
    references = []
    container = OmegaConf.to_container(cfg, resolve=False)

    def _search(obj, current_path):
        if isinstance(obj, str):
            # Look for ${key:...} patterns
            for match in KEY_REF_PATTERN.finditer(obj):
                entity, feature = match.group(1).strip(), match.group(2).strip()
                references.append((current_path, entity, feature))
        elif isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{current_path}.{key}" if current_path else key
                _search(value, new_path)
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                new_path = f"{current_path}[{i}]"
                _search(value, new_path)

    _search(container, path)
    return references


def _suggest_similar(name: str, candidates: List[str], n: int = 3) -> List[str]:
    """Find similar names for helpful error messages."""
    return get_close_matches(name, candidates, n=n, cutoff=0.4)


def _validate_key_references(cfg: DictConfig) -> None:
    """
    Validate that all ${key:entity,feature} references point to valid targets.

    Raises:
        ConfigValidationError: If any references are invalid
    """
    # Ensure entities are imported so registry is populated
    get_entity_class("Terrain")  # This triggers _ensure_entities_imported

    references = _find_key_references(cfg)
    errors = []

    # Get all registered entity names (lowercase for comparison)
    registered_entities = {name.lower(): name for name in entity_registry.names()}

    # Special keys that aren't entity.feature references
    special_keys = {"random", "shared_features"}

    for config_path, entity, feature in references:
        # Handle special cases
        if entity in special_keys:
            continue

        # Normalize entity name (lowercase in refs -> PascalCase in registry)
        entity_lower = entity.lower()
        registry_entity = registered_entities.get(entity_lower)

        if registry_entity is None:
            # Entity not found
            similar = _suggest_similar(entity_lower, list(registered_entities.keys()))
            suggestion = ""
            if similar:
                suggestion = f" Did you mean: {', '.join(similar)}?"
            errors.append(
                f"Unknown entity '{entity}' in reference at '{config_path}'.{suggestion}"
            )
            continue

        # Check feature exists in entity
        entity_features = entity_registry.get_features(registry_entity)
        feature_names = list(entity_features.keys())

        # Also check for computed/derived features that might not be in __features__
        # These are set dynamically (e.g., soil_water_saturation, surface_outflow)
        # For now, we'll allow any feature name for flexibility
        # TODO: Add a way to declare computed features

        if feature not in feature_names:
            # Feature might be a computed feature - allow but warn
            # Common computed features in the codebase
            computed_features = {
                "soil_water_saturation", "surface_outflow", "soil_sat_coeff",
                "growth_coeff"
            }
            if feature not in computed_features:
                similar = _suggest_similar(feature, feature_names)
                suggestion = ""
                if similar:
                    suggestion = f" Did you mean: {', '.join(similar)}?"
                # Make this a warning for now, not an error
                # (computed features are valid but not in registry)
                import warnings
                warnings.warn(
                    f"Feature '{feature}' not found in entity '{registry_entity}' "
                    f"at '{config_path}'.{suggestion} "
                    f"(This may be a computed feature, which is OK.)",
                    UserWarning
                )

    if errors:
        raise ConfigValidationError(errors)


def _ensure_entities_mapping(cfg: DictConfig) -> None:
    if "world" not in cfg or "entities" not in cfg.world:
        raise ValueError("Config must contain world.entities")
    if not cfg.world.entities:
        raise ValueError("world.entities must define at least one entity")


def _validate_cross_entity_dependencies(cfg: DictConfig) -> None:
    """
    Validate that cross-entity dependencies are satisfiable.

    Checks that if entity A's features depend on entity B, then entity B
    is also defined in the config.
    """
    # Ensure entities are imported so registry is populated
    get_entity_class("Terrain")

    config_entities = set(cfg.world.entities.keys())
    errors = []

    for entity_name in config_entities:
        entity_info = entity_registry.get(entity_name)

        for feature_name, feature_info in entity_info.features.items():
            # Check depends_on for cross-entity dependencies
            depends_on = feature_info.dependencies
            for dep_name, dep_target in depends_on.items():
                if isinstance(dep_target, tuple) and len(dep_target) == 2:
                    other_entity_lower, other_feature = dep_target
                    other_entity = entity_name_to_registry_name(other_entity_lower)

                    if other_entity not in config_entities:
                        errors.append(
                            f"Entity '{entity_name}' feature '{feature_name}' depends on "
                            f"'{other_entity}.{other_feature}', but '{other_entity}' is not "
                            f"defined in the config."
                        )

            # Also check ${key:...} references in default_config
            if feature_info.default_config:
                config_dict = OmegaConf.to_container(feature_info.default_config, resolve=False)
                for key, value in config_dict.items():
                    if isinstance(value, str):
                        for match in KEY_REF_PATTERN.finditer(value):
                            ref_entity_lower = match.group(1).strip()
                            ref_feature = match.group(2).strip()

                            # Skip special keys
                            if ref_entity_lower in {"random", "shared_features"}:
                                continue

                            ref_entity = entity_name_to_registry_name(ref_entity_lower)
                            if ref_entity not in config_entities and ref_entity != entity_name:
                                errors.append(
                                    f"Entity '{entity_name}' feature '{feature_name}' references "
                                    f"'{ref_entity}.{ref_feature}' via config key '{key}', but "
                                    f"'{ref_entity}' is not defined in the config."
                                )

    if errors:
        raise ConfigValidationError(errors)


def _validate_display(cfg: DictConfig) -> None:
    display = cfg.get("display", None)
    if not display:
        return
    data = OmegaConf.to_container(display, resolve=False)
    DisplayConfig.model_validate(data)


# Manually defined models for strict validation (preferred over auto-generated)
MANUAL_ENTITY_MODELS: Dict[str, Type[BaseModel]] = {
    "Terrain": TerrainEntityConfig,
    "SimpleTerrain": SimpleTerrainEntityConfig,
    "HydrodynamicPlant": HydrodynamicPlantEntityConfig,
    "SimplePlant": SimplePlantEntityConfig,
    "Herbivore": AnimalEntityConfig,
    "Predator": AnimalEntityConfig,
    "DiffusionToy": DiffusionToyEntityConfig,
    "Animal": AnimalEntityConfig,
}


def _get_entity_config_model(entity_name: str) -> Type[BaseModel]:
    """
    Get the Pydantic config model for an entity.

    Returns the manually defined model if available, otherwise auto-generates
    one from the entity's features and default_config.
    """
    return get_or_generate_entity_config_model(entity_name, MANUAL_ENTITY_MODELS)


def _validate_entities(cfg: DictConfig) -> None:
    for entity_name, entity_cfg in cfg.world.entities.items():
        # Check entity exists in registry (this also ensures imports have run)
        get_entity_class(entity_name)

        # Get the config model for this entity (manual or auto-generated)
        config_model = _get_entity_config_model(entity_name)

        data = OmegaConf.to_container(entity_cfg, resolve=False)
        config_model.model_validate(data)


def validate_config(cfg: DictConfig) -> None:
    _ensure_entities_mapping(cfg)
    data = OmegaConf.to_container(cfg, resolve=False)
    SimulationConfig.model_validate(data)
    _validate_entities(cfg)
    _validate_display(cfg)
    _validate_key_references(cfg)
    _validate_cross_entity_dependencies(cfg)


def _resolve_config_path(base_path: str, relative_path: str) -> str:
    """
    Resolve a config path relative to the base config file.

    If relative_path is absolute, returns it as-is.
    If relative_path starts with 'conf/', treats it as relative to project root.
    Otherwise, resolves relative to the directory containing base_path.
    """
    if os.path.isabs(relative_path):
        return relative_path

    # Check if it's a project-relative path (starts with conf/)
    if relative_path.startswith("conf/"):
        # Find project root by looking for conf/ directory
        base_dir = Path(base_path).resolve().parent
        while base_dir != base_dir.parent:
            if (base_dir / "conf").is_dir():
                # Found project root - use it directly
                return str(base_dir / relative_path)
            base_dir = base_dir.parent
        # Fallback: try from cwd (return absolute path to avoid double resolution)
        if Path("conf").is_dir():
            return str(Path(relative_path).resolve())

    # Default: resolve relative to the base config's directory
    base_dir = Path(base_path).resolve().parent
    return str(base_dir / relative_path)


def _load_config_with_inheritance(
    path: str,
    seen_paths: Optional[Set[str]] = None
) -> DictConfig:
    """
    Load a config file with inheritance support.

    If the config contains a `_base` key, the referenced config is loaded first
    and the current config is merged on top of it. Supports multiple levels
    of inheritance.

    Args:
        path: Path to the config file
        seen_paths: Set of already-loaded paths (for cycle detection)

    Returns:
        Merged DictConfig with inheritance resolved

    Raises:
        ValueError: If circular inheritance is detected
    """
    # Normalize path for cycle detection
    abs_path = os.path.abspath(path)

    if seen_paths is None:
        seen_paths = set()

    if abs_path in seen_paths:
        raise ValueError(f"Circular config inheritance detected: {abs_path}")

    seen_paths.add(abs_path)

    # Load the current config
    cfg = OmegaConf.load(path)

    # Check for inheritance
    if "_base" in cfg:
        base_ref = cfg._base

        # Handle single base or list of bases
        if isinstance(base_ref, str):
            base_refs = [base_ref]
        else:
            base_refs = list(base_ref)

        # Start with empty config, merge all bases in order
        merged = OmegaConf.create({})
        for ref in base_refs:
            base_path = _resolve_config_path(path, ref)
            base_cfg = _load_config_with_inheritance(base_path, seen_paths.copy())
            merged = OmegaConf.merge(merged, base_cfg)

        # Remove _base key before merging child on top
        del cfg._base

        # Merge child config on top of base(s)
        cfg = OmegaConf.merge(merged, cfg)

    return cfg


def load_config(path: str) -> DictConfig:
    """
    Load and validate a config file with inheritance support.

    Config files can specify a `_base` key to inherit from a parent config:

        _base: "base_config.yaml"  # Single parent
        # or
        _base:                      # Multiple parents (merged in order)
          - "base1.yaml"
          - "base2.yaml"

        # Override specific values
        world:
          size: [256, 256]

    Paths can be:
        - Relative to the current config file: "../base.yaml"
        - Project-relative (starting with conf/): "conf/base_config.yaml"
        - Absolute: "/path/to/config.yaml"

    Args:
        path: Path to the config file

    Returns:
        Validated DictConfig with inheritance resolved
    """
    register_resolvers()
    cfg = _load_config_with_inheritance(path)

    try:
        validate_config(cfg)
    except ValidationError as exc:
        raise ValueError(f"Config validation error: {exc}") from exc

    return cfg

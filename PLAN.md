# Tensor Beasts stabilization plan (no RL for now)

## Implementation Status

> **Last Updated**: January 2026

| Phase | Description | Status |
|-------|-------------|--------|
| Phase A | Decorator-based registration | **COMPLETE** |
| Phase B | Config validation of `${key:...}` refs | **COMPLETE** |
| Phase C | Automatic dependency resolution | **COMPLETE** |

### Completed Work

**Phase A (Registry Infrastructure)**:
- Created `tensor_beasts/registry.py` with `FeatureRegistry`, `EntityRegistry`, `@register_feature`, `@register_entity`
- Migrated all features and entities to use decorators
- Removed manual `ENTITY_NAMES` maintenance
- Added auto-generation of Pydantic config models from `default_config`

**Phase B (Config Validation)**:
- Added `ConfigValidationError` class with detailed error collection
- Implemented `_find_key_references()` to extract all `${key:entity,feature}` refs
- Implemented `_validate_key_references()` with typo suggestions via `difflib`
- Added `entity_name_to_registry_name()` for case normalization

**Phase C (Dependency Resolution)**:
- Added `depends_on` attribute to `Feature` base class
- Declared dependencies for all terrain features
- Implemented `_topological_sort()` using Kahn's algorithm with cycle detection
- Updated `Entity.initialize()` and `Entity.update()` to use automatic ordering
- Added `World._build_entity_dependency_graph()` for cross-entity deps
- Added `_validate_cross_entity_dependencies()` to config validation

**Tests**: 48 new tests across:
- `tests/test_registry.py` (16 tests)
- `tests/config/test_validation.py` (16 tests)
- `tests/test_dependencies.py` (14 tests)
- Plus updated fixtures in `tests/conftest.py`

---

## 1) Goals and scope
- **Primary goal**: establish a stable, extensible core architecture with clear boundaries between config, simulation, and rendering.
- **Out of scope (for now)**: RL subsystem and training scripts.
- **Success criteria**:
  - A single canonical config schema with validation and clear error messages.
  - Deterministic, headless simulation core that can run without rendering.
  - Rendering as a pure consumer of simulation state, with no side effects on sim.
  - Entity/feature definitions are consistent, discoverable, and easy to extend.

## 2) Proposed target architecture
```
+------------------+     +--------------------+     +---------------------+
| Config System    | --> | Simulation Core    | --> | Rendering / UI       |
| (schema + load)  |     | (World + Entities) |     | (OpenGL/PyGame)      |
+------------------+     +--------------------+     +---------------------+
           |                        |
           |                        v
           |                 +----------------+
           |                 | Observations   |
           |                 | (views/slices) |
           |                 +----------------+
           v
   +--------------------+
   | Diagnostics/Logging|
   +--------------------+
```

**Key principles**
- **Config is the source of truth**: strong schema, validation, explicit defaults.
- **Simulation is headless and deterministic**: no rendering dependencies.
- **Rendering reads state only**: it should not mutate simulation state.
- **Entities and features are explicit**: discoverable registry + stable naming.

## 3) Phased implementation plan

### Phase 0: Baseline hygiene (low risk, unblockers)
**Deliverables**
- Create `tensor_beasts/__main__.py` for `python -m tensor_beasts`.
- Fix `Feature.tags` bug so tags behave predictably.
- Remove or gate `print` calls in hot loops (switch to logging or a debug flag).

**Rationale**
These are quick fixes that reduce noise and failure points before deeper work.

### Phase 1: Config system stabilization
**Deliverables**
- Introduce a canonical config schema (Pydantic or OmegaConf + structured configs).
- Normalize entity naming (case strategy + validator for config keys).
- Fix config typos and missing keys (`infiltration_rate`, `field_capacity`).
- Centralize `${key:...}` resolver + remove implicit assumptions from feature code.

**Implementation outline**
- `tensor_beasts/config/schema.py`: define structured configs for world, entities, display.
- `tensor_beasts/config/load.py`: load and validate YAML, map/normalize entity names.
- Add explicit mapping for entity type names (e.g., `Plant` vs `plant`).

**Exit criteria**
- Running `main.py` with `beast_config.yaml` and `terrain_config.yaml` validates successfully.
- Schema errors point to exact keys and provide correction hints.

### Phase 2: Simulation core isolation
**Deliverables**
- Refactor `World` to be UI-agnostic and expose stable lifecycle methods:
  - `initialize()`, `reset()`, `step(action=None)`
- Implement `World.observable` or a dedicated `ObservationBuilder` in core.
- Add a `WorldSnapshot` or `StateView` to decouple TensorDict from render layer.

**Implementation outline**
- `tensor_beasts/world.py`: strict API, no display imports.
- `tensor_beasts/observations.py`: define composable observation views.
- Define a minimal, testable sim loop callable from CLI or headless entry.

**Exit criteria**
- Headless sim can run N steps without rendering.
- Rendering consumes snapshots without mutating state.

### Phase 3: Entity/feature system hardening
**Deliverables**
- Create a registry for entities and features (explicit imports and discovery).
- Add `Entity.get_feature(name)` and consistent feature access helpers.
- Define feature lifecycle ordering (e.g., update ordering, shared feature update).
- Consolidate feature defaults and enforce config-driven parameters.

**Implementation outline**
- `tensor_beasts/registry.py`: register entity/feature classes, validate config.
- `tensor_beasts/entities/entity.py`: add `get_feature`, `features()` helpers.
- `tensor_beasts/features/feature.py`: finalize base API; minimal required overrides.

**Exit criteria**
- New entity/feature can be added via config + registry with minimal boilerplate.
- Features update deterministically with clear order.

### Phase 4: Rendering boundary cleanup
**Deliverables**
- `DisplayManager` only consumes read-only snapshots.
- Rendering config uses the same schema validation pipeline as sim config.
- Clear separation between render loop and sim loop.

**Implementation outline**
- `tensor_beasts/display/` reads `WorldSnapshot` and `DisplayConfig`.
- Ensure display has no direct access to mutable `World` objects.

**Exit criteria**
- Rendering works with headless mode disabled but sim remains unaffected.

### Phase 5: Tests and documentation
**Deliverables**
- Align tests with new feature set (remove/replace invalid tests).
- Add tests for config validation and world initialization.
- Update `README.md` with new run commands and config examples.

**Exit criteria**
- Tests pass in a clean environment.
- Docs reflect the stabilized architecture and config schema.

## 4) Immediate work items (short-term backlog)
- Fix config naming and resolve `infiltration_rate` vs `infiltation_rate`.
- Implement a minimal `World.observable` that is explicitly documented.
- Introduce a `SimulationRunner` that can run headless with a fixed seed.
- Replace direct `print` calls with `logging` and a verbosity flag.

## 5) Risks and mitigation
- **Risk: schema churn breaks configs** → Add migration utilities and clear error messages.
- **Risk: feature update order is unclear** → Define explicit, tested ordering rules.
- **Risk: performance regressions** → Add benchmark script for sim steps/sec.

## 6) Decision points to confirm
- **Config framework**: stay with OmegaConf or move to Pydantic + OmegaConf interop.
- **Entity naming**: enforce canonical case or allow aliasing with a resolver.
- **Observation API**: TensorDict views vs. explicit typed dataclasses.

## 7) Proposed next step
Start with Phase 1 (config stabilization), since it unlocks consistent entity definitions and makes downstream refactors safer.

---

# Detailed Implementation Plan: Registry, Validation, and Dependency Resolution

This section provides atomic implementation steps for three high-value improvements identified during architecture review. These improvements align with Phase 1 (config stabilization) and Phase 3 (entity/feature hardening) above.

## Overview

| Phase | Improvement | Components Touched | Dependencies |
|-------|-------------|-------------------|--------------|
| A | Decorator-based registration | `registry.py`, `entities/__init__.py`, `config/models.py` | None |
| B | Config validation of `${key:...}` refs | `config/load.py`, `config/models.py` | Phase A |
| C | Automatic dependency resolution | `feature.py`, `entity.py`, `world.py` | Phase A |

---

## Phase A: Decorator-Based Registration

**Goal**: Replace manual registry maintenance with automatic discovery via decorators.

### A.1: Create Registry Infrastructure

**File**: `tensor_beasts/registry.py`

**Changes**:
1. Create `EntityRegistry` class to store entity metadata
2. Create `FeatureRegistry` class to store feature metadata
3. Add `@register_entity` decorator
4. Add `@register_feature` decorator
5. Maintain backward compatibility with existing `get_entity_class()` API

**Atomic Steps**:

```
A.1.1: Add FeatureInfo and EntityInfo dataclasses
       - FeatureInfo: name, cls, default_config, dtype, dependencies (placeholder for Phase C)
       - EntityInfo: name, cls, features (dict of feature_name -> FeatureInfo)

A.1.2: Add FeatureRegistry class
       - _features: Dict[str, FeatureInfo]
       - register(cls) -> cls: decorator that extracts metadata and stores it
       - get(name) -> FeatureInfo
       - all() -> Dict[str, FeatureInfo]
       - exists(name) -> bool

A.1.3: Add EntityRegistry class
       - _entities: Dict[str, EntityInfo]
       - register(cls) -> cls: decorator that extracts entity + feature metadata
       - get(name) -> EntityInfo
       - all() -> Dict[str, EntityInfo]
       - exists(name) -> bool
       - get_features(entity_name) -> Dict[str, FeatureInfo]

A.1.4: Create module-level singleton instances
       - feature_registry = FeatureRegistry()
       - entity_registry = EntityRegistry()
       - register_feature = feature_registry.register
       - register_entity = entity_registry.register

A.1.5: Update get_entity_class() to use entity_registry
       - Keep function signature identical for backward compat
       - Delegate to entity_registry.get(name).cls
```

**Testing** (`tests/test_registry.py`):
```
test_register_feature_decorator:
    - Define minimal test feature with @register_feature
    - Assert feature_registry.exists(feature.name)
    - Assert feature_registry.get(name).cls is feature class
    - Assert feature_registry.get(name).dtype matches

test_register_entity_decorator:
    - Define minimal test entity with @register_entity and annotated features
    - Assert entity_registry.exists(entity_name)
    - Assert entity_registry.get_features(name) contains expected features

test_get_entity_class_backward_compat:
    - Assert get_entity_class("Terrain") returns Terrain class
    - Assert get_entity_class("Unknown") raises ValueError with message

test_registry_rejects_duplicate_names:
    - Register feature with name "foo"
    - Attempt to register another with same name
    - Assert raises RegistryError
```

### A.2: Migrate Features to Use Decorators

**Files**: `tensor_beasts/features/*.py`

**Atomic Steps**:
```
A.2.1: Add @register_feature to terrain features
       - Elevation, AquiferElevation, SoilVolume, SoilWaterVolume, SurfaceWaterVolume

A.2.2: Add @register_feature to shared features
       - Energy, Scent

A.2.3: Add @register_feature to plant features
       - Seed, Crowding

A.2.4: Add @register_feature to animal features
       - IdFeature, OffspringCount

A.2.5: Add @register_feature to toy features
       - FluidDensity
```

**Testing** (`tests/test_registry.py`):
```
test_all_features_registered:
    - Import all feature modules
    - Assert feature_registry has entries for all known features
    - Expected: elevation, aquifer_elevation, soil_volume, soil_water_volume,
                surface_water_volume, energy, scent, seed, crowding,
                id_feature, offspring_count, fluid_density
```

### A.3: Migrate Entities to Use Decorators

**Files**: `tensor_beasts/entities/*.py`

**Atomic Steps**:
```
A.3.1: Add @register_entity to Terrain

A.3.2: Add @register_entity to Plant

A.3.3: Add @register_entity to Animal, Herbivore, Predator

A.3.4: Add @register_entity to DiffusionToy
```

**Testing** (`tests/test_registry.py`):
```
test_all_entities_registered:
    - Import all entity modules
    - Assert entity_registry has: Terrain, Plant, Herbivore, Predator, DiffusionToy

test_entity_features_discovered:
    - Assert entity_registry.get_features("Terrain") contains:
      elevation, aquifer_elevation, soil_volume, soil_water_volume, surface_water_volume
    - Assert entity_registry.get_features("Plant") contains:
      energy, scent, seed, crowding
```

### A.4: Remove Manual Registry Maintenance

**Atomic Steps**:
```
A.4.1: Remove ENTITY_NAMES set from registry.py
       - Delete the hardcoded set
       - get_entity_class() now uses entity_registry

A.4.2: Update entities/__init__.py
       - Keep imports for backward compat (other modules may import from here)
       - Add comment noting registration happens via decorators

A.4.3: Update config/load.py _validate_entities()
       - Replace hardcoded entity_models dict with registry lookup
       - entity_registry.exists(entity_name) replaces manual check
```

**Testing**:
```
test_adding_entity_requires_no_manual_edits:
    - Create a new test entity class with @register_entity in test file
    - Assert entity_registry.exists(new_entity_name)
    - Assert get_entity_class(new_entity_name) returns the class
    - No edits to registry.py or __init__.py needed
```

### A.5: Auto-Generate Pydantic Config Models (Optional Enhancement)

**File**: `tensor_beasts/config/models.py`

**Atomic Steps**:
```
A.5.1: Add generate_feature_config_model(feature_info) -> Type[BaseModel]
       - Inspect feature.default_config
       - Create Pydantic model with Optional fields for each config key
       - Cache generated models

A.5.2: Add generate_entity_config_model(entity_info) -> Type[BaseModel]
       - For each feature in entity, get/generate feature config model
       - Compose into entity config model

A.5.3: Update _validate_entities() to use generated models
       - If entity not in manual entity_models, try generated model
       - Log warning if using generated (less strict) model
```

**Testing**:
```
test_generate_feature_config_model:
    - Generate model for SoilWaterVolume
    - Assert model has fields: soil_porosity, infiltration_rate, etc.
    - Assert validation accepts valid config
    - Assert validation rejects unknown keys (extra="forbid")

test_new_feature_validates_without_manual_model:
    - Create test feature with default_config
    - Generate config model
    - Assert it validates matching config dict
```

---

## Phase B: Config Validation of `${key:...}` References

**Goal**: Validate that all `${key:entity,feature}` references point to valid targets at config load time.

### B.1: Extract Key References from Config

**File**: `tensor_beasts/config/load.py`

**Atomic Steps**:
```
B.1.1: Add find_key_references(cfg: DictConfig) -> List[Tuple[str, str, str]]
       - Recursively walk config tree
       - Find all strings matching pattern: ${key:entity,feature}
       - Return list of (path_in_config, entity, feature) tuples
       - Handle both resolved tuples and unresolved strings

B.1.2: Add parse_key_reference(ref: str) -> Tuple[str, str]
       - Parse "${key:terrain,elevation}" -> ("terrain", "elevation")
       - Handle edge cases: extra whitespace, missing parts
       - Raise ConfigError for malformed references
```

**Testing** (`tests/config/test_validation.py`):
```
test_find_key_references_simple:
    - Config with one ${key:terrain,elevation}
    - Assert returns [("path.to.key", "terrain", "elevation")]

test_find_key_references_nested:
    - Config with refs in nested dicts and lists
    - Assert all refs found with correct paths

test_find_key_references_none:
    - Config with no key refs
    - Assert returns empty list

test_parse_key_reference_valid:
    - Assert parse_key_reference("${key:terrain,elevation}") == ("terrain", "elevation")

test_parse_key_reference_malformed:
    - Assert raises ConfigError for "${key:terrain}"  (missing feature)
    - Assert raises ConfigError for "${key:}"  (missing both)
    - Assert raises ConfigError for "not a ref"
```

### B.2: Validate References Against Registry

**File**: `tensor_beasts/config/load.py`

**Atomic Steps**:
```
B.2.1: Add validate_key_references(cfg: DictConfig) -> None
       - Call find_key_references(cfg)
       - For each (path, entity, feature):
         - Check entity_registry.exists(entity) - note: entity names in config are PascalCase
         - Check feature exists in entity_registry.get_features(entity)
         - Collect all errors before raising

B.2.2: Add ConfigValidationError class
       - Stores list of validation errors
       - Pretty-prints all errors with config paths
       - Suggests corrections (did you mean "elevation"?)

B.2.3: Integrate into validate_config()
       - Call validate_key_references() after schema validation
       - Catch and re-raise with helpful context
```

**Testing**:
```
test_validate_key_references_valid:
    - Config with valid refs: ${key:terrain,elevation}
    - Assert no exception raised

test_validate_key_references_unknown_entity:
    - Config with ${key:unknown_entity,elevation}
    - Assert raises ConfigValidationError
    - Assert error message mentions "unknown_entity"
    - Assert error message suggests similar entity names

test_validate_key_references_unknown_feature:
    - Config with ${key:terrain,unknown_feature}
    - Assert raises ConfigValidationError
    - Assert error mentions path in config where ref appears
    - Assert error suggests similar feature names

test_validate_key_references_multiple_errors:
    - Config with multiple bad refs
    - Assert all errors collected and reported together
    - Assert each error has its config path

test_load_config_catches_bad_refs:
    - Call load_config() with bad refs
    - Assert raises ValueError with clear message
```

### B.3: Handle Entity Name Case Sensitivity

**Atomic Steps**:
```
B.3.1: Document naming convention
       - Entity names in registry: PascalCase ("Terrain")
       - Entity names in key refs: lowercase ("terrain")
       - Feature names: snake_case ("soil_water_volume")

B.3.2: Add entity_name_to_registry_name(name: str) -> str
       - "terrain" -> "Terrain"
       - "plant" -> "Plant"
       - Handle edge cases

B.3.3: Update validate_key_references() to use normalization
       - Normalize entity name before registry lookup
       - Error messages show both forms for clarity
```

**Testing**:
```
test_entity_name_normalization:
    - Assert entity_name_to_registry_name("terrain") == "Terrain"
    - Assert entity_name_to_registry_name("diffusion_toy") == "DiffusionToy"

test_validate_refs_handles_lowercase:
    - Config with ${key:terrain,elevation}  (lowercase entity)
    - Assert validation passes (finds Terrain entity)
```

---

## Phase C: Automatic Dependency Resolution

**Goal**: Declare feature dependencies explicitly and auto-order initialization/updates.

### C.1: Add Dependency Declaration to Features

**File**: `tensor_beasts/features/feature.py`

**Atomic Steps**:
```
C.1.1: Add depends_on class attribute to Feature
       - Type: Dict[str, Union[str, Tuple[str, str]]]
       - Key: dependency name (for documentation)
       - Value: "local_feature" or ("entity", "feature") for cross-entity
       - Default: empty dict

C.1.2: Add get_dependencies() method
       - Returns list of (entity, feature) tuples
       - Resolves "local" deps using self.key_prefix

C.1.3: Update FeatureInfo to include dependencies
       - Extract from cls.depends_on during registration
```

**Testing** (`tests/test_dependencies.py`):
```
test_feature_depends_on_default:
    - Feature with no depends_on
    - Assert get_dependencies() returns []

test_feature_depends_on_local:
    - Feature with depends_on = {"elev": "elevation"}
    - In entity "terrain"
    - Assert get_dependencies() returns [("terrain", "elevation")]

test_feature_depends_on_cross_entity:
    - Feature with depends_on = {"elev": ("terrain", "elevation")}
    - Assert get_dependencies() returns [("terrain", "elevation")]
```

### C.2: Declare Dependencies for Existing Features

**Files**: `tensor_beasts/features/terrain_features.py`, etc.

**Atomic Steps**:
```
C.2.1: Add depends_on to SoilVolume
       depends_on = {
           "elevation": "elevation",
           "surface_outflow": "surface_outflow",
       }

C.2.2: Add depends_on to SoilWaterVolume
       depends_on = {
           "soil_volume": "soil_volume",
           "elevation": "elevation",
           "surface_water_volume": "surface_water_volume",
       }

C.2.3: Add depends_on to SurfaceWaterVolume
       depends_on = {
           "elevation": "elevation",
       }

C.2.4: Add depends_on to AquiferElevation
       depends_on = {
           "elevation": "elevation",
       }

C.2.5: Add depends_on to shared features (Energy, Scent)
       - Review cross-entity dependencies
       - Document any

C.2.6: Add depends_on to plant features (Seed, Crowding)
       depends_on for Seed = {"energy": "energy", "crowding": "crowding"}
       depends_on for Crowding = {"energy": "energy"}
```

**Testing**:
```
test_terrain_feature_dependencies_declared:
    - Assert SoilWaterVolume.depends_on includes soil_volume, elevation
    - Assert SurfaceWaterVolume.depends_on includes elevation
    - Assert Elevation.depends_on is empty (no dependencies)
```

### C.3: Implement Topological Sort for Features

**File**: `tensor_beasts/entities/entity.py`

**Atomic Steps**:
```
C.3.1: Add _build_dependency_graph() method to Entity
       - For each feature, get dependencies
       - Filter to only local (same-entity) dependencies
       - Return dict: feature_name -> set of dependency feature_names

C.3.2: Add _topological_sort(graph) static method
       - Kahn's algorithm or DFS-based topo sort
       - Detect cycles and raise clear error
       - Return ordered list of feature names

C.3.3: Add _get_initialization_order() method
       - Build dependency graph
       - Return topologically sorted feature names
       - Cache result (dependencies don't change)

C.3.4: Add _get_update_order() method
       - May differ from init order (some features don't update)
       - Filter to features with non-trivial update()
       - Return topologically sorted list
```

**Testing**:
```
test_build_dependency_graph:
    - Terrain entity
    - Assert graph shows soil_water_volume depends on soil_volume

test_topological_sort_simple:
    - Graph: A -> B -> C
    - Assert order is [C, B, A] or [A, B, C] depending on direction

test_topological_sort_detects_cycle:
    - Graph: A -> B -> A
    - Assert raises DependencyCycleError with cycle path

test_get_initialization_order_terrain:
    - Assert elevation comes before soil_volume
    - Assert soil_volume comes before soil_water_volume
```

### C.4: Update Entity.initialize() to Use Ordering

**File**: `tensor_beasts/entities/entity.py`

**Atomic Steps**:
```
C.4.1: Update Entity.initialize() base implementation
       - Get initialization order via _get_initialization_order()
       - Call feature.initialize_data() in order
       - Log order if debug flag set

C.4.2: Update Terrain.initialize() to delegate to base
       - Remove manual ordering
       - Call super().initialize()
       - Verify behavior unchanged

C.4.3: Update Plant.initialize() similarly

C.4.4: Update Animal.initialize() similarly

C.4.5: Update DiffusionToy.initialize() similarly
```

**Testing**:
```
test_entity_initialize_respects_order:
    - Create Terrain entity
    - Mock feature.initialize_data() to record call order
    - Call entity.initialize()
    - Assert elevation initialized before soil_water_volume

test_terrain_initialize_still_works:
    - Integration test: create World with Terrain
    - Initialize
    - Assert all features have data tensors
    - Assert no exceptions
```

### C.5: Update Entity.update() to Use Ordering

**File**: `tensor_beasts/entities/entity.py`

**Atomic Steps**:
```
C.5.1: Update Entity.update() base implementation
       - Get update order via _get_update_order()
       - Call feature.update(step) in order

C.5.2: Update Terrain.update() to delegate
       - Remove manual ordering
       - Add any entity-specific logic before/after

C.5.3: Update other entities similarly
```

**Testing**:
```
test_entity_update_respects_order:
    - Mock feature.update() to record order
    - Call entity.update()
    - Assert correct order

test_terrain_update_conserves_water:
    - Existing test: water mass conserved after update
    - Verify still passes with automatic ordering
```

### C.6: Handle Cross-Entity Dependencies (World Level)

**File**: `tensor_beasts/world.py`

**Atomic Steps**:
```
C.6.1: Add _build_entity_dependency_graph() to World
       - For each entity's features, get cross-entity deps
       - Build graph: entity_name -> set of entity_names it depends on

C.6.2: Add _get_entity_initialization_order() to World
       - Topological sort of entity dependency graph
       - Return ordered list of entity names

C.6.3: Update World.initialize() to use ordering
       - Initialize entities in dependency order
       - Initialize shared features at appropriate point

C.6.4: Add _get_entity_update_order() to World
       - May differ from init order
       - Return ordered list
```

**Testing**:
```
test_world_entity_initialization_order:
    - Config with Plant depending on Terrain (via soil_water refs)
    - Assert Terrain initialized before Plant

test_world_handles_cross_entity_deps:
    - Plant's growth depends on Terrain's soil_water_saturation
    - Assert no KeyError during initialization
```

### C.7: Validate Dependencies at Config Load Time

**File**: `tensor_beasts/config/load.py`

**Atomic Steps**:
```
C.7.1: Add validate_dependencies(cfg: DictConfig) -> None
       - For each entity in config, get its features
       - For each feature, check declared dependencies are satisfiable:
         - Local deps: feature exists in same entity's config
         - Cross-entity deps: entity and feature exist in config
       - Collect and report all errors

C.7.2: Integrate into validate_config()
       - Call after validate_key_references()
       - Provides early feedback on missing dependencies
```

**Testing**:
```
test_validate_dependencies_satisfied:
    - Config with Terrain (has elevation)
    - Config with Plant (depends on terrain.elevation via soil refs)
    - Assert no error

test_validate_dependencies_missing_local:
    - Config with entity missing required feature
    - Assert error identifies missing feature

test_validate_dependencies_missing_cross_entity:
    - Config with Plant but no Terrain
    - Plant features depend on terrain.elevation
    - Assert error identifies missing entity
```

---

## Testing Strategy Summary

### Unit Tests (per phase)

| Phase | Test File | Coverage |
|-------|-----------|----------|
| A | `tests/test_registry.py` | Registry classes, decorators, backward compat |
| B | `tests/config/test_validation.py` | Ref extraction, validation, error messages |
| C | `tests/test_dependencies.py` | Dependency declaration, topo sort, ordering |

### Integration Tests

| Test | Description |
|------|-------------|
| `test_full_config_validation` | Load each config file, assert validation passes |
| `test_new_entity_workflow` | Add entity via decorators only, verify it works |
| `test_initialization_order_real_world` | Terrain features init in correct order |
| `test_update_order_real_world` | Water flows correctly with auto-ordering |

### Regression Tests

| Test | Description |
|------|-------------|
| `test_terrain_config_loads` | Existing terrain_config.yaml still works |
| `test_beast_config_loads` | Existing beast_config.yaml still works |
| `test_water_conservation` | Existing water mass test still passes |
| `test_simulation_deterministic` | Same seed produces same results |

### Test Fixtures

Add to `tests/conftest.py`:
```python
@pytest.fixture
def fresh_registry():
    """Reset registry state for isolated tests."""
    # Save current state
    # Yield
    # Restore state

@pytest.fixture
def minimal_entity_config():
    """Minimal valid config for testing."""
    return OmegaConf.create({...})

@pytest.fixture
def mock_feature():
    """Test feature class with configurable dependencies."""
    ...
```

---

## Implementation Order and Checkpoints

### Checkpoint 1: Registry Infrastructure (A.1-A.3) - COMPLETE
- [x] FeatureRegistry and EntityRegistry classes work
- [x] Decorators register metadata correctly
- [x] All existing features and entities migrated
- [x] All A.* tests pass

### Checkpoint 2: Registry Cleanup (A.4-A.5) - COMPLETE
- [x] Manual ENTITY_NAMES removed
- [x] get_entity_class() uses registry
- [x] Config validation uses registry
- [x] Existing configs still load

### Checkpoint 3: Reference Validation (B.1-B.3) - COMPLETE
- [x] Key references extracted from configs
- [x] Validation catches invalid refs
- [x] Error messages are helpful
- [x] All B.* tests pass

### Checkpoint 4: Dependency Declaration (C.1-C.2) - COMPLETE
- [x] depends_on attribute added to Feature
- [x] All terrain features have dependencies declared
- [x] Dependencies visible in registry
- [x] All C.1-C.2 tests pass

### Checkpoint 5: Automatic Ordering (C.3-C.5) - COMPLETE
- [x] Topological sort implemented
- [x] Entity.initialize() uses ordering
- [x] Entity.update() uses ordering
- [x] Existing behavior preserved
- [x] All C.3-C.5 tests pass

### Checkpoint 6: Cross-Entity and Final (C.6-C.7) - COMPLETE
- [x] World-level ordering works
- [x] Dependency validation at load time
- [x] All tests pass
- [x] All existing configs work

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking existing configs | Run all configs through validation after each phase |
| Circular dependencies | Detect and report clearly; existing code has none |
| Performance regression | Ordering computed once at init, not per-frame |
| Missing dependencies | Start with obvious ones; iterate based on runtime errors |

## Success Metrics - ACHIEVED

1. **Zero manual registry edits**: Adding new entity/feature requires only the class definition - **DONE**
2. **Config errors at load time**: All `${key:...}` typos caught before simulation starts - **DONE**
3. **No manual ordering**: Entity subclasses don't override initialize()/update() just for ordering - **DONE**
4. **All existing tests pass**: No regressions - **DONE** (48 new tests passing)
5. **Clear error messages**: Invalid config produces actionable error with line number and suggestion - **DONE**

> **ARCHIVED**: This document is a pre-implementation assessment from before the
> registry, config validation, and dependency resolution improvements were implemented.
> Many issues identified here have been addressed. Retained for historical context.
> See 00-stabilization-plan.md for implementation status.

---

# Tensor Beasts repo assessment (current state)

## 1) Executive snapshot
- **Purpose**: ecological simulation using mostly `uint8` Torch tensors + PyGame/OpenGL rendering, with experimental RL wrappers (`DQN`, `IQN`, `IQL`). Primary entry is `tensor_beasts/main.py`.
- **Top-level contents**: `README.md`, config YAMLs (`beast_config.yaml`, `terrain_config.yaml`, `toy_config.yaml`), core package `tensor_beasts/`, tests in `tests/`, plus runtime artifacts (`logs/`, `outputs/`, `wandb/`).
- **Immediate blockers** (details in later chapters): missing package entrypoint for `python -m tensor_beasts`, unimplemented `World.observable`, RL envs out of sync with `World`, config key mismatches, tests refer to missing features.

## 2) System architecture & simulation flow
The simulation is an Entity + Feature system backed by a single `TensorDict`. Each `Entity` composes `Feature`s, and some `Feature`s are shared (e.g., `Energy`, `Scent`).

**Core runtime flow (from `tensor_beasts/main.py` + `tensor_beasts/world.py`):**
```
+-----------------+        +---------------------+        +------------------+
| main.py         |        | WorldThread         |        | DisplayManager   |
| (CLI loop)      |        | (background update) |        | (OpenGL/PyGame)  |
+--------+--------+        +----------+----------+        +---------+--------+
         |                            |                             |
         | start WorldThread          |                             |
         |--------------------------->|                             |
         |                            | world.update() loop         |
         |                            |---------------------------->|
         |                            | update_screen_cb            |
         |<---------------------------| (renders TensorDict)         |
         | handle input/events        |                             |
         v                            v                             v
   pygame event loop          TensorDict state                 GL texture
```

**World update (from `tensor_beasts/world.py`):**
```
World.update()
  ├─ td["random"] = random uint8 tensor
  ├─ update shared features (Energy/Scent)
  └─ update entities (Plant/Herbivore/Predator/Terrain)
```

**Entity/Feature composition (from `tensor_beasts/entities/*.py` + `tensor_beasts/features/*.py`):**
```
Entity
  ├─ Feature: data lives at td[(entity_name, feature_name)]
  └─ SharedFeature: data lives at td[("shared_features", feature_name)]
```

## 3) Data model & configuration
The configs drive the world size, entities, and display layers. Keys are stored in a `TensorDict` and referenced by OmegaConf resolver `${key:...}` registered in `tensor_beasts/main.py`.

**TensorDict layout (intended)**
```
td
├─ ("random") : uint8[H,W]
├─ ("shared_features","energy") : uint8[H,W,C]
├─ ("shared_features","scent")  : uint8[H,W,C]
└─ ("plant"|"herbivore"|"predator"|"terrain", <feature>)
```

**Config highlights**
- `beast_config.yaml` (full sim) and `terrain_config.yaml` (terrain only) use **capitalized entity keys** (`Terrain`, `Plant`, `Herbivore`, `Predator`). `World.__init__` expects keys matching class names via `getattr(entities, entity_name)`.
- `toy_config.yaml` runs a `DiffusionToy` with `FluidDensity`.

**Schema and naming issues**
- `tensor_beasts/rl/dqn/tensor_beasts_config.yaml` uses lowercase `plant`, `herbivore`, `predator`. This will not resolve in `World.__init__`.
- `tensor_beasts/features/terrain_features.py` defines `SoilWaterVolume.default_config` with `infiltation_rate` while configs use `infiltration_rate` (typo mismatch).
- `SoilWaterVolume.initialize_data()` expects `field_capacity`, but it is not in `default_config` and only appears in YAML.

## 4) Rendering, UI, and diagnostics
Rendering uses OpenGL textures updated from Torch tensors.

**Render pipeline (from `tensor_beasts/display/*`):**
```
TensorDict -> dispatch_render(...) -> RGB tensor -> DisplayManager.update_screen()
```

**Display design**
- `dispatch_render` supports `default`, `layered`, and `cross_section` renderers (`tensor_beasts/display/rendering.py`).
- `DisplayManager` uses OpenGL with texture replacement per frame (`tensor_beasts/display/display_manager.py`).
- Input controls (in `tensor_beasts/main.py`): zoom, pan, toggle displays, seed entities, pause/step.

**State and usability notes**
- `DisplayManager.create_texture()` is called twice on initialization; this is likely redundant.
- `DisplayManager.render_cell_values` is expensive and currently commented-out by default.
- Several `print(...)` calls in water-related features will flood stdout during simulation steps, which can dominate performance (`tensor_beasts/features/terrain_features.py`).

## 5) RL subsystem (DQN / IQN / IQL)
The RL side appears experimental and currently out of sync with the simulation core.

**Intended RL data flow**
```
World -> Gym Env -> TorchRL Collector -> Replay Buffer -> Loss -> Optimizer
```

**Key mismatches and breakages**
- `TensorBeastsEnv` in `tensor_beasts/rl/envs/world_environment.py` calls `World([Predator, Plant, Herbivore], config=world_cfg)`, but `World` takes only a `DictConfig`.
- `World.observable` is declared but not implemented (`tensor_beasts/world.py`), yet RL depends on it.
- `World.entity_scores()` references `self.entities` and `entity.get_feature`, neither of which exist.
- Configs for RL environments use lowercase entity names and old schema.

**Status summary**: RL scripts (`tensor_beasts/rl/dqn/dqn_script.py`, `tensor_beasts/rl/iqn/*`, `tensor_beasts/rl/iql/*`) are not runnable as-is against the current `World` API.

## 6) Correctness and reliability risks
High-priority issues that likely break runtime:

1) **Missing module entrypoint**
   `README.md` says `python -m tensor_beasts`, but there is no `tensor_beasts/__main__.py`. The command will fail.
   - Affects: `README.md`, `tensor_beasts/` package.

2) **Unimplemented `World.observable`**
   `tensor_beasts/world.py:42` is `pass`; RL and observation logic rely on this.
   - Affects: `tensor_beasts/rl/envs/world_environment.py`.

3) **Broken API usage in RL**
   `TensorBeastsEnv` and RL configs mismatch `World` signature and schema.
   - Affects: `tensor_beasts/rl/envs/world_environment.py`, `tensor_beasts/rl/dqn/tensor_beasts_config.yaml`.

4) **Config key mismatch in soil water**
   `infiltation_rate` vs `infiltration_rate` breaks config usage.
   - Affects: `tensor_beasts/features/terrain_features.py`, `beast_config.yaml`.

5) **Feature tag bug**
   `Feature.__init__` assigns `self.tags = set(...).update(...)` which returns `None`, so tags are lost.
   - Affects: `tensor_beasts/features/feature.py`.

6) **`Plant.initialize()` can error on low `init_prob`**
   `torch.randint(0, int(init_prob*255), ...)` will error when `int(...) == 0`.
   - Affects: `tensor_beasts/entities/plant.py`.

7) **Tests target non-existent features**
   Tests reference `slopes`, `neighbor_distances`, `soil_saturation_gradient` not present in `Terrain` features.
   - Affects: `tests/conftest.py`, `tests/features/test_terrain_features.py`.

## 7) Tests, tooling, and runtime assets
- **Tests**: `tests/test_util.py` covers utility functions; `tests/features/test_terrain_features.py` appears invalid with current features. Expect test failures.
- **Tooling**: `pyproject.toml` includes `torchrl`, `gymnasium`, `hydra-core`, `wandb`, `moviepy`, `rich`, `matplotlib`. `omegaconf` and `tensordict` are used directly but not listed explicitly; these may be coming via dependencies.
- **Artifacts**: `logs/`, `outputs/`, `wandb/` and `__pycache__/` indicate prior runs are included in repo.

## 8) Refactor opportunities (no sacred cows)
Here are the biggest architectural moves worth considering:

1) **Formalize the data model and schema**
   - Define a single canonical schema for entity names and feature keys.
   - Validate config against schema at startup.
   - Move key resolver logic to a central `config.py` module.

2) **Split simulation core from rendering**
   - Core sim should be deterministic and headless.
   - Rendering should consume snapshots or views of `TensorDict`.
   - Makes testing and RL integration much easier.

3) **Rebuild RL integration around a stable API**
   - Implement `World.observable` explicitly, or create a dedicated `ObservationBuilder`.
   - Ensure `World` and RL envs agree on config schema and data types.
   - Provide a minimal "RL-ready" world config.

4) **Clean up Feature base class**
   - Fix `self.tags` bug; add `Feature.get_feature()` to entities.
   - Add explicit lifecycle (`initialize`, `update`, `post_update`) hooks.

5) **Consistency and performance**
   - Avoid `print` in hot loops; use `logging` with levels or callback hooks.
   - Replace repeated `create_texture` calls.
   - Use vectorized operations consistently and avoid autograd overhead if not needed.

---

## Appendices

### A) Repo structure (selected)
```
tensor_beasts/
  main.py
  world.py
  util.py
  display/
    display_manager.py
    rendering.py
  entities/
    animal.py
    plant.py
    terrain.py
    diffusion_toy.py
  features/
    feature.py
    shared_features.py
    animal_features.py
    plant_features.py
    terrain_features.py
    toy_features.py
  rl/
    envs/world_environment.py
    dqn/
    iqn/
    iql/
```

### B) Key files to audit first
- `tensor_beasts/world.py` — implement `observable`, fix `entity_scores`.
- `tensor_beasts/features/feature.py` — fix `tags` handling.
- `tensor_beasts/rl/envs/world_environment.py` — align with `World` API.
- `tensor_beasts/entities/plant.py` — fix init RNG edge case.
- `beast_config.yaml` — align with feature config names (`infiltration_rate`).

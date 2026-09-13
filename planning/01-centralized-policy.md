# Feature: Centralized Animal Policy System

**Status:** Planning
**Priority:** High
**Dependencies:** None
**Blocks:** Genetic Algorithm System (partial)

---

## Goal Statement

Centralize animal decision-making into a single well-abstracted and well-generalized **policy**. This policy:

- Takes environmental inputs (sensory scent inputs, current internal state)
- Outputs selected actions (movement direction preference, metabolic state, reproduction)
- Provides a well-thought-out method for defining relationships between inputs and outputs
- Has the generality of an MLP but with affordances for manually defining relationships
- Is easily extensible to additional inputs, outputs, and internal function mechanics
- Is compatible with Gymnasium interfaces for future interoperability

---

## Current Architecture Analysis

### Decision Flow (12-step `Animal.update()`)

```
Step 1:  Death Check          → biomass < threshold → die
Step 2:  Build Observation    → gather sensory inputs via perception config
Step 3:  Process Observation  → collapse to gradient + direction using weights
Step 4:  Update Gradient EMA  → smooth stimulus strength over time
Step 5:  Compute Metabolic Rate → based on gradient_ema + biomass
Step 5b: Compute Metabolic Efficiency → efficiency decreases with exertion
Step 6:  Metabolism           → convert biomass → energy
Step 7:  Energy Dissipation   → fixed floor + proportional loss
Step 8:  Movement Probability → probability = energy / 255.0
Step 9:  Movement Cost        → flat cost per movement
Step 10: Movement & Reproduction → move in direction + reproduce if able
Step 11: Eating               → try food sources in order
Step 12: Scent Emission       → emit scent based on biomass
```

### Current Input Gathering

**Observation System** (`observations.py`):
```python
# Configuration example
perception: [
    {"key": "hydrodynamicplant:scent", "kernel_size": 1},
    {"key": "herbivore:scent", "kernel_size": 1},
]

navigation_weights: {
    "hydrodynamicplant:scent": 1.0,    # Attractive
    "herbivore:scent": -0.1,           # Repulsive
}
```

**Observation Data Structure:**
- `directional`: Dict of (4, H, W) tensors per feature [up, down, left, right]
- `current`: Dict of (H, W) current values per feature
- `energy`, `biomass`: Current state tensors

### Current Decision Logic (Fragmented)

| Decision | Current Location | Logic |
|----------|-----------------|-------|
| Movement Direction | `process_observation()` | Weighted argmax of neighbor scents |
| Movement Probability | `Animal.update()` | `energy / 255.0` |
| Reproduction | `perform_move()` | `biomass > reproduction_threshold` |
| Metabolic Rate | `Animal.update()` | `basal + gradient_ema * sensitivity` |
| Eating | `Animal.update()` | Try food sources in config order |

### Key Files

| File | Relevance |
|------|-----------|
| `tensor_beasts/entities/animal.py` | Main decision loop (lines 88-426) |
| `tensor_beasts/observations.py` | Perception system |
| `tensor_beasts/entities/helpers/animal_helpers.py` | Movement/reproduction execution |
| `tensor_beasts/features/feature.py` | Feature base classes |

---

## Proposed Architecture

### Policy Interface

```python
from dataclasses import dataclass
from typing import Protocol, Dict, Any
import torch

@dataclass
class PolicyInput:
    """Standardized input to policy."""
    # Sensory inputs (from Observation)
    directional: Dict[str, torch.Tensor]  # {feature_key: (4, H, W)}
    current: Dict[str, torch.Tensor]      # {feature_key: (H, W)}

    # Internal state
    energy: torch.Tensor      # (H, W) uint8
    biomass: torch.Tensor     # (H, W) uint8
    gradient_ema: torch.Tensor  # (H, W) float32

    # Optional additional context
    metadata: Dict[str, Any] = None

@dataclass
class PolicyOutput:
    """Standardized output from policy."""
    # Movement
    move_direction: torch.Tensor  # (H, W) int: 0=stay, 1-4=cardinal
    move_probability: torch.Tensor  # (H, W) float [0,1]

    # Metabolism
    metabolic_rate: torch.Tensor  # (H, W) float

    # Reproduction
    reproduce: torch.Tensor  # (H, W) bool

    # Optional: eating priority (future)
    # eat_priority: torch.Tensor  # (H, W, num_food_sources)

class AnimalPolicy(Protocol):
    """Protocol for animal decision-making policies."""

    def __call__(self, input: PolicyInput) -> PolicyOutput:
        """Compute actions from inputs."""
        ...

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Return learnable/evolvable parameters."""
        ...

    def set_parameters(self, params: Dict[str, torch.Tensor]) -> None:
        """Set parameters (for genetic mutation)."""
        ...
```

### Gymnasium Compatibility

The interface aligns with Gymnasium's design:

```python
# Gymnasium-style spaces (for future integration)
observation_space = spaces.Dict({
    "directional": spaces.Dict({
        feature_key: spaces.Box(0, 1, shape=(4, H, W), dtype=np.float32)
        for feature_key in perception_keys
    }),
    "current": spaces.Dict({...}),
    "energy": spaces.Box(0, 255, shape=(H, W), dtype=np.uint8),
    "biomass": spaces.Box(0, 255, shape=(H, W), dtype=np.uint8),
})

action_space = spaces.Dict({
    "move_direction": spaces.MultiDiscrete([5] * H * W),  # 0-4 per cell
    "metabolic_rate": spaces.Box(0, max_rate, shape=(H, W)),
    "reproduce": spaces.MultiBinary(H * W),
})
```

### Policy Implementations

#### 1. RuleBasedPolicy (Default/Current Behavior)

```python
class RuleBasedPolicy:
    """Replicates current hardcoded behavior."""

    def __init__(self, config: DictConfig):
        self.navigation_weights = config.navigation_weights
        self.reproduction_threshold = config.reproduction_threshold
        self.basal_rate = config.basal_rate
        self.metabolic_sensitivity = config.metabolic_sensitivity
        # ... other config

    def __call__(self, input: PolicyInput) -> PolicyOutput:
        # Movement direction: weighted argmax (current logic)
        direction = self._compute_direction(input.directional)

        # Movement probability: energy-based (current logic)
        move_prob = input.energy.float() / 255.0

        # Metabolic rate: gradient-coupled (current logic)
        metabolic_rate = self.basal_rate + input.gradient_ema * self.metabolic_sensitivity

        # Reproduction: threshold-based (current logic)
        reproduce = input.biomass > self.reproduction_threshold

        return PolicyOutput(
            move_direction=direction,
            move_probability=move_prob,
            metabolic_rate=metabolic_rate,
            reproduce=reproduce,
        )
```

#### 2. ParameterizedPolicy (For Genetic Algorithm)

```python
class ParameterizedPolicy:
    """Policy with evolvable parameters."""

    def __init__(self, config: DictConfig):
        # Navigation weights become evolvable
        self.navigation_weights = nn.Parameter(
            torch.tensor([config.navigation_weights[k] for k in sorted(config.perception)])
        )

        # Thresholds become evolvable
        self.reproduction_threshold = nn.Parameter(torch.tensor(200.0))
        self.move_energy_threshold = nn.Parameter(torch.tensor(0.5))

        # Metabolic parameters
        self.basal_rate = nn.Parameter(torch.tensor(1.0))
        self.metabolic_sensitivity = nn.Parameter(torch.tensor(1.0))

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        return {
            "navigation_weights": self.navigation_weights.data,
            "reproduction_threshold": self.reproduction_threshold.data,
            "move_energy_threshold": self.move_energy_threshold.data,
            "basal_rate": self.basal_rate.data,
            "metabolic_sensitivity": self.metabolic_sensitivity.data,
        }

    def set_parameters(self, params: Dict[str, torch.Tensor]) -> None:
        for name, value in params.items():
            getattr(self, name).data = value
```

#### 3. NetworkPolicy (Future MLP/RL)

```python
class NetworkPolicy:
    """Neural network policy for RL training."""

    def __init__(self, config: DictConfig):
        input_dim = self._compute_input_dim(config)
        hidden_dim = config.get("hidden_dim", 64)

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Separate heads for each output
        self.move_head = nn.Linear(hidden_dim, 5)  # 5 directions
        self.metabolic_head = nn.Linear(hidden_dim, 1)
        self.reproduce_head = nn.Linear(hidden_dim, 1)
```

### Relationship Definition System

To allow manual definition of input-output relationships with MLP-like generality:

```python
class RelationshipGraph:
    """Define which inputs affect which outputs."""

    def __init__(self):
        self.edges: Dict[Tuple[str, str], Callable] = {}

    def add_relationship(
        self,
        input_key: str,
        output_key: str,
        transform: Callable[[torch.Tensor], torch.Tensor],
        weight: float = 1.0
    ):
        """Add a weighted relationship from input to output."""
        self.edges[(input_key, output_key)] = (transform, weight)

    def compute(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = defaultdict(list)
        for (in_key, out_key), (transform, weight) in self.edges.items():
            if in_key in inputs:
                contribution = weight * transform(inputs[in_key])
                outputs[out_key].append(contribution)

        return {k: sum(v) for k, v in outputs.items()}

# Example usage:
graph = RelationshipGraph()

# Food scent -> movement direction (attraction)
graph.add_relationship(
    "plant:scent", "move_direction",
    transform=lambda x: x,  # identity
    weight=1.0
)

# Predator scent -> movement direction (repulsion)
graph.add_relationship(
    "predator:scent", "move_direction",
    transform=lambda x: -x,  # negate
    weight=0.5
)

# Energy -> movement probability (threshold)
graph.add_relationship(
    "energy", "move_probability",
    transform=lambda x: (x > 50).float(),
    weight=1.0
)
```

---

## Integration Plan

### Phase 1: Extract Policy Interface

1. Create `tensor_beasts/policy/` directory
2. Define `PolicyInput`, `PolicyOutput` dataclasses
3. Define `AnimalPolicy` protocol
4. Create `RuleBasedPolicy` that replicates current behavior exactly

### Phase 2: Refactor Animal.update()

1. Collect inputs into `PolicyInput` (already mostly done via `get_observation()`)
2. Call `self.policy(input)` to get `PolicyOutput`
3. Execute actions from `PolicyOutput`

```python
# Before (scattered):
def update(self, step):
    # ... 200+ lines of mixed decision + execution

# After (centralized):
def update(self, step):
    # 1. Gather inputs
    observation = self.get_observation()
    policy_input = PolicyInput(
        directional=observation.directional,
        current=observation.current,
        energy=self.energy.data,
        biomass=self.biomass.data,
        gradient_ema=self.gradient_ema.data,
    )

    # 2. Compute decisions
    policy_output = self.policy(policy_input)

    # 3. Execute actions
    self._execute_metabolism(policy_output.metabolic_rate)
    self._execute_movement(policy_output.move_direction, policy_output.move_probability)
    self._execute_reproduction(policy_output.reproduce)
    self._execute_eating()  # May also use policy output in future
```

### Phase 3: Configuration Integration

```yaml
# conf/base/simulation.yaml
entities:
  Herbivore:
    policy:
      type: "rule_based"  # or "parameterized", "network"
      # Policy-specific config
      navigation_weights:
        "simpleplant:scent": 1.0
        "predator:scent": -0.5
      reproduction_threshold: 200
      metabolic_sensitivity: 1.0
```

### Phase 4: Gymnasium Wrapper (Optional)

```python
class SimulationEnv(gym.Env):
    """Gymnasium environment wrapper."""

    def __init__(self, world: World):
        self.world = world
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

    def step(self, action):
        # Convert action to PolicyOutput
        # Apply to all entities
        # Return observation, reward, done, info
        ...
```

---

## Risks and Mitigations

### Risk 1: Performance Degradation
**Risk:** Additional abstraction layers slow down the simulation.

**Mitigation:**
- Keep all operations vectorized (no per-cell loops)
- Policy is called once per entity type per step
- Use `torch.compile()` for network policies
- Profile before/after

### Risk 2: Breaking Existing Behavior
**Risk:** Refactoring changes simulation dynamics unexpectedly.

**Mitigation:**
- `RuleBasedPolicy` must replicate current behavior exactly
- Create comprehensive tests comparing old vs new outputs
- Run side-by-side simulations to verify identical trajectories

### Risk 3: Over-abstraction
**Risk:** Making the system too flexible makes it hard to use.

**Mitigation:**
- Start with minimal interface, extend as needed
- Keep `RuleBasedPolicy` as simple reference implementation
- Don't add features until they're needed

### Risk 4: Gymnasium Compatibility Friction
**Risk:** Gymnasium conventions may not fit well with grid-based simulation.

**Mitigation:**
- Focus on interface compatibility, not full integration
- Gymnasium wrapper is optional/future work
- Adapt conventions where necessary (batch actions across grid)

---

## Testing Strategy

### Unit Tests

```python
def test_rule_based_policy_matches_original():
    """Verify RuleBasedPolicy produces identical outputs."""
    # Setup identical inputs
    # Compare outputs exactly

def test_policy_input_construction():
    """Verify PolicyInput is correctly constructed from Observation."""

def test_policy_output_execution():
    """Verify PolicyOutput correctly drives entity actions."""
```

### Integration Tests

```python
def test_simulation_unchanged_with_policy():
    """Run full simulation, verify dynamics unchanged."""
    # Seed random state
    # Run N steps with old code
    # Run N steps with new policy code
    # Compare all tensor states
```

### Performance Tests

```python
def test_policy_performance():
    """Verify no significant performance regression."""
    # Benchmark old update()
    # Benchmark new update() with policy
    # Assert < 10% slowdown
```

---

## Open Questions

1. **Eating policy:** Should food source prioritization be part of policy output?
2. **Action masking:** Should policy output invalid actions (e.g., move into wall), or should execution handle this?
3. **Multi-entity coordination:** Should policies have access to other entities' states?
4. **Reward signal:** What signal should drive RL training? Biomass? Offspring count? Survival time?

---

## Success Criteria

- [ ] `PolicyInput` and `PolicyOutput` dataclasses defined
- [ ] `AnimalPolicy` protocol defined
- [ ] `RuleBasedPolicy` implemented and tested
- [ ] `Animal.update()` refactored to use policy
- [ ] All existing tests pass
- [ ] No performance regression > 10%
- [ ] `ParameterizedPolicy` ready for genetic algorithm integration

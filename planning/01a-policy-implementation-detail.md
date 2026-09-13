# Policy System: Detailed Implementation Plan

**Parent Document:** [01-centralized-policy.md](./01-centralized-policy.md)
**Status:** Planning
**Last Updated:** 2024

---

## Current Decision Points in Animal.update()

The `Animal.update()` method (lines 236-380) makes decisions at these points:

| Step | Decision | Current Logic | Config Parameters |
|------|----------|---------------|-------------------|
| 3 | Movement direction | `process_observation()` → weighted argmax | `navigation_weights`, `log_scale` |
| 4 | Gradient EMA update | `alpha * gradient + (1-alpha) * ema` | `gradient_ema_alpha` |
| 5 | Metabolic rate | `basal + gradient_ema * sensitivity`, capped by biomass | `basal_rate`, `metabolic_sensitivity`, `max_metabolic_rate`, `survival_threshold` |
| 5b | Metabolic efficiency | Linear interpolation basal→max | `min_efficiency`, `max_efficiency` |
| 8 | Movement probability | `energy / 255.0` | (none - hardcoded) |
| 10 | Reproduction trigger | `biomass > reproduction_threshold` | `reproduction_threshold` |
| 11 | Eating order | Sequential through `food_keys` | `food_keys`, `eat_max` |

---

## Exact Current Observation Structure

From `observations.py`:

```python
@dataclass
class Observation:
    # Spatial perception: (4, H, W) per feature [up, down, left, right]
    # Keys are tuples like ("herbivore", "scent")
    directional: Dict[Tuple[str, str], torch.Tensor]

    # Current cell values: (H, W) per feature
    current: Dict[Tuple[str, str], torch.Tensor]

    # Internal state
    energy: torch.Tensor   # (H, W) uint8
    biomass: torch.Tensor  # (H, W) uint8
```

**Built via `get_observation()`:**
```python
obs = get_observation(
    td=self.td,
    perception=[(p.key, p.kernel_size) for p in self.config.perception],
    energy=energy,
    biomass=biomass,
    log_scale=self.config.log_scale,
)
```

**Processed via `process_observation()`:**
```python
gradient_strength, direction = process_observation(obs, self.config.navigation_weights)
# gradient_strength: (H, W) float - absolute stimulus intensity
# direction: (H, W) long - 0=stay, 1=up, 2=down, 3=left, 4=right
```

---

## Proposed Policy Interface (Matching Config Structure)

### PolicyInput

```python
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import torch

@dataclass
class PolicyInput:
    """
    Input to animal policy, directly derived from Observation + internal state.

    All tensors are (H, W) unless otherwise noted.
    Keys in dicts are tuples: ("entity", "feature")
    """
    # === Sensory Input ===
    # Directional perception per feature: (4, H, W) for [up, down, left, right]
    directional: Dict[Tuple[str, str], torch.Tensor]

    # Current cell value per feature: (H, W)
    current: Dict[Tuple[str, str], torch.Tensor]

    # === Internal State ===
    energy: torch.Tensor       # (H, W) uint8 - current energy
    biomass: torch.Tensor      # (H, W) uint8 - current biomass
    gradient_ema: torch.Tensor # (H, W) float32 - smoothed stimulus history

    # === Contextual Info (optional, for advanced policies) ===
    alive_mask: Optional[torch.Tensor] = None  # (H, W) bool - which cells have living entities
    step: Optional[int] = None                  # Current simulation step

    @classmethod
    def from_observation(
        cls,
        obs: 'Observation',
        gradient_ema: torch.Tensor,
        alive_mask: torch.Tensor = None,
        step: int = None,
    ) -> 'PolicyInput':
        """Construct PolicyInput from existing Observation."""
        return cls(
            directional=obs.directional,
            current=obs.current,
            energy=obs.energy,
            biomass=obs.biomass,
            gradient_ema=gradient_ema,
            alive_mask=alive_mask,
            step=step,
        )
```

### PolicyOutput

```python
@dataclass
class PolicyOutput:
    """
    Output from animal policy - all decisions for current step.

    All tensors are (H, W) matching the input grid.
    """
    # === Movement ===
    move_direction: torch.Tensor   # (H, W) long: 0=stay, 1=up, 2=down, 3=left, 4=right
    move_probability: torch.Tensor # (H, W) float [0, 1]: probability of attempting move

    # === Metabolism ===
    metabolic_rate: torch.Tensor   # (H, W) float: biomass to burn this step

    # === Reproduction ===
    # Note: reproduction_threshold check happens in execution, not policy
    # Policy can override with explicit flag if needed

    # === Intermediate Values (for gradient_ema update) ===
    gradient_strength: torch.Tensor  # (H, W) float: current stimulus intensity
```

### Policy Protocol

```python
from typing import Protocol, Dict, Any
from omegaconf import DictConfig

class AnimalPolicy(Protocol):
    """Protocol defining the animal policy interface."""

    def __call__(self, input: PolicyInput) -> PolicyOutput:
        """
        Compute all decisions from sensory input and internal state.

        This is the main entry point - called once per update step.
        Must be fully vectorized (operates on entire grid).
        """
        ...

    def get_config(self) -> DictConfig:
        """Return current configuration (for serialization/inspection)."""
        ...

    # === For Genetic Algorithm Integration ===

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Return evolvable parameters as tensors.

        Keys should match config parameter names where possible.
        """
        ...

    def set_parameters(self, params: Dict[str, torch.Tensor]) -> None:
        """
        Set evolvable parameters from tensors.

        Used by genetic algorithm for mutation/crossover.
        """
        ...
```

---

## RuleBasedPolicy Implementation

This policy replicates the exact current behavior from `Animal.update()`:

```python
class RuleBasedPolicy:
    """
    Rule-based policy matching current Animal.update() behavior.

    Config parameters used:
    - navigation_weights: Dict[Tuple[str,str], float] - signed weights for direction
    - log_scale: float - perception log compression scale
    - basal_rate: int - minimum metabolic rate
    - metabolic_sensitivity: float - gradient -> metabolic rate scaling
    - max_metabolic_rate: int - ceiling on metabolic rate
    - survival_threshold: int - biomass floor for max rate scaling
    - min_efficiency: float - efficiency at max exertion (not used in policy, in execution)
    - max_efficiency: float - efficiency at rest (not used in policy, in execution)
    """

    def __init__(self, config: DictConfig):
        self.config = config

        # Extract relevant parameters
        self.navigation_weights = dict(config.navigation_weights)
        self.basal_rate = config.basal_rate
        self.metabolic_sensitivity = config.metabolic_sensitivity
        self.max_metabolic_rate = config.max_metabolic_rate
        self.survival_threshold = config.survival_threshold

    def __call__(self, input: PolicyInput) -> PolicyOutput:
        # === Step 1: Compute direction and gradient from weighted perception ===
        gradient_strength, move_direction = self._process_observation(input)

        # === Step 2: Compute metabolic rate ===
        # Current logic: basal + gradient_ema * sensitivity, capped by biomass
        metabolic_rate = self._compute_metabolic_rate(
            input.gradient_ema,
            input.biomass
        )

        # === Step 3: Compute movement probability ===
        # Current logic: energy / 255.0
        move_probability = input.energy.float() / 255.0

        return PolicyOutput(
            move_direction=move_direction,
            move_probability=move_probability,
            metabolic_rate=metabolic_rate,
            gradient_strength=gradient_strength,
        )

    def _process_observation(
        self,
        input: PolicyInput
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Replicate process_observation() logic.

        Combines weighted directional values to compute:
        - gradient_strength: absolute stimulus intensity (for metabolism)
        - direction: movement direction (0=stay, 1-4=cardinal)
        """
        combined_directional = None
        combined_current = None
        abs_directional = None
        abs_current = None

        for key, weight in self.navigation_weights.items():
            if key not in input.directional:
                continue

            # Signed weighted values (for direction)
            weighted_dir = input.directional[key].float() * weight
            weighted_cur = input.current[key].float() * weight

            # Absolute weighted values (for stimulus intensity)
            abs_weighted_dir = input.directional[key].float() * abs(weight)
            abs_weighted_cur = input.current[key].float() * abs(weight)

            if combined_directional is None:
                combined_directional = weighted_dir
                combined_current = weighted_cur
                abs_directional = abs_weighted_dir
                abs_current = abs_weighted_cur
            else:
                combined_directional = combined_directional + weighted_dir
                combined_current = combined_current + weighted_cur
                abs_directional = abs_directional + abs_weighted_dir
                abs_current = abs_current + abs_weighted_cur

        if combined_directional is None:
            h, w = input.energy.shape
            device = input.energy.device
            return (
                torch.zeros(h, w, device=device),
                torch.zeros(h, w, dtype=torch.long, device=device)
            )

        # Clamp for direction calculation
        combined_directional = combined_directional.clamp(min=0)
        combined_current = combined_current.clamp(min=0)

        # Direction from signed values
        direction = self._compute_direction(combined_current, combined_directional)

        # Gradient from absolute values
        gradient_strength = self._compute_gradient(abs_current, abs_directional)

        return gradient_strength, direction

    def _compute_direction(
        self,
        current: torch.Tensor,
        directional: torch.Tensor
    ) -> torch.Tensor:
        """Compute movement direction: argmax with random tie-breaking."""
        # Stack: [current, up, down, left, right] -> (5, H, W)
        stacked = torch.cat([current.unsqueeze(0), directional], dim=0)

        # Argmax with random tie-breaking
        max_values = stacked.max(dim=0).values
        masks = (stacked == max_values.unsqueeze(0))
        random_max_masks = masks * torch.rand_like(masks, dtype=torch.float32)
        direction = torch.argmax(random_max_masks, dim=0)

        return direction

    def _compute_gradient(
        self,
        current: torch.Tensor,
        directional: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient strength: max_neighbor - current."""
        max_neighbor = directional.max(dim=0).values
        gradient = (max_neighbor - current.float()).clamp(min=0)
        return gradient

    def _compute_metabolic_rate(
        self,
        gradient_ema: torch.Tensor,
        biomass: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute metabolic rate from gradient EMA and biomass.

        Biomass modulates maximum achievable rate:
        - At survival_threshold: can only do basal
        - At 255: can reach max_rate
        """
        basal = self.basal_rate
        sensitivity = self.metabolic_sensitivity
        max_rate = self.max_metabolic_rate
        threshold = self.survival_threshold

        # Biomass fraction above threshold
        biomass_range = 255.0 - threshold
        biomass_above = (biomass.float() - threshold).clamp(min=0)
        biomass_fraction = (biomass_above / biomass_range).clamp(0, 1)

        # Effective max rate based on biomass
        effective_max = basal + (max_rate - basal) * biomass_fraction

        # Rate from gradient sensitivity
        rate = basal + gradient_ema * sensitivity

        # Clamp to biomass-limited max
        rate = torch.min(rate, effective_max)

        return rate

    def get_config(self) -> DictConfig:
        return self.config

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Return evolvable parameters."""
        # Convert navigation weights to tensor
        keys = sorted(self.navigation_weights.keys())
        nav_weights = torch.tensor([self.navigation_weights[k] for k in keys])

        return {
            "navigation_weights": nav_weights,
            "basal_rate": torch.tensor(float(self.basal_rate)),
            "metabolic_sensitivity": torch.tensor(self.metabolic_sensitivity),
            "max_metabolic_rate": torch.tensor(float(self.max_metabolic_rate)),
        }

    def set_parameters(self, params: Dict[str, torch.Tensor]) -> None:
        """Set evolvable parameters."""
        if "navigation_weights" in params:
            keys = sorted(self.navigation_weights.keys())
            values = params["navigation_weights"].tolist()
            self.navigation_weights = dict(zip(keys, values))

        if "basal_rate" in params:
            self.basal_rate = int(params["basal_rate"].item())

        if "metabolic_sensitivity" in params:
            self.metabolic_sensitivity = params["metabolic_sensitivity"].item()

        if "max_metabolic_rate" in params:
            self.max_metabolic_rate = int(params["max_metabolic_rate"].item())
```

---

## Refactored Animal.update()

Here's how `Animal.update()` would look after policy integration:

```python
def update(self, action: Optional[torch.Tensor] = None):
    biomass = self.biomass.data
    energy = self.energy.data
    gradient_ema = self.gradient_ema.data

    # === Step 1: Death check ===
    dead = biomass < self.config.survival_threshold
    self._handle_death(dead)

    # === Step 2: Build observation ===
    perception = [(p.key, p.kernel_size) for p in self.config.perception]
    obs = get_observation(
        td=self.td,
        perception=perception,
        energy=energy,
        biomass=biomass,
        log_scale=self.config.log_scale,
    )

    # === Step 3: Construct policy input ===
    alive = biomass >= self.config.survival_threshold
    policy_input = PolicyInput.from_observation(
        obs=obs,
        gradient_ema=gradient_ema,
        alive_mask=alive,
        step=self.world.step,
    )

    # === Step 4: Run policy to get all decisions ===
    policy_output = self.policy(policy_input)

    # === Step 5: Update gradient EMA ===
    alpha = self.config.gradient_ema_alpha
    gradient_ema[:] = torch.where(
        alive,
        alpha * policy_output.gradient_strength + (1 - alpha) * gradient_ema,
        gradient_ema
    )

    # === Step 6: Execute metabolism ===
    self._execute_metabolism(policy_output.metabolic_rate)

    # === Step 7: Energy dissipation ===
    self._execute_dissipation()

    # === Step 8: Execute movement ===
    # Use external action if provided (for RL), otherwise policy output
    move_direction = action if action is not None else policy_output.move_direction
    self._execute_movement(
        direction=move_direction,
        move_probability=policy_output.move_probability,
    )

    # === Step 9: Execute eating ===
    self._eat()

    # === Step 10: Emit scent ===
    self.scent.emit(self.world.step)

def _execute_metabolism(self, metabolic_rate: torch.Tensor):
    """Convert biomass to energy based on metabolic rate."""
    biomass = self.biomass.data
    energy = self.energy.data

    # Compute efficiency (decreases at higher rates)
    efficiency = self._compute_efficiency(metabolic_rate)

    # Burn biomass, gain energy
    biomass_burned = torch.min(metabolic_rate, biomass.float()).to(torch.uint8)
    energy_gained = (biomass_burned.float() * efficiency).to(torch.uint8)

    safe_add(energy, energy_gained)
    safe_sub(biomass, biomass_burned)

def _execute_dissipation(self):
    """Apply energy dissipation."""
    energy = self.energy.data

    dissipation = (energy.float() * self.config.dissipation_rate).to(torch.uint8)
    dissipation = torch.clamp(dissipation, min=self.config.dissipation_floor)
    safe_sub(energy, dissipation)

def _execute_movement(
    self,
    direction: torch.Tensor,
    move_probability: torch.Tensor,
):
    """Execute movement and reproduction."""
    energy = self.energy.data
    biomass = self.biomass.data
    offspring_count = self.offspring_count.data
    id_feature = self.id_feature.data
    gradient_ema = self.gradient_ema.data
    random = self.world.td.get("random")

    # Stochastic movement based on probability
    move_mask = torch.rand_like(energy, dtype=torch.float32) < move_probability

    # Movement cost
    movement_cost = torch.full_like(energy, self.config.base_movement_cost)

    # Execute move (includes reproduction logic)
    move(
        primary_feature=energy,
        target=None,
        target_weights=None,
        divide_threshold=self.config.reproduction_threshold,
        divide_feature=biomass,
        divide_fn_self=lambda x: (x.float() * 0.5).to(x.dtype),
        divide_fn_offspring=lambda x: (x.float() * 0.5).to(x.dtype),
        carried_features_self=[offspring_count, id_feature, biomass, gradient_ema],
        carried_feature_fns_self=[
            lambda x: safe_add(x, 1),
            lambda x: x,
            lambda x: (x.float() * 0.5).to(x.dtype),
            lambda x: x
        ],
        carried_features_offspring=[id_feature, biomass, gradient_ema],
        carried_feature_fns_offspring=[
            lambda x: random,
            lambda x: (x.float() * 0.5).to(x.dtype),
            lambda x: x * 0.5
        ],
        agent_action=direction,
        move_mask=move_mask,
        move_cost=movement_cost,
    )
```

---

## Configuration Changes

### New Config Structure

```yaml
# conf/base/simulation.yaml
entities:
  Herbivore:
    # Policy configuration
    policy:
      type: "rule_based"  # Options: "rule_based", "parameterized", "network"

      # Navigation weights (used by all policy types)
      navigation_weights:
        "simpleplant:scent": 1.0
        "predator:scent": -1.0

    # Perception (unchanged)
    perception:
      - key: "simpleplant:scent"
        kernel_size: 1
      - key: "predator:scent"
        kernel_size: 1

    # Perception processing (unchanged)
    log_scale: 50.0

    # Metabolism parameters (used by policy)
    basal_rate: 2
    metabolic_sensitivity: 2.5
    max_metabolic_rate: 6
    gradient_ema_alpha: 0.9

    # Efficiency curve (used by execution, not policy)
    min_efficiency: 2.5
    max_efficiency: 3.5

    # ... rest unchanged ...
```

### Backward Compatibility

If `policy.type` is not specified, default to `"rule_based"` with parameters from existing config locations:

```python
def _create_policy(self) -> AnimalPolicy:
    """Create policy from config."""
    policy_config = self.config.get("policy", {})
    policy_type = policy_config.get("type", "rule_based")

    if policy_type == "rule_based":
        # Build config from existing parameters
        return RuleBasedPolicy(self.config)
    elif policy_type == "parameterized":
        return ParameterizedPolicy(self.config)
    elif policy_type == "network":
        return NetworkPolicy(self.config)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
```

---

## File Structure

```
tensor_beasts/
├── policy/
│   ├── __init__.py
│   ├── base.py          # PolicyInput, PolicyOutput, AnimalPolicy protocol
│   ├── rule_based.py    # RuleBasedPolicy
│   ├── parameterized.py # ParameterizedPolicy (for genetic algorithm)
│   └── network.py       # NetworkPolicy (future, for RL)
├── entities/
│   └── animal.py        # Modified to use policy
└── observations.py      # Unchanged
```

---

## Evolvable Parameters Summary

Parameters that should be evolvable for genetic algorithm:

| Parameter | Type | Range | Effect |
|-----------|------|-------|--------|
| `navigation_weights[*]` | float | [-10, 10] | Attraction/repulsion per sensory input |
| `basal_rate` | int | [0, 10] | Minimum metabolic rate |
| `metabolic_sensitivity` | float | [0, 5] | Gradient → metabolic rate scaling |
| `max_metabolic_rate` | int | [1, 15] | Ceiling on metabolic rate |
| `reproduction_threshold` | int | [100, 250] | Biomass level to reproduce |
| `survival_threshold` | int | [1, 50] | Biomass level for death |

**Not evolvable (execution details):**
- `min_efficiency`, `max_efficiency` - efficiency curve shape
- `dissipation_rate`, `dissipation_floor` - energy decay
- `base_movement_cost` - flat movement cost
- `eat_max`, `food_keys` - eating mechanics

---

## Testing Verification Plan

### Test 1: Exact Output Match

```python
def test_rule_based_policy_matches_current():
    """Verify RuleBasedPolicy produces identical outputs to current code."""
    # Setup
    world = create_test_world()
    herbivore = world.entity_dict["Herbivore"]

    # Get inputs
    obs = get_observation(...)
    gradient_ema = herbivore.gradient_ema.data.clone()

    # Run current process_observation
    old_gradient, old_direction = process_observation(obs, herbivore.config.navigation_weights)

    # Run policy
    policy = RuleBasedPolicy(herbivore.config)
    policy_input = PolicyInput.from_observation(obs, gradient_ema)
    policy_output = policy(policy_input)

    # Compare
    torch.testing.assert_close(policy_output.gradient_strength, old_gradient)
    torch.testing.assert_close(policy_output.move_direction, old_direction)
```

### Test 2: Full Simulation Match

```python
def test_simulation_unchanged():
    """Run full simulation, verify identical trajectories."""
    # Seed random state
    torch.manual_seed(42)

    # Run N steps with old code
    world_old = create_world(use_policy=False)
    for _ in range(100):
        world_old.step()

    # Reset and run with policy
    torch.manual_seed(42)
    world_new = create_world(use_policy=True)
    for _ in range(100):
        world_new.step()

    # Compare all feature tensors
    for key in world_old.td.keys():
        torch.testing.assert_close(
            world_old.td[key],
            world_new.td[key],
            msg=f"Mismatch in {key}"
        )
```

### Test 3: Parameter Round-Trip

```python
def test_parameter_get_set_roundtrip():
    """Verify get_parameters/set_parameters preserves behavior."""
    policy = RuleBasedPolicy(config)

    # Get parameters
    params = policy.get_parameters()

    # Create new policy, set parameters
    policy2 = RuleBasedPolicy(config)
    policy2.set_parameters(params)

    # Verify identical outputs
    output1 = policy(input)
    output2 = policy2(input)

    torch.testing.assert_close(output1.move_direction, output2.move_direction)
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (No Behavior Change)
1. Create `tensor_beasts/policy/` directory
2. Implement `PolicyInput`, `PolicyOutput`, `AnimalPolicy` protocol in `base.py`
3. Implement `RuleBasedPolicy` in `rule_based.py`
4. Add comprehensive tests verifying exact match

### Phase 2: Integration (No Behavior Change)
1. Add `policy` attribute to `Animal.__init__`
2. Refactor `Animal.update()` to use policy
3. Extract `_execute_metabolism`, `_execute_dissipation`, `_execute_movement`
4. Verify all tests still pass

### Phase 3: Parameterized Policy
1. Implement `ParameterizedPolicy` with `get_parameters()`/`set_parameters()`
2. Add parameter validation and clamping
3. Create tests for parameter manipulation

### Phase 4: Config Integration
1. Add `policy.type` to config schema
2. Update Pydantic models
3. Add policy factory in `Animal.__init__`
4. Update example configs

---

## Open Design Questions

1. **Gradient EMA ownership:** Should gradient EMA update be inside or outside policy?
   - Current: Outside (policy returns gradient_strength, update happens in Animal)
   - Alternative: Inside (policy has stateful EMA)
   - **Recommendation:** Outside - keeps policy stateless/pure

2. **Reproduction decision:** Should policy output explicit reproduce flag?
   - Current: Implicit (threshold check in execution)
   - Alternative: Policy decides
   - **Recommendation:** Keep implicit for now, add flag later if needed

3. **Eating policy:** Should eating order/amount be policy-controlled?
   - Current: Fixed by config
   - Alternative: Policy outputs food priorities
   - **Recommendation:** Keep fixed for now, consider for v2

4. **Move probability vs. certainty:** Should policy output deterministic direction + probability, or distribution?
   - Current: Deterministic direction + separate probability
   - Alternative: Distribution over actions
   - **Recommendation:** Keep current for compatibility, consider distribution for RL

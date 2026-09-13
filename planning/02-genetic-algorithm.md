# Feature: Genetic Algorithm System

**Status:** Planning
**Priority:** High
**Dependencies:** Centralized Animal Policy (partial)
**Blocks:** None

---

## Goal Statement

Create a genetic algorithm system that enables evolutionary dynamics in the simulation:

- Reserve multiple **slots** for each animal type (e.g., 8 herbivore slots)
- When reproduction occurs, offspring may be allocated to an empty slot (different "genome")
- Combined with centralized policy, offspring in new slots can have mutated policy parameters
- Enable genetic diversity and natural selection through differential survival/reproduction

---

## Current Architecture Analysis

### Entity Slot System (Current)

**One entity instance per species:**
```
World
├── entity_dict: {
│   "Herbivore": Herbivore(instance),  # ONE instance
│   "Predator": Predator(instance),    # ONE instance
│   ...
│ }
└── td (TensorDict):
    ├── ("herbivore", "energy"): (H, W)   # All herbivores share one 2D grid
    ├── ("herbivore", "biomass"): (H, W)
    └── ...
```

**Implications:**
- All individuals of a species share identical behavior (same policy)
- No genetic variation within species
- No way to track lineages or traits

### Reproduction System (Current)

**Mechanism:** Reproduction during movement (`animal_helpers.py:85-236`)

```python
# In perform_move():
offspring_mask = move_origin_mask & (biomass > divide_threshold)

# Resource division:
# Parent: 50% biomass, 50% energy
# Offspring: 50% biomass, 50% energy (placed in destination cell)

# Features carried:
# - id_feature: NEW random ID for offspring
# - offspring_count: Incremented for parent
# - gradient_ema: 50% of parent's value
```

**Limitations:**
- Offspring identical to parent (no mutation)
- No trait/gene storage
- No slot allocation logic

### Data Structures (Current)

**Features per entity:**
```python
class Animal(Entity):
    energy: Energy           # uint8, (H, W)
    biomass: Biomass        # uint8, (H, W)
    gradient_ema: GradientEMA  # float32, (H, W)
    scent: Scent            # float16, (H, W) - shared
    id_feature: IdFeature   # int32, (H, W)
    offspring_count: OffspringCount  # int32, (H, W)
```

**Key constraint:** All features are 2D (H, W). No mechanism for multiple "types" within a species.

---

## Proposed Architecture

### Multi-Slot Entity System

**Extend from 2D to 3D tensors:**

```
Before: ("herbivore", "energy"): (H, W)
After:  ("herbivore", "energy"): (num_slots, H, W)
```

**Entity Structure:**
```python
class Animal(Entity):
    num_slots: int = 8  # Configurable

    # Features now 3D: (num_slots, H, W)
    energy: Energy           # uint8
    biomass: Biomass        # uint8
    gradient_ema: GradientEMA  # float32
    slot_id: SlotId         # uint8 - which slot each individual belongs to

    # Per-slot policy parameters (could be separate or embedded in Policy)
    policies: List[ParameterizedPolicy]  # One per slot
```

### Slot Allocation on Reproduction

```python
def allocate_offspring_slot(
    parent_slots: torch.Tensor,      # (H, W) uint8 - slot IDs of parents
    reproduction_mask: torch.Tensor,  # (H, W) bool - which cells reproduce
    slot_populations: torch.Tensor,   # (num_slots,) int - population per slot
    mutation_probability: float = 0.1
) -> torch.Tensor:
    """
    Determine slot allocation for offspring.

    Returns:
        offspring_slots: (H, W) uint8 - slot IDs for offspring
    """
    # Find empty slots (population = 0)
    empty_slots = (slot_populations == 0).nonzero(as_tuple=True)[0]

    if len(empty_slots) == 0:
        # No empty slots: offspring inherit parent slot
        return parent_slots.clone()

    # For each reproducing cell, decide: same slot or new slot?
    offspring_slots = parent_slots.clone()

    # Random mask for mutation
    mutation_mask = torch.rand_like(reproduction_mask.float()) < mutation_probability
    mutate_cells = reproduction_mask & mutation_mask

    if mutate_cells.any():
        # Assign mutating offspring to random empty slot
        num_mutants = mutate_cells.sum().item()
        assigned_slots = empty_slots[torch.randint(len(empty_slots), (num_mutants,))]
        offspring_slots[mutate_cells] = assigned_slots

    return offspring_slots
```

### Genetic Parameter System

**Gene representation:**
```python
@dataclass
class Genome:
    """Evolvable parameters for an individual."""

    # Navigation weights (per sensory input)
    navigation_weights: Dict[str, float]

    # Metabolic parameters
    basal_rate: float
    metabolic_sensitivity: float

    # Thresholds
    reproduction_threshold: float
    survival_threshold: float

    # Movement parameters
    move_energy_threshold: float

    def to_tensor(self) -> torch.Tensor:
        """Flatten genome to tensor for mutation operations."""
        ...

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, template: 'Genome') -> 'Genome':
        """Reconstruct genome from tensor."""
        ...
```

**Per-slot genome storage:**
```python
class GeneticRegistry:
    """Manages genomes for all slots of an entity type."""

    def __init__(self, num_slots: int, base_genome: Genome):
        self.genomes = [base_genome.copy() for _ in range(num_slots)]

    def mutate(self, slot_id: int, mutation_rate: float, mutation_scale: float):
        """Apply random mutation to slot's genome."""
        genome = self.genomes[slot_id]
        tensor = genome.to_tensor()

        # Gaussian mutation
        mutation_mask = torch.rand_like(tensor) < mutation_rate
        noise = torch.randn_like(tensor) * mutation_scale
        tensor = tensor + mutation_mask * noise

        self.genomes[slot_id] = Genome.from_tensor(tensor, genome)

    def crossover(self, parent_a: int, parent_b: int) -> Genome:
        """Create child genome from two parents."""
        tensor_a = self.genomes[parent_a].to_tensor()
        tensor_b = self.genomes[parent_b].to_tensor()

        # Uniform crossover
        mask = torch.rand_like(tensor_a) > 0.5
        child_tensor = torch.where(mask, tensor_a, tensor_b)

        return Genome.from_tensor(child_tensor, self.genomes[parent_a])
```

### Integration with Policy System

```python
class GeneticPolicy:
    """Policy that uses genome parameters from GeneticRegistry."""

    def __init__(self, registry: GeneticRegistry):
        self.registry = registry

    def __call__(self, input: PolicyInput, slot_ids: torch.Tensor) -> PolicyOutput:
        """
        Compute policy output, using per-slot genomes.

        Args:
            input: Standard policy input
            slot_ids: (H, W) tensor indicating which slot each cell belongs to
        """
        # Gather genome parameters for each cell based on its slot
        params = self._gather_params_by_slot(slot_ids)

        # Compute outputs using slot-specific parameters
        return self._compute_with_params(input, params)

    def _gather_params_by_slot(self, slot_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Expand per-slot params to (H, W) tensors."""
        # Stack all slot params: (num_slots, param_dim)
        all_params = torch.stack([g.to_tensor() for g in self.registry.genomes])

        # Index by slot_id to get (H, W, param_dim)
        return all_params[slot_ids]
```

### Population Tracking

```python
class PopulationTracker:
    """Track population statistics per slot."""

    def __init__(self, num_slots: int):
        self.num_slots = num_slots
        self.history = []

    def update(self, biomass: torch.Tensor, slot_ids: torch.Tensor):
        """Record population counts per slot."""
        alive = biomass > 0
        counts = torch.zeros(self.num_slots, dtype=torch.int64)

        for slot in range(self.num_slots):
            counts[slot] = (alive & (slot_ids == slot)).sum()

        self.history.append(counts.clone())
        return counts

    def get_fitness(self, slot: int, window: int = 100) -> float:
        """Compute fitness as average population over window."""
        if len(self.history) < window:
            return 0.0

        recent = torch.stack(self.history[-window:])
        return recent[:, slot].float().mean().item()
```

---

## Slot System Design Options

### Option A: 3D Tensor Extension

**Structure:**
```python
# All features become 3D
energy: (num_slots, H, W)
biomass: (num_slots, H, W)
slot_active: (num_slots, H, W)  # Which slot is "active" at each cell
```

**Pros:**
- Clean extension of current architecture
- All operations remain vectorized
- Easy to index by slot

**Cons:**
- Memory scales with num_slots
- Must handle slot conflicts (multiple slots active at same cell)
- Significant refactor of feature system

### Option B: Slot ID per Cell

**Structure:**
```python
# Features remain 2D, add slot identifier
energy: (H, W)
biomass: (H, W)
slot_id: (H, W)  # uint8, which slot this cell belongs to
```

**Pros:**
- Minimal memory overhead
- No structural change to features
- Simpler collision handling

**Cons:**
- Policy lookup requires gather operations
- Can't have multiple individuals per cell
- Mutation requires careful handling

### Option C: Hybrid - Population Tensor + Slot ID

**Structure:**
```python
# Main features remain 2D
energy: (H, W)
biomass: (H, W)
slot_id: (H, W)

# Per-slot genomes stored separately
genomes: (num_slots, genome_dim)

# Population tracking
slot_populations: (num_slots,)  # Count per slot
```

**Pros:**
- Best of both worlds
- Minimal memory overhead
- Clear separation of concerns

**Cons:**
- Two-level lookup for some operations

**Recommendation:** Option C (Hybrid) - simplest to implement, cleanest separation.

---

## Reproduction Flow with Genetics

```python
def perform_move_with_genetics(
    self,
    direction_masks: torch.Tensor,
    reproduce_mask: torch.Tensor,
    genetic_registry: GeneticRegistry,
    mutation_prob: float = 0.1,
    mutation_scale: float = 0.1,
):
    """
    Extended perform_move with genetic offspring handling.
    """
    # 1. Standard movement
    # ... existing logic ...

    # 2. Determine offspring slots
    offspring_slots = allocate_offspring_slot(
        parent_slots=self.slot_id.data,
        reproduction_mask=reproduce_mask,
        slot_populations=self.population_tracker.current_counts(),
        mutation_probability=mutation_prob,
    )

    # 3. Apply mutations to newly occupied slots
    new_slot_mask = offspring_slots != self.slot_id.data[reproduce_mask]
    for new_slot in offspring_slots[new_slot_mask].unique():
        # Find a parent to base mutation on
        parent_slot = self.slot_id.data[reproduce_mask][new_slot_mask][0].item()

        # Copy parent genome to new slot
        genetic_registry.genomes[new_slot] = genetic_registry.genomes[parent_slot].copy()

        # Apply mutation
        genetic_registry.mutate(new_slot, mutation_scale=mutation_scale)

    # 4. Assign slot IDs to offspring
    # ... update slot_id feature for offspring cells ...
```

---

## Configuration Structure

```yaml
# conf/base/simulation.yaml
entities:
  Herbivore:
    # Genetic algorithm settings
    genetics:
      enabled: true
      num_slots: 8
      mutation_probability: 0.1  # Chance of slot change on reproduction
      mutation_scale: 0.1        # Magnitude of parameter mutations

      # Which policy parameters are evolvable
      evolvable_params:
        - navigation_weights
        - reproduction_threshold
        - basal_rate
        - metabolic_sensitivity

      # Fitness tracking
      fitness_window: 100  # Steps to average for fitness
      extinction_threshold: 10  # Min population before slot is "extinct"

    # Base genome (default parameters)
    base_genome:
      navigation_weights:
        "simpleplant:scent": 1.0
        "predator:scent": -0.5
      reproduction_threshold: 200
      basal_rate: 1.0
      metabolic_sensitivity: 1.0
```

---

## Evolutionary Dynamics

### Selection Pressure

Selection occurs naturally through:

1. **Survival:** Individuals with poor parameters die (low biomass, can't find food)
2. **Reproduction:** Individuals with good parameters reproduce more
3. **Slot competition:** When a slot goes extinct, its genome is lost

### Fitness Landscape

Fitness is implicit through population dynamics:
- Good navigation weights → find food → survive → reproduce
- Good metabolic parameters → efficient energy use → survive longer
- Good reproduction threshold → balance growth vs reproduction

### Drift and Speciation

With multiple slots, we can observe:
- **Genetic drift:** Random parameter changes accumulate
- **Adaptation:** Slots with better parameters outcompete others
- **Speciation:** Different slots may evolve different strategies (e.g., fast vs slow metabolism)

---

## Implementation Plan

### Phase 1: Slot Infrastructure

1. Add `slot_id` feature to Animal entities
2. Extend reproduction to assign slot IDs
3. Add population tracking per slot
4. Update configuration schema

### Phase 2: Genetic Registry

1. Implement `Genome` dataclass
2. Implement `GeneticRegistry` for genome management
3. Add mutation and crossover operators
4. Connect to configuration system

### Phase 3: Policy Integration

1. Create `GeneticPolicy` that uses slot-specific parameters
2. Modify `Animal.update()` to use genetic policy
3. Implement parameter gathering by slot ID

### Phase 4: Slot Allocation

1. Implement `allocate_offspring_slot()` function
2. Integrate into reproduction flow
3. Add slot extinction detection
4. Add slot re-colonization logic

### Phase 5: Monitoring and Visualization

1. Add population-per-slot metrics
2. Add genome diversity metrics
3. Extend rendering to show slot/genome differences
4. Add lineage tracking (optional)

---

## Risks and Mitigations

### Risk 1: Memory Explosion
**Risk:** 8 slots × large grid × many features = high memory.

**Mitigation:**
- Use Option C (slot ID per cell, not 3D tensors)
- Only store genomes separately (small)
- Profile memory usage early

### Risk 2: Slot Starvation
**Risk:** One slot dominates, others always empty (no diversity).

**Mitigation:**
- Tune mutation probability to maintain diversity
- Consider "immigration" to empty slots from successful ones
- Add mutation on reproduction even within same slot

### Risk 3: Performance Degradation
**Risk:** Slot-based indexing slows down computation.

**Mitigation:**
- Keep core features 2D, only genome lookup is indexed
- Use `torch.gather()` for efficient indexing
- Profile critical paths

### Risk 4: Genome Collapse
**Risk:** All genomes converge to same values (loss of diversity).

**Mitigation:**
- Track genome diversity metrics
- Increase mutation rate if diversity drops
- Consider occasional "random restart" of empty slots

### Risk 5: Incompatible with Policy System
**Risk:** Genetic system doesn't integrate cleanly with policy abstraction.

**Mitigation:**
- Design policy system with genetics in mind (Phase 1 dependency)
- `ParameterizedPolicy` must support `get_parameters()` / `set_parameters()`
- Test integration early

---

## Testing Strategy

### Unit Tests

```python
def test_slot_allocation():
    """Test offspring slot assignment logic."""

def test_genome_mutation():
    """Test mutation produces valid, bounded parameters."""

def test_population_tracking():
    """Test population counts are accurate per slot."""

def test_genetic_policy_slot_lookup():
    """Test policy uses correct genome per cell."""
```

### Integration Tests

```python
def test_reproduction_with_genetics():
    """Test full reproduction flow with slot allocation and mutation."""

def test_slot_extinction():
    """Test slot correctly marked extinct when population = 0."""

def test_diversity_maintenance():
    """Test genetic diversity maintained over many generations."""
```

### Simulation Tests

```python
def test_evolution_improves_fitness():
    """Run long simulation, verify population fitness increases."""
    # Note: This is a statistical test, may need many runs

def test_speciation_emergence():
    """Test that different slots evolve different strategies."""
```

---

## Metrics and Monitoring

### Population Metrics
- Population per slot over time
- Extinction events
- Slot diversity index

### Genetic Metrics
- Mean/variance of each parameter per slot
- Inter-slot parameter distance
- Effective population size

### Fitness Metrics
- Average lifetime per slot
- Offspring per individual per slot
- Biomass accumulation rate

---

## Open Questions

1. **Crossover:** Should reproduction between different slots allow crossover? (Requires spatial proximity detection)

2. **Migration:** Should individuals change slots during lifetime? (Simpler: only at birth)

3. **Carrying capacity:** Should there be a max population per slot? Or let natural competition handle it?

4. **Visualization:** How to visualize genetic differences? (Color coding? Separate layers?)

5. **Serialization:** How to save/load genomes for reproducibility?

---

## Success Criteria

- [ ] Slot ID feature added to Animal entities
- [ ] Reproduction assigns slots correctly
- [ ] Population tracked per slot
- [ ] Genome dataclass implemented
- [ ] GeneticRegistry with mutation working
- [ ] GeneticPolicy uses slot-specific parameters
- [ ] Slot allocation on reproduction working
- [ ] No performance regression > 15%
- [ ] Diversity maintained over 1000+ steps
- [ ] Observable evolution in test simulations

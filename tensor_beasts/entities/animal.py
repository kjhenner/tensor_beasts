from typing import Optional, Union, List, Dict, Callable, Tuple

import torch
from omegaconf import DictConfig

from tensor_beasts.entities import Entity
from tensor_beasts.registry import register_entity
from tensor_beasts.entities.helpers.animal_helpers import move
from tensor_beasts.features.shared_features import Energy, Scent
from tensor_beasts.observations import get_observation
from tensor_beasts.policy.base import Action, AnimalPolicy
from tensor_beasts.policy.rule_based import RuleBasedPolicy
from tensor_beasts.policy.parameterized import ParameterizedPolicy

from tensor_beasts.features.animal_features import IdFeature, OffspringCount, Biomass, GradientEMA, SlotId
from tensor_beasts.util import safe_sub, safe_add


@register_entity
class Animal(Entity):
    energy: Energy
    biomass: Biomass
    gradient_ema: GradientEMA
    scent: Scent
    id_feature: IdFeature
    offspring_count: OffspringCount
    slot_id: SlotId
    default_config = DictConfig({
        # Initialization
        "initial_energy": 50,
        "initial_biomass": 150,
        "init_prob": 0.001,
        "toy_init": False,

        # Eating - list of food sources tried in order
        "eat_max": 10,
        "food_keys": ["hydrodynamicplant:energy"],  # Tried in order until satiated

        # Perception: list of {key, kernel_size}
        # key format: "entity:feature"
        # kernel_size: 1 = immediate neighbors, >1 = wedge-shaped perception
        "perception": [
            {"key": "hydrodynamicplant:scent", "kernel_size": 1},
            {"key": "herbivore:scent", "kernel_size": 1},
        ],

        # Navigation weights: signed weights for movement decisions
        # key format: "entity:feature"
        # Positive = attractive, negative = repulsive
        "navigation_weights": {
            "hydrodynamicplant:scent": 1.0,
            "herbivore:scent": -0.1,
        },

        # Perception processing
        "log_scale": 10.0,  # Scale factor before log compression (higher = better gradient detection at low scent)

        # Metabolism - gradient-based
        "basal_rate": 1,              # minimum biomass burn per step
        "metabolic_sensitivity": 1.0, # biomass burn increase per unit gradient_ema
        "max_metabolic_rate": 5,      # ceiling on biomass burn per step
        "gradient_ema_alpha": 0.1,    # EMA smoothing (0.1 = slow, 0.5 = fast)

        # Metabolic efficiency curve - diminishing returns at higher metabolic rates
        # At basal_rate: efficiency = max_efficiency (resting, efficient)
        # At max_metabolic_rate: efficiency = min_efficiency (sprinting, costly)
        # Total energy always increases with rate, but each extra biomass gives less
        "min_efficiency": 2.5,        # energy per biomass at max exertion
        "max_efficiency": 3.5,        # energy per biomass at rest

        # Dissipation
        "dissipation_rate": 0.05,  # fraction of energy lost per step
        "dissipation_floor": 1,    # minimum energy loss per step

        # Movement cost
        "base_movement_cost": 1,   # flat energy cost to move

        # Death
        "survival_threshold": 5,   # die if biomass below this
        "carrion_key": None,  # where dead biomass goes (carrion layer)

        # Reproduction
        "reproduction_threshold": 200,     # biomass level to reproduce
        # Note: On reproduction, both parent and offspring receive 50% of biomass/energy

        # Genetic algorithm settings
        "genetics": {
            "enabled": False,         # Whether genetic system is active
            "num_slots": 8,           # Number of genetic slots
            "mutation_probability": 0.1,  # Chance of slot change on reproduction
            "mutation_rate": 0.1,     # Probability of mutating each parameter
            "mutation_scale": 0.1,    # Magnitude of parameter mutations
            "log_interval": 0,        # Log genetic status every N steps (0 = disabled)
        },

        # Debugging
        "verbose": False,  # Log metabolism details per step
    })

    def __init__(
        self,
        world: 'World',
        config: DictConfig
    ):
        super().__init__(world, config)

        # Set up genetic system if enabled
        self.genetic_registry = None
        genetics_config = getattr(self.config, 'genetics', None)
        if genetics_config and getattr(genetics_config, 'enabled', False):
            self._setup_genetics(genetics_config)

        # Create policy for decision-making
        self.policy: AnimalPolicy = self._create_policy()

    def _setup_genetics(self, genetics_config) -> None:
        """Initialize the genetic registry with base genome from config."""
        from tensor_beasts.genetic import Genome, GeneticRegistry

        # Create base genome from entity config
        base_genome = Genome.from_config(self.config)

        # Determine base hue from entity type (blue for herbivores, red for predators)
        class_name = self.__class__.__name__.lower()
        if 'predator' in class_name:
            base_hue = 0.0  # Red
        else:
            base_hue = 0.6  # Blue (default for herbivores)

        # Create registry
        num_slots = getattr(genetics_config, 'num_slots', 8)
        self.genetic_registry = GeneticRegistry(
            num_slots=num_slots,
            base_genome=base_genome,
            base_hue=base_hue,
        )

        # Register slot_colors in TensorDict for rendering (use tuple key for consistency)
        slot_colors_key = (class_name, "slot_colors")
        self.td.set(slot_colors_key, self.genetic_registry.slot_colors)

    def _create_policy(self) -> AnimalPolicy:
        """Create the decision-making policy from config."""
        # Use ParameterizedPolicy if genetics enabled, otherwise RuleBasedPolicy
        if self.genetic_registry is not None:
            return ParameterizedPolicy(self.config)
        return RuleBasedPolicy(self.config)

    def initialize(self):
        self.energy.initialize_data()
        self.biomass.initialize_data()
        self.gradient_ema.initialize_data()
        self.scent.initialize_data()
        self.offspring_count.initialize_data()
        self.id_feature.initialize_data()
        self.slot_id.initialize_data()

        energy = self.energy.data
        biomass = self.biomass.data

        if self.config.toy_init:
            # Single entity at random position
            h = torch.randint(0, self.world.size[0], (1,)).item()
            w = torch.randint(0, self.world.size[1], (1,)).item()
            energy[h, w] = self.config.initial_energy
            biomass[h, w] = self.config.initial_biomass
        else:
            spawn_mask = (
                torch.rand_like(energy, dtype=torch.float32) < self.config.init_prob
            )
            energy[:] = spawn_mask * self.config.initial_energy
            biomass[:] = spawn_mask * self.config.initial_biomass

        # Update genetic registry populations if enabled
        if self.genetic_registry is not None:
            alive_mask = biomass >= self.config.survival_threshold
            self.genetic_registry.update_populations(self.slot_id.data, alive_mask)

    def _compute_metabolic_rate(
        self,
        gradient_ema: torch.Tensor,
        biomass: torch.Tensor
    ) -> torch.Tensor:
        """Compute metabolic rate (biomass burn) based on gradient EMA and biomass.

        Higher gradient = more active pursuit = higher metabolism.
        Higher biomass = more fit = higher max metabolic rate.
        Low biomass animals can't sprint - they slow down before dying.

        Biomass scaling uses survival_threshold as baseline:
        - At survival_threshold: can only do basal metabolism
        - At 255: can reach full max_rate

        Returns biomass units to burn per step.
        """
        basal = self.config.basal_rate
        sensitivity = self.config.metabolic_sensitivity
        max_rate = self.config.max_metabolic_rate
        threshold = self.config.survival_threshold

        # Biomass modulates the maximum achievable metabolic rate
        # Scale from survival_threshold (0% capacity) to 255 (100% capacity)
        biomass_range = 255.0 - threshold
        biomass_above_threshold = (biomass.float() - threshold).clamp(min=0)
        biomass_fraction = biomass_above_threshold / biomass_range
        biomass_fraction = biomass_fraction.clamp(0, 1)

        effective_max_rate = basal + (max_rate - basal) * biomass_fraction

        # Direct sensitivity: each unit of gradient adds to metabolic rate
        rate = basal + gradient_ema * sensitivity

        # Clamp to biomass-limited max rate
        rate = torch.min(rate, effective_max_rate)

        return rate

    def _compute_efficiency(self, metabolic_rate: torch.Tensor) -> torch.Tensor:
        """Compute metabolic efficiency based on current metabolic rate.

        Efficiency decreases as metabolic rate increases:
        - At basal_rate (resting): max_efficiency
        - At max_metabolic_rate (sprinting): min_efficiency

        This models the trade-off between aerobic (efficient) and
        anaerobic (inefficient) metabolism during high exertion.
        """
        basal = self.config.basal_rate
        max_rate = self.config.max_metabolic_rate
        min_eff = self.config.min_efficiency
        max_eff = self.config.max_efficiency

        # Normalize rate to [0, 1] where 0 = basal, 1 = max
        # Avoid division by zero if basal == max_rate
        rate_range = max_rate - basal
        if rate_range <= 0:
            return torch.full_like(metabolic_rate, max_eff)

        normalized_rate = (metabolic_rate - basal) / rate_range
        normalized_rate = normalized_rate.clamp(0, 1)

        # Linear interpolation: max_eff at 0, min_eff at 1
        efficiency = max_eff - normalized_rate * (max_eff - min_eff)

        return efficiency

    def _handle_death(self, dead: torch.Tensor):
        """Handle death: transfer biomass to carrion, zero all features.

        Note: Only processes positions where an entity actually existed (biomass > 0).
        Empty positions (biomass=0) are not "dead" - they never had an entity.
        SharedFeatures (like scent) are NOT zeroed because they represent a field
        that persists independently of entity presence.
        """
        # Only consider positions with actual entities, not empty cells
        biomass = self.biomass.data
        actually_dead = dead & (biomass > 0)

        if not actually_dead.any():
            return

        # Transfer biomass to carrion layer
        if self.config.carrion_key is not None:
            try:
                carrion = self.td.get(self.config.carrion_key)
                if carrion is not None:
                    # Add dead animal's biomass to carrion
                    dead_biomass = (biomass * actually_dead).to(torch.uint8)
                    carrion += dead_biomass
                    carrion.clamp_(max=255)
            except KeyError:
                pass  # No carrion feature, biomass just disappears

        # Zero non-shared features at dead positions
        # SharedFeatures (like scent) represent fields that persist independently
        from tensor_beasts.features.feature import SharedFeature
        alive = ~actually_dead
        for feature in self.features():
            if not isinstance(feature, SharedFeature):
                feature.data *= alive

    def _log_metabolism(self, positions, label, **values):
        """Log metabolism values for entities at given positions."""
        if not self.config.verbose:
            return
        for y, x in positions:
            def fmt_val(k, v):
                if hasattr(v, '__getitem__'):
                    val = v[y, x].item()
                    # Show 2 decimal places for values in [0,1] range, integers otherwise
                    if 0 <= val <= 1 and val != int(val):
                        return f"{k}={val:.2f}"
                    return f"{k}={val:.0f}"
                return f"{k}={v}"
            vals = ", ".join(fmt_val(k, v) for k, v in values.items())
            print(f"  [{self.__class__.__name__}@({x},{y})] {label}: {vals}")

    def update(self, action: Optional[torch.Tensor] = None):
        """
        Main update loop for animal entities.

        Uses the policy to make all decisions, then executes them.

        Args:
            action: Optional external action override (for RL). If provided,
                   overrides the policy's move_direction output.
        """
        # Capture external action before policy call produces local 'action' variable
        external_action = action
        biomass = self.biomass.data
        energy = self.energy.data
        gradient_ema = self.gradient_ema.data

        # Find living entity positions for logging
        verbose = self.config.verbose
        if verbose:
            alive_mask = biomass >= self.config.survival_threshold
            positions = list(zip(*torch.where(alive_mask)))
            if positions:
                print(f"\n=== {self.__class__.__name__} Step {self.world.step} ===")
                self._log_metabolism(positions, "START", biomass=biomass, energy=energy, gradient_ema=gradient_ema)

        # Step 1: Death check - biomass < survival_threshold
        dead = biomass < self.config.survival_threshold
        self._handle_death(dead)

        # Step 2: Build observation (complete input to policy)
        perception = [(p.key, p.kernel_size) for p in self.config.perception]
        obs = get_observation(
            td=self.td,
            perception=perception,
            energy=energy,
            biomass=biomass,
            gradient_ema=gradient_ema,
            survival_threshold=self.config.survival_threshold,
            log_scale=self.config.log_scale,
            step=self.world.step,
        )

        # Step 3: Run policy to get all decisions (includes gradient EMA update)
        alive = obs.alive_mask
        action = self.policy(obs)

        # Step 4: Write back updated gradient EMA from policy
        gradient_ema[:] = action.gradient_ema

        if verbose and positions:
            self._log_metabolism(
                positions, "GRADIENT",
                gradient_ema=action.gradient_ema,
                metabolic_rate=action.metabolic_rate,
                efficiency=self._compute_efficiency(action.metabolic_rate)
            )

        # Step 5: Execute metabolism using action's metabolic_rate
        self._execute_metabolism(action.metabolic_rate, verbose, positions if verbose else None)

        # Step 6: Execute energy dissipation
        self._execute_dissipation(verbose, positions if verbose else None)

        # Step 7: Execute movement using action's direction and probability
        # External action overrides policy direction (for RL)
        move_direction = external_action if external_action is not None else action.move_direction
        did_move = self._execute_movement(
            direction=move_direction,
            move_probability=action.move_probability,
            verbose=verbose,
            positions=positions if verbose else None,
        )

        # After movement, find new positions for logging
        if verbose:
            new_alive_mask = biomass >= self.config.survival_threshold
            new_positions = list(zip(*torch.where(new_alive_mask)))
            if positions:
                for y, x in positions:
                    if did_move[y, x]:
                        print(f"  [{self.__class__.__name__}@({x},{y})] MOVED to new position")
            positions = new_positions

        if verbose and positions:
            self._log_metabolism(positions, "POST_MOVE", biomass=biomass, energy=energy)

        # Step 8: Eating (fills biomass, not energy)
        biomass_before_eat = biomass.clone() if verbose else None
        self._eat()

        if verbose and positions:
            eaten = biomass - biomass_before_eat
            self._log_metabolism(positions, "EAT", eaten=eaten, biomass=biomass)
            self._log_metabolism(positions, "END", biomass=biomass, energy=energy)

        # Step 9: Emit scent (based on biomass); diffusion handled by World
        self.scent.emit(self.world.step)

        # Step 10: Update genetic registry populations if enabled
        if self.genetic_registry is not None:
            alive = biomass >= self.config.survival_threshold
            self.genetic_registry.update_populations(self.slot_id.data, alive)

            # Log genetic status at configured interval
            log_interval = getattr(self.config.genetics, 'log_interval', 0)
            if log_interval > 0 and self.world.step % log_interval == 0:
                self.genetic_registry.log_status(
                    entity_name=self.__class__.__name__,
                    step=self.world.step
                )

    def _execute_metabolism(
        self,
        metabolic_rate: torch.Tensor,
        verbose: bool = False,
        positions: Optional[List[Tuple[int, int]]] = None
    ):
        """
        Execute metabolism: convert biomass to energy.

        Args:
            metabolic_rate: (H, W) float - biomass to burn per cell
            verbose: Whether to log metabolism details
            positions: Positions to log (if verbose)
        """
        biomass = self.biomass.data
        energy = self.energy.data

        # Compute efficiency (decreases at higher metabolic rates)
        efficiency = self._compute_efficiency(metabolic_rate)

        # Burn biomass, limited by available biomass
        biomass_burned = torch.min(metabolic_rate, biomass.float()).to(torch.uint8)

        # Energy gained = biomass * efficiency
        energy_gained = (biomass_burned.float() * efficiency).to(torch.uint8)

        safe_add(energy, energy_gained)
        safe_sub(biomass, biomass_burned)

        if verbose and positions:
            self._log_metabolism(
                positions, "METABOLISM",
                biomass_burned=biomass_burned,
                energy_gained=energy_gained,
                biomass=biomass,
                energy=energy
            )

    def _execute_dissipation(
        self,
        verbose: bool = False,
        positions: Optional[List[Tuple[int, int]]] = None
    ):
        """
        Execute energy dissipation.

        Args:
            verbose: Whether to log dissipation details
            positions: Positions to log (if verbose)
        """
        energy = self.energy.data

        dissipation = (energy.float() * self.config.dissipation_rate).to(torch.uint8)
        dissipation = torch.clamp(dissipation, min=self.config.dissipation_floor)
        safe_sub(energy, dissipation)

        if verbose and positions:
            self._log_metabolism(positions, "DISSIPATION", dissipation=dissipation, energy=energy)

    def _execute_movement(
        self,
        direction: torch.Tensor,
        move_probability: torch.Tensor,
        verbose: bool = False,
        positions: Optional[List[Tuple[int, int]]] = None,
    ) -> torch.Tensor:
        """
        Execute movement and reproduction.

        Args:
            direction: (H, W) long - movement direction per cell
            move_probability: (H, W) float - probability of attempting move
            verbose: Whether to log movement details
            positions: Positions to log (if verbose)

        Returns:
            did_move: (H, W) bool - which cells actually moved
        """
        energy = self.energy.data
        biomass = self.biomass.data
        offspring_count = self.offspring_count.data
        id_feature = self.id_feature.data
        gradient_ema = self.gradient_ema.data
        slot_id = self.slot_id.data
        random = self.world.td.get("random")

        # Stochastic movement based on probability
        move_mask = torch.rand_like(energy, dtype=torch.float32) < move_probability

        if verbose and positions:
            self._log_metabolism(positions, "MOVE_PROB", move_prob=move_probability, move_mask=move_mask.float())

        # Movement cost (flat cost per move)
        movement_cost = torch.full_like(energy, self.config.base_movement_cost)

        # Prepare offspring slot assignment function
        offspring_slot_fn = self._make_offspring_slot_fn()

        # Execute move with reproduction
        did_move = move(
            primary_feature=energy,
            target=None,  # Direction pre-computed
            target_weights=None,
            divide_threshold=self.config.reproduction_threshold,
            divide_feature=biomass,
            divide_fn_self=lambda x: (x.float() * 0.5).to(x.dtype),
            divide_fn_offspring=lambda x: (x.float() * 0.5).to(x.dtype),
            carried_features_self=[offspring_count, id_feature, biomass, gradient_ema, slot_id],
            carried_feature_fns_self=[
                lambda x: safe_add(x, 1),
                lambda x: x,
                lambda x: (x.float() * 0.5).to(x.dtype),
                lambda x: x,
                lambda x: x,  # slot_id unchanged for parent
            ],
            carried_features_offspring=[id_feature, biomass, gradient_ema, slot_id],
            carried_feature_fns_offspring=[
                lambda x: random,
                lambda x: (x.float() * 0.5).to(x.dtype),
                lambda x: x * 0.5,
                offspring_slot_fn,  # slot assignment for offspring
            ],
            agent_action=direction,
            move_mask=move_mask,
            move_cost=movement_cost,
        )

        return did_move

    def _make_offspring_slot_fn(self) -> Callable:
        """
        Create function for assigning slots to offspring.

        If genetics not enabled, offspring inherit parent slot.
        If enabled, offspring may be assigned to empty slots with mutation.

        Uses lazy colonization: only creates genomes for slots that actually
        receive offspring, avoiding wasted work.
        """
        if self.genetic_registry is None:
            # No genetics: offspring inherit parent slot
            return lambda x: x

        genetics_config = self.config.genetics
        mutation_prob = getattr(genetics_config, 'mutation_probability', 0.1)
        mutation_rate = getattr(genetics_config, 'mutation_rate', 0.1)
        mutation_scale = getattr(genetics_config, 'mutation_scale', 0.1)

        def assign_offspring_slot(parent_slots: torch.Tensor) -> torch.Tensor:
            """Assign slots to offspring, potentially mutating to new slots."""
            # Early exit: check mutation mask BEFORE any slot queries
            mutation_mask = torch.rand_like(parent_slots.float()) < mutation_prob
            if not mutation_mask.any():
                return parent_slots  # No clone needed if no mutations

            offspring_slots = parent_slots.clone()

            # Now query slots (only when we know mutations will occur)
            empty_slots = self.genetic_registry.get_empty_slots()

            if len(empty_slots) > 0:
                # Assign mutating offspring to random empty slots
                # Only generate random indices for cells that are actually mutating
                mutating_indices = mutation_mask.nonzero(as_tuple=True)
                num_mutating = len(mutating_indices[0])
                random_slot_indices = torch.randint(0, len(empty_slots), (num_mutating,))
                new_slots = empty_slots[random_slot_indices]
                offspring_slots[mutating_indices] = new_slots.to(offspring_slots.dtype)

                # Lazy colonization: only create genomes for slots that were actually assigned
                used_empty_slots = new_slots.unique()
                occupied_slots = self.genetic_registry.get_occupied_slots()

                if len(occupied_slots) > 0:
                    for slot in used_empty_slots:
                        slot_int = slot.item()
                        # Only colonize if this slot is still empty (population == 0)
                        if self.genetic_registry.populations[slot_int] == 0:
                            source = occupied_slots[torch.randint(len(occupied_slots), (1,))].item()
                            self.genetic_registry.colonize_slot(
                                slot_int, source, mutation_rate, mutation_scale
                            )
            else:
                # No empty slots - assign to random occupied slots
                occupied_slots = self.genetic_registry.get_occupied_slots()
                if len(occupied_slots) > 0:
                    mutating_indices = mutation_mask.nonzero(as_tuple=True)
                    num_mutating = len(mutating_indices[0])
                    random_slot_indices = torch.randint(0, len(occupied_slots), (num_mutating,))
                    new_slots = occupied_slots[random_slot_indices]
                    offspring_slots[mutating_indices] = new_slots.to(offspring_slots.dtype)

            return offspring_slots

        return assign_offspring_slot

    def _eat_from(self, food: torch.Tensor, amount: int) -> torch.Tensor:
        """
        Attempt to eat from a food source.

        Args:
            food: Food tensor to eat from
            amount: Max amount to eat

        Returns:
            Amount actually eaten (tensor)
        """
        biomass = self.biomass.data
        alive = biomass >= self.config.survival_threshold

        old_food = food.clone()
        safe_sub(food, alive.to(torch.uint8) * amount)
        eaten = old_food - food

        # Food goes 100% to biomass
        safe_add(biomass, eaten)
        return eaten

    def _eat(self):
        """
        Eating fills biomass (not energy directly).

        Tries food sources in config order until satiated.
        """
        eat_max = self.config.eat_max
        total_eaten = torch.zeros_like(self.biomass.data, dtype=torch.uint8)

        for food_key in self.config.food_keys:
            food = self.td.get(food_key)
            if food is None:
                continue

            remaining_appetite = eat_max - total_eaten
            still_hungry = remaining_appetite > 0
            if not still_hungry.any():
                break  # Everyone is full

            # Eat up to remaining appetite from this source
            eaten = self._eat_from(food, remaining_appetite.to(torch.uint8))
            total_eaten = total_eaten + eaten


@register_entity
class Herbivore(Animal):
    """Herbivore that eats plants."""
    default_config = DictConfig({
        "food_keys": ["hydrodynamicplant:energy"],
        "perception": [
            {"key": "hydrodynamicplant:scent", "kernel_size": 1},
            {"key": "herbivore:scent", "kernel_size": 1},
        ],
        "navigation_weights": {
            "hydrodynamicplant:scent": 1.0,
            "herbivore:scent": -0.1,
        },
    })


@register_entity
class Predator(Animal):
    """Predator that eats herbivores."""
    default_config = DictConfig({
        # Predators have higher base/max metabolism
        "basal_rate": 2,
        "max_metabolic_rate": 6,

        "food_keys": ["herbivore:energy"],
        "perception": [
            {"key": "herbivore:scent", "kernel_size": 1},
            {"key": "predator:scent", "kernel_size": 1},
        ],
        "navigation_weights": {
            "herbivore:scent": 1.0,
            "predator:scent": -0.1,
        },
    })

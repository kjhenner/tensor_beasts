from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Callable, Tuple

import torch
from omegaconf import DictConfig

from tensor_beasts.entities import Entity
from tensor_beasts.registry import register_entity
from tensor_beasts.entities.helpers.animal_helpers import move
from tensor_beasts.features.shared_features import Energy, Scent, ENERGY_MAX
from tensor_beasts.observations import get_observation
from tensor_beasts.policy.base import Action, AnimalPolicy
from tensor_beasts.policy.metabolism import clamp_metabolic_rate, effective_max_metabolic_rate
from tensor_beasts.policy.rule_based import RuleBasedPolicy
from tensor_beasts.policy.parameterized import ParameterizedPolicy

from tensor_beasts.features.animal_features import Memory, IdFeature, OffspringCount, Biomass, GradientEMA, SlotId


@dataclass
class TransitionInfo:
    """Where each acting individual ended up during one step.

    This is what makes per-individual reinforcement learning trajectories
    possible. An individual is identified by the cell it occupied when it chose
    its action; ``successor`` says which cell that same individual occupies once
    the step is over, so a learner can follow it through time and bootstrap a
    value from the right place.

    Held as an attribute rather than written into the world TensorDict, so that
    turning tracking on cannot change simulation state or the golden hashes.

    Attributes:
        acted: (H, W) bool, cells that held an individual which chose an action.
        successor: (H, W) int64, flat index of that individual's cell once the
            step is over, or -1 where nothing acted. An individual that stayed
            put, or whose move was blocked, is its own successor.
        reproduced: (H, W) bool, indexed by the cell the individual acted from,
            true where it divided this step.
    """

    acted: torch.Tensor
    successor: torch.Tensor
    reproduced: torch.Tensor
    # (H, W) int64, flat index of the cell the offspring occupies after this
    # step, or -1 where nothing divided. Indexed, like every other field here,
    # by the cell the parent acted from. Reproduction leaves the offspring in
    # exactly that cell while the parent moves on to its successor, so the edge
    # is free: it is the parent's own origin. A learner that credits an
    # individual for its descendants needs this edge, and without it division
    # is only ever a cost, since it halves the parent's biomass.
    offspring: Optional[torch.Tensor] = None
    # (H, W) float32, biomass gained by eating this step, indexed by the cell
    # the individual occupies AFTER moving, i.e. its successor cell. What
    # foraging actually is, as distinct from net biomass change, which also
    # counts what metabolism burned and would punish using the throttle.
    eaten: Optional[torch.Tensor] = None
    # (H, W) float32, biomass burned by metabolism this step, indexed by the
    # cell the individual acted from, since metabolism runs before movement.
    # With ``eaten`` and the reserve lost at death this is what makes the sum
    # of every individual's stock change equal the species' stock change
    # exactly (planning/11).
    burned: Optional[torch.Tensor] = None


@register_entity
class Animal(Entity):
    # Most recent TransitionInfo, or None when track_transitions is off.
    # Deliberately unannotated: EntityMeta scans annotations to collect features.
    last_transition = None
    energy: Energy
    biomass: Biomass
    gradient_ema: GradientEMA
    scent: Scent
    id_feature: IdFeature
    offspring_count: OffspringCount
    slot_id: SlotId
    memory: Memory
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

        # Reinforcement learning support
        # Records, each step, which individuals acted, where each ended up, and
        # which reproduced. Needed to stitch per-individual trajectories: the id
        # feature cannot do it, because offspring draw their id from the world's
        # uint8 random field, so only 256 distinct ids ever exist.
        "track_transitions": False,

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
        self.memory.initialize_data()

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

        # Biomass-limited maximum, shared with the policies and the override.
        effective_max_rate = effective_max_metabolic_rate(biomass, basal, max_rate, threshold)

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

        Every per-animal feature is zeroed wherever ``dead`` is true, energy
        included; scent is the one field that persists, because it is a
        diffusing trace rather than a property of the animal.
        """
        # Every cell that is not alive is cleaned, not only cells that still
        # hold biomass. The guard used to be `dead & (biomass > 0)`, and it had
        # a hole: a predator bite that takes the last unit leaves biomass at
        # exactly zero, so the prey never counted as dead and its id, gradient
        # EMA, offspring count, slot and memory sat on an empty cell until the
        # next arrival was summed into them (perform_move adds arrivals).
        # Cleaning empty cells is a no-op on their zeros, so the cheap and
        # correct rule is the same one: nothing but a living animal holds
        # per-cell state.
        biomass = self.biomass.data

        # Transfer biomass to carrion layer
        if self.config.carrion_key is not None:
            try:
                carrion = self.td.get(self.config.carrion_key)
                if carrion is not None:
                    # Add dead animal's biomass to carrion, exactly.
                    carrion += biomass * dead
                    carrion.clamp_(max=ENERGY_MAX)
            except KeyError:
                pass  # No carrion feature, biomass just disappears

        # Zero every per-animal feature at dead positions. Scent is the one
        # field that persists after death: it is a diffusing trace, not a
        # property of the animal. Energy is stored as a SharedFeature for
        # tensor-layout reasons only; the slice is this entity's own, and
        # leaving it on a dead cell made a ghost that kept moving, paid move
        # costs, and blocked living animals through the clearance kernel until
        # dissipation drained it, 35 steps for a herbivore and over 200 for a
        # predator. The energy of a dead animal is destroyed; its biomass is
        # what becomes carrion.
        alive = ~dead
        for feature in self.features():
            if feature is self.scent:
                continue
            # Trailing channel axes (memory is (H, W, K)) broadcast against
            # the 2-D mask only if the mask gains matching trailing dims.
            extra = feature.data.ndim - alive.ndim
            feature.data *= alive.reshape(*alive.shape, *([1] * extra)) if extra > 0 else alive

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

    @staticmethod
    def _split_external_action(action):
        """Return (direction, metabolic_rate, memory) from an external action override.

        Two forms are accepted. A bare (H, W) int64 tensor is a direction
        override and nothing else, which is what the first learned policies
        sent and what must keep working unchanged. A mapping (dict or
        TensorDict) with key ``"direction"`` and optionally ``"metabolic_rate"``
        overrides the throttle as well. Either entry may be None.
        """
        if action is None:
            return None, None, None
        if isinstance(action, torch.Tensor):
            return action, None, None
        return (
            action.get("direction", None),
            action.get("metabolic_rate", None),
            action.get("memory", None),
        )

    def update(self, action: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None):
        """
        Main update loop for animal entities.

        Uses the policy to make all decisions, then executes them.

        Args:
            action: Optional external action override (for RL). Either a bare
                (H, W) int64 direction tensor, which overrides only the
                policy's move_direction, or a mapping with ``"direction"``
                (H, W) int64 and optionally ``"metabolic_rate"`` (H, W)
                float32, the desired biomass to burn this step. A learned rate
                is clamped to ``[basal_rate, effective_max(biomass)]`` through
                the same biomass cap the rule-based policy applies: an animal
                cannot burn what it does not carry, whoever sets the throttle.
        """
        # Capture external action before policy call produces local 'action' variable
        external_action, external_rate, external_memory = self._split_external_action(action)
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

        # A learned policy's memory write lands here, before movement, so the
        # value carried to the individual's next cell is the one it just wrote.
        # The rule-based policy never writes memory. See features Memory.
        if external_memory is not None and self.memory.size > 0:
            # Only living cells hold memory. An unmasked write would leave stale
            # values on empty cells, and movement ADDS an arriving animal's
            # carried features onto its destination, so a stale value would be
            # summed into whoever moved there next.
            new_memory = external_memory.reshape(self.memory.data.shape).to(self.memory.data.dtype)
            self.memory.data[:] = torch.where(alive.unsqueeze(-1), new_memory, torch.zeros_like(new_memory))

        # External metabolic rate overrides the policy's, subject to the same
        # biomass cap the policy applies to itself (see policy/metabolism.py).
        # obs.biomass is what the policy capped against, so the two agree.
        metabolic_rate = action.metabolic_rate
        if external_rate is not None:
            metabolic_rate = clamp_metabolic_rate(
                external_rate.reshape(biomass.shape),
                obs.biomass,
                self.config.basal_rate,
                self.config.max_metabolic_rate,
                self.config.survival_threshold,
            )

        if verbose and positions:
            self._log_metabolism(
                positions, "GRADIENT",
                gradient_ema=action.gradient_ema,
                metabolic_rate=metabolic_rate,
                efficiency=self._compute_efficiency(metabolic_rate)
            )

        # Step 5: Execute metabolism using the chosen metabolic_rate
        burned = self._execute_metabolism(metabolic_rate, verbose, positions if verbose else None)

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
            acting_mask=alive if self.config.track_transitions else None,
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
        biomass_before_eat = biomass.clone() if self.config.track_transitions else None
        self._eat()
        if self.last_transition is not None and biomass_before_eat is not None:
            self.last_transition.eaten = (biomass.float() - biomass_before_eat.float()).clamp(min=0)
            self.last_transition.burned = burned.float()

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
    ) -> torch.Tensor:
        """
        Execute metabolism: convert biomass to energy.

        Args:
            metabolic_rate: (H, W) float - biomass to burn per cell
            verbose: Whether to log metabolism details
            positions: Positions to log (if verbose)

        Returns:
            (H, W) biomass burned at each cell.
        """
        biomass = self.biomass.data
        energy = self.energy.data

        # Compute efficiency (decreases at higher metabolic rates)
        efficiency = self._compute_efficiency(metabolic_rate)

        # Burn biomass, limited by available biomass. Exact: a rate of 2.5
        # costs 2.5. When this was uint8 the burn truncated (2.5 cost 2) and
        # the energy conversion truncated too, which made the throttle a cliff
        # rather than a trade-off: at exactly the basal rate an animal got
        # 2 * 3.5 = 7 energy, while at 2.05 it got int(6.975) = 6 for the same
        # 2 biomass, a 14% tax on every setting except one exact value.
        biomass_burned = torch.minimum(metabolic_rate.to(biomass.dtype), biomass)
        energy_gained = biomass_burned * efficiency

        energy += energy_gained
        energy.clamp_(max=ENERGY_MAX)
        biomass -= biomass_burned

        if verbose and positions:
            self._log_metabolism(
                positions, "METABOLISM",
                biomass_burned=biomass_burned,
                energy_gained=energy_gained,
                biomass=biomass,
                energy=energy
            )
        return biomass_burned

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

        dissipation = torch.clamp(energy * self.config.dissipation_rate, min=self.config.dissipation_floor)
        energy -= dissipation
        energy.clamp_(min=0)

        if verbose and positions:
            self._log_metabolism(positions, "DISSIPATION", dissipation=dissipation, energy=energy)

    def _execute_movement(
        self,
        direction: torch.Tensor,
        move_probability: torch.Tensor,
        verbose: bool = False,
        positions: Optional[List[Tuple[int, int]]] = None,
        acting_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Execute movement and reproduction.

        Args:
            direction: (H, W) long - movement direction per cell
            move_probability: (H, W) float - probability of attempting move
            verbose: Whether to log movement details
            positions: Positions to log (if verbose)
            acting_mask: (H, W) bool - individuals that chose an action this
                step. When given, a TransitionInfo is recorded on the entity.

        Returns:
            did_move: (H, W) bool - which cells actually moved
        """
        energy = self.energy.data
        biomass = self.biomass.data
        offspring_count = self.offspring_count.data
        id_feature = self.id_feature.data
        gradient_ema = self.gradient_ema.data
        slot_id = self.slot_id.data
        # Each memory channel rides along as its own carried 2-D slice (a view).
        memory_slices = [self.memory.data[..., k] for k in range(self.memory.size)]
        random = self.world.td.get("random")

        # Stochastic movement based on probability
        move_mask = torch.rand_like(energy, dtype=torch.float32) < move_probability

        if verbose and positions:
            self._log_metabolism(positions, "MOVE_PROB", move_prob=move_probability, move_mask=move_mask.float())

        # Movement cost (flat cost per move)
        movement_cost = torch.full_like(energy, self.config.base_movement_cost)

        # Reproduction is decided inside perform_move against biomass as it
        # stands right now, after metabolism and before eating, so capture it
        # here to reconstruct that same decision.
        biomass_at_move = biomass.clone() if acting_mask is not None else None

        # Prepare offspring slot assignment function
        offspring_slot_fn = self._make_offspring_slot_fn()

        # Execute move with reproduction
        did_move = move(
            primary_feature=energy,
            target=None,  # Direction pre-computed
            target_weights=None,
            divide_threshold=self.config.reproduction_threshold,
            divide_feature=biomass,
            divide_fn_self=lambda x: x * 0.5,
            divide_fn_offspring=lambda x: x * 0.5,
            carried_features_self=[offspring_count, id_feature, biomass, gradient_ema, slot_id, *memory_slices],
            carried_feature_fns_self=[
                lambda x: x + 1,  # must be pure; see perform_move
                lambda x: x,
                lambda x: x * 0.5,
                lambda x: x,
                lambda x: x,  # slot_id unchanged for parent
                *([lambda x: x] * len(memory_slices)),  # memory travels unchanged
            ],
            carried_features_offspring=[offspring_count, id_feature, biomass, gradient_ema, slot_id, *memory_slices],
            carried_feature_fns_offspring=[
                # A newborn has no offspring. The origin cell is not vacated
                # on division, so without this the offspring kept the parent's
                # count as it stood before the parent's own increment.
                lambda x: torch.zeros_like(x),
                lambda x: random,
                lambda x: x * 0.5,
                lambda x: x * 0.5,
                offspring_slot_fn,  # slot assignment for offspring
                *([lambda x: x] * len(memory_slices)),  # offspring inherit a copy
            ],
            agent_action=direction,
            move_mask=move_mask,
            # Presence for the clearance check is anything that carries
            # biomass, not only anything with energy. The two agree for every
            # living animal in the shipped configs, but energy is the mobile
            # currency and can in principle reach zero on a living cell, and
            # a mover landing on such a cell would merge the two animals.
            obstacle_mask=(biomass > 0).to(torch.uint8),
            move_cost=movement_cost,
        )
        # Several movers can land on one cell; perform_move saturates energy
        # itself, biomass is a carried feature and has the same ceiling.
        biomass.clamp_(max=ENERGY_MAX)

        if acting_mask is not None:
            self._record_transition(direction, did_move, acting_mask, biomass_at_move)

        return did_move

    def _record_transition(
        self,
        direction: torch.Tensor,
        did_move: torch.Tensor,
        acting_mask: torch.Tensor,
        biomass_at_move: torch.Tensor,
    ) -> None:
        """Record where each acting individual ended up. See TransitionInfo.

        Direction encoding matches pad_matrix: 1 moves to the row above, 2 to
        the row below, 3 one column left, 4 one column right.
        """
        height, width = did_move.shape[-2:]
        # Indices are per world, not per batch: cell (h, w) of every world is
        # h * width + w, and a consumer gathers within one world at a time by
        # reshaping to (..., H * W). The alternative, offsetting world b by
        # b * H * W, would make a bare `.reshape(-1)` in a consumer appear to
        # work while silently gathering across worlds, which is the failure
        # this whole refactor has to avoid. See planning/07-batched-worlds.md.
        flat = torch.arange(height * width, device=did_move.device).reshape(height, width)
        flat = flat.expand_as(did_move)

        moved = did_move.bool()
        chosen = direction.reshape(did_move.shape).long()
        successor = flat.clone()
        for code, offset in ((1, -width), (2, width), (3, -1), (4, 1)):
            successor = torch.where(moved & (chosen == code), flat + offset, successor)
        # Moves off the edge are already blocked by the clearance check, which
        # treats the boundary as occupied; clamp anyway so a bad index can never
        # escape into a gather.
        successor = successor.clamp_(0, height * width - 1)

        reproduced = moved & (biomass_at_move > self.config.reproduction_threshold)
        divided = reproduced & acting_mask

        self.last_transition = TransitionInfo(
            acted=acting_mask.clone(),
            successor=torch.where(acting_mask, successor, torch.full_like(successor, -1)),
            reproduced=divided,
            # An individual that divides leaves its offspring behind in the cell
            # it is vacating, which is `flat`, the cell it acted from.
            offspring=torch.where(divided, flat, torch.full_like(flat, -1)),
        )

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

    def _eat_from(self, food: torch.Tensor, amount: torch.Tensor) -> torch.Tensor:
        """
        Attempt to eat from a food source.

        Args:
            food: Food tensor to eat from, on the 0..255 scale
            amount: (H, W) max amount to eat per cell

        Returns:
            Amount actually eaten (tensor)
        """
        biomass = self.biomass.data
        alive = biomass >= self.config.survival_threshold

        eaten = torch.minimum(food, alive * amount).to(food.dtype)
        food -= eaten

        # Food goes 100% to biomass. What does not fit above the ceiling is
        # still taken from the food, as it always was.
        biomass += eaten
        biomass.clamp_(max=ENERGY_MAX)
        return eaten

    def _eat(self):
        """
        Eating fills biomass (not energy directly).

        Tries food sources in config order until satiated.
        """
        eat_max = self.config.eat_max
        total_eaten = torch.zeros_like(self.biomass.data)

        for food_key in self.config.food_keys:
            food = self.td.get(food_key)
            if food is None:
                continue

            remaining_appetite = eat_max - total_eaten
            still_hungry = remaining_appetite > 0
            if not still_hungry.any():
                break  # Everyone is full

            # Eat up to remaining appetite from this source
            eaten = self._eat_from(food, remaining_appetite)
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

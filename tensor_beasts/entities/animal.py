from typing import Optional, Union, List, Dict, Callable, Tuple

import torch
from omegaconf import DictConfig

from tensor_beasts.entities import Entity
from tensor_beasts.registry import register_entity
from tensor_beasts.entities.helpers.animal_helpers import move
from tensor_beasts.features.shared_features import Energy, Scent
from tensor_beasts.observations import get_observation, process_observation

from tensor_beasts.features.animal_features import IdFeature, OffspringCount, Biomass, GradientEMA
from tensor_beasts.util import safe_sub, safe_add


@register_entity
class Animal(Entity):
    energy: Energy
    biomass: Biomass
    gradient_ema: GradientEMA
    scent: Scent
    id_feature: IdFeature
    offspring_count: OffspringCount
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

        # Debugging
        "verbose": False,  # Log metabolism details per step
    })

    def __init__(
        self,
        world: 'World',
        config: DictConfig
    ):
        super().__init__(world, config)

    def initialize(self):
        self.energy.initialize_data()
        self.biomass.initialize_data()
        self.gradient_ema.initialize_data()
        self.scent.initialize_data()
        self.offspring_count.initialize_data()
        self.id_feature.initialize_data()

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

        # Step 2: Build observation (sensorium)
        # Convert PerceptionConfig objects to (key_tuple, kernel_size) pairs
        perception = [(p.key, p.kernel_size) for p in self.config.perception]
        obs = get_observation(
            td=self.td,
            perception=perception,
            energy=energy,
            biomass=biomass,
            log_scale=self.config.log_scale,
        )

        # Step 3: Process observation into gradient and direction
        # navigation_weights already has tuple keys from Pydantic validation
        gradient_strength, direction = process_observation(obs, self.config.navigation_weights)

        # Step 4: Update gradient EMA
        alpha = self.config.gradient_ema_alpha
        alive = biomass >= self.config.survival_threshold

        gradient_ema[:] = torch.where(
            alive,
            alpha * gradient_strength + (1 - alpha) * gradient_ema,
            gradient_ema
        )

        # Step 5: Compute metabolic rate based on gradient EMA and biomass
        # Biomass modulates max rate - low biomass animals can't sprint
        metabolic_rate = self._compute_metabolic_rate(gradient_ema, biomass)

        # Step 5b: Compute metabolic efficiency (decreases at higher rates)
        efficiency = self._compute_efficiency(metabolic_rate)

        if verbose and positions:
            self._log_metabolism(positions, "GRADIENT", gradient_strength=gradient_strength, metabolic_rate=metabolic_rate, efficiency=efficiency)

        # Step 6: Metabolism - convert biomass to energy
        # biomass_burned is limited by metabolic rate and available biomass
        biomass_burned = torch.min(metabolic_rate, biomass.float()).to(torch.uint8)
        # energy gained = biomass burned * efficiency (dynamic based on exertion)
        energy_gained = (biomass_burned.float() * efficiency).to(torch.uint8)
        safe_add(energy, energy_gained)
        safe_sub(biomass, biomass_burned)

        if verbose and positions:
            self._log_metabolism(positions, "METABOLISM", biomass_burned=biomass_burned, energy_gained=energy_gained, biomass=biomass, energy=energy)

        # Step 7: Energy dissipation
        dissipation = (energy.float() * self.config.dissipation_rate).to(torch.uint8)
        dissipation = torch.clamp(dissipation, min=self.config.dissipation_floor)
        safe_sub(energy, dissipation)

        if verbose and positions:
            self._log_metabolism(positions, "DISSIPATION", dissipation=dissipation, energy=energy)

        offspring_count = self.offspring_count.data
        id_feature = self.id_feature.data
        random = self.world.td.get("random")

        # Step 8: Compute movement probability based on energy
        # stimulus -> metabolism -> energy -> movement
        # Energy accumulates over time, providing natural smoothing
        move_prob = energy.float() / 255.0
        move_mask = torch.rand_like(energy, dtype=torch.float32) < move_prob

        if verbose and positions:
            self._log_metabolism(positions, "MOVE_PROB", move_prob=move_prob, move_mask=move_mask.float())

        # Step 9: Movement cost (flat cost per move)
        movement_cost = torch.full_like(energy, self.config.base_movement_cost)

        # Step 10: Movement
        # Direction comes from observation processing (or external action for RL)
        # Reproduction: both parent and offspring get 50% of biomass and energy
        # Movement cost is subtracted from energy during the move
        move_direction = action if action is not None else direction
        did_move = move(
            primary_feature=energy,
            target=None,  # Direction pre-computed
            target_weights=None,
            divide_threshold=self.config.reproduction_threshold,
            divide_feature=biomass,  # Check biomass, not energy, for reproduction
            divide_fn_self=lambda x: (x.float() * 0.5).to(x.dtype),  # energy: parent gets 50%
            divide_fn_offspring=lambda x: (x.float() * 0.5).to(x.dtype),  # energy: offspring gets 50%
            carried_features_self=[offspring_count, id_feature, biomass, gradient_ema],
            carried_feature_fns_self=[
                lambda x: safe_add(x, 1),  # offspring_count increments
                lambda x: x,                # id_feature unchanged
                lambda x: (x.float() * 0.5).to(x.dtype),  # biomass: parent gets 50%
                lambda x: x                 # gradient_ema unchanged
            ],
            carried_features_offspring=[id_feature, biomass, gradient_ema],
            carried_feature_fns_offspring=[
                lambda x: random,           # id_feature: new random id
                lambda x: (x.float() * 0.5).to(x.dtype),  # biomass: offspring gets 50%
                lambda x: x * 0.5           # gradient_ema: offspring starts with half
            ],
            agent_action=move_direction,
            move_mask=move_mask,
            move_cost=movement_cost,
        )

        # After movement, find new positions for logging
        if verbose:
            new_alive_mask = biomass >= self.config.survival_threshold
            new_positions = list(zip(*torch.where(new_alive_mask)))
            if positions:
                # Show movement info at old positions
                for y, x in positions:
                    if did_move[y, x]:
                        print(f"  [{self.__class__.__name__}@({x},{y})] MOVED to new position")
            positions = new_positions  # Update to new positions

        if verbose and positions:
            self._log_metabolism(positions, "POST_MOVE", biomass=biomass, energy=energy)

        # Step 11: Eating (fills biomass, not energy)
        # Try food sources in order until satiated
        biomass_before_eat = biomass.clone() if verbose else None
        self._eat()

        if verbose and positions:
            eaten = biomass - biomass_before_eat
            self._log_metabolism(positions, "EAT", eaten=eaten, biomass=biomass)
            self._log_metabolism(positions, "END", biomass=biomass, energy=energy)

        # Step 12: Emit scent (based on biomass); diffusion handled by World
        self.scent.emit(self.world.step)

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

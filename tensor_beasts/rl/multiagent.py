"""Per-individual reinforcement learning over a tensor-beasts world.

The framing
-----------

Every living herbivore is its own agent. They all share one set of policy
weights, they each get their own reward, and each one's episode runs from birth
to death. This is parameter-sharing multi-agent reinforcement learning, and it
is the framing the simulation was already built for:

* The observation each individual receives is local. Nobody sees the global map.
* The rule-based policy is already a shared, translation-equivariant function
  from that local observation to a direction. In substance it is a linear model:
  navigation weights dotted against local scent gradients. A convolutional
  network over the same inputs can represent it exactly, so a learned policy
  that loses to it lost on optimization rather than on expressiveness.
* Because the policy is shared and the observation is spatially laid out, every
  agent's action comes from a single convolutional forward pass over the grid.
  Treating individuals as separate agents therefore costs nothing at runtime
  compared to treating the world as one controller. Only the reward and the
  trajectory bookkeeping differ, and those are the parts that decide whether
  learning works at all.

What this module does not do is pretend the agents are independent. They eat the
same plants and they are each other's scent field, so every agent's environment
shifts as the shared policy changes. That non-stationarity is inherent to an
ecology and is the most likely source of training instability.

Identity
--------

Following an individual through time needs identity, and the simulation's ``id``
feature cannot provide it: offspring draw their id from the world's uint8 random
field, so only 256 distinct ids exist across thousands of animals. Instead the
``Animal`` entity records a :class:`~tensor_beasts.entities.animal.TransitionInfo`
each step, giving the cell each acting individual moved to. Measured over 55,000
agent-steps at 256x256, no two individuals ever share a successor cell, so the
mapping is one-to-one in practice as well as in intent.

Reward
------

Three terms, all per individual:

* ``survival_reward`` for each step the individual is still alive afterwards.
* ``reproduction_reward`` when it divides.
* ``foraging_reward`` times the biomass it ate this step.
* Its episode ends when it dies.

The metric this is a surrogate for is the one the project actually cares about,
total herbivore-steps survived, which is what ``evaluate_policy.py`` reports for
the rule-based baseline. Survival reward tracks it directly; reproduction reward
credits an individual for the future population it creates, which survival
reward alone would attribute entirely to the offspring.

Foraging reward exists because the first two are nearly useless as a learning
signal on their own. Herbivores survive about 99.4% of steps, so the per-step
reward is 1.0 with a standard deviation of 0.076: almost all of an individual's
return is fixed no matter what it does, and the part that responds to its
choices is buried under that. Reproduction is rarer still, around 0.7% of
agent-steps. Biomass change, by contrast, responds immediately and directly to
whether the individual moved somewhere with food. It is off by default, because
turning it on is reward shaping and changes what is being optimized; what it
must never change is the *evaluation*, which stays herbivore-steps survived.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from tensordict import TensorDict

from tensor_beasts.config import load_config
from tensor_beasts.observations import get_observation
from tensor_beasts.world import World

# Direction encoding shared with the simulation.
NUM_ACTIONS = 5
DIRECTION_NAMES = ("stay", "up", "down", "left", "right")

DEFAULT_CONFIG = "conf/basic_config.yaml"
DEFAULT_ENTITY = "Herbivore"

# Multiplier on the explicit gradient channels (neighbour minus own cell).
# Chosen so a measured standard deviation of 0.010 becomes 0.25 and the 99th
# percentile of 0.037 lands near 0.9. See _observe for why they exist.
GRADIENT_GAIN = 25.0
# Gradient channels are clipped to this magnitude. Typical values sit within
# +-1 after the gain, but a fresh scent source in a young world can produce a
# neighbour difference several times larger, and an unbounded input is a poor
# thing to hand a network.
GRADIENT_CLIP = 4.0


@dataclass
class AgentBatch:
    """One step of per-individual transitions, kept in grid layout.

    Every field is indexed by the cell an individual acted from, so they line up
    with each other and with the observation. ``acted`` selects the entries that
    refer to a real individual; everything else is padding and must be masked
    out before it reaches a loss.

    Attributes:
        observation: (C, H, W) float32, the field the policy reads.
        acted: (H, W) bool, cells holding an individual that chose an action.
        action: (H, W) int64, the direction each one chose.
        reward: (H, W) float32.
        done: (H, W) bool, true where the individual died during this step.
        successor: (H, W) int64, flat index of the cell that individual occupies
            next, for bootstrapping a value from the right place. Meaningless
            where ``done``.
        reproduced: (H, W) bool.
    """

    observation: torch.Tensor
    acted: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    successor: torch.Tensor
    reproduced: torch.Tensor
    # (H, W) int64: the direction the simulation's own rule-based policy would
    # have chosen from this same observation. Free to compute, since the rules
    # are a function of the observation, and it is what lets a learner anchor
    # to the baseline before the real rewards take over. Tie-breaks inside the
    # rules are random, so this is a sample from the rule policy, not its mode.
    rule_action: Optional[torch.Tensor] = None
    # (5, H, W) float32: the rule-based policy's score for each of
    # [stay, up, down, left, right] at each cell, i.e. the quantity it takes an
    # argmax over. Distilling toward a softmax of these, rather than toward the
    # argmax, is what makes the rules learnable: their decisions are knife-edge,
    # and a soft target turns a near-tie into a near-uniform distribution that
    # costs nothing to disagree with.
    rule_scores: Optional[torch.Tensor] = None
    # (H, W) int64: the metabolic level each individual chose, when the
    # environment was built with ``num_metabolic_levels`` and the caller sent
    # one. None otherwise, meaning the rule-based policy set the throttle.
    metabolic_action: Optional[torch.Tensor] = None
    # (H, W) int64: the rule-based policy's own metabolic rate on this same
    # observation, mapped to the nearest level. The anchor target for the
    # metabolic lever, as ``rule_action`` is for the direction. None when the
    # environment has no metabolic levels.
    rule_metabolic_level: Optional[torch.Tensor] = None

    @property
    def num_agents(self) -> int:
        return int(self.acted.sum())


def metabolic_level_rates(basal_rate: float, max_rate: float, num_levels: int) -> torch.Tensor:
    """Rate for each discrete metabolic level, ``(num_levels,)`` float32.

    Level ``i`` maps to ``basal + i / (L - 1) * (max - basal)``: level 0 is
    resting, the top level is the configured ceiling, evenly spaced between.
    The biomass cap is applied afterwards by the simulation, not here, so a
    starving individual that picks the top level still burns only what it can.
    """
    if num_levels < 2:
        raise ValueError(f"num_metabolic_levels must be at least 2, got {num_levels}")
    fraction = torch.arange(num_levels, dtype=torch.float32) / (num_levels - 1)
    return float(basal_rate) + fraction * (float(max_rate) - float(basal_rate))


def rate_to_metabolic_level(rate: torch.Tensor, level_rates: torch.Tensor) -> torch.Tensor:
    """Nearest level to each rate, ``rate.shape`` int64.

    Argmin over the absolute distance rather than rounding, so a rate exactly
    halfway between two levels resolves to the lower one deterministically
    instead of depending on round-half-even.
    """
    level_rates = level_rates.to(rate.device)
    shape = (level_rates.numel(),) + (1,) * rate.dim()
    distance = (rate.float().unsqueeze(0) - level_rates.reshape(shape)).abs()
    return distance.argmin(dim=0)


class MultiAgentWorldEnv:
    """A tensor-beasts world exposed as many individuals sharing one policy.

    Unlike a Gymnasium environment this has no global episode. The world is a
    persistent ecology that runs indefinitely; episodes belong to individuals,
    who are born and die inside it. A learner collects fixed-length segments of
    world time and reads per-individual episode boundaries out of ``done``.

    Args:
        config_path: Simulation config to load.
        size: Optional (height, width) override. The ecology is strongly
            size-dependent: below roughly 256 the predator population collapses
            and the three-species dynamic degenerates, so prefer 512 for
            anything whose result is meant to mean something.
        entity_name: Which entity the policy controls.
        survival_reward: Reward per step an individual remains alive.
        reproduction_reward: Reward for dividing.
        foraging_reward: Reward per unit of biomass eaten this step, read at
            the individual's own new cell. Dense and strongly
            action-dependent, unlike the other two. Zero by default because it
            is reward shaping; see the module docstring.
        device: Torch device for the simulation.
        num_metabolic_levels: Number of discrete metabolic levels the policy
            may choose from, or 0 to leave the throttle to the rule-based
            policy. See :func:`metabolic_level_rates` for the mapping.
    """

    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG,
        size: Optional[Tuple[int, int]] = None,
        entity_name: str = DEFAULT_ENTITY,
        survival_reward: float = 1.0,
        reproduction_reward: float = 10.0,
        foraging_reward: float = 0.0,
        offspring_credit: float = 0.0,
        device: Optional[str] = None,
        num_metabolic_levels: int = 0,
        memory_size: int = 0,
    ):
        self.config_path = config_path
        self.entity_name = entity_name
        self.survival_reward = survival_reward
        self.reproduction_reward = reproduction_reward
        self.foraging_reward = foraging_reward
        self.offspring_credit = offspring_credit
        self.device = torch.device(device) if device is not None else torch.get_default_device()
        self.num_metabolic_levels = int(num_metabolic_levels)

        config = load_config(config_path)
        if size is not None:
            config.world.size = list(size)
        self.world_config = config.world
        self.size: Tuple[int, int] = tuple(config.world.size)

        if entity_name not in config.world.entities:
            raise ValueError(
                f"Entity {entity_name!r} is not in {config_path}. "
                f"Available: {sorted(config.world.entities)}"
            )
        # Transition tracking is what makes per-individual trajectories possible.
        config.world.entities[entity_name].track_transitions = True
        if memory_size:
            # Learned memory lives on the entity as a feature; its width is set
            # here, before the world exists, and read back by memory_size.
            config.world.entities[entity_name].memory = {"size": int(memory_size)}

        self.world = World(self.world_config)
        self.world.initialize()

        self._flat_index = torch.arange(
            self.size[0] * self.size[1], device=self.device
        ).reshape(*self.size)

        self.observation_channels = self._build_observation().shape[0]
        self.channel_names = self._channel_names()
        self._level_rates = self._build_level_rates()

    @classmethod
    def attach(
        cls,
        world: World,
        entity_name: str = DEFAULT_ENTITY,
        num_metabolic_levels: int = 0,
    ) -> "MultiAgentWorldEnv":
        """Wrap an already-built World, for driving it with a learned policy.

        The interactive viewer in tensor_beasts/main.py owns its World and its
        display thread; this gives it the same observation encoding a trained
        network expects, without constructing a second world. Transition
        tracking is not enabled here: acting needs only the observation.
        """
        env = cls.__new__(cls)
        env.config_path = None
        env.entity_name = entity_name
        env.survival_reward = 0.0
        env.reproduction_reward = 0.0
        env.foraging_reward = 0.0
        env.offspring_credit = 0.0
        env.world_config = world.config
        env.size = tuple(world.size)
        env.world = world
        if entity_name not in world.entity_dict:
            raise ValueError(
                f"Entity {entity_name!r} is not in this world. Available: {sorted(world.entity_dict)}"
            )
        env.device = world.entity_dict[entity_name].biomass.data.device
        env._flat_index = torch.arange(env.size[0] * env.size[1], device=env.device).reshape(*env.size)
        env.num_metabolic_levels = int(num_metabolic_levels)
        env.observation_channels = env._build_observation().shape[0]
        env.channel_names = env._channel_names()
        env._level_rates = env._build_level_rates()
        return env

    # ------------------------------------------------------------------
    # Entity access
    # ------------------------------------------------------------------
    @property
    def entity(self):
        return self.world.entity_dict[self.entity_name]

    @property
    def memory_size(self) -> int:
        """Channels of learned memory the controlled entity carries (0 = none)."""
        return int(self.entity.memory.size)

    def _alive(self) -> torch.Tensor:
        entity = self.entity
        return entity.biomass.data >= entity.config.survival_threshold

    def population(self) -> int:
        return int(self._alive().sum())

    # ------------------------------------------------------------------
    # Metabolic levels
    # ------------------------------------------------------------------
    def _build_level_rates(self) -> Optional[torch.Tensor]:
        if self.num_metabolic_levels <= 0:
            return None
        config = self.entity.config
        return metabolic_level_rates(
            config.basal_rate, config.max_metabolic_rate, self.num_metabolic_levels
        ).to(self.device)

    @property
    def metabolic_level_rates(self) -> Optional[torch.Tensor]:
        """``(num_metabolic_levels,)`` rate per level, or None when off."""
        return self._level_rates

    def metabolic_level_to_rate(self, level: torch.Tensor) -> torch.Tensor:
        """Map chosen levels (H, W) int64 to the desired rate (H, W) float32.

        This is the rate *before* the simulation's biomass cap.
        """
        if self._level_rates is None:
            raise ValueError("This environment was built without metabolic levels.")
        level = level.reshape(*self.size).to(device=self.device, dtype=torch.long)
        return self._level_rates[level.clamp(0, self.num_metabolic_levels - 1)]

    def metabolic_rate_to_level(self, rate: torch.Tensor) -> torch.Tensor:
        """Nearest level to each rate. Used to discretize the rule's throttle."""
        if self._level_rates is None:
            raise ValueError("This environment was built without metabolic levels.")
        return rate_to_metabolic_level(rate, self._level_rates)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _perception(self) -> List[Tuple[Tuple[str, str], int]]:
        return [(p.key, p.kernel_size) for p in self.entity.config.perception]

    def _channel_names(self) -> List[str]:
        names: List[str] = []
        for key, _ in self._perception():
            label = ":".join(key)
            names.append(f"{label}/here")
            names.extend(f"{label}/{d}" for d in ("up", "down", "left", "right"))
        for key, _ in self._perception():
            label = ":".join(key)
            names.extend(f"{label}/grad_{d}" for d in ("up", "down", "left", "right"))
        names.extend(["self/energy", "self/biomass", "self/gradient_ema", "self/alive"])
        names.extend(f"memory/{k}" for k in range(self.memory_size))
        return names

    def _rule_decision(self, observation):
        """The entity's own rule-based policy's full Action from ``observation``.

        Called once per step so the direction and the metabolic rate recorded
        as anchor targets come from the same evaluation. The direction has a
        random tie-break, the rate does not, but the rate is computed from the
        gradient history the same way the simulation is about to compute it.
        """
        with torch.no_grad():
            return self.entity.policy(observation)

    def _rule_action(self, observation) -> torch.Tensor:
        """What the entity's own rule-based policy would do from ``observation``."""
        return self._rule_decision(observation).move_direction.to(torch.long)

    def _rule_metabolic_level(self, decision) -> Optional[torch.Tensor]:
        """The rule's metabolic rate as the nearest discrete level, or None."""
        if self._level_rates is None:
            return None
        return self.metabolic_rate_to_level(decision.metabolic_rate)

    def _rule_scores(self, observation) -> torch.Tensor:
        """The rule-based policy's per-action scores, (5, H, W).

        Reproduces RuleBasedPolicy._process_observation's signed, clamped
        weighted sum over perceived features for [stay, up, down, left, right].
        An analytic check against the policy's own action agrees 100%.
        """
        weights = dict(self.entity.config.navigation_weights)
        combined = None
        for key, weight in weights.items():
            if key not in observation.directional:
                continue
            part = torch.cat(
                [
                    (observation.current[key].float() * weight).unsqueeze(0),
                    observation.directional[key].float() * weight,
                ]
            )
            combined = part if combined is None else combined + part
        if combined is None:
            return torch.zeros(NUM_ACTIONS, *self.size, device=self.device)
        return combined.clamp(min=0)

    def _build_observation(self) -> torch.Tensor:
        """Stack the individual's local view into a (C, H, W) field."""
        return self._observe()[0]

    def _observe(self):
        """Return the (C, H, W) policy input and the raw Observation it came from.

        Each perceived feature contributes its value at the individual's own
        cell plus the four neighbouring values, which are exactly the inputs the
        rule-based policy uses. The raw neighbour values are included rather
        than only a summary so a convolutional policy starts from the same
        information the baseline has, and own state is appended because
        metabolism and movement both depend on it.

        Values are scaled to roughly the unit interval. Perceived features are
        NOT raw uint8: get_observation log-compresses them as log1p(x * log_scale)
        for the rule-based policy's benefit, so their ceiling is
        log1p(255 * log_scale), about 9.4 at the default log_scale of 50. An
        earlier version divided these by 255 as if they were bytes, which left
        every perceived channel in roughly [0, 0.04]; a linear model then could
        not fit the rule action above 0.46 agreement on a label that is 99.8%
        self-consistent, because it had to grow its weights thirty-fold first.
        Own energy and biomass are floats on a 0..255 scale and are divided by 255.

        After the perceived values come explicit gradient channels, neighbour
        minus own cell for each direction, scaled by GRADIENT_GAIN. The rule
        compares a cell against its neighbours and nothing else, and scent is a
        smooth field, so those differences are tiny relative to the values:
        measured on a settled 256x256 world the difference has a standard
        deviation of 0.010 and a 99th percentile near 0.037. A linear model can
        represent the rule exactly from the raw channels, and an analytic check
        confirms the label is exactly linear in them, yet cross-entropy on logit
        margins that small is nearly flat and the linear fit stalled at 0.46
        agreement. Handing the learner the differences at unit scale removes an
        optimization problem the rule-based policy never had to face.

        Even so, do not expect any learner to match the rule action exactly.
        The rule's decisions are knife-edge: measured on a settled world the
        median relative margin between its best and second-best direction is
        0.18%, 93% of decisions are settled by under 1%, and perturbing one
        navigation weight by 1% flips 5% of them. Around 0.9 agreement is the
        practical ceiling for an approximate model, which is why the default
        imitation target sits below it.
        """
        entity = self.entity
        observation = get_observation(
            td=self.world.td,
            perception=self._perception(),
            energy=entity.energy.data,
            biomass=entity.biomass.data,
            gradient_ema=entity.gradient_ema.data,
            survival_threshold=entity.config.survival_threshold,
            log_scale=entity.config.log_scale,
            step=self.world.step,
        )

        # Ceiling of the log-compressed perceived values; see the docstring.
        perceived_scale = float(torch.log1p(torch.tensor(255.0 * entity.config.log_scale)))

        channels: List[torch.Tensor] = []
        for key, _ in self._perception():
            current = observation.current[key].float() / perceived_scale
            directional = observation.directional[key].float() / perceived_scale
            channels.append(current)
            channels.extend(directional[i] for i in range(directional.shape[0]))

        for key, _ in self._perception():
            here = observation.current[key].float() / perceived_scale
            directional = observation.directional[key].float() / perceived_scale
            channels.extend(
                ((directional[i] - here) * GRADIENT_GAIN).clamp(-GRADIENT_CLIP, GRADIENT_CLIP)
                for i in range(directional.shape[0])
            )

        channels.append(observation.energy.float() / 255.0)
        channels.append(observation.biomass.float() / 255.0)
        channels.append(observation.gradient_ema.float())
        channels.append(observation.alive_mask.float())
        if self.memory_size > 0:
            # What this individual wrote last step, carried to wherever it is now.
            memory = entity.memory.data
            channels.extend(memory[..., k].float() for k in range(self.memory_size))

        return torch.stack(channels, dim=0), observation

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------
    def _reward(self, transition, biomass_before: torch.Tensor):
        """Reward, liveness and successor for one completed step.

        Shared by :meth:`step` and :meth:`rule_based_step` so the learned policy
        and the baseline are scored by exactly the same rules. If these ever
        drift apart the comparison stops meaning anything.
        """
        acted = transition.acted
        successor = transition.successor.clamp(min=0)

        entity = self.entity
        biomass_flat = entity.biomass.data.reshape(-1)
        # An individual is alive afterwards if the cell it moved into holds
        # enough biomass to survive. Same predicate the simulation applies at
        # the top of the next step.
        alive_after = (biomass_flat[successor] >= entity.config.survival_threshold) & acted

        reward = (
            alive_after.float() * self.survival_reward
            + transition.reproduced.float() * self.reproduction_reward
        )

        if self.foraging_reward:
            # What the individual ATE, read at its successor cell since eating
            # happens after the move. Net biomass change was the first version
            # and it punished the metabolic lever: burning biomass into energy
            # is what metabolism does, so every unit burned cost reward and the
            # learned throttle collapsed onto the coldest setting.
            eaten = transition.eaten.reshape(-1)[successor].reshape(*self.size)
            reward = reward + alive_after.float() * eaten * self.foraging_reward

        if self.offspring_credit and transition.offspring is not None:
            reward = reward + self._offspring_credit(transition, successor, acted)

        return reward, alive_after, successor, acted

    def _offspring_credit(self, transition, successor: torch.Tensor, acted: torch.Tensor) -> torch.Tensor:
        """Credit an individual with a share of the biomass it endowed its child.

        Division is a cost under every reward this project has used: it halves
        the parent's biomass, and a reward in biomass alone is therefore
        maximized by eating and never dividing. That is the same failure the
        metabolic lever hit when its reward counted net biomass change. The
        metric being approximated is not an individual's own mass but its
        lineage's, so an individual is credited with what it handed on.

        The credit is the offspring's biomass at birth, which is half the
        parent's, times ``offspring_credit``. First generation only, and the
        argument for stopping there is variance: crediting a lineage without a
        generational bound makes an early ancestor's return depend on
        descendants it never saw, growing without limit in a growing
        population. One generation captures the investment and stays bounded.
        The coefficient is the discount: 0 disables the term, and 1.0 values a
        unit of offspring biomass exactly as a unit of the individual's own.
        """
        offspring = transition.offspring
        divided = transition.reproduced & acted
        if not bool(divided.any()):
            return torch.zeros(self.size, dtype=torch.float32, device=self.device)
        biomass_flat = self.entity.biomass.data.reshape(-1)
        endowment = torch.zeros(self.size, dtype=torch.float32, device=self.device)
        cells = offspring.clamp(min=0)
        endowment = torch.where(divided, biomass_flat[cells].reshape(*self.size), endowment)
        return endowment * float(self.offspring_credit)

    def reset(self, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Restart the ecology. Returns (observation, acted mask).

        Note this resets the *world*, not an agent episode. Individual episodes
        begin and end inside a running world.
        """
        if seed is not None:
            torch.manual_seed(seed)
        self.world.reset()
        return self._build_observation(), self._alive()

    def step(
        self,
        action: torch.Tensor,
        metabolic_action: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
    ) -> AgentBatch:
        """Advance one simulation step under a per-cell direction field.

        Args:
            action: (H, W) int64 in [0, 5). Entries at cells with no individual
                are ignored by the simulation.
            memory: Optional (K, H, W) float32 in [-1, 1], the memory each
                individual writes for its next step. None leaves memory as is.
            metabolic_action: Optional (H, W) int64 in [0, num_metabolic_levels).
                Mapped to a rate through :meth:`metabolic_level_to_rate` and
                sent to the simulation, which clamps it to what the individual's
                biomass allows. None leaves the throttle to the rule-based
                policy, which is what a direction-only learner wants.

        Returns:
            An :class:`AgentBatch` whose ``observation`` is the state *before*
            the step, matching the actions that were taken from it.
        """
        action = action.reshape(*self.size).to(device=self.device, dtype=torch.long)
        observation, raw = self._observe()
        decision = self._rule_decision(raw)
        rule_action = decision.move_direction.to(torch.long)
        rule_scores = self._rule_scores(raw)
        rule_metabolic_level = self._rule_metabolic_level(decision)
        biomass_before = self.entity.biomass.data.clone()

        if metabolic_action is None and memory is None:
            entity_action = action
        else:
            fields = {"direction": action}
            if metabolic_action is not None:
                metabolic_action = metabolic_action.reshape(*self.size).to(
                    device=self.device, dtype=torch.long
                )
                fields["metabolic_rate"] = self.metabolic_level_to_rate(metabolic_action)
            if memory is not None:
                # Network layout is (K, H, W); the feature is (H, W, K).
                fields["memory"] = memory.reshape(self.memory_size, *self.size).permute(1, 2, 0).to(self.device)
            entity_action = TensorDict(fields, batch_size=[])
        self.world.update(TensorDict({self.entity_name: entity_action}, batch_size=[]))

        transition = self.entity.last_transition
        if transition is None:
            raise RuntimeError(
                f"{self.entity_name} recorded no transition. track_transitions "
                "should have been enabled by this environment's constructor."
            )

        reward, alive_after, successor, acted = self._reward(transition, biomass_before)
        done = acted & ~alive_after

        return AgentBatch(
            observation=observation,
            acted=acted,
            action=action,
            reward=reward,
            done=done,
            successor=successor,
            reproduced=transition.reproduced,
            rule_action=rule_action,
            rule_scores=rule_scores,
            metabolic_action=metabolic_action,
            rule_metabolic_level=rule_metabolic_level,
        )

    def step_with_policy(self, decide) -> AgentBatch:
        """Advance one step, asking ``decide`` for the action at the right moment.

        :meth:`step` takes an action chosen before the world updated at all.
        Entities update in dependency order and each builds its observation
        inside its own update, so the rule-based predator sees the prey field
        *after* the herbivores have moved this step while a learner using
        :meth:`step` saw it before. Measured at 512 that halved the learned
        predator's hunting success, 1.46% of steps against 2.95%, and drove the
        population extinct where the rules recover, purely from the timing.

        ``decide`` is called with this environment's observation at the instant
        the controlled entity updates, and returns
        ``(direction, metabolic_action, memory)``, any of which may be None. The
        comparison against the rules is then like for like.

        Args:
            decide: Callable taking the ``(C, H, W)`` observation and returning
                ``(direction, metabolic_action, memory)``.
        """
        captured: Dict[str, object] = {}

        def build_action():
            observation, raw = self._observe()
            captured["observation"] = observation
            decision = self._rule_decision(raw)
            captured["rule_action"] = decision.move_direction.to(torch.long)
            captured["rule_scores"] = self._rule_scores(raw)
            captured["rule_metabolic_level"] = self._rule_metabolic_level(decision)

            direction, metabolic_action, memory = decide(observation)
            captured["action"] = direction
            captured["metabolic_action"] = metabolic_action
            return self._entity_action(direction, metabolic_action, memory)

        biomass_before = self.entity.biomass.data.clone()
        self.world.update(action_fns={self.entity_name: build_action})

        transition = self.entity.last_transition
        if transition is None:
            raise RuntimeError(
                f"{self.entity_name} recorded no transition. track_transitions "
                "should have been enabled by this environment's constructor."
            )
        reward, alive_after, successor, acted = self._reward(transition, biomass_before)
        return AgentBatch(
            observation=captured["observation"],
            acted=acted,
            action=captured["action"].reshape(*self.size).to(self.device, torch.long),
            reward=reward,
            done=acted & ~alive_after,
            successor=successor,
            reproduced=transition.reproduced,
            rule_action=captured["rule_action"],
            rule_scores=captured["rule_scores"],
            metabolic_action=captured["metabolic_action"],
            rule_metabolic_level=captured["rule_metabolic_level"],
        )

    def _entity_action(self, action, metabolic_action, memory):
        """The TensorDict or bare tensor ``Animal.update`` expects."""
        action = action.reshape(*self.size).to(device=self.device, dtype=torch.long)
        if metabolic_action is None and memory is None:
            return action
        fields = {"direction": action}
        if metabolic_action is not None:
            metabolic_action = metabolic_action.reshape(*self.size).to(
                device=self.device, dtype=torch.long
            )
            fields["metabolic_rate"] = self.metabolic_level_to_rate(metabolic_action)
        if memory is not None:
            fields["memory"] = memory.reshape(self.memory_size, *self.size).permute(1, 2, 0).to(self.device)
        return TensorDict(fields, batch_size=[])

    def rule_based_step(self) -> AgentBatch:
        """Advance one step using the simulation's own policy.

        The baseline has to be scored through identical reward bookkeeping for
        the comparison to mean anything, so this returns the same AgentBatch as
        :meth:`step`, differing only in where the movement decision came from.
        """
        observation, raw = self._observe()
        decision = self._rule_decision(raw)
        rule_action = decision.move_direction.to(torch.long)
        rule_scores = self._rule_scores(raw)
        rule_metabolic_level = self._rule_metabolic_level(decision)
        biomass_before = self.entity.biomass.data.clone()
        self.world.update()

        transition = self.entity.last_transition
        reward, alive_after, successor, acted = self._reward(transition, biomass_before)

        # The simulation's chosen direction is not reported back, so record
        # "stay" rather than inventing one. Callers that need real actions from
        # the baseline should read them from the policy directly.
        return AgentBatch(
            observation=observation,
            acted=acted,
            action=torch.zeros(self.size, dtype=torch.long, device=self.device),
            reward=reward,
            done=acted & ~alive_after,
            successor=successor,
            reproduced=transition.reproduced,
            rule_action=rule_action,
            rule_scores=rule_scores,
            rule_metabolic_level=rule_metabolic_level,
        )

    def stats(self) -> Dict[str, float]:
        """Cheap per-step diagnostics, for logging during training."""
        world = self.world
        out: Dict[str, float] = {"population": float(self.population())}
        for name, entity in world.entity_dict.items():
            if hasattr(entity, "biomass"):
                out[f"{name.lower()}_population"] = float((entity.biomass.data > 0).sum())
            elif hasattr(entity, "energy"):
                out[f"{name.lower()}_cells"] = float((entity.energy.data > 0).sum())
        return out

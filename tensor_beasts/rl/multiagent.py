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

One quantity, with no coefficients: the change in the individual's own stock
of biomass over the step,

    r_t(i) = eat_t(i) - burn_t(i) - loss_t(i)

where ``eat`` is what it ate at its new cell, ``burn`` what its metabolism
consumed, and ``loss`` the reserve that becomes carrion when it ends the step
below the survival threshold. Division is neutral: the parent keeps half and
the offspring carries the other half, so nothing is lost and nothing is paid.
Summed over every individual this is exactly the species' stock change,
``B_{t+1} - B_t`` with ``B`` the biomass carried by living individuals, except
for what newborns eat in their birth step, which nobody is paid for. That
makes the reward the evaluation metric's own increment.

An individual's own outcomes are blind to what its decisions do through the
shared prey field: a control run (planning/STATE.md) found a collapse under which
every per-individual quantity was unchanged while the world sustained a
quarter fewer individuals. So the reward is pooled spatially: with ``rho_t``
the field holding each individual's ``r`` at its successor cell,

    R_t(i) = sum_c K_R(c - c_{t+1}(i)) rho_t(c)

with ``K_R`` a box of radius ``reward_radius``. Radius 0 is the individual
reward; a radius covering the world pays everyone the species' stock change;
a few cells pays each individual for its neighbourhood's stock, which scales
with how many neighbours there are. The radius is the one knob.

Exact for the predator, whose biomass nothing else takes. A herbivore also
loses biomass to predator bites, which it did not choose and which are not
recorded per individual, so for the herbivore the identity above does not
hold.
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
        reward: (H, W) float32, the individual's stock change pooled over its
            neighbourhood; see the module docstring.
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
    # (H, W) float32 in [0, 1]: the normalised throttle each individual chose,
    # when the environment was built with ``metabolic`` and the caller sent one.
    # None otherwise, meaning the rule-based policy set the throttle. 0 is the
    # basal rate and 1 the configured maximum; the environment maps it onto the
    # rate through :meth:`MultiAgentWorldEnv.metabolic_unit_to_rate`.
    metabolic_unit: Optional[torch.Tensor] = None
    # (H, W) float32 in [0, 1]: the rule-based policy's own metabolic rate on
    # this same observation, expressed in those same units. The anchor target
    # for the metabolic lever, as ``rule_action`` is for the direction. None
    # when the environment has no metabolic head.
    rule_metabolic_unit: Optional[torch.Tensor] = None

    @property
    def num_agents(self) -> int:
        return int(self.acted.sum())


def gather_per_world(field: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``field`` read at ``index``, where both are grids and ``index`` is per world.

    The simulation's successor and offspring maps hold a flat cell index within
    one world: cell (h, w) is ``h * width + w`` in every world. Flattening a
    batched ``(B, H, W)`` field with ``reshape(-1)`` and indexing it with those
    would read world 0's cells for every world, and produce entirely plausible
    numbers while doing it. Reshaping to ``(worlds, H * W)`` and gathering along
    the last axis keeps each world to itself, and at one world is exactly the
    old arithmetic.
    """
    grid = index.shape
    cells = grid[-2] * grid[-1]
    worlds = index.numel() // cells
    gathered = field.reshape(worlds, cells).gather(1, index.reshape(worlds, cells))
    return gathered.reshape(grid)


def scatter_per_world(values: torch.Tensor, index: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The inverse of :func:`gather_per_world`: ``values`` at ``mask`` summed
    into a grid at ``index``, within each world. Entries outside ``mask`` are
    dropped, and two individuals sharing a cell add rather than overwrite."""
    grid = index.shape
    cells = grid[-2] * grid[-1]
    worlds = index.numel() // cells
    out = torch.zeros(worlds, cells, dtype=values.dtype, device=values.device)
    contribution = torch.where(mask, values, torch.zeros_like(values)).reshape(worlds, cells)
    out.scatter_add_(1, index.reshape(worlds, cells), contribution)
    return out.reshape(grid)


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
        reward_radius: Radius in cells of the box each individual's reward is
            pooled over, around its new cell. 0 pays each individual its own
            stock change; see the module docstring.
        device: Torch device for the simulation.
        metabolic: Let the policy set its own metabolic rate, as a continuous
            throttle in [0, 1] mapped onto [basal_rate, max_metabolic_rate].
            False leaves the throttle to the rule-based policy.
    """

    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG,
        size: Optional[Tuple[int, int]] = None,
        entity_name: str = DEFAULT_ENTITY,
        reward_radius: int = 0,
        worlds: int = 1,
        device: Optional[str] = None,
        metabolic: bool = False,
        memory_size: int = 0,
    ):
        self.config_path = config_path
        self.entity_name = entity_name
        self.reward_radius = int(reward_radius)
        self.device = torch.device(device) if device is not None else torch.get_default_device()
        self.metabolic = bool(metabolic)

        config = load_config(config_path)
        if size is not None:
            config.world.size = list(size)
        if worlds and worlds > 1:
            config.world.batch = int(worlds)
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
        ).reshape(*self.size).expand(self.field_shape)

        # Indexed from the end: shape[0] is the batch axis on a batched world, so
        # reading it there would size the network with B input channels.
        self.observation_channels = self._build_observation().shape[-3]
        self.channel_names = self._channel_names()
        self._metabolic_range = self._metabolic_bounds()

    @classmethod
    def attach(
        cls,
        world: World,
        entity_name: str = DEFAULT_ENTITY,
        metabolic: bool = False,
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
        env.reward_radius = 0
        env.world_config = world.config
        env.size = tuple(world.size)
        env.world = world
        if entity_name not in world.entity_dict:
            raise ValueError(
                f"Entity {entity_name!r} is not in this world. Available: {sorted(world.entity_dict)}"
            )
        env.device = world.entity_dict[entity_name].biomass.data.device
        env._flat_index = torch.arange(env.size[0] * env.size[1], device=env.device).reshape(*env.size)
        env.metabolic = bool(metabolic)
        env.observation_channels = env._build_observation().shape[-3]
        env.channel_names = env._channel_names()
        env._metabolic_range = env._metabolic_bounds()
        return env

    # ------------------------------------------------------------------
    # Entity access
    # ------------------------------------------------------------------
    def _memory_field(self, memory: torch.Tensor) -> torch.Tensor:
        """Network memory ``(K, H, W)`` or ``(B, K, H, W)`` to the feature's layout.

        The feature stores memory with its channel axis trailing, ``(H, W, K)``
        or ``(B, H, W, K)``, so the channel axis moves to the end rather than
        being permuted by fixed positions.
        """
        memory = memory.reshape(*self.world.batch_shape, self.memory_size, *self.size)
        return memory.movedim(-3, -1).to(self.device)

    @property
    def entity(self):
        return self.world.entity_dict[self.entity_name]

    @property
    def field_shape(self) -> Tuple[int, ...]:
        """Shape of one per-cell field: ``(H, W)``, or ``(B, H, W)`` batched.

        Every action, reward and mask the learner exchanges with the simulation
        has this shape. ``self.size`` stays the spatial extent so that code
        reading height and width keeps working.
        """
        return self.world.batch_shape + self.size

    @property
    def num_worlds(self) -> int:
        """Independent worlds stepped together. 1 when unbatched."""
        return self.world.num_worlds

    @property
    def memory_size(self) -> int:
        """Channels of learned memory the controlled entity carries (0 = none)."""
        return int(self.entity.memory.size)

    def _alive(self) -> torch.Tensor:
        entity = self.entity
        return entity.biomass.data >= entity.config.survival_threshold

    def population(self) -> int:
        """Living individuals across every world. See population_per_world."""
        return int(self._alive().sum())

    def population_per_world(self) -> torch.Tensor:
        """Living individuals in each world, ``(worlds,)``.

        Evaluation needs this rather than the total: each world is its own
        seed, and summing them before they are reported hides the spread
        between seeds, which is the quantity every claim here is hedged
        against.
        """
        return self._alive().sum(dim=(-2, -1)).reshape(self.num_worlds).float()

    def stock_per_world(self) -> torch.Tensor:
        """Biomass carried by living individuals in each world, ``(worlds,)``.

        The quantity the metric integrates and the reward increments.
        Living individuals only: a cell below the survival
        threshold still holds the reserve of an animal that died this step,
        and that reserve becomes carrion at the top of the next one.
        """
        biomass = self.entity.biomass.data
        alive = biomass >= self.entity.config.survival_threshold
        return (biomass * alive).sum(dim=(-2, -1)).reshape(self.num_worlds).float()

    # ------------------------------------------------------------------
    # Metabolic throttle
    # ------------------------------------------------------------------
    def _metabolic_bounds(self) -> Optional[Tuple[float, float]]:
        """``(basal, max)`` rate for the controlled entity, or None when off."""
        if not self.metabolic:
            return None
        config = self.entity.config
        return float(config.basal_rate), float(config.max_metabolic_rate)

    @property
    def metabolic_range(self) -> Optional[Tuple[float, float]]:
        return self._metabolic_range

    def metabolic_unit_to_rate(self, unit: torch.Tensor) -> torch.Tensor:
        """Map a normalised throttle in [0, 1] to a rate, ``field_shape`` float32.

        The rate is continuous, because it is a continuous quantity: an
        individual burns some amount of biomass, not one of four settings. This
        is the rate *before* the simulation's biomass cap, which the entity
        applies afterwards, so a starving animal that asks for the maximum still
        burns only what it carries.
        """
        if self._metabolic_range is None:
            raise ValueError("This environment was built without a metabolic head.")
        basal, top = self._metabolic_range
        unit = unit.reshape(self.field_shape).to(device=self.device, dtype=torch.float32)
        return basal + unit.clamp(0.0, 1.0) * (top - basal)

    def metabolic_rate_to_unit(self, rate: torch.Tensor) -> torch.Tensor:
        """The inverse: a rate to [0, 1]. Used to express the rule's throttle
        as an anchor target in the same units the policy emits."""
        if self._metabolic_range is None:
            raise ValueError("This environment was built without a metabolic head.")
        basal, top = self._metabolic_range
        span = max(top - basal, 1e-6)
        return ((rate.float() - basal) / span).clamp(0.0, 1.0)

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

    def rule_spec(self) -> Dict[str, object]:
        """What the rule-parametrised actor needs to be the rule at start.

        Everything :class:`~tensor_beasts.rl.networks.RulePolicy` reads from
        the entity's config, keyed the way the observation channels are named,
        so the actor can be rebuilt from a checkpoint against any world with
        the same perception.
        """
        config = self.entity.config
        weights = {}
        for key, weight in dict(config.navigation_weights).items():
            label = ":".join(key) if isinstance(key, (tuple, list)) else str(key)
            weights[label] = float(weight)
        return {
            "channel_names": list(self.channel_names),
            "features": [":".join(key) for key, _ in self._perception()],
            "navigation_weights": weights,
            "perceived_scale": float(torch.log1p(torch.tensor(255.0 * config.log_scale))),
            "gradient_ema_alpha": float(config.gradient_ema_alpha),
            "basal_rate": float(config.basal_rate),
            "max_metabolic_rate": float(config.max_metabolic_rate),
            "metabolic_sensitivity": float(config.metabolic_sensitivity),
            "survival_threshold": float(config.survival_threshold),
        }

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

    def _rule_metabolic_unit(self, decision) -> Optional[torch.Tensor]:
        """The rule's metabolic rate as a normalised throttle in [0, 1], or None."""
        if not self.metabolic:
            return None
        return self.metabolic_rate_to_unit(decision.metabolic_rate)

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
            # The action axis is built leading, as get_directional_values
            # produces it, then moved to -3 so the scores line up with the
            # policy's logits: (5, H, W) unbatched, (B, 5, H, W) batched.
            part = torch.cat(
                [
                    (observation.current[key].float() * weight).unsqueeze(0),
                    observation.directional[key].float() * weight,
                ]
            )
            combined = part if combined is None else combined + part
        if combined is None:
            return torch.zeros(*self.world.batch_shape, NUM_ACTIONS, *self.size, device=self.device)
        return combined.clamp(min=0).movedim(0, -3)

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

        # Stacked at -3, the channel axis a convolution expects, rather than at
        # 0. Each channel is (H, W) unbatched and (B, H, W) batched, so dim=0
        # would give (C, B, H, W) and hand the network the batch as its
        # channels. At one world the two are identical.
        return torch.stack(channels, dim=-3), observation

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------
    def _reward(self, transition, biomass_before: torch.Tensor):
        """Reward, liveness and successor for one completed step.

        Shared by :meth:`step` and :meth:`rule_based_step` so the learned policy
        and the baseline are scored by exactly the same rules. If these ever
        drift apart the comparison stops meaning anything.

        The reward is each individual's own stock change, pooled over a box of
        ``reward_radius`` cells around its new cell; see the module docstring.
        """
        acted = transition.acted
        successor = transition.successor.clamp(min=0)

        entity = self.entity
        # An individual is alive afterwards if the cell it moved into holds
        # enough biomass to survive. Same predicate the simulation applies at
        # the top of the next step.
        biomass_after = gather_per_world(entity.biomass.data, successor)
        alive_after = (biomass_after >= entity.config.survival_threshold) & acted

        # What it ate, read at its successor cell since eating happens after
        # the move; what it burned, at the cell it acted from since metabolism
        # happens before; and, if it ends the step below the threshold, the
        # reserve it leaves behind, which the simulation turns into carrion at
        # the top of the next step.
        eaten = gather_per_world(transition.eaten, successor)
        burned = transition.burned
        loss = torch.where(acted & ~alive_after, biomass_after, torch.zeros_like(biomass_after))
        own = torch.where(acted, eaten - burned - loss, torch.zeros_like(eaten))

        reward = self._pool(own, successor, acted)
        return reward, alive_after, successor, acted

    def _pool(self, own: torch.Tensor, successor: torch.Tensor, acted: torch.Tensor) -> torch.Tensor:
        """Each individual's reward summed over its neighbourhood at its new cell.

        Scatter every individual's own reward to its successor cell, sum that
        field over a box of ``reward_radius`` cells, and read it back at the
        successor. Radius 0 is the identity. The box is unnormalised, so a
        radius covering the world hands everyone the species' total.
        """
        radius = self.reward_radius
        if radius <= 0:
            return own
        field = scatter_per_world(own, successor, acted)
        worlds = self.num_worlds
        kernel = torch.ones(1, 1, 2 * radius + 1, 2 * radius + 1, device=own.device, dtype=own.dtype)
        pooled = torch.nn.functional.conv2d(
            field.reshape(worlds, 1, *self.size), kernel, padding=radius
        ).reshape(self.field_shape)
        pooled = gather_per_world(pooled, successor)
        return torch.where(acted, pooled, torch.zeros_like(pooled))

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
        metabolic_unit: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
    ) -> AgentBatch:
        """Advance one simulation step under a per-cell direction field.

        Args:
            action: (H, W) int64 in [0, 5). Entries at cells with no individual
                are ignored by the simulation.
            memory: Optional (K, H, W) float32 in [-1, 1], the memory each
                individual writes for its next step. None leaves memory as is.
            metabolic_unit: Optional (H, W) float32 in [0, 1], the normalised
                throttle. Mapped to a rate through
                :meth:`metabolic_unit_to_rate` and sent to the simulation,
                which clamps it to what the individual's biomass allows. None
                leaves the throttle to the rule-based policy, which is what a
                direction-only learner wants.

        Returns:
            An :class:`AgentBatch` whose ``observation`` is the state *before*
            the step, matching the actions that were taken from it.
        """
        action = action.reshape(self.field_shape).to(device=self.device, dtype=torch.long)
        observation, raw = self._observe()
        decision = self._rule_decision(raw)
        rule_action = decision.move_direction.to(torch.long)
        rule_scores = self._rule_scores(raw)
        rule_metabolic_unit = self._rule_metabolic_unit(decision)
        biomass_before = self.entity.biomass.data.clone()

        if metabolic_unit is None and memory is None:
            entity_action = action
        else:
            fields = {"direction": action}
            if metabolic_unit is not None:
                metabolic_unit = metabolic_unit.reshape(self.field_shape).to(
                    device=self.device, dtype=torch.float32
                )
                fields["metabolic_rate"] = self.metabolic_unit_to_rate(metabolic_unit)
            if memory is not None:
                # Network layout is (K, H, W); the feature is (H, W, K).
                fields["memory"] = self._memory_field(memory)
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
            metabolic_unit=metabolic_unit,
            rule_metabolic_unit=rule_metabolic_unit,
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
        ``(direction, metabolic_unit, memory)``, any of which may be None. The
        comparison against the rules is then like for like.

        Args:
            decide: Callable taking the ``(C, H, W)`` observation and returning
                ``(direction, metabolic_unit, memory)``.
        """
        captured: Dict[str, object] = {}

        def build_action():
            observation, raw = self._observe()
            captured["observation"] = observation
            decision = self._rule_decision(raw)
            captured["rule_action"] = decision.move_direction.to(torch.long)
            captured["rule_scores"] = self._rule_scores(raw)
            captured["rule_metabolic_unit"] = self._rule_metabolic_unit(decision)

            direction, metabolic_unit, memory = decide(observation)
            captured["action"] = direction
            captured["metabolic_unit"] = metabolic_unit
            return self._entity_action(direction, metabolic_unit, memory)

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
            action=captured["action"].reshape(self.field_shape).to(self.device, torch.long),
            reward=reward,
            done=acted & ~alive_after,
            successor=successor,
            reproduced=transition.reproduced,
            rule_action=captured["rule_action"],
            rule_scores=captured["rule_scores"],
            metabolic_unit=captured["metabolic_unit"],
            rule_metabolic_unit=captured["rule_metabolic_unit"],
        )

    def _entity_action(self, action, metabolic_unit, memory):
        """The TensorDict or bare tensor ``Animal.update`` expects."""
        action = action.reshape(self.field_shape).to(device=self.device, dtype=torch.long)
        if metabolic_unit is None and memory is None:
            return action
        fields = {"direction": action}
        if metabolic_unit is not None:
            metabolic_unit = metabolic_unit.reshape(self.field_shape).to(
                device=self.device, dtype=torch.float32
            )
            fields["metabolic_rate"] = self.metabolic_unit_to_rate(metabolic_unit)
        if memory is not None:
            fields["memory"] = self._memory_field(memory)
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
        rule_metabolic_unit = self._rule_metabolic_unit(decision)
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
            action=torch.zeros(self.field_shape, dtype=torch.long, device=self.device),
            reward=reward,
            done=acted & ~alive_after,
            successor=successor,
            reproduced=transition.reproduced,
            rule_action=rule_action,
            rule_scores=rule_scores,
            rule_metabolic_unit=rule_metabolic_unit,
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

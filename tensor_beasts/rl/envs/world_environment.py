"""
Gymnasium environment wrapping the tensor-beasts :class:`~tensor_beasts.world.World`.

The environment exposes *herbivore* control: on every step the agent supplies a
movement direction for every cell of the grid, the simulation advances one tick,
and the agent is rewarded for keeping herbivores alive.

This is the *single-controller* view: one agent receives the whole world and
emits one direction per cell, rewarded by total population. It is a valid
Gymnasium environment and is kept for comparison and for off-the-shelf
algorithms that expect the Gymnasium API, but it is **not** the framing the
project trains against. A single scalar reward against a per-cell action field
makes credit assignment close to hopeless.

For the framing that is actually trained, where every herbivore is its own
agent with its own reward and its own episode, see
``tensor_beasts/rl/multiagent.py``.

Contract
--------

**Observation** -- ``Box(-inf, inf, (H, W, C), float32)``.

    This is exactly ``World.observable``: every feature tagged ``"observable"``
    on every entity, concatenated along a trailing channel axis (see
    ``tensor_beasts.observations.build_observation``). ``C`` depends on the
    entities in the config and is measured once at construction time. Values are
    *not* normalised -- terrain features are unbounded floats, energy/biomass/
    scent live in ``[0, 255]`` -- hence the infinite bounds. This is the
    simulation's existing "what can be seen" surface, so using it avoids
    inventing a second, divergent notion of observability.

    Note this is a *global* view, not a per-animal egocentric view. The world is
    fully observed; partial observability is not modelled here.

**Action** -- ``MultiDiscrete(nvec=5, shape=(H, W))``, dtype ``int64``.

    One movement direction per grid cell, using the simulation's own encoding:
    ``0 = stay, 1 = up, 2 = down, 3 = left, 4 = right``. The action is handed to
    ``World.update`` as ``TensorDict({entity_name: action})``, which routes it to
    ``Animal.update(action=...)`` where it *overrides* the rule-based policy's
    movement direction (metabolism, eating, death and reproduction still run
    normally). Cells with no herbivore in them simply have their action ignored,
    so the action space is deliberately dense and position-indexed: it matches
    the tensor the simulation already consumes, with no packing/unpacking layer
    that could silently mis-map cells.

**Reward** -- ``float``: the number of living herbivore cells after the step.

    "Living" means ``biomass >= survival_threshold``, the same test the
    simulation uses to kill animals. The undiscounted return is therefore the
    total number of herbivore-steps survived during the episode, which is the
    quantity we actually want to maximise when asking "can a learned policy beat
    the rule-based policy at herbivore survival?". It needs no tuning constants
    and is directly comparable between a learned policy and the rule-based
    baseline run through the same env.

**Termination** -- the herbivore population reaches zero (no cell has
``biomass >= survival_threshold``). There is nothing left to control and the
population can never recover, so the episode ends.

**Truncation** -- after ``max_steps`` simulation steps. A tensor-beasts world
otherwise runs forever.

**Info** -- ``{"population": int, "step": int}``.

Caveats
-------

* The world is built once and reused across episodes via ``World.reset()``,
  which is about five times cheaper than rebuilding. ``reset()`` re-randomises
  from the current global RNG state, so a fresh episode is not bit-identical to
  a freshly constructed world; seeding makes it repeatable, which is what
  matters here.
* The env does not touch ``torch.set_default_device``. If you want the
  simulation on a non-CPU device, set the default device yourself before
  constructing the env, the way ``tensor_beasts/main.py`` does.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

try:  # gymnasium is an optional import so this module can be imported without it
    import gymnasium as gym
    from gymnasium import spaces

    _GYM_IMPORT_ERROR: Optional[Exception] = None
    _EnvBase = gym.Env
except ImportError as exc:  # pragma: no cover - exercised only without gymnasium
    gym = None
    spaces = None
    _GYM_IMPORT_ERROR = exc
    _EnvBase = object

from tensor_beasts.world import World

# Direction encoding used by tensor_beasts.entities.helpers.animal_helpers.
NUM_ACTIONS = 5
DIRECTIONS = ("stay", "up", "down", "left", "right")

DEFAULT_ENTITY = "Herbivore"
DEFAULT_MAX_STEPS = 1000


class TensorBeastsEnv(_EnvBase):
    """Single-agent Gymnasium view of a tensor-beasts world (herbivore control).

    See the module docstring for the observation/action/reward/termination
    contract.

    Args:
        world_config: The ``world`` subtree of a loaded config, i.e.
            ``load_config(path).world``.
        entity_name: Registry name of the controlled entity. Defaults to
            ``"Herbivore"``.
        max_steps: Episode length before truncation.
    """

    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        world_config: DictConfig,
        entity_name: str = DEFAULT_ENTITY,
        max_steps: int = DEFAULT_MAX_STEPS,
    ):
        if gym is None:  # pragma: no cover - exercised only without gymnasium
            raise ImportError(
                "TensorBeastsEnv requires gymnasium. Install it with "
                "`pip install gymnasium`."
            ) from _GYM_IMPORT_ERROR

        super().__init__()

        self.world_config = world_config
        self.entity_name = entity_name
        self.max_steps = max_steps
        self.size: Tuple[int, int] = tuple(world_config.size)

        self.world = self._build_world()
        if entity_name not in self.world.entity_dict:
            raise ValueError(
                f"Entity '{entity_name}' is not present in the world config. "
                f"Available entities: {sorted(self.world.entity_dict)}"
            )

        obs = self._observation()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(
            np.full(self.size, NUM_ACTIONS, dtype=np.int64),
            dtype=np.int64,
        )
        self.reward_range = (0.0, float(self.size[0] * self.size[1]))

        self._elapsed_steps = 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_world(self) -> World:
        """Construct and initialize a World."""
        world = World(self.world_config)
        world.initialize()
        return world

    @property
    def entity(self):
        return self.world.entity_dict[self.entity_name]

    def _alive_mask(self) -> torch.Tensor:
        entity = self.entity
        return entity.biomass.data >= entity.config.survival_threshold

    def population(self) -> int:
        """Number of living cells of the controlled entity."""
        return int(self._alive_mask().sum().item())

    def _observation(self) -> np.ndarray:
        obs = self.world.observable
        return obs.detach().to("cpu", torch.float32).numpy()

    def _info(self) -> Dict[str, Any]:
        return {"population": self.population(), "step": self._elapsed_steps}

    def _to_action_tensor(self, action) -> torch.Tensor:
        """Coerce a sampled action into the (H, W) int64 tensor World expects."""
        if isinstance(action, torch.Tensor):
            tensor = action.detach()
        else:
            tensor = torch.as_tensor(np.asarray(action))

        tensor = tensor.reshape(*self.size).to(
            device=self.entity.biomass.data.device, dtype=torch.long
        )
        return tensor

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            # World initialization and every simulation step draw from torch's
            # global RNG, so this is what actually makes an episode reproducible.
            torch.manual_seed(seed)

        # Reusing the world costs about a fifth of rebuilding it, which matters
        # when an agent resets thousands of times during training.
        self.world.reset()
        self._elapsed_steps = 0
        return self._observation(), self._info()

    def step(
        self, action
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action_tensor = self._to_action_tensor(action)
        self.world.update(
            TensorDict({self.entity_name: action_tensor}, batch_size=[])
        )
        self._elapsed_steps += 1

        population = self.population()
        reward = float(population)
        terminated = population == 0
        truncated = (not terminated) and self._elapsed_steps >= self.max_steps

        return self._observation(), reward, terminated, truncated, self._info()

    def step_builtin(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance one step letting the simulation's own policy choose movement.

        This is a baseline-only escape hatch, not part of the Gymnasium API. It
        exists so the rule-based policy can be scored through exactly the same
        reward, termination and truncation rules as a learned policy, which is
        what makes the comparison meaningful. Everything except the source of
        the movement decision is identical to ``step``.
        """
        self.world.update()
        self._elapsed_steps += 1

        population = self.population()
        reward = float(population)
        terminated = population == 0
        truncated = (not terminated) and self._elapsed_steps >= self.max_steps

        return self._observation(), reward, terminated, truncated, self._info()

    def close(self):
        return None


def make_env(
    config_path: str = "conf/base/simulation.yaml",
    size: Optional[Tuple[int, int]] = None,
    entity_name: str = DEFAULT_ENTITY,
    max_steps: int = DEFAULT_MAX_STEPS,
    device: Optional[str] = None,
) -> TensorBeastsEnv:
    """Load a simulation config from disk and wrap it in a TensorBeastsEnv.

    Args:
        config_path: Path to a config file (validated by
            ``tensor_beasts.config.load_config``).
        size: Optional ``(height, width)`` override for the world.
        entity_name: Registry name of the controlled entity.
        max_steps: Episode length before truncation.
        device: Optional override for ``world.device``. This only sets the
            config value; set ``torch.set_default_device`` yourself if you want
            tensors allocated somewhere other than the process default.
    """
    from tensor_beasts.config import load_config

    config = load_config(config_path)
    if size is not None:
        config.world.size = list(size)
    if device is not None:
        config.world.device = device
    return TensorBeastsEnv(
        config.world, entity_name=entity_name, max_steps=max_steps
    )

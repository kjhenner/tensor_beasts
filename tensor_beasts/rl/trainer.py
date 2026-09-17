"""Training loop: collect a segment of world time, learn from it, repeat.

Three things here are specific to this simulation rather than boilerplate.

**There is no episode boundary to collect up to.** The world runs forever and
individuals are born and die inside it, so the loop collects fixed-length
segments of *world* time and lets ``done`` mark the ends of individual
episodes. Nothing is ever reset between segments; the ecology carries on.

**The comparison against the rule-based policy is the experiment.** It is run
on a separate world through :meth:`MultiAgentWorldEnv.rule_based_step`, which
applies the identical reward bookkeeping, from the same seed as the learned
policy's evaluation world, and both numbers are logged side by side on every
evaluation. A training curve that is not next to that number does not say
whether anything was learned.

**Per-individual episode statistics have to follow the individuals.** An
animal's accumulated return lives in the cell it is standing in, and next step
it is standing somewhere else, so the accumulator is scattered through the same
successor map that :func:`~tensor_beasts.rl.rollout.compute_gae` uses. See
:class:`EpisodeTracker`.

Resuming restores the network, the optimizer and the step counters, but not the
world. The ecology restarts from a fresh initialization, which is the honest
thing to say about it: world state is a large tensordict and the simulation
offers no serialization for it. For a persistent-world task that is a real
discontinuity in the data distribution at every resume, and it is a reason to
prefer one long run over several resumed ones.
"""

import configparser
import json
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from tensor_beasts.rl.multiagent import MultiAgentWorldEnv, NUM_ACTIONS
from tensor_beasts.rl.networks import ActorCritic, build_network
from tensor_beasts.rl.normalization import ValueNormalizer
from tensor_beasts.rl.ppo import MAX_LOG_STD, METABOLIC_ANCHOR_STD, MIN_LOG_STD, PPO, PPOConfig
from tensor_beasts.rl.rollout import RolloutBuffer, compute_gae

# The project's default W&B server. A different value on TrainerConfig is taken
# as an explicit choice; this one defers to the user's own wandb settings, since
# that is the host their API key is stored against. See resolve_wandb_host.
DEFAULT_WANDB_HOST = "http://localhost:8080"


@dataclass
class TrainerConfig:
    """Everything the loop needs that is not a PPO hyperparameter.

    Attributes:
        config_path: Simulation config. ``conf/basic_config.yaml`` is the
            baseline world.
        size: World side length. Below roughly 256 the predator population
            collapses and the three-species dynamic degenerates, so 256 is for
            iteration and 512 is for results.
        entity: Which entity the policy controls.
        survival_reward: Reward per step an individual stays alive. Leave at 1
            so that the summed reward is literally herbivore-steps survived,
            the number ``tools/evaluate_policy.py`` reports.
        reproduction_reward: Reward for dividing.
        foraging_reward: Reward per unit of biomass an individual gains in a
            step. Dense and action-dependent, unlike survival, which is nearly
            constant at 99.4% per step. Zero by default because it is reward
            shaping; evaluation stays herbivore-steps survived either way.
        arch: Network name from ``tensor_beasts.rl.networks.ARCHITECTURES``.
        arch_kwargs: Extra constructor arguments for that network.
        metabolic: Let the policy set its own metabolic rate through a second
            network head. The throttle is a continuous rate the policy emits,
            learned alongside movement and anchored on the same schedule, not a
            choice among discrete settings. False leaves the metabolic rate to
            the simulation's rules and learns movement only, which is the
            default so existing runs reproduce.
        device: "auto", "cpu", "mps", "cuda".
        seed: Seed for the training world and the torch RNG.
        total_world_steps: Length of the run, in world steps.
        segment_steps: World steps per collected segment, i.e. the PPO batch.
        warmup_steps: World steps to run under the rule-based policy before
            training starts, so learning does not begin on a just-seeded world
            that has not settled into its ecology yet.
        eval_interval: World steps between evaluations. 0 disables.
        eval_steps: World steps per evaluation run.
        eval_seeds: Number of paired evaluation worlds, each with its own seed.
            The learned and rule-based policies are scored on the same seeds.
        eval_deterministic: Take the argmax action at evaluation instead of
            sampling.
        checkpoint_interval: World steps between checkpoints. 0 disables.
        output_dir: Where checkpoints and the log go.
        log_name: JSONL log filename inside ``output_dir``.
        wandb: Whether to mirror the log to Weights & Biases. Off by default and
            entirely optional; the repo has wandb installed and old sweep
            directories, but nothing here requires it.
        wandb_project: Project name if wandb is on.
        wandb_host: W&B server to log to. Defaults to the local server; None
            defers to wandb's own resolution (WANDB_BASE_URL, then wandb.ai).
    """

    config_path: str = "conf/basic_config.yaml"
    size: int = 256
    entity: str = "Herbivore"
    survival_reward: float = 1.0
    reproduction_reward: float = 10.0
    foraging_reward: float = 0.0
    # Fraction of an offspring's biomass at birth credited back to its parent.
    # Division halves the parent's biomass, so a reward in biomass alone is
    # maximized by never dividing; this makes it an investment. First
    # generation only, for bounded variance. 0 disables. See planning/06.
    offspring_credit: float = 0.0
    # Independent worlds stepped together, so each update's batch is drawn
    # across decorrelated ecologies rather than through one world's timeline.
    # Measured at 512 on a 3090: four worlds cost 5% more wall-clock than one,
    # 13.1 ms a step against 12.5, because the simulation is launch-bound. The
    # card saturates between four and eight. 1 reproduces every earlier run.
    # See planning/07-batched-worlds.md.
    worlds: int = 1

    arch: str = "conv"
    arch_kwargs: Dict[str, object] = field(default_factory=dict)
    metabolic: bool = False
    # Channels of learned memory each individual carries; 0 disables it.
    memory_size: int = 0
    # Reset a world whose controlled population has been extinct for this many
    # consecutive segments. An extinct world produces no transitions and, the
    # simulation having no immigration, never repopulates, so leaving it in
    # the batch trains on nothing: a predator run once spent 89% of its world
    # steps that way. It used to stop the run instead, which treated one
    # world's fate as the run's verdict and, worse, punished exactly the runs
    # that met a bust early. The world is reset and warmed up under the rules
    # in place, the other worlds are untouched, and the run spends its whole
    # budget. Extinction stays visible as `world_resets` in the log. 0
    # disables the reset and an extinct world simply stays empty.
    extinction_patience: int = 3
    # Supervised updates on rule-based rollouts before RL starts, each over one
    # segment of world steps. Zero skips it. See Trainer.pretrain for why a
    # small population needs it.
    pretrain_updates: int = 0
    # Evaluation only. Hold the learned policy's throttle fixed at this unit in
    # [0, 1], where 0 is the basal rate and 1 the configured maximum, so the
    # throttle's contribution can be separated from movement's. None leaves the
    # throttle to the network, or to the rules for a direction-only policy.
    # Requires an environment with a metabolic head.
    eval_pin_metabolic: Optional[float] = None
    device: str = "auto"
    seed: int = 0

    total_world_steps: int = 20_000
    segment_steps: int = 64
    warmup_steps: int = 100

    eval_interval: int = 2_000
    eval_steps: int = 500
    eval_seeds: int = 2
    eval_deterministic: bool = False
    # World steps each evaluation world runs under the rules, from its reset,
    # before either policy is scored on it. A fresh world spends its first
    # several hundred steps in a startup transient (predators dip to a third
    # of their starting count and recover), and a 400-step evaluation from the
    # reset measured that transient rather than the settled cycles training
    # happens in: the rules' smoothed biomass at step 400 was a fifth of their
    # steady-state level. Both policies start from the same warmed state, so
    # the comparison stays paired. 0 reproduces the old behaviour.
    eval_warmup_steps: int = 0

    checkpoint_interval: int = 2_000
    output_dir: str = "outputs/rl"
    log_name: str = "train_log.jsonl"

    # World steps between individual-following films. Deliberately much rarer
    # than evaluation: a film costs a full world snapshot per recorded step,
    # which is the most memory-hungry thing the trainer does, and its value is
    # in watching the policy change over a run rather than every few minutes.
    # 0 disables.
    film_interval: int = 0
    # World steps recorded per film. The followed individuals must live and die
    # inside this window for their returns to be complete.
    film_steps: int = 300
    # Crop side length in world cells, and the nearest-neighbour upscale.
    film_window: int = 48
    film_scale: int = 5
    # Which display config to render through, by title, from the simulation's
    # own color_displays. "layers" shows plants, herbivores and predators at once.
    film_display: str = "layers"

    # Predict normalized values. Off makes the ablation runnable; see
    # tensor_beasts/rl/normalization.py for why it should normally stay on.
    normalize_values: bool = True

    wandb: bool = False
    wandb_project: str = "tensor-beasts-rl"
    wandb_host: Optional[str] = DEFAULT_WANDB_HOST

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# Metrics that are only produced on an evaluation step, grouped so they form
# their own sparse series instead of gaps in a dense one.
_EVAL_PREFIXES = ("learned_", "rule_based_")
_EVAL_KEYS = frozenset({"learned_over_rule_based", "biomass_over_rule_based", "score"})
# Run-level summaries of the ratio, computed across evaluations. A sweep should
# optimise eval/ratio_mean_late rather than the last value; see
# Trainer._record_eval_summary.
_EVAL_SUMMARY_PREFIX = "eval_score_"
# Metrics produced on a film step.
_FILM_PREFIX = "film_"


def wandb_record(record: Dict[str, object]) -> Dict[str, object]:
    """Group one log record into stable W&B namespaces.

    Two things went wrong without this, and both show up as charts that look
    broken rather than as errors.

    **A metric must keep one name for the whole run.** Pretraining and training
    both report ``population`` and ``argmax_agreement``. If one phase logs them
    under a prefix and the other does not, W&B draws two half-empty charts for
    one quantity: one that stops when pretraining ends and one that starts
    there. Phase goes in a ``phase`` metric, never in the metric names.

    **Sparse metrics must be separated from dense ones.** Evaluation runs every
    few thousand world steps, so ``learned_over_rule_based`` has a value on two
    rows out of a hundred and fifty. Mixed in with per-segment metrics that is
    read as a line with enormous gaps, and W&B's step interpolation fills the
    space between two distant points as though the value held there. Putting
    them under ``eval/`` keeps them a series of their own, plotted as the
    handful of points they actually are.
    """
    out: Dict[str, object] = {}
    for key, value in record.items():
        if not isinstance(value, (int, float, bool)) or isinstance(value, bool):
            # Strings such as the checkpoint path and the film paths are not
            # metrics; W&B renders them as a table column, which is noise on a
            # chart. Keep the phase, as a metric, so a chart can be split on it.
            if key == "phase":
                out["phase"] = 0 if value == "pretrain" else 1
            continue
        if value != value:  # NaN: a metric that was not measured this step
            continue
        if key.startswith(_EVAL_SUMMARY_PREFIX) or key == "eval_count":
            out[f"eval/{key[5:] if key.startswith('eval_') else key}"] = value
        elif key.startswith(_EVAL_PREFIXES) or key in _EVAL_KEYS:
            out[f"eval/{key}"] = value
        elif key.startswith(_FILM_PREFIX):
            out[f"film/{key[len(_FILM_PREFIX):]}"] = value
        else:
            out[key] = value
    return out


def resolve_wandb_host(configured: Optional[str]) -> Optional[str]:
    """The W&B server to log to, preferring what the user has already set up.

    The credential lookup is by exact host string. A key stored for
    ``0.0.0.0:8080`` is not found when the base URL says ``localhost:8080``,
    even though both reach the same server, and wandb then fails with "No API
    key configured" rather than saying the host did not match. This project
    defaulted to ``localhost`` while the local server here was configured as
    ``0.0.0.0``, so ``--wandb`` could not log to it at all.

    So the user's own ``~/.config/wandb/settings`` wins over this project's
    default, which is right in general: whatever host they logged in against is
    the host their key is stored under. An explicit ``--wandb-host`` still wins
    over both, and None defers to wandb's own resolution.
    """
    if configured is not None and configured != DEFAULT_WANDB_HOST:
        return configured

    settings_path = Path.home() / ".config" / "wandb" / "settings"
    try:
        parser = configparser.ConfigParser()
        parser.read(settings_path)
        for section in parser.sections():
            base_url = parser[section].get("base_url")
            if base_url:
                return base_url.strip()
    except (OSError, configparser.Error):
        pass
    return configured


def resolve_device(name: str) -> torch.device:
    """Pick a device, preferring the accelerator with room to work in.

    ``"cuda"`` without an index means ``cuda:0``, and which physical card that
    is depends on ``CUDA_DEVICE_ORDER``. On a mixed machine the two orderings
    disagree: with ``PCI_BUS_ID``, which is what ``nvidia-smi`` prints,
    ``cuda:0`` is whichever card sits at the lower bus address, while CUDA's own
    default of ``FASTEST_FIRST`` puts the fastest card there instead. So an
    index is not a stable name for a card, and a sweep agent that inherits no
    ``CUDA_VISIBLE_DEVICES`` lands on whatever ``cuda:0`` happens to mean. Here
    that was an 11 GB card already holding 4.5 GB of someone else's work, and a
    512 world needs about 6 GB, so the trial died in pretraining.

    ``"auto"`` and a bare ``"cuda"`` therefore select by free memory rather than
    by index. An explicit ``"cuda:1"`` is left alone: that is the caller naming
    a card, and second-guessing it would be worse than obeying it.
    """
    if name not in ("auto", "cuda"):
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{_most_free_cuda_device()}")
    if name == "cuda":
        raise RuntimeError("CUDA was requested but torch reports no CUDA device.")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _most_free_cuda_device() -> int:
    """Index of the CUDA device with the most free memory right now.

    Free rather than total: a big card that another process has filled is worse
    than a small idle one, and this machine runs other things on its GPUs.
    """
    best, best_free = 0, -1
    for index in range(torch.cuda.device_count()):
        try:
            free, _total = torch.cuda.mem_get_info(index)
        except Exception:  # noqa: BLE001 - a device that cannot be queried is not a candidate
            continue
        if free > best_free:
            best, best_free = index, free
    return best


class EpisodeTracker:
    """Accumulates per-individual return and length, following them as they move.

    The accumulator is a flat ``(H*W,)`` buffer indexed by the cell an
    individual currently occupies. After each step, survivors' totals are
    scattered to their successor cells; newborns start from a zeroed cell, which
    is correct, since an offspring's episode begins at birth. Totals of
    individuals that died are harvested and cleared.

    Episodes still running when the segment ends are simply not reported, which
    biases the reported mean episode length short in exactly the way every
    on-policy logger does.
    """

    def __init__(self, size: Tuple[int, ...], device: torch.device):
        # ``size`` may carry a leading batch of worlds. The accumulator is
        # (worlds, H * W) rather than a flat (H * W,): successor indices are per
        # world, so a single flat buffer would scatter one world's survivors
        # into another world's cells. At one world this is (1, H * W) and the
        # arithmetic is unchanged. See planning/07-batched-worlds.md.
        self.cells = size[-2] * size[-1]
        self.worlds = 1
        for extent in size[:-2]:
            self.worlds *= extent
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.ret = torch.zeros(self.worlds, self.cells, device=self.device)
        self.length = torch.zeros(self.worlds, self.cells, device=self.device)
        self.finished_return: List[float] = []
        self.finished_length: List[float] = []

    def _per_world(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(self.worlds, self.cells)

    def update(self, batch) -> None:
        acted = self._per_world(batch.acted)
        done = self._per_world(batch.done)
        successor = self._per_world(batch.successor)

        current_return = self.ret + self._per_world(batch.reward) * acted
        current_length = self.length + acted.float()

        finished = done
        if bool(finished.any()):
            self.finished_return.extend(current_return[finished].tolist())
            self.finished_length.extend(current_length[finished].tolist())

        survivors = acted & ~done
        next_return = torch.zeros_like(self.ret)
        next_length = torch.zeros_like(self.length)
        # scatter within each world's own row
        world_index = torch.arange(self.worlds, device=self.device).unsqueeze(1).expand_as(successor)
        rows = world_index[survivors]
        index = successor[survivors]
        next_return[rows, index] = current_return[survivors]
        next_length[rows, index] = current_length[survivors]
        self.ret = next_return
        self.length = next_length

    def summary(self) -> Dict[str, float]:
        if not self.finished_return:
            return {
                "episode_return": float("nan"),
                "episode_length": float("nan"),
                "episodes_finished": 0.0,
            }
        returns = torch.tensor(self.finished_return)
        lengths = torch.tensor(self.finished_length)
        return {
            "episode_return": float(returns.mean()),
            "episode_length": float(lengths.mean()),
            "episodes_finished": float(len(self.finished_return)),
        }


@dataclass
class EvalResult:
    """What one scored run of a policy produced.

    The headline is ``biomass_ema``: the controlled entity's total carried
    biomass, exponentially smoothed over the run. It is a physical quantity
    with a unit, measurable without reference to any other policy.

    What it replaced was a ratio against the hand-written rule-based policy,
    and the reason is that the denominator was arbitrary. That policy's
    navigation weights, metabolic sensitivity and log scale were all chosen by
    hand, so dividing by its score made every result a statement about those
    particular constants. This document's own record contains a case of that
    going wrong: the throttle finding looked like a discovery about metabolism
    and turned out to be integer truncation in the baseline. The ratio moved
    because the denominator was broken.

    Worse, the baseline is not a fixed reference. Predators and herbivores
    share a world, so changing the learned predator changes the prey
    population, which changes what the rule-based predator would have scored.
    The denominator moved in response to the numerator.

    Biomass rather than population count because it is what the ecology
    actually conserves: an individual carries biomass, eats it, burns it and
    halves it into its offspring. A population count weights a starving animal
    about to die the same as a thriving one about to divide.

    Smoothed rather than averaged because the predator-prey cycle swings by a
    factor of four within a single run, so a plain mean is dominated by which
    phase the window happened to catch. The EMA weights recent state more
    heavily and settles toward the level the policy sustains.
    """

    biomass_ema: List[float]
    mean_biomass: List[float]
    final_biomass: List[float]
    mean_population: List[float]
    reproductions: List[float]
    survived_agent_steps: List[float]
    total_reward: List[float]
    episode_return: float
    episode_length: float
    episodes_finished: float

    def to_dict(self, prefix: str) -> Dict[str, float]:
        """Mean over worlds for each per-world field, scalars passed through."""
        out: Dict[str, float] = {}
        for key, value in asdict(self).items():
            if isinstance(value, list):
                out[f"{prefix}_{key}"] = sum(value) / len(value) if value else 0.0
            else:
                out[f"{prefix}_{key}"] = value
        return out

    def per_world(self, prefix: str) -> Dict[str, List[float]]:
        """The per-world values themselves, for reporting the spread."""
        return {
            f"{prefix}_{key}": value
            for key, value in asdict(self).items()
            if isinstance(value, list)
        }


def _observe(env: MultiAgentWorldEnv) -> torch.Tensor:
    """Current observation.

    The environment returns the observation an action was taken *from* out of
    ``step``, and only ``reset`` hands one back up front, so there is no public
    way to look at the state before choosing the next action. A public
    ``observation()`` accessor on ``MultiAgentWorldEnv`` would be the clean fix;
    until then this is the single place that reaches for the private builder.
    """
    return env._build_observation()


def policy_input(observation: torch.Tensor) -> torch.Tensor:
    """Round-trip the observation through float16 before the network sees it.

    ``RolloutBuffer`` stores observations as float16 to keep a segment in
    memory, and the update reads them back as float32. If the network saw the
    full-precision tensor at collection time, the log-probabilities recomputed
    in the update would differ from the stored ones in the last few bits, and
    the PPO ratio would not be exactly 1 on the first epoch. That is a tiny
    numerical error but it destroys the cheapest correctness check available,
    so collection is made to see exactly the tensor the update will see.
    """
    return observation.to(torch.float16).float()


def checkpoint_metabolic(payload: Dict[str, object]) -> bool:
    """Whether a checkpoint carries a metabolic head.

    Newer checkpoints record ``metabolic`` directly. Ones from the discrete
    design recorded ``num_metabolic_levels``, where any positive count meant the
    head was present, so those still load without a translation step.
    """
    if "metabolic" in payload:
        return bool(payload["metabolic"])
    if "num_metabolic_levels" in payload:
        return int(payload["num_metabolic_levels"]) > 0
    config = payload.get("trainer_config") or {}
    if "metabolic" in config:
        return bool(config["metabolic"])
    return int(config.get("metabolic_levels", 0)) > 0


class Trainer:
    """Collect, learn, evaluate, checkpoint."""

    def __init__(
        self,
        config: Optional[TrainerConfig] = None,
        ppo_config: Optional[PPOConfig] = None,
    ):
        self.config = config or TrainerConfig()
        self.ppo_config = ppo_config or PPOConfig()
        self.device = resolve_device(self.config.device)

        # The simulation places its tensors on the global default device rather
        # than taking one as an argument (see tensor_beasts/main.py, which does
        # the same thing), so this has to be set before any world is built or
        # the world lands on CPU while the network sits on the accelerator.
        torch.set_default_device(self.device)
        torch.manual_seed(self.config.seed)

        self.env = self._make_env()
        self.observation_channels = self.env.observation_channels
        # Build on CPU and move: orthogonal init goes through torch.linalg.qr,
        # which MPS does not implement, and the global default device is now the
        # accelerator.
        with torch.device("cpu"):
            network = build_network(
                self.config.arch,
                self.observation_channels,
                metabolic=self.config.metabolic,
                memory_size=self.config.memory_size,
                **dict(self.config.arch_kwargs),
            )
        self.network: ActorCritic = network.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.ppo_config.learning_rate
        )
        # The network predicts normalized values; see rl/normalization.py for the
        # measurement that motivates it. Without this the value term is 100% of
        # the gradient norm and global clipping starves the policy.
        self.value_normalizer = ValueNormalizer(enabled=config.normalize_values)
        self.algorithm = PPO(self.ppo_config, value_normalizer=self.value_normalizer)

        self.size = self.env.size
        self.world_steps = 0
        self.updates = 0
        # Worlds replaced after extinction, over the run; see reset_extinct_worlds.
        self.world_resets = 0
        self._extinct_segments: Optional[torch.Tensor] = None
        self.agent_steps = 0
        self.start_time = time.time()
        self._next_eval = 0
        self._next_checkpoint = self.config.checkpoint_interval
        self._next_film = self.config.film_interval

        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / self.config.log_name

        self._eval_envs: Dict[int, MultiAgentWorldEnv] = {}
        self._eval_ratios: List[Tuple[int, float]] = []
        self._wandb = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _make_env(self, worlds: Optional[int] = None) -> MultiAgentWorldEnv:
        config = self.config
        return MultiAgentWorldEnv(
            config_path=config.config_path,
            size=(config.size, config.size),
            entity_name=config.entity,
            survival_reward=config.survival_reward,
            reproduction_reward=config.reproduction_reward,
            foraging_reward=config.foraging_reward,
            offspring_credit=config.offspring_credit,
            worlds=config.worlds if worlds is None else worlds,
            device=str(self.device),
            # Pinning needs the unit-to-rate mapping even for a direction-only
            # policy, so the environment carries the bounds either way.
            metabolic=bool(config.metabolic) or config.eval_pin_metabolic is not None,
            memory_size=config.memory_size,
        )

    def _init_wandb(self) -> None:
        if not self.config.wandb or self._wandb is not None:
            return
        import wandb  # imported lazily: wandb is optional and off by default

        host = resolve_wandb_host(self.config.wandb_host)
        run = wandb.init(
            project=self.config.wandb_project,
            config={**self.config.to_dict(), **self.ppo_config.to_dict()},
            settings=wandb.Settings(base_url=host) if host else None,
        )
        # Plot everything against world steps rather than against W&B's own
        # increment-per-log-call counter. Without this the x-axis counts log
        # calls, so pretraining's thirty updates and training's hundreds share
        # an axis that means nothing, and the step a metric appears at does not
        # match the step in the JSONL log.
        run.define_metric("world_steps")
        run.define_metric("*", step_metric="world_steps")
        print(f"wandb: {run.url}", flush=True)
        self._wandb = wandb

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------
    @staticmethod
    def _sample(logits: torch.Tensor, deterministic: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample (or argmax) one categorical per cell from ``(1, K, H, W)`` logits.

        Returns (action, log_prob), both ``(1, H, W)``.
        """
        log_probs = torch.log_softmax(logits, dim=1)
        if deterministic:
            action = log_probs.argmax(dim=1)
        else:
            flat = log_probs.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
            action = torch.multinomial(flat.exp(), 1).reshape(logits.shape[0], *logits.shape[2:])
        log_prob = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        return action, log_prob

    @staticmethod
    def _sample_metabolic(
        mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One continuous throttle per cell from a Gaussian around ``mean``.

        The throttle is a rate, not a choice among settings, so the policy is a
        Gaussian on the unit interval rather than a categorical. Returns
        (unit, log_prob), both the shape of ``mean``. The sample is clamped to
        [0, 1] because that is the interval the environment maps onto
        [basal, max]; the log-probability is the unclamped Gaussian's, which is
        the quantity PPO's ratio is defined over on both sides.
        """
        log_std = log_std.clamp(MIN_LOG_STD, MAX_LOG_STD)
        std = log_std.exp()
        if deterministic:
            unit = mean
        else:
            unit = (mean + std * torch.randn_like(mean)).clamp(0.0, 1.0)
        log_prob = (
            -0.5 * ((unit - mean) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)
        )
        return unit, log_prob

    @torch.no_grad()
    def act(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
        deterministic_metabolic: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Sample an action for every cell.

        Returns (action, log_prob, value, metabolic_unit, memory).
        ``metabolic_unit`` is None unless the network has a metabolic head, in
        which case it is a continuous throttle in [0, 1] sampled from a
        Gaussian around the head's mean and ``log_prob`` is the joint
        log-probability, the quantity the PPO ratio is defined over.
        ``deterministic_metabolic`` takes the head's mean as the throttle
        while leaving the direction as ``deterministic`` says; None follows
        ``deterministic`` for both.

        Actions are produced for the whole grid, empty cells included. The
        simulation ignores directions at cells with nobody in them, and the loss
        masks those entries out, so the wasted work is a few hundred thousand
        multiply-adds that the convolution was doing anyway.
        """
        # A batched world already presents (B, C, H, W); an unbatched one is
        # (C, H, W) and needs a singleton for the convolution. Whichever was
        # added here is what gets taken off again, so a real batch of worlds is
        # never mistaken for the singleton and silently collapsed.
        batched = observation.dim() == 4
        network_input = observation if batched else observation.unsqueeze(0)

        def unwrap(tensor):
            return tensor if batched else tensor.squeeze(0)

        out = self.network.forward_all(network_input)
        action, log_prob = self._sample(out["logits"], deterministic)
        metabolic_unit = None
        if "metabolic_mean" in out:
            if deterministic_metabolic is None:
                deterministic_metabolic = deterministic
            metabolic_unit, metabolic_log_prob = self._sample_metabolic(
                out["metabolic_mean"], out["metabolic_log_std"], deterministic_metabolic
            )
            log_prob = log_prob + metabolic_log_prob
            metabolic_unit = unwrap(metabolic_unit)
        # The advantage recursion mixes rewards and bootstrapped values, so it
        # has to run in real return units, not normalized ones.
        value = self.value_normalizer.denormalize(out["value"])
        # Deterministic memory write; not an action in the policy-gradient
        # sense, so it carries no log-probability. Stage 1 of the memory
        # design: the read is learnable, the write is a fixed function of the
        # observation until recurrent training exists.
        memory = unwrap(out["memory"]) if "memory" in out else None
        return unwrap(action), unwrap(log_prob), unwrap(value), metabolic_unit, memory

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------
    def collect(self, steps: int) -> Tuple[object, Dict[str, float]]:
        """Run ``steps`` world steps under the current policy."""
        buffer = RolloutBuffer(steps)
        tracker = EpisodeTracker(self.env.field_shape, self.device)

        populations: List[float] = []
        reproductions = 0.0
        survived = 0.0

        for _ in range(steps):
            # The policy is asked for its action at the point in the step where
            # the entity's own policy would run, so a learner sees the same
            # world the rule-based baseline sees. Deciding beforehand, which is
            # what env.step does, hands an entity whose food updates before it
            # a stale observation: for the predator that halved hunting success
            # and drove the population extinct. See multiagent.step_with_policy.
            decided: Dict[str, object] = {}

            def decide(observation, decided=decided):
                observation = policy_input(observation)
                action, log_prob, value, metabolic_unit, memory = self.act(observation)
                decided.update(
                    observation=observation, log_prob=log_prob, value=value,
                )
                return action, metabolic_unit, memory

            batch = self.env.step_with_policy(decide)
            # The env rebuilt the observation itself; overwrite it with the
            # exact tensor the network saw, so stored and recomputed
            # log-probabilities agree bit for bit.
            batch.observation = decided["observation"]
            log_prob, value = decided["log_prob"], decided["value"]
            buffer.add(batch, log_prob, value)
            tracker.update(batch)

            populations.append(float(self.env.population()))
            reproductions += float(batch.reproduced.sum())
            survived += float((batch.acted & ~batch.done).sum())
            self.agent_steps += batch.num_agents

        rollout = buffer.build()
        with torch.no_grad():
            # Same rule as `act`: a batched world already has its leading axis,
            # and whichever axis is added here is the one taken off again.
            final = policy_input(_observe(self.env))
            batched = final.dim() == 4
            _, last_value = self.network(final if batched else final.unsqueeze(0))
            last_value = self.value_normalizer.denormalize(last_value)
            if not batched:
                last_value = last_value.squeeze(0)
        rollout = compute_gae(
            rollout,
            last_value,
            gamma=self.ppo_config.gamma,
            gae_lambda=self.ppo_config.gae_lambda,
            normalize=self.ppo_config.normalize_advantage,
        )

        agent_steps = rollout.num_agent_steps
        per_world = self.env.population_per_world()
        stats = {
            "population": sum(populations) / max(len(populations), 1),
            # The extremes across worlds, so a batch whose worlds sit at
            # different phases of the cycle can be seen to, and an extinct
            # world shows up before the reset does.
            "population_min_world": float(per_world.min()),
            "population_max_world": float(per_world.max()),
            "reward_per_agent_step": float(rollout.reward.sum()) / max(agent_steps, 1),
            "reproduction_rate": reproductions / max(agent_steps, 1),
            "survival_rate": survived / max(agent_steps, 1),
            "segment_agent_steps": float(agent_steps),
            **tracker.summary(),
        }
        return rollout, stats

    # ------------------------------------------------------------------
    # Evaluation: the actual experiment
    # ------------------------------------------------------------------
    def _eval_env(self, worlds: int) -> MultiAgentWorldEnv:
        """A cached evaluation world holding ``worlds`` independent seeds.

        Separate from the training environment because evaluation's batch width
        is the seed count, which has nothing to do with how many worlds the
        training loop steps.
        """
        if worlds not in self._eval_envs:
            self._eval_envs[worlds] = self._make_env(worlds=worlds)
        return self._eval_envs[worlds]

    @torch.no_grad()
    def _score(self, env: MultiAgentWorldEnv, steps: int, policy: str) -> EvalResult:
        tracker = EpisodeTracker(env.field_shape, self.device)
        worlds = env.num_worlds
        zeros = torch.zeros(worlds, device=self.device)
        total_reward = zeros.clone()
        survived = zeros.clone()
        reproductions = zeros.clone()
        populations: List[torch.Tensor] = []
        biomasses: List[torch.Tensor] = []
        # Smoothing constant for the headline. 2/(N+1) with N the window, so
        # this is a 100-step window: long enough that a single boom or crash
        # does not set the number, short enough that the last quarter of the
        # run dominates it, which is the level the policy actually sustains.
        ema_alpha = 2.0 / (100.0 + 1.0)
        biomass_ema: Optional[torch.Tensor] = None

        for _ in range(steps):
            if policy == "rule_based":
                batch = env.rule_based_step()
            else:
                # Decided at the same point in the step the rule-based baseline
                # decides, or the comparison measures the observation's timing
                # rather than the policy. See multiagent.step_with_policy.
                def decide(observation, env=env):
                    observation = policy_input(observation)
                    # The throttle is scored at the head's mean whatever the
                    # direction does. Its Gaussian spread is exploration, and
                    # sampling it is not merely noisy: the clamp to [0, 1]
                    # rectifies the noise into a hotter throttle than the
                    # policy chose, which is a bias, not a variance.
                    action, _, _, metabolic_unit, memory = self.act(
                        observation,
                        deterministic=self.config.eval_deterministic,
                        deterministic_metabolic=True,
                    )
                    if self.config.eval_pin_metabolic is not None:
                        # Evaluation-only: hold the throttle at a fixed unit so
                        # the throttle's effect separates from movement's.
                        # field_shape, not size: evaluation runs one world per
                        # seed, so an (H, W) tensor is the wrong shape for every
                        # eval_seeds > 1 and reshaping it raises.
                        metabolic_unit = torch.full(
                            env.field_shape, float(self.config.eval_pin_metabolic),
                            dtype=torch.float32, device=self.device,
                        )
                    # Memory must be written during evaluation exactly as in
                    # training. It was not, once: this call dropped `memory`, so
                    # every memory checkpoint was scored with its memory stuck
                    # at zero, and said nothing about memory either way.
                    return action, metabolic_unit, memory

                batch = env.step_with_policy(decide)

            tracker.update(batch)
            # Summed over the grid but NOT over the batch: each world is an
            # independent evaluation seed and must produce its own number, or
            # the batch silently averages worlds together before anyone can see
            # the spread between them.
            grid = (-2, -1)
            total_reward += batch.reward.sum(dim=grid).reshape(worlds)
            survived += (batch.acted & ~batch.done).sum(dim=grid).reshape(worlds).float()
            reproductions += batch.reproduced.sum(dim=grid).reshape(worlds).float()
            populations.append(env.population_per_world())

            # Total carried biomass, the quantity the ecology conserves.
            carried = env.entity.biomass.data.sum(dim=grid).reshape(worlds)
            biomasses.append(carried)
            biomass_ema = carried if biomass_ema is None else (
                ema_alpha * carried + (1.0 - ema_alpha) * biomass_ema
            )

        summary = tracker.summary()
        # Each field is one number per world, so the caller can average over
        # seeds AND see the spread between them. Episode statistics are pooled
        # across worlds on purpose: an episode is one individual's life, and
        # lives are comparable wherever they happened.
        stacked_biomass = torch.stack(biomasses) if biomasses else zeros.unsqueeze(0)
        stacked_population = torch.stack(populations) if populations else zeros.unsqueeze(0)
        ema = biomass_ema if biomass_ema is not None else zeros
        return EvalResult(
            biomass_ema=ema.tolist(),
            mean_biomass=stacked_biomass.mean(dim=0).tolist(),
            final_biomass=stacked_biomass[-1].tolist(),
            mean_population=stacked_population.mean(dim=0).tolist(),
            reproductions=reproductions.tolist(),
            survived_agent_steps=survived.tolist(),
            total_reward=total_reward.tolist(),
            episode_return=summary["episode_return"],
            episode_length=summary["episode_length"],
            episodes_finished=summary["episodes_finished"],
        )

    def evaluate(self) -> Dict[str, float]:
        """Score the learned policy and the rule-based baseline on the same worlds.

        Each seed gets a freshly reset world, and both policies are run from the
        same reset, so the comparison is paired. They diverge immediately
        afterwards, of course, because the policies consume the shared RNG
        differently and because the ecology is chaotic; the seeds control the
        starting conditions, not the trajectory.
        """
        # Evaluation reseeds the global torch RNG (MultiAgentWorldEnv.reset does,
        # by design, so eval worlds are comparable), which would otherwise make
        # the training stream depend on the evaluation schedule.
        rng_state = torch.get_rng_state()
        self.network.eval()
        seeds = max(self.config.eval_seeds, 1)
        seed = self.config.seed + 10_000

        summary: Dict[str, float] = {}
        spread: Dict[str, List[float]] = {}
        for policy in ("learned", "rule_based"):
            # One batched world holding every evaluation seed, rather than one
            # world per seed run in sequence. The seeds become the batch axis.
            # Both policies are scored from the same reset, so the comparison
            # stays paired exactly as it was when this was a loop.
            env = self._eval_env(seeds)
            env.reset(seed=seed)
            # Past the startup transient before anything is scored. The
            # reset seeded the RNG, so both policies see the same warmed
            # worlds.
            for _ in range(self.config.eval_warmup_steps):
                env.rule_based_step()
            scored = self._score(env, self.config.eval_steps, policy)
            summary.update(scored.to_dict(policy))
            spread.update(scored.per_world(policy))
        self.network.train()
        torch.set_rng_state(rng_state)

        # The spread across seeds is the quantity every claim in this project is
        # hedged against, so it is reported rather than averaged away.
        learned = spread.get("learned_biomass_ema") or []
        if len(learned) > 1:
            mean = sum(learned) / len(learned)
            variance = sum((value - mean) ** 2 for value in learned) / (len(learned) - 1)
            summary["score_spread"] = variance ** 0.5
            summary["score_min"] = min(learned)
            summary["score_max"] = max(learned)
        # The headline is the learned policy's own smoothed biomass, an absolute
        # quantity in the units the ecology conserves. It replaced a ratio
        # against the rule-based policy, whose constants were hand-chosen, so
        # every result was a statement about those constants rather than about
        # the ecology; see EvalResult for the full argument.
        summary["score"] = summary.get("learned_biomass_ema", 0.0)

        # The baseline is still scored and reported, as context rather than as
        # a divisor. The ratio is kept because it is what every result recorded
        # before this change is quoted in, so the record stays readable, but it
        # is no longer what a sweep optimises.
        baseline = summary.get("rule_based_survived_agent_steps", 0.0)
        summary["learned_over_rule_based"] = (
            summary.get("learned_survived_agent_steps", 0.0) / baseline if baseline else float("nan")
        )
        biomass_baseline = summary.get("rule_based_biomass_ema", 0.0)
        summary["biomass_over_rule_based"] = (
            summary["score"] / biomass_baseline if biomass_baseline else float("nan")
        )
        return summary

    # ------------------------------------------------------------------
    # Films
    # ------------------------------------------------------------------
    @torch.no_grad()
    def record_film(self, seed: Optional[int] = None) -> Dict[str, object]:
        """Follow two individuals through a run and write a video of each.

        Every other number this trainer reports is a sum over thousands of
        animals, which is the right way to decide whether a policy is better and
        a poor way to see *how*. This records one film from the typical band of
        the return distribution and one from the top decile, so what is watched
        is a representative life next to a good one rather than the single
        luckiest animal, which in a chaotic ecology looks impressive under any
        policy.

        Returns paths and the followed individuals' statistics, for the log.
        Never raises: a missing video encoder or a run where nothing died
        inside the window is a reason to skip the film, not to end training.
        """
        from tensor_beasts.config import load_config
        from tensor_beasts.rl.film import (
            IndividualTracker,
            display_keys,
            film_life,
            select_bands,
            thin_snapshot,
            write_video,
        )

        rng_state = torch.get_rng_state()
        self.network.eval()
        try:
            display_config = self._film_display_config(load_config)
            if display_config is None:
                return {}

            # One world, whatever the training batch is: a film follows one
            # individual through one ecology, and the tracker, the snapshots and
            # the crop all address a single (H, W) grid.
            env = self._make_env(worlds=1)
            env.reset(seed=self.config.seed + 20_000 if seed is None else seed)
            tracker = IndividualTracker(env.size, self.device)
            tracker.begin(env._alive())

            # Only the grids the renderer and the meters read, held on CPU. A
            # full World.snapshot is 21.8 MB a step at 512, so a 300-step film
            # would pin 6.4 GB of the accelerator the training is using.
            entity_key = self.config.entity.lower()
            keys = display_keys(display_config) + [
                (entity_key, "energy"),
                (entity_key, "biomass"),
            ]

            snapshots = []
            for index in range(self.config.film_steps):
                snapshots.append(thin_snapshot(env.world, keys, step=index))

                def decide(observation):
                    observation = policy_input(observation)
                    action, _, _, metabolic_unit, memory = self.act(
                        observation, deterministic=self.config.eval_deterministic
                    )
                    return action, metabolic_unit, memory

                batch = env.step_with_policy(decide)
                tracker.update(batch)
                tracker.observe_newcomers(env._alive())

            lives = tracker.completed_lives(min_steps=8)
            bands = select_bands(lives)
            if not bands:
                return {"film_skipped": "no individual both lived and died inside the window"}

            out: Dict[str, object] = {}
            films_dir = self.output_dir / "films"
            for band, life in bands.items():
                frames = film_life(
                    snapshots,
                    life,
                    display_config,
                    window=self.config.film_window,
                    scale=self.config.film_scale,
                    entity_name=self.config.entity.lower(),
                )
                path = write_video(
                    frames, films_dir / f"step{self.world_steps:07d}_{band}.mp4", fps=10
                )
                out[f"film_{band}_steps"] = life.steps_survived
                out[f"film_{band}_reward"] = life.reward
                out[f"film_{band}_reproductions"] = life.reproductions
                if path is not None:
                    out[f"film_{band}_path"] = str(path)
                    self._log_film(band, path)
            out["film_candidates"] = len(lives)
            return out
        except Exception as exc:  # noqa: BLE001 - a film is never worth a dead run
            return {"film_error": f"{type(exc).__name__}: {exc}"}
        finally:
            self.network.train()
            torch.set_rng_state(rng_state)

    def _record_eval_summary(self, record: Dict[str, object]) -> None:
        """Keep the summary statistics a sweep should actually optimize.

        W&B's summary holds the *last* value of each metric, and with this
        metric's measured 16% noise floor across evaluation seeds the last
        evaluation is one noisy draw. A sweep told to maximize it is largely
        ranking luck.

        Three summaries instead. ``eval/ratio_best`` is the best evaluation the
        run reached, which is what a checkpoint-selecting workflow would keep.
        ``eval/ratio_mean_late`` averages the evaluations from the second half
        of the run, which is the one to optimize: it is the level the policy
        settled at rather than a single sample of it, and averaging several
        evaluations is the only lever that beats the noise floor without more
        seeds. ``eval/ratio_last`` is kept for continuity.
        """
        score = record.get("score")
        if not isinstance(score, (int, float)) or score != score:
            return
        self._eval_ratios.append((int(self.world_steps), float(score)))

        scores = [value for _, value in self._eval_ratios]
        late = scores[len(scores) // 2:] or scores
        record["eval_score_best"] = max(scores)
        record["eval_score_mean_late"] = sum(late) / len(late)
        record["eval_score_last"] = scores[-1]
        record["eval_count"] = float(len(scores))

    def _film_display_config(self, load_config):
        """The display entry the films render through, or None if it is absent."""
        config = load_config(self.config.config_path)
        displays = config.display.color_displays
        for entry in displays:
            if entry.get("title") == self.config.film_display:
                return entry
        return displays[0] if displays else None

    def _log_film(self, band: str, path: Path) -> None:
        if self._wandb is None:
            return
        try:
            self._wandb.log(
                {f"film/{band}": self._wandb.Video(str(path), fps=10, format="mp4")},
                step=self.world_steps,
            )
        except Exception:  # noqa: BLE001 - logging a video must not end a run
            pass

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def checkpoint_payload(self) -> Dict[str, object]:
        return {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "trainer_config": self.config.to_dict(),
            "ppo_config": self.ppo_config.to_dict(),
            "observation_channels": self.observation_channels,
            # Recorded on its own as well as inside trainer_config, so the
            # viewer's controller rebuilds the right head without knowing the
            # trainer's field names.
            "metabolic": self.config.metabolic,
            "memory_size": self.config.memory_size,
            "world_steps": self.world_steps,
            "agent_steps": self.agent_steps,
            "updates": self.updates,
        }

    def save_checkpoint(self, path: Optional[Path] = None) -> Path:
        path = Path(path) if path is not None else self.output_dir / "checkpoint.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), path)
        return path

    def load_checkpoint(self, path: Path, load_optimizer: bool = True) -> None:
        """Restore weights, optimizer and counters. The world is not restored."""
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload["observation_channels"] != self.observation_channels:
            raise ValueError(
                f"Checkpoint was trained on {payload['observation_channels']} observation "
                f"channels, this environment has {self.observation_channels}. The config or "
                "the entity's perception changed."
            )
        metabolic = checkpoint_metabolic(payload)
        if metabolic != bool(self.config.metabolic):
            raise ValueError(
                f"Checkpoint was trained with metabolic={metabolic}, this trainer "
                f"has metabolic={bool(self.config.metabolic)}. Pass "
                f"{'--metabolic' if metabolic else '--no-metabolic'}."
            )
        memory = int(payload.get("memory_size", 0))
        if memory != self.config.memory_size:
            raise ValueError(
                f"Checkpoint was trained with memory size {memory}, this trainer has "
                f"{self.config.memory_size}. Pass --memory-size {memory}."
            )
        self.network.load_state_dict(payload["network"])
        if load_optimizer:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.world_steps = int(payload.get("world_steps", 0))
        self.agent_steps = int(payload.get("agent_steps", 0))
        self.updates = int(payload.get("updates", 0))
        # The anchor's fade counts RL updates; a resumed run must not re-anchor.
        if hasattr(self.algorithm, "rl_updates"):
            self.algorithm.rl_updates = self.updates
        self._next_eval = self.world_steps
        self._next_checkpoint = self.world_steps + self.config.checkpoint_interval

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def log(self, record: Dict[str, object]) -> None:
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        if self._wandb is not None:
            self._wandb.log(wandb_record(record), step=int(record.get("world_steps", self.world_steps)))

    @staticmethod
    def format_record(record: Dict[str, object]) -> str:
        parts = [f"step {record['world_steps']:>8}"]
        for key, label in (
            ("episode_return", "ep_ret"),
            ("episode_length", "ep_len"),
            ("population", "pop"),
            ("entropy", "H"),
            ("approx_kl", "kl"),
            ("explained_variance", "ev"),
            ("argmax_agreement", "agree"),
            ("metabolic_error", "m_err"),
            ("metabolic_head_mean", "m_rate"),
            ("metabolic_rule_mean", "m_rule"),
            ("metabolic_head_spread", "m_spread"),
            ("metabolic_std", "m_std"),
            ("world_steps_per_sec", "w/s"),
            ("agent_steps_per_sec", "a/s"),
        ):
            value = record.get(key)
            if isinstance(value, (int, float)) and value == value:  # skip NaN
                parts.append(f"{label}={value:.3g}")
        return "  ".join(parts)

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------
    def warmup(self) -> None:
        """Let the ecology settle under its own policy before learning starts."""
        for _ in range(self.config.warmup_steps):
            self.env.rule_based_step()

    def reset_extinct_worlds(self) -> int:
        """Replace every world whose controlled population has been extinct for
        ``extinction_patience`` consecutive segments with a fresh one, warmed
        up under the rules for ``warmup_steps``. Returns how many were reset.

        The fresh world is built as its own unbatched environment, run under
        the rules, and copied leaf by leaf into the extinct world's slice of
        the batched TensorDict, so the other worlds are not touched. Every
        batched leaf is the unbatched leaf with a leading world axis, which is
        what makes the copy a slice assignment; the one scalar (the water
        oscillator's phase) is shared by all worlds and is left alone.

        The fresh world's seed comes from the trainer's, offset by the reset
        count, and the global RNG is restored afterwards so the training
        stream does not depend on when a world happened to die.
        """
        patience = int(self.config.extinction_patience)
        if patience <= 0:
            return 0
        per_world = self.env.population_per_world()
        if self._extinct_segments is None or self._extinct_segments.numel() != per_world.numel():
            self._extinct_segments = torch.zeros_like(per_world)
        self._extinct_segments = torch.where(
            per_world <= 0, self._extinct_segments + 1, torch.zeros_like(per_world)
        )
        due = torch.nonzero(self._extinct_segments >= patience).flatten().tolist()
        if not due:
            return 0

        rng_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        destination = self.env.world.td
        batched = bool(self.env.world.batch_shape)
        for index in due:
            self.world_resets += 1
            fresh = self._make_env(worlds=1)
            fresh.reset(seed=self.config.seed + 20_000 + self.world_resets)
            for _ in range(self.config.warmup_steps):
                fresh.rule_based_step()
            source = fresh.world.td
            for key in source.keys(True, True):
                value = source.get(key)
                target = destination.get(key)
                if value.dim() == 0 or target.shape != (
                    (self.env.num_worlds,) + tuple(value.shape) if batched else tuple(value.shape)
                ):
                    continue
                if batched:
                    target[index].copy_(value)
                else:
                    target.copy_(value)
            self._extinct_segments[index] = 0
        torch.set_rng_state(rng_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        return len(due)

    def pretrain(self, verbose: bool = True) -> Dict[str, float]:
        """Supervised pretraining on the rule-based policy before any RL.

        The world runs under its own rules while the network learns to
        reproduce the rule's choices: soft distillation toward the rule's
        per-action scores when the imitation temperature is positive, hard
        cross-entropy to the rule's direction otherwise, plus the rule's
        metabolic throttle when the network has that head.

        Exists because a near-random initial policy is fatal to a small
        population. Twice, a learned predator took its population from 486 to
        zero within 200 steps at 512, before the imitation anchor could pull the
        policy toward anything that hunts. Herbivores survived the same start
        only because there are thousands of them. Starting RL from a policy
        that already behaves like the rule removes that cliff.

        The anchor's fade during RL is a schedule of updates (see PPOConfig),
        so it starts at full weight on the first RL update regardless of what
        pretraining reached, and is gone once the schedule runs out.
        """
        import torch.nn.functional as F

        updates = int(self.config.pretrain_updates)
        if updates <= 0:
            return {}
        ppo = self.ppo_config
        temperature = ppo.imitation_temperature
        steps = self.config.segment_steps
        last: Dict[str, float] = {}

        for update in range(updates):
            # Each step's fields are moved off the device as it is taken.
            # Holding the whole segment's AgentBatch objects on the GPU, which
            # is what this used to do, keeps every observation resident: at 512
            # with four worlds that is 3.88 GB of observations before anything
            # is stacked, and pretraining peaked at 16 GB against the 3.8 GB
            # that collection and the update need together. That made it the
            # most memory-hungry phase of a run, and the one the memory estimate
            # did not model at all.
            batches = []
            for _ in range(steps):
                batch = self.env.rule_based_step()
                for name in ("observation", "acted", "rule_action", "rule_scores",
                             "rule_metabolic_unit"):
                    field = getattr(batch, name, None)
                    if field is not None:
                        setattr(batch, name, field.detach().to("cpu", non_blocking=True))
                batches.append(batch)

            def stacked(select, half: bool = False):
                """Stack a field over the segment, folding worlds into the batch.

                Each field is (H, W) or (B, H, W), so stacking gives (T, H, W)
                or (T, B, H, W). The convolution wants one batch axis, and a
                timestep and a world are both just independent samples for a
                supervised fit, so the two fold together. Same rule as the RL
                minibatch iterator in ppo.py.

                The stack is held on the CPU and each minibatch is moved to the
                device as it is used. Measured at 512 with four worlds, holding
                it on the device made pretraining peak at 16 GB while collection
                and the update together needed 3.8 GB, so pretraining was by far
                the most memory-hungry phase of a run and the one the memory
                estimate did not model at all. It is a supervised pass over
                stored data, so the transfer is not on any critical path.

                ``half`` stores it as float16, as RolloutBuffer already does for
                the same tensors, which halves it again at no cost in fidelity:
                policy_input has already round-tripped the observation through
                float16 by this point.
                """
                stack = torch.stack([select(b) for b in batches])
                if half:
                    stack = stack.to(torch.float16)
                if self.env.num_worlds > 1:
                    stack = stack.reshape(
                        stack.shape[0] * self.env.num_worlds, *stack.shape[2:]
                    )
                return stack.to("cpu", non_blocking=True)

            observations = stacked(lambda b: policy_input(b.observation), half=True)
            acted = stacked(lambda b: b.acted)
            rule_action = stacked(lambda b: b.rule_action)
            rule_scores = stacked(lambda b: b.rule_scores, half=True) if batches[0].rule_scores is not None else None
            rule_unit = (
                stacked(lambda b: b.rule_metabolic_unit)
                if batches[0].rule_metabolic_unit is not None else None
            )
            self.world_steps += steps

            totals = {"loss": 0.0, "argmax_agreement": 0.0, "metabolic_error": 0.0}
            weight = 0.0
            for _ in range(max(ppo.epochs, 1)):
                # On CPU because the stacked segment is: an index tensor has to
                # live on the same device as the tensor it indexes.
                order = torch.randperm(steps, device="cpu")
                for start in range(0, steps, ppo.minibatch_steps):
                    index = order[start : start + ppo.minibatch_steps]
                    mask = acted[index].to(self.device)
                    count = float(mask.sum())
                    if count == 0:
                        continue
                    out = self.network.forward_all(
                        observations[index].to(self.device).float()
                    )
                    log_probs = F.log_softmax(out["logits"], dim=1)
                    if rule_scores is not None and temperature > 0:
                        target = F.softmax(rule_scores[index].to(self.device).float() / temperature, dim=1)
                        per_cell = (target * (torch.log(target + 1e-12) - log_probs)).sum(dim=1)
                    else:
                        per_cell = -log_probs.gather(1, rule_action[index].to(self.device).unsqueeze(1)).squeeze(1)
                    loss = (per_cell * mask).sum() / count
                    metabolic_error = float("nan")
                    if "metabolic_mean" in out and rule_unit is not None:
                        # The same fixed-scale anchor PPO applies, at the same
                        # weight as the direction term: the two heads imitate
                        # one rule on one schedule. Only the mean is fitted;
                        # the policy's std is exploration and keeps its
                        # initialisation.
                        target = rule_unit[index].to(self.device).float()
                        mean = out["metabolic_mean"]
                        met = 0.5 * ((target - mean) / METABOLIC_ANCHOR_STD) ** 2
                        loss = loss + (met * mask).sum() / count
                        metabolic_error = float(((mean - target).abs() * mask).sum() / count)
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), ppo.max_grad_norm)
                    self.optimizer.step()
                    agreement = float(((out["logits"].argmax(1) == rule_action[index].to(self.device)) & mask).sum() / count)
                    totals["loss"] += float(loss) * count
                    totals["argmax_agreement"] += agreement * count
                    if metabolic_error == metabolic_error:
                        totals["metabolic_error"] += metabolic_error * count
                    weight += count

            if weight == 0:
                # No living individuals in this segment, which happens when a
                # small world's population dies out. There is nothing to
                # measure; keep the last real measurement rather than report
                # zeros that would then seed the anchor.
                continue
            last = {key: value / weight for key, value in totals.items()}
            last["pretrain_update"] = float(update + 1)
            last["population"] = float(self.env.population())
            self.log({"phase": "pretrain", "world_steps": self.world_steps, **last})
            if verbose:
                print(
                    f"pretrain {update + 1}/{updates}  loss {last['loss']:.3f}  "
                    f"agreement {last['argmax_agreement']:.3f}  metabolic err {last['metabolic_error']:.3f}  "
                    f"population {last['population']:.0f}"
                )

        return last

    def train(self, verbose: bool = True) -> Dict[str, object]:
        """Run until ``total_world_steps``. Returns the last logged record."""
        self._init_wandb()
        self.env.reset(seed=self.config.seed)
        self.warmup()
        self.pretrain(verbose=verbose)
        self.start_time = time.time()
        # Throughput is measured over this process only. After a resume the
        # counters carry over from the previous run, so rates have to be taken
        # from the delta or they report the average of a run that is over.
        steps_at_start = self.world_steps
        agent_steps_at_start = self.agent_steps

        record: Dict[str, object] = {}
        while self.world_steps < self.config.total_world_steps:
            steps = min(self.config.segment_steps, self.config.total_world_steps - self.world_steps)

            collect_start = time.time()
            rollout, collect_stats = self.collect(steps)
            collect_time = time.time() - collect_start

            update_start = time.time()
            diagnostics = self.algorithm.update(self.network, rollout, self.optimizer)
            update_time = time.time() - update_start

            self.world_steps += steps
            self.updates += 1
            elapsed = max(time.time() - self.start_time, 1e-9)

            record = {
                "world_steps": self.world_steps,
                "updates": self.updates,
                "total_agent_steps": self.agent_steps,
                "elapsed_sec": elapsed,
                "world_steps_per_sec": (self.world_steps - steps_at_start) / elapsed,
                "agent_steps_per_sec": (self.agent_steps - agent_steps_at_start) / elapsed,
                "collect_sec": collect_time,
                "update_sec": update_time,
                **collect_stats,
                **diagnostics,
            }

            if self.config.eval_interval and self.world_steps >= self._next_eval:
                record.update(self.evaluate())
                self._next_eval = self.world_steps + self.config.eval_interval
                self._record_eval_summary(record)

            if self.config.checkpoint_interval and self.world_steps >= self._next_checkpoint:
                record["checkpoint"] = str(self.save_checkpoint())
                self._next_checkpoint = self.world_steps + self.config.checkpoint_interval

            if self.config.film_interval and self.world_steps >= self._next_film:
                record.update(self.record_film())
                self._next_film = self.world_steps + self.config.film_interval

            # A world that has gone extinct is reset, not mourned. Its dead
            # individuals already entered the loss as the low returns they
            # earned; what remains is an empty world that can teach nothing,
            # so it is replaced by a fresh one and the run goes on. Done here,
            # after the update, so no segment straddles a reset.
            reset_now = self.reset_extinct_worlds()
            record["worlds_reset_now"] = float(reset_now)
            record["world_resets"] = float(self.world_resets)
            if reset_now and verbose:
                print(
                    f"    reset {reset_now} extinct world(s) at world step {self.world_steps}; "
                    f"{self.world_resets} resets so far",
                    flush=True,
                )

            self.log(record)
            if verbose:
                print(self.format_record(record), flush=True)
                if "score" in record:
                    print(
                        f"    eval  biomass={record['score']:.0f}  "
                        f"pop={record.get('learned_mean_population', 0):.0f}  "
                        f"repro={record.get('learned_reproductions', 0):.0f}  "
                        f"lifespan={record.get('learned_episode_length', 0):.1f}  "
                        f"(rules {record.get('rule_based_biomass_ema', 0):.0f})",
                        flush=True,
                    )

        # Always finish on an evaluation. Without this the returned record is
        # whatever the last segment happened to log, so a run whose final step
        # was not an evaluation step reports no score at all, and a sweep
        # reading the summary gets nothing to optimise. Skipped when the
        # population is already gone, since there is nothing left to score.
        if self.config.eval_interval and "score" not in record:
            record.update(self.evaluate())
            self._record_eval_summary(record)
            self.log(record)
            if verbose and "score" in record:
                # Only the evaluation line: the step line was already printed
                # for this segment by the loop above.
                print(
                    f"    eval  biomass={record['score']:.0f}  "
                    f"pop={record.get('learned_mean_population', 0):.0f}  "
                    f"repro={record.get('learned_reproductions', 0):.0f}  "
                    f"(rules {record.get('rule_based_biomass_ema', 0):.0f})",
                    flush=True,
                )

        if self.config.checkpoint_interval:
            self.save_checkpoint()
        return record

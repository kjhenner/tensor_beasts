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

**Every world starts from the bank.** A run's first act is a batched run
under the rules whose states are banked (:mod:`tensor_beasts.rl.bank`).
Training worlds, the replacement for an extinct world, the film world and
the evaluation worlds all start from bank states, and evaluation uses a
fixed seeded subset of them, so every evaluation scores the same starts.
Resuming restores the network, the optimizer and the counters; the worlds
start again from the bank, which is a discontinuity in the data at every
resume and a reason to prefer one long run over several resumed ones.
"""

import configparser
import json
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from tensor_beasts.rl.bank import StartBank, cached_bank
from tensor_beasts.rl.multiagent import MultiAgentWorldEnv, NUM_ACTIONS
from tensor_beasts.rl.networks import RULE_ARCHITECTURE, LinearPolicy, RulePolicy, build_network
from tensor_beasts.rl.memory import format_bytes
from tensor_beasts.rl.normalization import ValueNormalizer
from tensor_beasts.rl.ppo import MAX_LOG_STD, MIN_LOG_STD, PPO, PPOConfig
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
        reward_radius: Radius in cells of the box each individual's stock
            reward is pooled over, around its new cell. 0 pays each individual
            its own stock change. See tensor_beasts/rl/multiagent.py.
        arch: Network name: one of ``tensor_beasts.rl.networks.ARCHITECTURES``
            or ``"rule"``, the rule with free values (planning/11).
        arch_kwargs: Extra constructor arguments for that network. For
            ``"rule"``: ``critic`` ("conv" or "linear"), ``hidden_channels``
            and ``depth`` of the conv critic.
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
        bank_worlds, bank_steps, bank_warmup, bank_stride: The start bank, a
            rule-based run of ``bank_worlds`` worlds for ``bank_steps`` steps
            whose state is banked every ``bank_stride`` steps once
            ``bank_warmup`` steps are past. Every world the run touches
            starts from a bank state. See tensor_beasts/rl/bank.py.
        bank_cache: Directory the bank is written to under a key made from
            everything that shapes it, and read back from by the next
            process that asks for the same bank. None, the default here,
            builds it every time; conf/rl/ppo.yaml sets outputs/bank.
        eval_interval: World steps between evaluations. 0 disables.
        eval_steps: The metric's window T: world steps each evaluation world
            runs for, scored as the mean stock over the window. At least
            4,000, longer than a collapse (planning/11).
        eval_seeds: Number of evaluation worlds, a fixed seeded subset of the
            bank. The learned and rule-based policies are scored from the
            same states.
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
    # The one reward knob: how far each individual's stock reward is pooled.
    reward_radius: int = 0
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
    # Offline distillation of the rule-based policy before RL starts, the
    # learner's initialisation: the network is fitted to the bank's labelled
    # grids for at most pretrain_epochs, stopping once agreement on a
    # held-out split has plateaued. Zero epochs skips it. See
    # tensor_beasts/rl/distill.py.
    pretrain_epochs: int = 0
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

    # The start bank. Eight worlds for 3,000 rule steps, banked every 100
    # steps after the first thousand: 160 states from the settled cycles, out
    # of phase with each other. A fresh world spends its first several
    # hundred steps in a startup transient, and everything scored inside it
    # was a score of the transient (planning/10).
    bank_worlds: int = 8
    bank_steps: int = 3_000
    bank_warmup: int = 1_000
    bank_stride: int = 100
    bank_cache: Optional[str] = None

    # Two evaluations per run plus the final one, at the default budget.
    eval_interval: int = 10_000
    eval_steps: int = 4_000
    eval_seeds: int = 8
    eval_deterministic: bool = False

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
_EVAL_KEYS = frozenset({"score", "score_spread", "score_min", "score_max"})
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
    few thousand world steps, so ``score`` has a value on two rows out of a
    hundred and fifty. Mixed in with per-segment metrics that is
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

    The headline is ``mean_biomass``: the metric M_T of planning/11, the
    stock of biomass carried by the controlled species' living individuals,
    averaged over the T steps of the window, from a banked start. Extinction
    is absorbing, so a world that dies contributes zeros for the rest of the
    window, and ``extinct`` records that it did. Persistence and growth are
    one quantity at a long horizon; every earlier tension between them was an
    artefact of windows shorter than a collapse, which is why T is 4,000 and
    why the smoothed headline this replaced is gone.

    Biomass rather than population count because it is what the ecology
    conserves: an individual carries biomass, eats it, burns it and halves
    it into its offspring. A population count weights a starving animal
    about to die the same as a thriving one about to divide. Absolute rather
    than a ratio against the rules: their constants were chosen by hand, and
    the rules' own M_T on the same starts is reported beside it as a row.
    """

    mean_biomass: List[float]
    final_biomass: List[float]
    extinct: List[float]
    mean_population: List[float]
    reproductions: List[float]
    survived_agent_steps: List[float]
    total_reward: List[float]
    episode_return: float
    episode_length: float
    episodes_finished: float

    def to_dict(self, prefix: str) -> Dict[str, float]:
        """Mean over worlds for each per-world field, scalars passed through.
        The mean of ``extinct`` is the extinction fraction and is named so."""
        out: Dict[str, float] = {}
        for key, value in asdict(self).items():
            name = "extinct_fraction" if key == "extinct" else key
            if isinstance(value, list):
                out[f"{prefix}_{name}"] = sum(value) / len(value) if value else 0.0
            else:
                out[f"{prefix}_{name}"] = value
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
        arch_kwargs = dict(self.config.arch_kwargs)
        if self.config.arch == RULE_ARCHITECTURE:
            # The rule with free values starts as the rule: the spec is read
            # from the environment, not stored, so a checkpoint rebuilds it
            # against any world with the same perception.
            arch_kwargs.setdefault("rule", self.env.rule_spec())
            arch_kwargs.setdefault("temperature", self.ppo_config.imitation_temperature)
        with torch.device("cpu"):
            network = build_network(
                self.config.arch,
                self.observation_channels,
                metabolic=self.config.metabolic,
                memory_size=self.config.memory_size,
                **arch_kwargs,
            )
        self.network = network.to(self.device)
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
        self.verbose = True

        # The bank is built on first use, under its own seed, and every draw
        # from it goes through one seeded generator so which state a world
        # starts from never depends on anything else the run did.
        self._bank: Optional[StartBank] = None
        self._bank_generator = torch.Generator().manual_seed(self.config.seed + 50_000)
        # The rules' score on the evaluation starts is a fixed reference,
        # scored once per process.
        self._rule_result: Optional[EvalResult] = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _make_env(self, worlds: Optional[int] = None) -> MultiAgentWorldEnv:
        config = self.config
        return MultiAgentWorldEnv(
            config_path=config.config_path,
            size=(config.size, config.size),
            entity_name=config.entity,
            reward_radius=config.reward_radius,
            worlds=config.worlds if worlds is None else worlds,
            device=str(self.device),
            # Pinning needs the unit-to-rate mapping even for a direction-only
            # policy, so the environment carries the bounds either way.
            metabolic=bool(config.metabolic) or config.eval_pin_metabolic is not None,
            memory_size=config.memory_size,
        )

    @property
    def bank(self) -> StartBank:
        """The start bank, built on first use. See tensor_beasts/rl/bank.py."""
        if self._bank is None:
            config = self.config
            # What shapes the bank beyond its own parameters: the simulation
            # config, the world the environment builds and the device type.
            # The reward radius is left out; it never reaches the state.
            key_parts = {
                "config_path": config.config_path, "size": int(config.size), "entity": config.entity,
                "metabolic": bool(config.metabolic) or config.eval_pin_metabolic is not None,
                "memory_size": int(config.memory_size), "device": self.device.type,
            }
            self._bank = cached_bank(
                self._make_env, config.bank_cache, key_parts,
                worlds=int(config.bank_worlds), steps=int(config.bank_steps),
                warmup=int(config.bank_warmup), stride=int(config.bank_stride),
                seed=config.seed + 40_000, verbose=self.verbose,
            )
            if self.verbose:
                steps = self._bank.steps
                print(
                    f"bank: {len(self._bank)} states from steps {min(steps)} to {max(steps)}, "
                    f"{format_bytes(self._bank.nbytes)} of host memory",
                    flush=True,
                )
        return self._bank

    def eval_indices(self) -> List[int]:
        """The fixed subset of the bank every evaluation starts from."""
        generator = torch.Generator().manual_seed(self.config.seed + 10_000)
        return self.bank.draw(max(int(self.config.eval_seeds), 1), generator)

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
        extinct = torch.zeros(worlds, dtype=torch.bool, device=self.device)
        populations: List[torch.Tensor] = []
        stocks: List[torch.Tensor] = []

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
            # independent evaluation start and must produce its own number, or
            # the batch silently averages worlds together before anyone can see
            # the spread between them.
            grid = (-2, -1)
            total_reward += batch.reward.sum(dim=grid).reshape(worlds)
            survived += (batch.acted & ~batch.done).sum(dim=grid).reshape(worlds).float()
            reproductions += batch.reproduced.sum(dim=grid).reshape(worlds).float()
            population = env.population_per_world()
            populations.append(population)
            extinct |= population <= 0
            # The stock: biomass carried by living individuals, the quantity
            # the metric integrates. Zero for good once a world is extinct,
            # which the simulation guarantees, so extinction is absorbing.
            stocks.append(env.stock_per_world())

        summary = tracker.summary()
        # Each field is one number per world, so the caller can average over
        # starts AND see the spread between them. Episode statistics are pooled
        # across worlds on purpose: an episode is one individual's life, and
        # lives are comparable wherever they happened.
        stacked_stock = torch.stack(stocks) if stocks else zeros.unsqueeze(0)
        stacked_population = torch.stack(populations) if populations else zeros.unsqueeze(0)
        return EvalResult(
            mean_biomass=stacked_stock.mean(dim=0).tolist(),
            final_biomass=stacked_stock[-1].tolist(),
            extinct=extinct.float().tolist(),
            mean_population=stacked_population.mean(dim=0).tolist(),
            reproductions=reproductions.tolist(),
            survived_agent_steps=survived.tolist(),
            total_reward=total_reward.tolist(),
            episode_return=summary["episode_return"],
            episode_length=summary["episode_length"],
            episodes_finished=summary["episodes_finished"],
        )

    def evaluate(self) -> Dict[str, float]:
        """Score the learned policy and the rule-based baseline on the same starts.

        The evaluation worlds are a fixed seeded subset of the bank, loaded
        afresh for each policy from the same RNG seed, so the comparison is
        paired and every evaluation of the run, and every run with the same
        seed, scores the same starts. The two runs diverge immediately, of
        course, because the policies consume the RNG differently and the
        ecology is chaotic; the starts are controlled, not the trajectories.

        The rules are a fixed reference on those starts and are scored once
        per process; every later evaluation reports that same number.
        """
        # Both policies start from a reseeded RNG, which would otherwise make
        # the training stream depend on the evaluation schedule.
        rng_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        self.network.eval()
        seeds = max(self.config.eval_seeds, 1)
        seed = self.config.seed + 10_000
        indices = self.eval_indices()
        env = self._eval_env(seeds)

        summary: Dict[str, float] = {}
        spread: Dict[str, List[float]] = {}
        torch.manual_seed(seed)
        self.bank.load(env, indices)
        learned = self._score(env, self.config.eval_steps, "learned")
        if self._rule_result is None:
            torch.manual_seed(seed)
            self.bank.load(env, indices)
            self._rule_result = self._score(env, self.config.eval_steps, "rule_based")
        for policy, scored in (("learned", learned), ("rule_based", self._rule_result)):
            summary.update(scored.to_dict(policy))
            spread.update(scored.per_world(policy))
        self.network.train()
        torch.set_rng_state(rng_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        # Evaluation is the run's allocation peak, and the caching allocator
        # would otherwise hold that peak for the rest of the run. Two agents
        # sharing one card each hoarding an evaluation's worth is what stops
        # the second from fitting.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # The headline is the learned policy's own M_T, the mean stock over
        # the window, averaged over starts; the spread across starts is the
        # quantity every claim in this project is hedged against, so it is
        # reported rather than averaged away. See EvalResult.
        values = spread.get("learned_mean_biomass") or []
        summary["score"] = summary.get("learned_mean_biomass", 0.0)
        if len(values) > 1:
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            summary["score_spread"] = variance ** 0.5
            summary["score_min"] = min(values)
            summary["score_max"] = max(values)
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
            generator = torch.Generator().manual_seed(self.config.seed + 20_000 if seed is None else seed)
            self.bank.load(env, self.bank.draw(1, generator))
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
            # Readable beside the weights, for the rule with free values.
            "rule_values": self.network.values() if isinstance(self.network, RulePolicy) else None,
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

    def format_evaluation(self, record: Dict[str, object]) -> str:
        line = (
            f"    eval  M_T={record['score']:.0f}  "
            f"extinct={record.get('learned_extinct_fraction', 0):.2f}  "
            f"pop={record.get('learned_mean_population', 0):.0f}  "
            f"repro={record.get('learned_reproductions', 0):.0f}  "
            f"lifespan={record.get('learned_episode_length', 0):.1f}  "
            f"(rules M_T={record.get('rule_based_mean_biomass', 0):.0f} "
            f"extinct={record.get('rule_based_extinct_fraction', 0):.2f})"
        )
        if isinstance(self.network, RulePolicy):
            line += f"\n    rule  {self.network.describe()}"
        return line

    def _rule_values(self) -> Dict[str, float]:
        """The rule actor's values, as metrics, so a drift is a sentence in
        the log. Empty for every other architecture."""
        if not isinstance(self.network, RulePolicy):
            return {}
        out = {}
        for key, value in self.network.values().items():
            name = key.replace("w:", "w_").replace(":", "_").replace("/", "_")
            out[f"rule_{name}"] = value
        return out

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------
    def start_worlds(self) -> None:
        """Start every training world from a state drawn from the bank."""
        self.bank.load(self.env, self.bank.draw(self.env.num_worlds, self._bank_generator))
        self._extinct_segments = None

    def reset_extinct_worlds(self) -> int:
        """Replace every world whose controlled population has been extinct for
        ``extinction_patience`` consecutive segments with a state drawn from
        the bank. Returns how many were reset.

        The state is copied leaf by leaf into the extinct world's slice of
        the batched TensorDict, so the other worlds are not touched; the
        world clock, shared by the batch, is left alone. The draw comes from
        the bank's own generator, so the training stream does not depend on
        when a world happened to die.
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
        for index in due:
            self.world_resets += 1
            state = self.bank.draw(1, self._bank_generator)[0]
            self.bank.load_into(self.env, state, index)
            self._extinct_segments[index] = 0
        return len(due)

    def pretrain(self, verbose: bool = True) -> Dict[str, float]:
        """Fit the network to the rule-based policy before any RL.

        Offline, on the bank's labelled grids: the rule's own direction,
        scores and throttle at every banked state, so the fit is on exactly
        the states training and evaluation start from. The direction head is
        fitted by soft distillation toward the rule's per-action scores and
        the throttle head by a fixed-scale fit to the rule's rate, under the
        eight symmetries of the grid, until agreement on a held-out split
        plateaus or ``pretrain_epochs`` is reached. See
        tensor_beasts/rl/distill.py.

        This is the learner's initialisation and it is run to convergence. It
        used to run in lockstep with the training world, a fixed ten
        segments, which left the conv network at whatever agreement ten
        updates reached and made 0.88 look like a property of the network.

        Exists because a near-random initial policy is fatal to a small
        population: twice, a learned predator took its population from 486 to
        zero within 200 steps at 512. And it turned out to matter more than
        that. Frozen after pretraining, the policy carried 1.8 times the
        rules' biomass on the settled ecology (planning/10), so what
        pretraining reaches is the bar every RL result is read against.

        The training world is untouched, and the global RNG is restored, so
        the RL stream does not depend on how long the fit took.
        """
        from tensor_beasts.rl.distill import distill

        epochs = int(self.config.pretrain_epochs)
        if epochs <= 0:
            return {}
        ppo = self.ppo_config
        rng_state = torch.get_rng_state()

        grids = self.bank.grids
        if verbose:
            print(f"distilling the rules from {len(grids)} labelled grids", flush=True)

        if isinstance(self.network, LinearPolicy):
            # The linear network is the rule by construction; start it there
            # and let the fit refine the throttle and the rest.
            perceived_scale = float(torch.log1p(torch.tensor(255.0 * self.env.entity.config.log_scale)))
            temperature = float(ppo.imitation_temperature)
            self.network.initialise_from_rule(
                self.env.channel_names, self.env.entity.config.navigation_weights,
                scale=perceived_scale / temperature if temperature > 0 else 100.0,
            )

        def on_epoch(record: Dict[str, float]) -> None:
            self.log({"phase": "pretrain", "world_steps": self.world_steps, **record})
            if verbose:
                error = record["metabolic_error"]
                throttle = f"  metabolic err {error:.3f}" if error == error else ""
                print(
                    f"pretrain epoch {record['pretrain_epoch']:.0f}/{epochs}  loss {record['loss']:.3f}  "
                    f"agreement {record['argmax_agreement']:.3f} (train {record['train_agreement']:.3f}){throttle}",
                    flush=True,
                )

        last = distill(
            self.network, self.optimizer, grids,
            epochs=epochs,
            minibatch=max(1, int(ppo.minibatch_steps) * int(self.config.worlds)),
            temperature=ppo.imitation_temperature,
            max_grad_norm=ppo.max_grad_norm,
            device=self.device,
            on_epoch=on_epoch,
            generator=torch.Generator().manual_seed(self.config.seed),
        )
        if verbose and last.get("converged"):
            print(f"pretraining converged at epoch {last['pretrain_epoch']:.0f}: agreement {last['argmax_agreement']:.3f}", flush=True)
        torch.set_rng_state(rng_state)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return last

    def train(self, verbose: bool = True) -> Dict[str, object]:
        """Run until ``total_world_steps``. Returns the last logged record."""
        self.verbose = verbose
        self._init_wandb()
        self.bank  # built first, so its cost is not charged to the first segment
        torch.manual_seed(self.config.seed)
        self.start_worlds()
        if self.updates == 0:
            # The learner's initialisation, kept beside the run so any drifted
            # policy can be compared with its own start on the same worlds.
            self.pretrain(verbose=verbose)
            pretrained = self.save_checkpoint(self.output_dir / "pretrained.pt")
            if verbose:
                print(f"initial policy saved to {pretrained}", flush=True)
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
                **self._rule_values(),
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
                    print(self.format_evaluation(record), flush=True)

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
                print(self.format_evaluation(record), flush=True)

        if self.config.checkpoint_interval:
            self.save_checkpoint()
        return record

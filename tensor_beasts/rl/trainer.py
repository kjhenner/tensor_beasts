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

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from tensor_beasts.rl.multiagent import MultiAgentWorldEnv, NUM_ACTIONS
from tensor_beasts.rl.networks import ActorCritic, build_network
from tensor_beasts.rl.normalization import ValueNormalizer
from tensor_beasts.rl.ppo import PPO, PPOConfig
from tensor_beasts.rl.rollout import RolloutBuffer, compute_gae


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
            the number ``evaluate_policy.py`` reports.
        reproduction_reward: Reward for dividing.
        foraging_reward: Reward per unit of biomass an individual gains in a
            step. Dense and action-dependent, unlike survival, which is nearly
            constant at 99.4% per step. Zero by default because it is reward
            shaping; evaluation stays herbivore-steps survived either way.
        arch: Network name from ``tensor_beasts.rl.networks.ARCHITECTURES``.
        arch_kwargs: Extra constructor arguments for that network.
        metabolic_levels: Number of discrete metabolic levels the policy
            controls through a second network head, or 0 to leave the
            metabolic rate to the simulation's rules and learn movement only.
            Zero is the default so existing runs reproduce.
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

    arch: str = "conv"
    arch_kwargs: Dict[str, object] = field(default_factory=dict)
    metabolic_levels: int = 0
    # Channels of learned memory each individual carries; 0 disables it.
    memory_size: int = 0
    # Stop the run when the controlled population has been extinct for this many
    # consecutive segments. An extinct population produces no transitions, so
    # every gradient, every diagnostic and every evaluation after that point is
    # empty: a predator run once spent 89% of its world steps training on a
    # world with no predators in it and reported NaN agreement the whole way.
    # 0 disables the guard.
    extinction_patience: int = 3
    # Supervised updates on rule-based rollouts before RL starts, each over one
    # segment of world steps. Zero skips it. See Trainer.pretrain for why a
    # small population needs it.
    pretrain_updates: int = 0
    # Evaluation only. Hold the learned policy's metabolic level fixed at this
    # value, so the throttle's contribution can be separated from movement's.
    # None leaves the throttle to the network, or to the rules for a
    # direction-only policy. Requires an environment with metabolic levels.
    eval_pin_metabolic_level: Optional[int] = None
    device: str = "auto"
    seed: int = 0

    total_world_steps: int = 20_000
    segment_steps: int = 64
    warmup_steps: int = 100

    eval_interval: int = 2_000
    eval_steps: int = 500
    eval_seeds: int = 2
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
    wandb_host: Optional[str] = "http://localhost:8080"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def resolve_device(name: str) -> torch.device:
    """Pick a device, preferring the accelerator when asked for "auto"."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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

    def __init__(self, size: Tuple[int, int], device: torch.device):
        self.cells = size[0] * size[1]
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.ret = torch.zeros(self.cells, device=self.device)
        self.length = torch.zeros(self.cells, device=self.device)
        self.finished_return: List[float] = []
        self.finished_length: List[float] = []

    def update(self, batch) -> None:
        acted = batch.acted.reshape(-1)
        done = batch.done.reshape(-1)
        successor = batch.successor.reshape(-1)

        current_return = self.ret + batch.reward.reshape(-1) * acted
        current_length = self.length + acted.float()

        finished = done
        if bool(finished.any()):
            self.finished_return.extend(current_return[finished].tolist())
            self.finished_length.extend(current_length[finished].tolist())

        survivors = acted & ~done
        next_return = torch.zeros_like(self.ret)
        next_length = torch.zeros_like(self.length)
        index = successor[survivors]
        next_return[index] = current_return[survivors]
        next_length[index] = current_length[survivors]
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
    """What one scored run of a policy produced."""

    total_reward: float
    survived_agent_steps: float
    reproductions: float
    mean_population: float
    episode_return: float
    episode_length: float
    episodes_finished: float

    def to_dict(self, prefix: str) -> Dict[str, float]:
        return {f"{prefix}_{key}": value for key, value in asdict(self).items()}


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
                num_metabolic_levels=self.config.metabolic_levels,
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
        self.agent_steps = 0
        self.start_time = time.time()
        self._next_eval = 0
        self._next_checkpoint = self.config.checkpoint_interval
        self._next_film = self.config.film_interval

        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / self.config.log_name

        self._eval_envs: Dict[int, MultiAgentWorldEnv] = {}
        self._wandb = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _make_env(self) -> MultiAgentWorldEnv:
        config = self.config
        return MultiAgentWorldEnv(
            config_path=config.config_path,
            size=(config.size, config.size),
            entity_name=config.entity,
            survival_reward=config.survival_reward,
            reproduction_reward=config.reproduction_reward,
            foraging_reward=config.foraging_reward,
            device=str(self.device),
            # Pinning needs a level-to-rate mapping even for a direction-only
            # policy; two levels make level 0 exactly the basal rate.
            num_metabolic_levels=max(
                config.metabolic_levels, 2 if config.eval_pin_metabolic_level is not None else 0
            ),
            memory_size=config.memory_size,
        )

    def _init_wandb(self) -> None:
        if not self.config.wandb or self._wandb is not None:
            return
        import wandb  # imported lazily: wandb is optional and off by default

        wandb.init(
            project=self.config.wandb_project,
            config={**self.config.to_dict(), **self.ppo_config.to_dict()},
            settings=wandb.Settings(base_url=self.config.wandb_host) if self.config.wandb_host else None,
        )
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

    @torch.no_grad()
    def act(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Sample an action for every cell.

        Returns (action, log_prob, value, metabolic_action, memory). ``metabolic_action``
        is None unless the network has a metabolic head, in which case both
        heads are sampled independently and ``log_prob`` is the joint
        log-probability, the quantity the PPO ratio is defined over.

        Actions are produced for the whole grid, empty cells included. The
        simulation ignores directions at cells with nobody in them, and the loss
        masks those entries out, so the wasted work is a few hundred thousand
        multiply-adds that the convolution was doing anyway.
        """
        out = self.network.forward_all(observation.unsqueeze(0))
        action, log_prob = self._sample(out["logits"], deterministic)
        metabolic_action = None
        if "metabolic_logits" in out:
            metabolic_action, metabolic_log_prob = self._sample(out["metabolic_logits"], deterministic)
            log_prob = log_prob + metabolic_log_prob
            metabolic_action = metabolic_action.squeeze(0)
        # The advantage recursion mixes rewards and bootstrapped values, so it
        # has to run in real return units, not normalized ones.
        value = self.value_normalizer.denormalize(out["value"])
        # Deterministic memory write; not an action in the policy-gradient
        # sense, so it carries no log-probability. Stage 1 of the memory
        # design: the read is learnable, the write is a fixed function of the
        # observation until recurrent training exists.
        memory = out["memory"].squeeze(0) if "memory" in out else None
        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0), metabolic_action, memory

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------
    def collect(self, steps: int) -> Tuple[object, Dict[str, float]]:
        """Run ``steps`` world steps under the current policy."""
        buffer = RolloutBuffer(steps)
        tracker = EpisodeTracker(self.size, self.device)

        populations: List[float] = []
        reproductions = 0.0
        survived = 0.0

        for _ in range(steps):
            observation = policy_input(_observe(self.env))
            action, log_prob, value, metabolic_action, memory = self.act(observation)
            batch = self.env.step(action, metabolic_action, memory)
            # The env rebuilt the observation itself; overwrite it with the
            # exact tensor the network saw, so stored and recomputed
            # log-probabilities agree bit for bit.
            batch.observation = observation
            buffer.add(batch, log_prob, value)
            tracker.update(batch)

            populations.append(float(self.env.population()))
            reproductions += float(batch.reproduced.sum())
            survived += float((batch.acted & ~batch.done).sum())
            self.agent_steps += batch.num_agents

        rollout = buffer.build()
        with torch.no_grad():
            _, last_value = self.network(policy_input(_observe(self.env)).unsqueeze(0))
            last_value = self.value_normalizer.denormalize(last_value)
        rollout = compute_gae(
            rollout,
            last_value.squeeze(0),
            gamma=self.ppo_config.gamma,
            gae_lambda=self.ppo_config.gae_lambda,
            normalize=self.ppo_config.normalize_advantage,
        )

        agent_steps = rollout.num_agent_steps
        stats = {
            "population": sum(populations) / max(len(populations), 1),
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
    def _eval_env(self, seed: int) -> MultiAgentWorldEnv:
        if seed not in self._eval_envs:
            self._eval_envs[seed] = self._make_env()
        return self._eval_envs[seed]

    @torch.no_grad()
    def _score(self, env: MultiAgentWorldEnv, steps: int, policy: str) -> EvalResult:
        tracker = EpisodeTracker(env.size, self.device)
        total_reward = 0.0
        survived = 0.0
        reproductions = 0.0
        populations: List[float] = []

        for _ in range(steps):
            if policy == "rule_based":
                batch = env.rule_based_step()
            else:
                observation = policy_input(_observe(env))
                action, _, _, metabolic_action, memory = self.act(
                    observation, deterministic=self.config.eval_deterministic
                )
                if self.config.eval_pin_metabolic_level is not None:
                    # Evaluation-only: hold the throttle at a fixed level so the
                    # effect of the throttle can be separated from movement.
                    metabolic_action = torch.full(
                        env.size, int(self.config.eval_pin_metabolic_level), dtype=torch.long, device=self.device
                    )
                # Memory must be written during evaluation exactly as in
                # training. It was not, once: this call dropped `memory`, so
                # every memory checkpoint was scored with its memory stuck at
                # zero, and the evaluation numbers said nothing about memory.
                batch = env.step(action, metabolic_action, memory)

            tracker.update(batch)
            total_reward += float(batch.reward.sum())
            survived += float((batch.acted & ~batch.done).sum())
            reproductions += float(batch.reproduced.sum())
            populations.append(float(env.population()))

        summary = tracker.summary()
        return EvalResult(
            total_reward=total_reward,
            survived_agent_steps=survived,
            reproductions=reproductions,
            mean_population=sum(populations) / max(len(populations), 1),
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
        results: Dict[str, List[float]] = {}
        for index in range(max(self.config.eval_seeds, 1)):
            seed = self.config.seed + 10_000 + index
            for policy in ("learned", "rule_based"):
                env = self._eval_env(index)
                env.reset(seed=seed)
                scored = self._score(env, self.config.eval_steps, policy)
                for key, value in scored.to_dict(policy).items():
                    results.setdefault(key, []).append(value)
        self.network.train()
        torch.set_rng_state(rng_state)

        summary = {
            key: (sum(values) / len(values)) for key, values in results.items()
        }
        # The headline ratio is defined on herbivore-steps survived, never on
        # total reward. Reward may be shaped (foraging_reward makes it signed),
        # and a sweep once reported ratios of -0.72 because this divided shaped
        # rewards. What is optimized may change; what is judged must not.
        baseline = summary.get("rule_based_survived_agent_steps", 0.0)
        summary["learned_over_rule_based"] = (
            summary.get("learned_survived_agent_steps", 0.0) / baseline if baseline else float("nan")
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

            env = self._make_env()
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
                observation = policy_input(_observe(env))
                action, _, _, metabolic_action, memory = self.act(
                    observation, deterministic=self.config.eval_deterministic
                )
                batch = env.step(action, metabolic_action, memory)
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
            "num_metabolic_levels": self.config.metabolic_levels,
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
        levels = int(payload.get("num_metabolic_levels", 0))
        if levels != self.config.metabolic_levels:
            raise ValueError(
                f"Checkpoint was trained with {levels} metabolic levels, this trainer "
                f"has {self.config.metabolic_levels}. Pass --metabolic-levels {levels}."
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
            self._wandb.log(record)

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
            ("metabolic_agreement", "m_agree"),
            ("metabolic_level_mean", "m_lvl"),
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

    def pretrain(self, verbose: bool = True) -> Dict[str, float]:
        """Supervised pretraining on the rule-based policy before any RL.

        The world runs under its own rules while the network learns to
        reproduce the rule's choices: soft distillation toward the rule's
        per-action scores when the imitation temperature is positive, hard
        cross-entropy to the rule's direction otherwise, plus the rule's
        metabolic level when the network has that head.

        Exists because a near-random initial policy is fatal to a small
        population. Twice, a learned predator took its population from 486 to
        zero within 200 steps at 512, before the imitation anchor could pull the
        policy toward anything that hunts. Herbivores survived the same start
        only because there are thousands of them. Starting RL from a policy
        that already behaves like the rule removes that cliff.

        The anchor's conformance is seeded from the measured agreement, so the
        cross-fade starts where pretraining left off instead of at full weight.
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
            batches = [self.env.rule_based_step() for _ in range(steps)]
            observations = torch.stack([policy_input(b.observation) for b in batches])
            acted = torch.stack([b.acted for b in batches])
            rule_action = torch.stack([b.rule_action for b in batches])
            rule_scores = torch.stack([b.rule_scores for b in batches]) if batches[0].rule_scores is not None else None
            rule_level = (
                torch.stack([b.rule_metabolic_level for b in batches])
                if batches[0].rule_metabolic_level is not None else None
            )
            self.world_steps += steps

            totals = {"loss": 0.0, "argmax_agreement": 0.0, "metabolic_agreement": 0.0}
            weight = 0.0
            for _ in range(max(ppo.epochs, 1)):
                order = torch.randperm(steps)
                for start in range(0, steps, ppo.minibatch_steps):
                    index = order[start : start + ppo.minibatch_steps]
                    mask = acted[index]
                    count = float(mask.sum())
                    if count == 0:
                        continue
                    out = self.network.forward_all(observations[index])
                    log_probs = F.log_softmax(out["logits"], dim=1)
                    if rule_scores is not None and temperature > 0:
                        target = F.softmax(rule_scores[index].float() / temperature, dim=1)
                        per_cell = (target * (torch.log(target + 1e-12) - log_probs)).sum(dim=1)
                    else:
                        per_cell = -log_probs.gather(1, rule_action[index].unsqueeze(1)).squeeze(1)
                    loss = (per_cell * mask).sum() / count
                    metabolic_agreement = float("nan")
                    if "metabolic_logits" in out and rule_level is not None:
                        met_log_probs = F.log_softmax(out["metabolic_logits"], dim=1)
                        met = -met_log_probs.gather(1, rule_level[index].unsqueeze(1)).squeeze(1)
                        loss = loss + ppo.metabolic_imitation_scale * (met * mask).sum() / count
                        metabolic_agreement = float(
                            ((out["metabolic_logits"].argmax(1) == rule_level[index]) & mask).sum() / count
                        )
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), ppo.max_grad_norm)
                    self.optimizer.step()
                    agreement = float(((out["logits"].argmax(1) == rule_action[index]) & mask).sum() / count)
                    totals["loss"] += float(loss) * count
                    totals["argmax_agreement"] += agreement * count
                    if metabolic_agreement == metabolic_agreement:
                        totals["metabolic_agreement"] += metabolic_agreement * count
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
                    f"agreement {last['argmax_agreement']:.3f}  metabolic {last['metabolic_agreement']:.3f}  "
                    f"population {last['population']:.0f}"
                )

        # Seed the anchor's cross-fade from what pretraining achieved.
        if hasattr(self.algorithm, "conformance") and last:
            self.algorithm.conformance = last["argmax_agreement"]
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
        extinct_segments = 0
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

            if self.config.checkpoint_interval and self.world_steps >= self._next_checkpoint:
                record["checkpoint"] = str(self.save_checkpoint())
                self._next_checkpoint = self.world_steps + self.config.checkpoint_interval

            if self.config.film_interval and self.world_steps >= self._next_film:
                record.update(self.record_film())
                self._next_film = self.world_steps + self.config.film_interval

            # An extinct population is the end of the experiment, not a bad
            # patch to train through: with nothing alive there are no
            # transitions, so the loss has nothing to act on and the world can
            # never repopulate, the simulation having no immigration.
            if self.config.extinction_patience:
                if collect_stats.get("population", 1.0) <= 0.0:
                    extinct_segments += 1
                else:
                    extinct_segments = 0
                if extinct_segments >= self.config.extinction_patience:
                    record["extinct"] = True
                    record["extinct_at_world_step"] = self.world_steps
                    self.log(record)
                    if verbose:
                        print(self.format_record(record), flush=True)
                        print(
                            f"\n{self.config.entity} went extinct: no living individual for "
                            f"{extinct_segments} consecutive segments, stopping at world step "
                            f"{self.world_steps} of {self.config.total_world_steps}. Every "
                            "gradient from here would be empty.",
                            flush=True,
                        )
                    break

            self.log(record)
            if verbose:
                print(self.format_record(record), flush=True)
                if "learned_total_reward" in record:
                    print(
                        f"    eval  learned={record['learned_total_reward']:.0f}  "
                        f"rule_based={record['rule_based_total_reward']:.0f}  "
                        f"ratio={record['learned_over_rule_based']:.3f}",
                        flush=True,
                    )

        if self.config.checkpoint_interval:
            self.save_checkpoint()
        return record

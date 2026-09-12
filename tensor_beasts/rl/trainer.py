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
        arch: Network name from ``tensor_beasts.rl.networks.ARCHITECTURES``.
        arch_kwargs: Extra constructor arguments for that network.
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
    """

    config_path: str = "conf/basic_config.yaml"
    size: int = 256
    entity: str = "Herbivore"
    survival_reward: float = 1.0
    reproduction_reward: float = 10.0

    arch: str = "conv"
    arch_kwargs: Dict[str, object] = field(default_factory=dict)
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

    wandb: bool = False
    wandb_project: str = "tensor-beasts-rl"

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
                **dict(self.config.arch_kwargs),
            )
        self.network: ActorCritic = network.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.ppo_config.learning_rate
        )
        self.algorithm = PPO(self.ppo_config)

        self.size = self.env.size
        self.world_steps = 0
        self.updates = 0
        self.agent_steps = 0
        self.start_time = time.time()
        self._next_eval = 0
        self._next_checkpoint = self.config.checkpoint_interval

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
            device=str(self.device),
        )

    def _init_wandb(self) -> None:
        if not self.config.wandb or self._wandb is not None:
            return
        import wandb  # imported lazily: wandb is optional and off by default

        wandb.init(
            project=self.config.wandb_project,
            config={**self.config.to_dict(), **self.ppo_config.to_dict()},
        )
        self._wandb = wandb

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a direction for every cell. Returns (action, log_prob, value).

        Actions are produced for the whole grid, empty cells included. The
        simulation ignores directions at cells with nobody in them, and the loss
        masks those entries out, so the wasted work is a few hundred thousand
        multiply-adds that the convolution was doing anyway.
        """
        logits, value = self.network(observation.unsqueeze(0))
        log_probs = torch.log_softmax(logits, dim=1)
        if deterministic:
            action = log_probs.argmax(dim=1)
        else:
            flat = log_probs.permute(0, 2, 3, 1).reshape(-1, NUM_ACTIONS)
            action = torch.multinomial(flat.exp(), 1).reshape(logits.shape[0], *logits.shape[2:])
        log_prob = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)

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
            action, log_prob, value = self.act(observation)
            batch = self.env.step(action)
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
                action, _, _ = self.act(
                    observation, deterministic=self.config.eval_deterministic
                )
                batch = env.step(action)

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
        baseline = summary.get("rule_based_total_reward", 0.0)
        summary["learned_over_rule_based"] = (
            summary.get("learned_total_reward", 0.0) / baseline if baseline else float("nan")
        )
        return summary

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
            ("world_steps_per_sec", "w/s"),
            ("agent_steps_per_sec", "a/s"),
        ):
            value = record.get(key)
            if isinstance(value, (int, float)):
                parts.append(f"{label}={value:.3g}")
        return "  ".join(parts)

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------
    def warmup(self) -> None:
        """Let the ecology settle under its own policy before learning starts."""
        for _ in range(self.config.warmup_steps):
            self.env.rule_based_step()

    def train(self, verbose: bool = True) -> Dict[str, object]:
        """Run until ``total_world_steps``. Returns the last logged record."""
        self._init_wandb()
        self.env.reset(seed=self.config.seed)
        self.warmup()
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

            if self.config.checkpoint_interval and self.world_steps >= self._next_checkpoint:
                record["checkpoint"] = str(self.save_checkpoint())
                self._next_checkpoint = self.world_steps + self.config.checkpoint_interval

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

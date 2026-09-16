#!/usr/bin/env python
"""Train a shared policy for individual herbivores, and score it against the rules.

The experiment this exists for is one sentence: can a learned policy beat the
simulation's own rule-based policy at keeping herbivores alive? Training and the
comparison are the same command, because a training curve on its own does not
answer that question. Every evaluation prints both numbers.

Start with --arch linear. That network is a single 1x1 convolution over the same
channels the rule-based policy reads, so it can represent the baseline exactly.
If PPO cannot bring it near the baseline, the learning setup is broken and
nothing larger is worth running.

Usage:
    source venv/bin/activate && python train_rl.py
    python train_rl.py --arch linear --size 256 --steps 20000
    python train_rl.py --eval-only outputs/rl/checkpoint.pt --size 256
    python train_rl.py --resume outputs/rl/checkpoint.pt --steps 40000

Sizes below about 256 are not a valid ecology: the predators die out and the
three-species dynamic degenerates into herbivores grazing an empty world.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from tensor_beasts.rl.ppo import PPOConfig
from tensor_beasts.rl.memory import estimate_training_bytes, format_bytes
from tensor_beasts.rl.trainer import Trainer, TrainerConfig

DEFAULT_CONFIG = "conf/rl/ppo.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="hyperparameter YAML")

    world = parser.add_argument_group("world")
    world.add_argument("--sim-config", default=None, help="simulation config YAML")
    world.add_argument("--size", type=int, default=None, help="world side length")
    world.add_argument("--entity", default=None)
    world.add_argument("--survival-reward", type=float, default=None)
    world.add_argument("--reproduction-reward", type=float, default=None)
    world.add_argument(
        "--foraging-reward",
        type=float,
        default=None,
        help=(
            "Reward per unit of biomass gained per step. Dense and strongly "
            # Escaped: argparse runs help text through %-formatting, and a bare
            # percent sign makes --help raise ValueError instead of printing.
            "action-dependent, unlike survival, which sits near 99.4%% per step "
            "and so carries almost no signal. Reward shaping: it changes what is "
            "optimized, never what is evaluated."
        ),
    )
    world.add_argument(
        "--offspring-credit",
        type=float,
        default=None,
        metavar="FRACTION",
        help=(
            "Credit an individual with this fraction of its offspring's biomass "
            "at birth. Division halves the parent's biomass, so a reward in "
            "biomass alone is maximised by never dividing; this makes it an "
            "investment instead. First generation only, so the credit stays "
            "bounded in a growing population. 0 (default) disables it."
        ),
    )
    world.add_argument(
        "--no-normalize-values",
        dest="normalize_values",
        action="store_const",
        const=False,
        default=None,
        help=(
            "Ablation. Predict raw returns instead of normalized ones. Expect the "
            "policy to stop moving: the value term becomes the whole gradient norm "
            "and global clipping scales the policy's share away with it. See "
            "tensor_beasts/rl/normalization.py."
        ),
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "--arch", default=None, choices=["linear", "conv", "residual", "dilated"]
    )
    model.add_argument("--hidden-channels", type=int, default=None)
    model.add_argument(
        "--metabolic-levels",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Let the policy set its own metabolic rate through a second head over "
            "N discrete levels from basal to max_metabolic_rate, still capped by "
            "carried biomass. 0 (the default) leaves the rate to the rules and "
            "learns movement only, so existing runs reproduce. --eval-only and "
            "--resume take the value from the checkpoint when this is not given."
        ),
    )
    model.add_argument(
        "--memory-size",
        type=int,
        default=None,
        help=(
            "Channels of learned memory each individual writes at one step and "
            "reads at the next, carried with it when it moves and copied into its "
            "offspring. 0 disables (default). Stage 1: the read is learnable, the "
            "write is a fixed function of the observation; see planning/04."
        ),
    )
    model.add_argument("--device", default=None, help="auto, cpu, mps or cuda")

    loop = parser.add_argument_group("loop")
    loop.add_argument("--steps", type=int, default=None, help="total world steps")
    loop.add_argument("--segment-steps", type=int, default=None)
    loop.add_argument("--warmup-steps", type=int, default=None)
    loop.add_argument("--seed", type=int, default=None)
    loop.add_argument(
        "--extinction-patience",
        type=int,
        default=None,
        metavar="SEGMENTS",
        help=(
            "Stop when the controlled population has been extinct for this many "
            "consecutive segments (default 3, 0 disables). The simulation has no "
            "immigration, so an extinct entity never returns and every gradient "
            "after that point is empty."
        ),
    )

    hyper = parser.add_argument_group("ppo")
    hyper.add_argument("--lr", type=float, default=None)
    hyper.add_argument("--clip-range", type=float, default=None)
    hyper.add_argument("--value-clip-range", type=float, default=None)
    hyper.add_argument("--entropy-coef", type=float, default=None)
    hyper.add_argument("--value-coef", type=float, default=None)
    hyper.add_argument("--epochs", type=int, default=None)
    hyper.add_argument("--minibatch-steps", type=int, default=None)
    hyper.add_argument("--max-grad-norm", type=float, default=None)
    hyper.add_argument("--gamma", type=float, default=None)
    hyper.add_argument("--gae-lambda", type=float, default=None)
    hyper.add_argument("--target-kl", type=float, default=None)
    hyper.add_argument(
        "--imitation-coef",
        type=float,
        default=None,
        help=(
            "Anchor to the rule-based policy: cross-entropy to its action, cross-"
            "faded to zero as conformance rises to --imitation-target. Starts the "
            "learner near the baseline instead of at random."
        ),
    )
    hyper.add_argument("--imitation-temperature", type=float, default=None,
                       help="softmax temperature over the rule's scores for soft distillation; 0 = hard argmax (default 0.01)")
    hyper.add_argument("--imitation-floor", type=float, default=None,
                       help="minimum anchor weight as a fraction of --imitation-coef, so the rules never fully let go (default 0)")
    hyper.add_argument("--metabolic-imitation-scale", type=float, default=None,
                       help="weight of the metabolic-level anchor relative to the direction anchor; 0 anchors direction only (default 1)")
    hyper.add_argument("--pretrain-updates", type=int, default=None,
                       help="supervised updates on rule-based rollouts before RL, one segment each (0 = off). "
                            "Needed for small populations such as predators, which a near-random start starves.")
    hyper.add_argument("--recurrent-window", type=int, default=None,
                       help="with --memory-size, backpropagate through this many steps of the individual's own memory writes (0 = off, stage 1)")
    hyper.add_argument("--imitation-target", type=float, default=None,
                       help="conformance at which the imitation weight reaches zero (default 0.8)")

    evaluation = parser.add_argument_group("evaluation")
    evaluation.add_argument(
        "--pin-metabolic-level",
        type=int,
        default=None,
        help=(
            "Evaluation only: hold the learned policy's metabolic level fixed, e.g. 0 "
            "for basal, to separate the throttle's effect from movement's. Works with "
            "a direction-only checkpoint too."
        ),
    )
    evaluation.add_argument("--eval-interval", type=int, default=None)
    evaluation.add_argument("--eval-steps", type=int, default=None)
    evaluation.add_argument("--eval-seeds", type=int, default=None)
    evaluation.add_argument(
        "--eval-deterministic", action="store_true", default=None,
        help="argmax at evaluation instead of sampling",
    )
    evaluation.add_argument(
        "--eval-only", default=None, metavar="CHECKPOINT",
        help="score a checkpoint against the rule-based baseline and exit",
    )

    film = parser.add_argument_group("films")
    film.add_argument(
        "--film-interval",
        type=int,
        default=None,
        metavar="STEPS",
        help=(
            "World steps between individual-following films, 0 to disable "
            "(default). Each film follows two individuals through their whole "
            "lives, one sampled from the typical band of the return "
            "distribution and one from the top decile, and writes an mp4 per "
            "individual. Rare on purpose: it holds one world snapshot per "
            "recorded step. With --eval-only it records one film and exits."
        ),
    )
    film.add_argument("--film-steps", type=int, default=None, metavar="N",
                      help="world steps recorded per film (default 300)")
    film.add_argument("--film-window", type=int, default=None, metavar="CELLS",
                      help="side length of the crop that follows the individual (default 48)")
    film.add_argument("--film-scale", type=int, default=None, metavar="N",
                      help="nearest-neighbour upscale of each frame (default 5)")
    film.add_argument("--film-display", default=None, metavar="TITLE",
                      help="which display config to render through (default 'layers')")

    output = parser.add_argument_group("output")
    output.add_argument("--out", default=None, help="output directory")
    output.add_argument("--checkpoint-interval", type=int, default=None)
    output.add_argument("--resume", default=None, metavar="CHECKPOINT")
    output.add_argument("--wandb", action="store_true", default=None,
                        help="mirror the log to W&B (on by default in conf/rl/ppo.yaml)")
    output.add_argument("--no-wandb", dest="wandb", action="store_false",
                        help="keep this run out of W&B; the JSONL log is written either way")
    output.add_argument("--wandb-project", default=None)
    output.add_argument("--wandb-host", default=None, help="W&B server URL (default http://localhost:8080)")
    output.add_argument("--quiet", action="store_true")
    return parser


def apply_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge YAML with CLI flags. Anything given on the command line wins."""
    config = OmegaConf.load(args.config)
    trainer = OmegaConf.to_container(config.trainer, resolve=True)
    ppo = OmegaConf.to_container(config.ppo, resolve=True)

    trainer_flags = {
        "config_path": args.sim_config,
        "size": args.size,
        "entity": args.entity,
        "survival_reward": args.survival_reward,
        "reproduction_reward": args.reproduction_reward,
        "foraging_reward": args.foraging_reward,
        "offspring_credit": args.offspring_credit,
        "normalize_values": args.normalize_values,
        "arch": args.arch,
        "metabolic_levels": args.metabolic_levels,
        "memory_size": args.memory_size,
        "pretrain_updates": args.pretrain_updates,
        "eval_pin_metabolic_level": args.pin_metabolic_level,
        "device": args.device,
        "seed": args.seed,
        "extinction_patience": args.extinction_patience,
        "total_world_steps": args.steps,
        "segment_steps": args.segment_steps,
        "warmup_steps": args.warmup_steps,
        "eval_interval": args.eval_interval,
        "eval_steps": args.eval_steps,
        "eval_seeds": args.eval_seeds,
        "eval_deterministic": args.eval_deterministic,
        "film_interval": args.film_interval,
        "film_steps": args.film_steps,
        "film_window": args.film_window,
        "film_scale": args.film_scale,
        "film_display": args.film_display,
        "checkpoint_interval": args.checkpoint_interval,
        "output_dir": args.out,
        "wandb": args.wandb,
        "wandb_project": args.wandb_project,
        "wandb_host": args.wandb_host,
    }
    ppo_flags = {
        "learning_rate": args.lr,
        "clip_range": args.clip_range,
        "value_clip_range": args.value_clip_range,
        "entropy_coef": args.entropy_coef,
        "value_coef": args.value_coef,
        "epochs": args.epochs,
        "minibatch_steps": args.minibatch_steps,
        "max_grad_norm": args.max_grad_norm,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "target_kl": args.target_kl,
        "imitation_coef": args.imitation_coef,
        "imitation_target_conformance": args.imitation_target,
        "imitation_temperature": args.imitation_temperature,
        "imitation_floor": args.imitation_floor,
        "metabolic_imitation_scale": args.metabolic_imitation_scale,
        "recurrent_window": args.recurrent_window,
    }

    trainer.update({k: v for k, v in trainer_flags.items() if v is not None})
    ppo.update({k: v for k, v in ppo_flags.items() if v is not None})

    # A checkpoint fixes the network's shape. When loading one and the flag was
    # not given, take the head configuration from the checkpoint rather than
    # refusing to load it.
    checkpoint = args.eval_only or args.resume
    if checkpoint and args.metabolic_levels is None:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        trainer["metabolic_levels"] = int(
            payload.get("num_metabolic_levels", payload["trainer_config"].get("metabolic_levels", 0))
        )

    if args.hidden_channels is not None:
        arch_kwargs = dict(trainer.get("arch_kwargs") or {})
        if trainer["arch"] == "linear":
            raise SystemExit("--hidden-channels does not apply to the linear architecture")
        arch_kwargs["hidden_channels"] = args.hidden_channels
        trainer["arch_kwargs"] = arch_kwargs

    return {"trainer": trainer, "ppo": ppo}


def print_evaluation(summary: Dict[str, float]) -> None:
    header = f"{'policy':12} {'return':>12} {'survived':>12} {'repro':>10} {'pop':>9} {'ep_ret':>9} {'ep_len':>9}"
    print(header)
    print("-" * len(header))
    for policy, label in (("learned", "learned"), ("rule_based", "rule-based")):
        print(
            f"{label:12} "
            f"{summary[f'{policy}_total_reward']:12.0f} "
            f"{summary[f'{policy}_survived_agent_steps']:12.0f} "
            f"{summary[f'{policy}_reproductions']:10.0f} "
            f"{summary[f'{policy}_mean_population']:9.1f} "
            f"{summary[f'{policy}_episode_return']:9.1f} "
            f"{summary[f'{policy}_episode_length']:9.1f}"
        )
    print(f"\nlearned / rule-based = {summary['learned_over_rule_based']:.3f}x")
    if summary["learned_over_rule_based"] < 1.0:
        print("The learned policy has not beaten the baseline.")


def check_device_headroom(trainer, estimate: int) -> None:
    """Refuse to start a run the device cannot hold, and say so plainly.

    A run that will not fit fails somewhere inside pretraining with a CUDA
    out-of-memory traceback pointing at whatever allocation happened to be last,
    which says nothing about the cause. Under a sweep agent that reads as a
    failed trial rather than as a machine that was already full, and the agent
    cheerfully starts the next one.

    This is a warning rather than an error when it is close, because the
    estimate is an estimate; it only refuses when the shortfall is large enough
    that the run is certain to die.
    """
    import torch

    device = trainer.device
    if device.type != "cuda":
        return
    free, total = torch.cuda.mem_get_info(device.index or 0)
    name = torch.cuda.get_device_properties(device.index or 0).name
    print(
        f"device {device} ({name}) has {format_bytes(free)} free of {format_bytes(total)}"
    )
    if estimate > free:
        raise SystemExit(
            f"\nThis run needs about {format_bytes(estimate)} and {device} has "
            f"{format_bytes(free)} free, so it would die partway through.\n"
            f"Another process may be holding the card: check nvidia-smi.\n"
            f"Otherwise lower --minibatch-steps, --segment-steps or --size, or "
            f"name a different card with --device cuda:N.\n"
            f"Note that a bare --device cuda picks the card with the most free "
            f"memory, not cuda:0, because index ordering is not stable."
        )
    if estimate > 0.8 * free:
        print(
            f"warning: this run needs about {format_bytes(estimate)} of the "
            f"{format_bytes(free)} free. It may not fit alongside anything else."
        )


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    merged = apply_overrides(args)
    trainer_config = TrainerConfig(**merged["trainer"])
    ppo_config = PPOConfig(**merged["ppo"])

    trainer = Trainer(trainer_config, ppo_config)

    # Say this out loud before starting. A fully convolutional policy holds one
    # full-resolution activation per convolution and the minibatch multiplies
    # every one of them, so the backward pass, not the stored rollout, is what
    # runs a machine out of memory. Lower --minibatch-steps if this is large.
    estimate = estimate_training_bytes(
        trainer.network,
        minibatch_steps=ppo_config.minibatch_steps,
        segment_steps=trainer_config.segment_steps,
        observation_channels=trainer.env.observation_channels,
        height=trainer_config.size,
        width=trainer_config.size,
    )
    print(f"estimated peak memory {format_bytes(estimate)}")
    check_device_headroom(trainer, estimate)
    print(
        f"arch={trainer_config.arch} params={trainer.network.num_parameters()} "
        f"receptive_field={trainer.network.receptive_field} "
        f"channels={trainer.observation_channels} device={trainer.device} "
        f"size={trainer_config.size}x{trainer_config.size} "
        f"metabolic_levels={trainer_config.metabolic_levels}"
    )

    if args.eval_only:
        trainer.load_checkpoint(Path(args.eval_only), load_optimizer=False)
        summary = trainer.evaluate()
        print(
            f"\n{args.eval_only}: {trainer_config.eval_seeds} seeds x "
            f"{trainer_config.eval_steps} world steps"
            f"{' (argmax)' if trainer_config.eval_deterministic else ' (sampled)'}\n"
        )
        print_evaluation(summary)
        if trainer_config.film_interval:
            film = trainer.record_film()
            summary.update({k: v for k, v in film.items() if isinstance(v, (int, float, str))})
            print()
            for band in ("typical", "high"):
                path = film.get(f"film_{band}_path")
                if path:
                    print(
                        f"{band:8} life: {film[f'film_{band}_steps']:.0f} steps, "
                        f"reward {film[f'film_{band}_reward']:.0f}, "
                        f"{film[f'film_{band}_reproductions']:.0f} offspring -> {path}"
                    )
            for key in ("film_skipped", "film_error"):
                if key in film:
                    print(f"{key}: {film[key]}")
        (Path(trainer_config.output_dir) / "eval.json").write_text(json.dumps(summary, indent=2))
        return 0

    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
        print(f"resumed from {args.resume} at world step {trainer.world_steps}")

    print(f"logging to {trainer.log_path}")
    record = trainer.train(verbose=not args.quiet)

    if "learned_total_reward" in record:
        print()
        print_evaluation(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
            "action-dependent, unlike survival, which sits near 99.4% per step "
            "and so carries almost no signal. Reward shaping: it changes what is "
            "optimized, never what is evaluated."
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
    model.add_argument("--device", default=None, help="auto, cpu, mps or cuda")

    loop = parser.add_argument_group("loop")
    loop.add_argument("--steps", type=int, default=None, help="total world steps")
    loop.add_argument("--segment-steps", type=int, default=None)
    loop.add_argument("--warmup-steps", type=int, default=None)
    loop.add_argument("--seed", type=int, default=None)

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
    hyper.add_argument("--imitation-target", type=float, default=None,
                       help="conformance at which the imitation weight reaches zero (default 0.8)")

    evaluation = parser.add_argument_group("evaluation")
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

    output = parser.add_argument_group("output")
    output.add_argument("--out", default=None, help="output directory")
    output.add_argument("--checkpoint-interval", type=int, default=None)
    output.add_argument("--resume", default=None, metavar="CHECKPOINT")
    output.add_argument("--wandb", action="store_true", default=None)
    output.add_argument("--wandb-project", default=None)
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
        "normalize_values": args.normalize_values,
        "arch": args.arch,
        "metabolic_levels": args.metabolic_levels,
        "device": args.device,
        "seed": args.seed,
        "total_world_steps": args.steps,
        "segment_steps": args.segment_steps,
        "warmup_steps": args.warmup_steps,
        "eval_interval": args.eval_interval,
        "eval_steps": args.eval_steps,
        "eval_seeds": args.eval_seeds,
        "eval_deterministic": args.eval_deterministic,
        "checkpoint_interval": args.checkpoint_interval,
        "output_dir": args.out,
        "wandb": args.wandb,
        "wandb_project": args.wandb_project,
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
    trainer = Trainer(trainer_config, ppo_config)
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

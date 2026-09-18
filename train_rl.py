#!/usr/bin/env python
"""Train a shared policy for the individuals of one species, and score it.

The experiment this exists for is one sentence: can a learned policy beat the
simulation's own rule-based policy at sustaining its species? The score is
the metric of planning/11: the stock of biomass carried by the species' living
individuals, averaged over a window of eval-steps world steps from a bank of
warmed start states, with the extinction fraction, the rules' score on the
same starts and the spread over starts beside it. Training and the comparison
are the same command, because a training curve on its own does not answer
that question.

The reward is that metric's own increment, each individual's stock change,
pooled over a box of --reward-radius cells around it. There are no reward
coefficients.

Start with --arch rule: the rule with free values, five parameters that begin
at the rule's own, so a drift reads as a sentence. Then --arch linear, then
conv, each compared against the rung below on the same starts.

Usage:
    source venv/bin/activate && python train_rl.py
    python train_rl.py --arch rule --metabolic --entity Predator --size 512
    python train_rl.py --eval-only outputs/rl/pretrained.pt --eval-deterministic
    python train_rl.py --resume outputs/rl/checkpoint.pt --steps 40000

Sizes below about 256 are not a valid ecology: the predators die out and the
three-species dynamic degenerates into herbivores grazing an empty world.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from tensor_beasts.rl.networks import architecture_names
from tensor_beasts.rl.ppo import PPOConfig
from tensor_beasts.rl.memory import estimate_training_bytes, format_bytes
from tensor_beasts.rl.trainer import Trainer, TrainerConfig

DEFAULT_CONFIG = "conf/rl/ppo.yaml"


def _boolean(value: str) -> bool:
    """Parse a boolean the way both a person and a sweep agent write one."""
    text = str(value).strip().lower()
    if text in ("true", "t", "yes", "y", "1"):
        return True
    if text in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="hyperparameter YAML")

    world = parser.add_argument_group("world")
    world.add_argument("--sim-config", default=None, help="simulation config YAML")
    world.add_argument("--size", type=int, default=None, help="world side length")
    world.add_argument("--entity", default=None)
    world.add_argument(
        "--reward-radius",
        type=int,
        default=None,
        metavar="CELLS",
        help=(
            "Radius of the box each individual's stock reward is pooled over, "
            "around its new cell. 0 (default) pays each individual its own stock "
            "change; a few cells pays it for its neighbourhood's, which is what "
            "registers the collapse the controls in planning/11 saw. The one "
            "reward knob; there are no coefficients."
        ),
    )
    world.add_argument(
        "--worlds",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Independent worlds stepped together, so each update's batch is "
            "drawn across decorrelated ecologies rather than through one "
            "world's timeline. Otherwise what is learned in a population boom "
            "is unlearned in the bust. Measured at 512: four worlds cost 5%% "
            "more wall-clock than one, and the GPU saturates between four and "
            "eight. 1 (the default) reproduces earlier runs exactly."
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
        "--arch", default=None, choices=list(architecture_names()),
        help=(
            "rule: the rule with free values, one weight per perceived feature, "
            "a sharpness and (with --metabolic) a throttle sensitivity, starting "
            "at the rule's own. linear: a 1x1 convolution, five weights per "
            "channel. conv, residual, dilated: convolutional trunks."
        ),
    )
    model.add_argument("--hidden-channels", type=int, default=None,
                       help="trunk width; for --arch rule, the conv critic's width")
    model.add_argument("--critic", default=None, choices=["conv", "linear"],
                       help="--arch rule only: the critic's architecture, chosen separately from the actor's (default conv)")
    model.add_argument(
        "--metabolic",
        # nargs="?" with a const so both spellings work: "--metabolic" as a
        # person types it, and "--metabolic=true" as a W&B sweep agent emits it.
        # A plain store_true rejects the second with "ignored explicit argument",
        # which would have failed every trial of a sweep that sets it.
        nargs="?",
        type=_boolean,
        const=True,
        default=None,
        help=(
            "Let the policy set its own metabolic rate through a second head, as "
            "a continuous throttle from basal_rate to max_metabolic_rate, still "
            "capped by carried biomass. Off by default, which leaves the rate to "
            "the rules and learns movement only, so existing runs reproduce. "
            "--eval-only and --resume take the setting from the checkpoint when "
            "neither this nor --no-metabolic is given."
        ),
    )
    model.add_argument(
        "--no-metabolic",
        dest="metabolic",
        action="store_false",
        default=None,
        help="leave the metabolic rate to the rules and learn movement only",
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
    loop.add_argument("--seed", type=int, default=None)
    loop.add_argument("--bank-worlds", type=int, default=None,
                      help="worlds in the rule-based run that fills the start bank (default 8)")
    loop.add_argument("--bank-steps", type=int, default=None,
                      help="length of that run in world steps (default 3000)")
    loop.add_argument("--bank-warmup", type=int, default=None,
                      help="steps of it before the first state is banked (default 1000)")
    loop.add_argument("--bank-stride", type=int, default=None,
                      help="steps between banked states (default 100)")
    loop.add_argument("--bank-cache", default=None, metavar="DIR",
                      help="directory banks are cached in, keyed by everything that shapes them (default outputs/bank)")
    loop.add_argument("--no-bank-cache", action="store_true", default=False,
                      help="build the bank in this process and write nothing")
    loop.add_argument(
        "--extinction-patience",
        type=int,
        default=None,
        metavar="SEGMENTS",
        help=(
            "Reset a world whose controlled population has been extinct for this many "
            "consecutive segments (default 3, 0 disables). The simulation has no "
            "immigration, so an extinct world never repopulates; it is replaced by a "
            "state drawn from the start bank, and the run continues. Resets are "
            "logged as world_resets."
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
    hyper.add_argument("--pretrain-epochs", type=int, default=None,
                       help="ceiling on epochs of offline distillation of the rule-based policy before RL, "
                            "on the bank's labelled grids; stops early once held-out agreement plateaus "
                            "(0 = off). The learner's initialisation, saved as pretrained.pt either way.")
    hyper.add_argument("--imitation-temperature", type=float, default=None,
                       help="pretraining: softmax temperature over the rule's scores for soft distillation; "
                            "0 = hard argmax (default 0.01)")
    hyper.add_argument("--recurrent-window", type=int, default=None,
                       help="with --memory-size, backpropagate through this many steps of the individual's own memory writes (0 = off, stage 1)")

    evaluation = parser.add_argument_group("evaluation")
    evaluation.add_argument(
        "--pin-metabolic",
        type=float,
        default=None,
        metavar="UNIT",
        help=(
            "Evaluation only: hold the learned policy's throttle fixed at this unit "
            "in [0, 1], where 0 is the basal rate and 1 the configured maximum, to "
            "separate the throttle's effect from movement's. Works with a "
            "direction-only checkpoint too."
        ),
    )
    evaluation.add_argument("--eval-interval", type=int, default=None)
    evaluation.add_argument("--eval-steps", type=int, default=None,
                            help="the metric's window T in world steps (default 4000, longer than a collapse)")
    evaluation.add_argument("--eval-seeds", type=int, default=None,
                            help="evaluation worlds, a fixed seeded subset of the bank (default 8)")
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
        "reward_radius": args.reward_radius,
        "worlds": args.worlds,
        "normalize_values": args.normalize_values,
        "arch": args.arch,
        "metabolic": args.metabolic,
        "memory_size": args.memory_size,
        "pretrain_epochs": args.pretrain_epochs,
        "bank_worlds": args.bank_worlds,
        "bank_steps": args.bank_steps,
        "bank_warmup": args.bank_warmup,
        "bank_stride": args.bank_stride,
        "bank_cache": args.bank_cache,
        "eval_pin_metabolic": args.pin_metabolic,
        "device": args.device,
        "seed": args.seed,
        "extinction_patience": args.extinction_patience,
        "total_world_steps": args.steps,
        "segment_steps": args.segment_steps,
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
        "imitation_temperature": args.imitation_temperature,
        "recurrent_window": args.recurrent_window,
    }

    trainer.update({k: v for k, v in trainer_flags.items() if v is not None})
    ppo.update({k: v for k, v in ppo_flags.items() if v is not None})
    if args.no_bank_cache:
        trainer["bank_cache"] = None

    # A checkpoint fixes the network's shape. When loading one, whatever was
    # not given on the command line is taken from the checkpoint rather than
    # from the YAML's defaults, so scoring a checkpoint needs no flags beyond
    # the path: the architecture and its kwargs, the entity, the memory width
    # and the metabolic head.
    checkpoint = args.eval_only or args.resume
    if checkpoint:
        import torch

        from tensor_beasts.rl.trainer import checkpoint_metabolic

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        saved = payload.get("trainer_config") or {}
        if args.metabolic is None:
            trainer["metabolic"] = checkpoint_metabolic(payload)
        if args.arch is None and "arch" in saved:
            trainer["arch"] = saved["arch"]
            trainer["arch_kwargs"] = dict(saved.get("arch_kwargs") or {})
        if args.entity is None and "entity" in saved:
            trainer["entity"] = saved["entity"]
        if args.memory_size is None:
            trainer["memory_size"] = int(payload.get("memory_size", saved.get("memory_size", 0)))

    arch_kwargs = dict(trainer.get("arch_kwargs") or {})
    if args.hidden_channels is not None:
        if trainer["arch"] == "linear":
            raise SystemExit("--hidden-channels does not apply to the linear architecture")
        arch_kwargs["hidden_channels"] = args.hidden_channels
    if args.critic is not None:
        if trainer["arch"] != "rule":
            raise SystemExit("--critic applies to --arch rule only")
        arch_kwargs["critic"] = args.critic
    trainer["arch_kwargs"] = arch_kwargs

    return {"trainer": trainer, "ppo": ppo}


def print_evaluation(summary: Dict[str, float]) -> None:
    """The metric, with the rules as a row beside it rather than a denominator.

    M_T is the mean stock of biomass carried by living individuals over the
    evaluation window, from the same banked starts for both policies. The
    rules' constants were chosen by hand, so a ratio against them would
    describe those constants as much as the ecology.
    """
    header = (
        f"{'policy':12} {'M_T':>11} {'extinct':>8} {'pop':>9} "
        f"{'repro':>8} {'lifespan':>9} {'survived':>12}"
    )
    print(header)
    print("-" * len(header))
    for policy, label in (("learned", "learned"), ("rule_based", "rule-based")):
        print(
            f"{label:12} "
            f"{summary[f'{policy}_mean_biomass']:11.0f} "
            f"{summary[f'{policy}_extinct_fraction']:8.2f} "
            f"{summary[f'{policy}_mean_population']:9.1f} "
            f"{summary[f'{policy}_reproductions']:8.0f} "
            f"{summary[f'{policy}_episode_length']:9.1f} "
            f"{summary[f'{policy}_survived_agent_steps']:12.0f}"
        )
    print(f"\nscore (M_T, mean stock over the window) = {summary['score']:.0f}")
    if "score_spread" in summary:
        print(
            f"spread across starts: {summary['score_spread']:.0f} "
            f"({summary['score_min']:.0f} to {summary['score_max']:.0f})"
        )


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
        worlds=trainer_config.worlds,
        eval_seeds=trainer_config.eval_seeds if trainer_config.eval_interval else 0,
    )
    print(f"estimated peak memory {format_bytes(estimate)}")
    check_device_headroom(trainer, estimate)
    print(
        f"arch={trainer_config.arch} params={trainer.network.num_parameters()} "
        f"receptive_field={trainer.network.receptive_field} "
        f"channels={trainer.observation_channels} device={trainer.device} "
        f"size={trainer_config.size}x{trainer_config.size} "
        f"metabolic={trainer_config.metabolic}"
    )

    if args.eval_only:
        trainer.load_checkpoint(Path(args.eval_only), load_optimizer=False)
        summary = trainer.evaluate()
        print(
            f"\n{args.eval_only}: {trainer_config.eval_seeds} banked starts x "
            f"{trainer_config.eval_steps} world steps"
            f"{' (argmax)' if trainer_config.eval_deterministic else ' (sampled)'}\n"
        )
        print_evaluation(summary)
        if hasattr(trainer.network, "describe"):
            print(f"values: {trainer.network.describe()}")
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

    if "score" in record:
        print()
        print_evaluation(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

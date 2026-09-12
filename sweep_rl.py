#!/usr/bin/env python
"""Hyperparameter search for the herbivore policy.

Runs many short training runs in parallel and ranks them by the only number
that matters: the learned policy's return divided by the rule-based policy's
return, scored on the same seeds.

    source venv/bin/activate && python sweep_rl.py --trials 16
    python sweep_rl.py --grid --workers 4
    python sweep_rl.py --report sweeps/sweep_20260911.json

Random search rather than grid search by default. With this many interacting
hyperparameters a grid spends most of its budget varying things that do not
matter, and random search finds the few that do with far fewer trials.

Two warnings about interpreting the output.

**Size.** Trials default to a 256 world for speed. Below roughly 256 the
predator population collapses and the three-species dynamic degenerates, so 128
is not a cheap version of this problem, it is a different one. Even 256 is a
compromise; confirm the winner at 512 before believing it.

**Noise.** A short run's final ratio is a noisy estimate. The ranking here is
for narrowing the field, not for declaring a winner. Re-run the top few with
more seeds and a longer budget.
"""

import argparse
import itertools
import json
import os
import random
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# Ranges to sample from. Each entry is a list of candidate values; random search
# picks one per trial, grid search takes the product.
#
# These are chosen around what the setting actually needs rather than generic
# PPO defaults. Notably reproduction_reward is swept widely because the balance
# between staying alive and dividing is the least-understood knob here, and
# entropy_coef because per-step survival is about 99.4%, so the reward signal is
# nearly constant and exploration has to come from somewhere.
SEARCH_SPACE: Dict[str, List[Any]] = {
    "arch": ["conv", "dilated", "residual"],
    "learning_rate": [1e-4, 3e-4, 1e-3, 3e-3],
    "entropy_coef": [0.0, 0.003, 0.01, 0.03],
    "clip_range": [0.1, 0.2, 0.3],
    "value_coef": [0.25, 0.5, 1.0],
    "gamma": [0.95, 0.99, 0.997],
    "gae_lambda": [0.9, 0.95, 0.99],
    "epochs": [2, 4, 8],
    "reproduction_reward": [0.0, 3.0, 10.0, 30.0],
    "segment_steps": [32, 64, 128],
}

# A deliberately small grid, for when a full product is wanted.
GRID_SPACE: Dict[str, List[Any]] = {
    "arch": ["conv", "dilated"],
    "learning_rate": [3e-4, 1e-3],
    "entropy_coef": [0.003, 0.01],
    "reproduction_reward": [0.0, 10.0],
}

TRAINER_KEYS = {
    "arch",
    "size",
    "seed",
    "segment_steps",
    "reproduction_reward",
    "survival_reward",
    "total_world_steps",
    "normalize_values",
}


@dataclass
class Trial:
    index: int
    params: Dict[str, Any]
    size: int
    steps: int
    eval_steps: int
    eval_seeds: int
    threads: int
    output_root: str
    result: Dict[str, Any] = field(default_factory=dict)


def _split(params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    trainer = {k: v for k, v in params.items() if k in TRAINER_KEYS}
    ppo = {k: v for k, v in params.items() if k not in TRAINER_KEYS}
    return {"trainer": trainer, "ppo": ppo}


def run_trial(trial: Trial) -> Dict[str, Any]:
    """One training run. Executed in a worker process."""
    import torch

    torch.set_num_threads(trial.threads)

    from tensor_beasts.rl.ppo import PPOConfig
    from tensor_beasts.rl.trainer import Trainer, TrainerConfig

    split = _split(trial.params)
    started = time.time()
    record: Dict[str, Any] = {"index": trial.index, "params": trial.params}

    try:
        trainer_config = TrainerConfig(
            size=trial.size,
            total_world_steps=trial.steps,
            eval_interval=trial.steps,  # one evaluation, at the end
            eval_steps=trial.eval_steps,
            eval_seeds=trial.eval_seeds,
            checkpoint_interval=0,
            output_dir=os.path.join(trial.output_root, f"trial_{trial.index:03d}"),
            device="cpu",  # workers share the machine; one process per core
            **split["trainer"],
        )
        ppo_config = PPOConfig(**split["ppo"])

        trainer = Trainer(trainer_config, ppo_config)
        trainer.train(verbose=False)
        summary = trainer.evaluate()

        record.update(
            {
                "ratio": float(summary["learned_over_rule_based"]),
                "learned_return": float(summary["learned_total_reward"]),
                "rule_based_return": float(summary["rule_based_total_reward"]),
                "learned_episode_length": float(summary["learned_episode_length"]),
                "rule_based_episode_length": float(summary["rule_based_episode_length"]),
                "seconds": time.time() - started,
            }
        )
    except Exception as exc:  # noqa: BLE001 - a failed trial is a result
        record.update(
            {
                "ratio": None,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=5),
                "seconds": time.time() - started,
            }
        )

    return record


def sample_params(space: Dict[str, List[Any]], rng: random.Random) -> Dict[str, Any]:
    return {key: rng.choice(values) for key, values in space.items()}


def grid_params(space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = sorted(space)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(space[k] for k in keys))]


def report(records: Sequence[Dict[str, Any]], top: int = 10) -> None:
    ok = [r for r in records if r.get("ratio") is not None]
    failed = [r for r in records if r.get("ratio") is None]

    ok.sort(key=lambda r: r["ratio"], reverse=True)
    print(f"\n{len(ok)} trials completed, {len(failed)} failed\n")

    if ok:
        print(f"{'rank':>4} {'ratio':>7} {'learned':>12} {'baseline':>12} {'mins':>6}  params")
        print("-" * 100)
        for rank, record in enumerate(ok[:top], start=1):
            params = ", ".join(f"{k}={v}" for k, v in sorted(record["params"].items()))
            print(
                f"{rank:>4} {record['ratio']:7.3f} {record['learned_return']:12.0f} "
                f"{record['rule_based_return']:12.0f} {record['seconds']/60:6.1f}  {params}"
            )
        best = ok[0]["ratio"]
        print()
        if best < 1.0:
            print(f"Best trial reached {best:.3f}x of the rule-based policy. Nothing beat it.")
        else:
            print(f"Best trial beat the rule-based policy by {best:.3f}x. Re-run it at 512 with")
            print("more seeds before believing it; a single short run is a noisy estimate.")

    for record in failed[:5]:
        print(f"\nFAILED trial {record['index']}: {record.get('error')}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--trials", type=int, default=12, help="random-search trials")
    parser.add_argument("--grid", action="store_true", help="sweep GRID_SPACE exhaustively instead")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=6000, help="world steps per trial")
    parser.add_argument("--eval-steps", type=int, default=400)
    parser.add_argument("--eval-seeds", type=int, default=2)
    parser.add_argument("--workers", type=int, default=None, help="default: cores // 2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="sweeps")
    parser.add_argument("--report", type=str, default=None, help="print a saved result file and exit")
    args = parser.parse_args()

    if args.report:
        report(json.loads(Path(args.report).read_text()))
        return 0

    cores = os.cpu_count() or 1
    # Half the cores by default: each worker still wants two threads for the
    # simulation, and oversubscribing makes every trial slower without
    # finishing the sweep any sooner.
    workers = args.workers or max(1, cores // 2)
    threads = max(1, cores // max(workers, 1))

    rng = random.Random(args.seed)
    if args.grid:
        combos = grid_params(GRID_SPACE)
    else:
        combos = [sample_params(SEARCH_SPACE, rng) for _ in range(args.trials)]

    output_root = os.path.join(args.out, time.strftime("sweep_%Y%m%d_%H%M%S"))
    os.makedirs(output_root, exist_ok=True)

    trials = [
        Trial(
            index=i,
            params=params,
            size=args.size,
            steps=args.steps,
            eval_steps=args.eval_steps,
            eval_seeds=args.eval_seeds,
            threads=threads,
            output_root=output_root,
        )
        for i, params in enumerate(combos)
    ]

    print(f"{len(trials)} trials, {workers} workers, {threads} torch threads each")
    print(f"world {args.size}x{args.size}, {args.steps} world steps per trial")
    print(f"writing to {output_root}\n")

    started = time.time()
    context = get_context("spawn")
    with context.Pool(workers) as pool:
        records = []
        for record in pool.imap_unordered(run_trial, trials):
            records.append(record)
            ratio = record.get("ratio")
            shown = f"{ratio:.3f}x" if ratio is not None else "FAILED"
            print(f"  [{len(records):>3}/{len(trials)}] trial {record['index']:>3}  {shown}")

    results_path = os.path.join(output_root, "results.json")
    Path(results_path).write_text(json.dumps(records, indent=2))

    print(f"\nfinished in {(time.time() - started)/60:.1f} minutes -> {results_path}")
    report(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

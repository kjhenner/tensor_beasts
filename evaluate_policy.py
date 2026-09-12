#!/usr/bin/env python
"""Score a policy on herbivore survival, through the RL environment.

The project goal is "a learned policy beats the rule-based policy at herbivore
survival". That sentence only means something once the rule-based policy has a
number, measured under exactly the rules a learned policy will face. This
script produces that number, plus two controls.

Policies:
    builtin  the simulation's own rule-based policy (the baseline to beat)
    random   uniform random direction per cell
    stay     every cell commanded to stay put

`stay` is the control that matters. If standing still scores as well as the
rule-based policy, then the ecology does not reward moving, and no amount of
learning machinery will produce interesting behaviour until that changes.

Usage:
    source venv/bin/activate && python evaluate_policy.py
    python evaluate_policy.py --episodes 10 --steps 500 --size 64
"""

import argparse
import statistics
from typing import Callable, Dict, List

import numpy as np
import torch

from tensor_beasts.rl.envs import make_env


def run_episode(env, policy: str, rng: np.random.Generator) -> Dict[str, float]:
    total_reward = 0.0
    steps = 0
    moved_cells = 0
    move_opportunities = 0

    obs, info = env.reset()
    entity = env.entity

    while True:
        occupied_before = (entity.biomass.data > 0).clone()
        position_before = occupied_before.nonzero()

        if policy == "builtin":
            obs, reward, terminated, truncated, info = env.step_builtin()
        elif policy == "random":
            action = rng.integers(0, 5, size=env.size, dtype=np.int64)
            obs, reward, terminated, truncated, info = env.step(action)
        elif policy == "stay":
            action = np.zeros(env.size, dtype=np.int64)
            obs, reward, terminated, truncated, info = env.step(action)
        else:
            raise ValueError(f"unknown policy {policy!r}")

        # A crude movement rate: how much the occupancy pattern changed.
        occupied_after = entity.biomass.data > 0
        moved_cells += int((occupied_before != occupied_after).sum())
        move_opportunities += int(position_before.shape[0])

        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    return {
        "return": total_reward,
        "steps": steps,
        "final_population": float(info["population"]),
        "movement_rate": moved_cells / max(move_opportunities, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="conf/base/simulation.yaml")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500, help="max steps per episode")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy",
        action="append",
        default=None,
        choices=["builtin", "random", "stay"],
        help="repeatable; defaults to all three",
    )
    args = parser.parse_args()

    policies: List[str] = args.policy or ["builtin", "random", "stay"]
    env = make_env(args.config, size=(args.size, args.size), max_steps=args.steps)

    print(f"config={args.config} size={args.size}x{args.size} "
          f"max_steps={args.steps} episodes={args.episodes}\n")
    header = f"{'policy':10} {'return':>12} {'+/-':>10} {'steps':>7} {'final pop':>10} {'move rate':>10}"
    print(header)
    print("-" * len(header))

    results = {}
    for policy in policies:
        runs = []
        for episode in range(args.episodes):
            torch.manual_seed(args.seed + episode)
            rng = np.random.default_rng(args.seed + episode)
            runs.append(run_episode(env, policy, rng))

        returns = [r["return"] for r in runs]
        mean_return = statistics.mean(returns)
        spread = statistics.stdev(returns) if len(returns) > 1 else 0.0
        results[policy] = mean_return
        print(f"{policy:10} {mean_return:12.1f} {spread:10.1f} "
              f"{statistics.mean(r['steps'] for r in runs):7.0f} "
              f"{statistics.mean(r['final_population'] for r in runs):10.1f} "
              f"{statistics.mean(r['movement_rate'] for r in runs):10.4f}")

    if "builtin" in results and "stay" in results and results["stay"] > 0:
        ratio = results["builtin"] / results["stay"]
        print(f"\nrule-based / stay-put = {ratio:.2f}x")
        if ratio < 1.1:
            print("The rule-based policy barely beats standing still. The ecology does")
            print("not reward moving, so there is little for a learned policy to find.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

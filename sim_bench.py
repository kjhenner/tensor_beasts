#!/usr/bin/env python
"""Benchmark and determinism harness for the simulation core.

Two jobs:

  golden  Hash full world state after a fixed number of steps from a fixed
          seed. Use this to prove a refactor or optimization did not change
          simulation behaviour: capture hashes before, capture after, diff.

  bench   Report steps per second across world sizes and devices.

Usage:
    source venv/bin/activate && python sim_bench.py golden
    python sim_bench.py golden --save baseline.json
    python sim_bench.py golden --check baseline.json
    python sim_bench.py bench --device cpu --device mps
"""

import argparse
import hashlib
import json
import sys
import time
from typing import Dict, List, Optional

import torch

from tensor_beasts.config import load_config
from tensor_beasts.world import World

DEFAULT_CONFIGS = [
    "conf/basic_config.yaml",
    "conf/terrain_config.yaml",
    "conf/toy_config.yaml",
    "conf/beast_config.yaml",
    "conf/base/simulation.yaml",
]

DEFAULT_SIZES = [64, 128, 256]


def state_hash(world: World) -> str:
    """SHA-256 over every tensor in the world, ordered by key."""
    digest = hashlib.sha256()
    for key in sorted(world.td.keys(True, True), key=str):
        value = world.td.get(key)
        if isinstance(value, torch.Tensor):
            digest.update(str(key).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def build(config_path: str, size: int, seed: int) -> World:
    torch.manual_seed(seed)
    config = load_config(config_path)
    config.world.size = [size, size]
    world = World(config.world)
    world.initialize()
    return world


def golden(configs: List[str], steps: int, size: int, seed: int) -> Dict[str, str]:
    results = {}
    for config_path in configs:
        try:
            world = build(config_path, size, seed)
            for _ in range(steps):
                world.update()
            results[config_path] = state_hash(world)
        except Exception as exc:  # noqa: BLE001 - a broken config is a result, not a crash
            results[config_path] = f"ERROR {type(exc).__name__}: {exc}"
    return results


def bench(config_path: str, size: int, steps: int, warmup: int, device: str) -> float:
    world = build(config_path, size, 0)
    for _ in range(warmup):
        world.update()
    if device == "mps":
        torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        world.update()
    if device == "mps":
        torch.mps.synchronize()
    return steps / (time.perf_counter() - start)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["golden", "bench"])
    parser.add_argument("--config", action="append", default=None, help="Config path (repeatable)")
    parser.add_argument("--size", type=int, action="append", default=None, help="World size (repeatable)")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", action="append", default=None, help="cpu or mps (repeatable)")
    parser.add_argument("--save", type=str, default=None, help="golden: write hashes to this file")
    parser.add_argument("--check", type=str, default=None, help="golden: compare against this file, exit 1 on drift")
    args = parser.parse_args()

    if args.mode == "golden":
        configs = args.config or DEFAULT_CONFIGS
        results = golden(configs, steps=args.steps or 60, size=(args.size or [64])[0], seed=args.seed)
        print(json.dumps(results, indent=2))
        if args.save:
            with open(args.save, "w") as handle:
                json.dump(results, handle, indent=2)
            print(f"\nsaved -> {args.save}", file=sys.stderr)
        if args.check:
            with open(args.check) as handle:
                expected = json.load(handle)
            drift = {k: (expected.get(k), v) for k, v in results.items() if expected.get(k) != v}
            if drift:
                print("\nSTATE DRIFT:", file=sys.stderr)
                for key, (was, now) in drift.items():
                    print(f"  {key}\n    was {was}\n    now {now}", file=sys.stderr)
                return 1
            print("\nno drift", file=sys.stderr)
        return 0

    devices = args.device or ["cpu"]
    sizes = args.size or DEFAULT_SIZES
    config_path = (args.config or ["conf/basic_config.yaml"])[0]
    for device in devices:
        torch.set_default_device(device)
        for size in sizes:
            try:
                rate = bench(config_path, size, args.steps or 200, args.warmup, device)
                cells = rate * size * size / 1e6
                print(f"{device:4} {size:4d}: {rate:8.1f} steps/s  ({cells:6.2f} Mcell-steps/s)")
            except Exception as exc:  # noqa: BLE001
                print(f"{device:4} {size:4d}: FAILED {type(exc).__name__}: {str(exc)[:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

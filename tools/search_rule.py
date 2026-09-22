#!/usr/bin/env python
"""Direct search on the metric over the rule's free values.

    venv/bin/python tools/search_rule.py --entity Predator --size 512 --device cuda --out outputs/search
    venv/bin/python tools/search_rule.py --values '{"sharpness": 30}' --out outputs/search    # one setting
    venv/bin/python tools/search_rule.py --report outputs/search/search.jsonl                 # read it back

The rule-parametrised actor has about five values: one navigation weight per
perceived feature, a sharpness, and with --metabolic a throttle sensitivity
(tensor_beasts/rl/networks.py, RulePolicy). Five values need no gradient. This
scores settings directly with the metric M_T, the mean stock over a window of
--eval-steps world steps from the same banked starts every evaluation uses,
with the extinction fraction and the spread over starts beside it, and the
rules' own score once. No reward, no critic, no credit assignment. The result
is the best rule the ecology admits, and the standard every reward-based
learner on the same actor has to reach.

The search is a coordinate sweep. Each round visits every value in turn,
tries it at each of --factors times its current setting (a value at zero is
stepped additively), keeps the best, and moves on; the factors are pulled
toward 1 every round. One evaluation is about three minutes at 512 with
eight starts over 4,000 steps on a 3090 (half that with --eval-seeds 4), so
a full search of a few hundred evaluations is a night.
Every evaluation is appended to search.jsonl in --out as it finishes, and a
rerun with the same --out skips settings already scored, so the search can
be stopped and resumed.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensor_beasts.rl.ppo import PPOConfig  # noqa: E402
from tensor_beasts.rl.trainer import Trainer, TrainerConfig  # noqa: E402

DEFAULT_CONFIG = "conf/rl/ppo.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="hyperparameter YAML, for the world and the bank")
    parser.add_argument("--sim-config", default=None)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None, help="the bank's and the evaluation subset's seed")
    parser.add_argument("--no-metabolic", dest="metabolic", action="store_false", default=True,
                        help="leave the throttle to the rules and search the direction values only")
    parser.add_argument("--eval-steps", type=int, default=None, help="the metric's window T")
    parser.add_argument("--eval-seeds", type=int, default=None, help="banked starts per evaluation")
    parser.add_argument("--eval-deterministic", action="store_true", default=False,
                        help="argmax directions; sharpness then means nothing and is not searched")
    parser.add_argument("--bank-worlds", type=int, default=None)
    parser.add_argument("--bank-steps", type=int, default=None)
    parser.add_argument("--bank-warmup", type=int, default=None)
    parser.add_argument("--bank-stride", type=int, default=None)
    parser.add_argument("--bank-cache", default=None, metavar="DIR",
                        help="directory banks are cached in (default outputs/bank)")
    parser.add_argument("--no-bank-cache", action="store_true", default=False,
                        help="build the bank in this process and write nothing")
    parser.add_argument("--rounds", type=int, default=3, help="coordinate sweeps over the values (default 3)")
    parser.add_argument("--factors", default="0.5,0.75,1.25,1.5,2",
                        help="multipliers tried on each value in the first round; pulled toward 1 each round")
    parser.add_argument("--only", default=None,
                        help="comma-separated value names to search; the rest stay at their start")
    parser.add_argument("--start", default=None, metavar="JSON",
                        help="values to start the search from, e.g. '{\"sharpness\": 30}'; default the rule's own")
    parser.add_argument("--values", default=None, metavar="JSON",
                        help="score this one setting (merged over the rule's own) and exit")
    parser.add_argument("--budget", type=int, default=0, help="stop after this many evaluations (0 = no limit)")
    parser.add_argument("--out", default="outputs/search", help="where search.jsonl and best.json go")
    parser.add_argument("--report", default=None, metavar="JSONL", help="print a table from a log and exit")
    return parser


def make_trainer(args: argparse.Namespace) -> Trainer:
    config = OmegaConf.load(args.config)
    trainer = OmegaConf.to_container(config.trainer, resolve=True)
    ppo = OmegaConf.to_container(config.ppo, resolve=True)
    overrides = {
        "config_path": args.sim_config, "size": args.size, "entity": args.entity, "device": args.device,
        "seed": args.seed, "eval_steps": args.eval_steps, "eval_seeds": args.eval_seeds,
        "bank_worlds": args.bank_worlds, "bank_steps": args.bank_steps,
        "bank_warmup": args.bank_warmup, "bank_stride": args.bank_stride, "bank_cache": args.bank_cache,
    }
    trainer.update({k: v for k, v in overrides.items() if v is not None})
    if args.no_bank_cache:
        trainer["bank_cache"] = None
    trainer.update({
        "arch": "rule", "arch_kwargs": {"critic": "linear"}, "metabolic": bool(args.metabolic),
        "memory_size": 0, "pretrain_epochs": 0, "eval_interval": 0, "checkpoint_interval": 0,
        "film_interval": 0, "wandb": False, "output_dir": args.out,
        "eval_deterministic": bool(args.eval_deterministic),
    })
    return Trainer(TrainerConfig(**trainer), PPOConfig(**ppo))


def key_of(values: Dict[str, float]) -> str:
    """Five significant figures: a value read back through the actor's
    float32 parameters must key the same as the value that was set."""
    return json.dumps({k: float(f"{float(v):.5g}") for k, v in sorted(values.items())})


def load_log(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Search:
    """Scores settings, caches by value, and appends every result to the log."""

    def __init__(self, trainer: Trainer, log_path: Path, budget: int = 0):
        self.trainer = trainer
        self.log_path = log_path
        self.budget = budget
        self.records = load_log(log_path)
        self.cache = {r["key"]: r for r in self.records if "key" in r}
        self.evaluations = 0
        self.rules: Optional[Dict[str, float]] = None

    def score(self, values: Dict[str, float], tag: str = "") -> Dict[str, object]:
        key = key_of(values)
        if key in self.cache:
            return self.cache[key]
        if self.budget and self.evaluations >= self.budget:
            raise StopIteration
        self.trainer.network.set_values(values)
        started = time.time()
        summary = self.trainer.evaluate()
        seconds = time.time() - started
        self.evaluations += 1
        record = {
            "key": key, "values": dict(values), "tag": tag,
            "score": float(summary["score"]),
            "spread": float(summary.get("score_spread", 0.0)),
            "min": float(summary.get("score_min", summary["score"])),
            "max": float(summary.get("score_max", summary["score"])),
            "extinct": float(summary["learned_extinct_fraction"]),
            "population": float(summary["learned_mean_population"]),
            "lifespan": float(summary["learned_episode_length"]),
            "rules": float(summary["rule_based_mean_biomass"]),
            "rules_extinct": float(summary["rule_based_extinct_fraction"]),
            "seconds": seconds,
        }
        self.rules = {"score": record["rules"], "extinct": record["rules_extinct"]}
        self.records.append(record)
        self.cache[key] = record
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(format_record(record), flush=True)
        return record


def format_record(record: Dict[str, object]) -> str:
    values = "  ".join(f"{k}={v:.4g}" for k, v in record["values"].items())
    return (
        f"M_T={record['score']:9.0f}  spread={record['spread']:7.0f}  extinct={record['extinct']:.2f}  "
        f"pop={record['population']:6.0f}  {record['seconds']:5.0f}s  [{record['tag']}]  {values}"
    )


def candidates(value: float, factor: float) -> float:
    if value == 0.0:
        return factor - 1.0
    return value * factor


def coordinate_search(search: Search, start: Dict[str, float], names: List[str], rounds: int, factors: List[float]) -> Dict[str, object]:
    best = search.score(start, tag="start")
    current = dict(best["values"])
    for round_index in range(rounds):
        # Factors pulled toward 1 each round: 2 -> 1.41 -> 1.19.
        exponent = 0.5 ** round_index
        round_factors = [f ** exponent for f in factors]
        for name in names:
            for factor in round_factors:
                trial = dict(current)
                trial[name] = candidates(current[name], factor)
                try:
                    record = search.score(trial, tag=f"round {round_index + 1} {name} x{factor:.3g}")
                except StopIteration:
                    print(f"budget of {search.budget} evaluations spent", flush=True)
                    return best
                if record["score"] > best["score"]:
                    best = record
                    current = dict(record["values"])
            print(f"round {round_index + 1}, after {name}: best M_T {best['score']:.0f} at {best['values']}", flush=True)
    return best


def report(records: List[Dict[str, object]]) -> None:
    if not records:
        print("nothing scored yet")
        return
    ranked = sorted(records, key=lambda r: r["score"], reverse=True)
    print(f"{len(records)} evaluations; rules scored {records[0]['rules']:.0f} "
          f"(extinct {records[0]['rules_extinct']:.2f}) on the same starts\n")
    for record in ranked[:20]:
        print(format_record(record))
    best = ranked[0]
    print(f"\nbest: M_T {best['score']:.0f} ({best['min']:.0f} to {best['max']:.0f}), "
          f"{best['score'] / best['rules']:.2f}x the rules, at {best['values']}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.report:
        report(load_log(Path(args.report)))
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trainer = make_trainer(args)
    network = trainer.network
    start = network.values()
    if args.start:
        start.update({k: float(v) for k, v in json.loads(args.start).items()})
    print(f"searching from {start} on {trainer.config.eval_seeds} banked starts x {trainer.config.eval_steps} steps", flush=True)
    search = Search(trainer, out / "search.jsonl", budget=args.budget)

    if args.values:
        values = dict(start)
        values.update({k: float(v) for k, v in json.loads(args.values).items()})
        search.score(values, tag="requested")
        report(search.records)
        return 0

    names = list(start)
    if args.eval_deterministic and "sharpness" in names:
        names.remove("sharpness")
    if args.only:
        wanted = [n.strip() for n in args.only.split(",") if n.strip()]
        unknown = [n for n in wanted if n not in names]
        if unknown:
            raise SystemExit(f"unknown values {unknown}; the actor has {names}")
        names = wanted
    factors = [float(f) for f in args.factors.split(",") if f.strip()]
    if any(f <= 0 for f in factors):
        raise SystemExit("factors must be positive")

    best = coordinate_search(search, start, names, args.rounds, factors)
    (out / "best.json").write_text(json.dumps(best, indent=2))
    print()
    report(search.records)
    print(f"\nlog: {search.log_path}\nbest: {out / 'best.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

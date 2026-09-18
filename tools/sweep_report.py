#!/usr/bin/env python
"""Read a W&B sweep back as a table, paired differences and axis marginals.

    venv/bin/python tools/sweep_report.py kfb90mco
    venv/bin/python tools/sweep_report.py kfb90mco --pair metabolic --by seed
    venv/bin/python tools/sweep_report.py kfb90mco --metric eval/score_last --columns world_resets,eval/score_best
    venv/bin/python tools/sweep_report.py ypprc6q9,abc123,def456 --pair metabolic   # a sweep plus its make-ups

Every sweep so far has been read by an ad hoc script; this is that script,
kept. The default metric is the sweep's own, eval/score_mean_late. `--pair`
names a two-level axis and `--by` the axis that identifies a pair (usually
the seed), so a paired design reads as differences at matched seeds rather
than as a ranking, which is the only honest way to read one: two trials
that differ by less than a single trial's ~6% standard error are the same
trial. Marginals are printed for every swept axis, with the caveat that a
marginal is only honest when the cells behind it are unimodal.
"""

import argparse
import statistics
from collections import defaultdict
from typing import Any, Dict, List

DEFAULT_COLUMNS = [
    "eval/score_mean_late", "eval/score_last", "eval/score_best", "eval/score_spread",
    "eval/learned_extinct_fraction", "eval/rule_based_mean_biomass",
    "eval/learned_mean_population", "eval/learned_reproductions",
    "world_resets", "argmax_agreement", "approx_kl", "explained_variance",
    "metabolic_head_mean", "metabolic_rule_mean", "metabolic_head_spread", "metabolic_rule_corr",
    "world_steps", "_runtime",
]
SHORT = {
    "eval/score_mean_late": "late", "eval/score_last": "last", "eval/score_best": "best",
    "eval/score_spread": "spread", "eval/learned_extinct_fraction": "extinct",
    "eval/rule_based_mean_biomass": "rules",
    "eval/learned_mean_population": "pop", "eval/learned_reproductions": "repro",
    "world_resets": "resets", "argmax_agreement": "agree", "approx_kl": "kl",
    "explained_variance": "ev", "metabolic_head_mean": "m_head", "metabolic_rule_mean": "m_rule",
    "metabolic_head_spread": "m_spr", "metabolic_rule_corr": "m_corr", "world_steps": "steps",
    "_runtime": "sec",
}


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e6:
            return str(int(value))
        if value != value:
            return "nan"
        if abs(value) >= 100:
            return f"{value:.0f}"
        if abs(value) >= 1:
            return f"{value:.2f}"
        return f"{value:.3f}"
    return str(value)


def load(project: str, sweep_ids: str) -> tuple:
    """Rows from one sweep, or from several given comma-separated, merged.

    Several because a make-up sweep for cells a crashed agent burned is the
    same experiment as the sweep it patches, and should be read with it. The
    swept axes are the union, so a make-up sweep that fixes an axis at one
    value still lines up under the original's columns.
    """
    import wandb
    api = wandb.Api()
    sweeps = [api.sweep(f"{project}/{sweep_id.strip()}") for sweep_id in sweep_ids.split(",")]
    swept: List[str] = []
    for sweep in sweeps:
        for k, v in (sweep.config or {}).get("parameters", {}).items():
            if "values" in v and k not in swept:
                swept.append(k)
    rows = []
    for sweep in sweeps:
        for run in sweep.runs:
            summary = {k: v for k, v in run.summary._json_dict.items() if not isinstance(v, dict)}
            rows.append({"id": run.id, "state": run.state, "config": dict(run.config), "summary": summary})
    return sweeps[0], swept, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep", help="sweep id, or several comma-separated to read as one")
    parser.add_argument("--project", default="tensor-beasts-rl")
    parser.add_argument("--metric", default=None, help="summary key to compare (default: the sweep's metric)")
    parser.add_argument("--columns", default=None, help="comma-separated summary keys to show")
    parser.add_argument("--pair", default=None, help="two-level axis to difference, e.g. metabolic")
    parser.add_argument("--by", default="seed", help="axis identifying a pair (default seed)")
    args = parser.parse_args()

    sweep, swept, rows = load(args.project, args.sweep)
    metric = args.metric or (sweep.config or {}).get("metric", {}).get("name", "eval/score_mean_late")
    columns = args.columns.split(",") if args.columns else DEFAULT_COLUMNS
    if metric not in columns:
        columns = [metric] + columns
    axes = [a for a in swept if a != args.by] + ([args.by] if args.by in swept else [])
    print(f"sweep {args.sweep}: {len(rows)} runs, state {sweep.state}, metric {metric}, axes {axes}")

    rows.sort(key=lambda r: tuple(str(r["config"].get(a)) for a in axes))
    header = "".join(f"{a[:10]:>11}" for a in axes) + f"{'state':>10}" + "".join(f"{SHORT.get(c, c)[-8:]:>9}" for c in columns)
    print(header)
    for r in rows:
        line = "".join(f"{fmt(r['config'].get(a))[:10]:>11}" for a in axes) + f"{r['state'][:9]:>10}"
        line += "".join(f"{fmt(r['summary'].get(c))[-8:]:>9}" for c in columns)
        print(line)

    def val(r):
        v = r["summary"].get(metric)
        return float(v) if isinstance(v, (int, float)) and v == v else None

    # Marginals per axis level.
    print(f"\nmarginals of {metric} (mean, sd, n):")
    for axis in axes:
        groups: Dict[Any, List[float]] = defaultdict(list)
        for r in rows:
            v = val(r)
            if v is not None:
                groups[r["config"].get(axis)].append(v)
        for level, values in sorted(groups.items(), key=lambda kv: str(kv[0])):
            sd = statistics.stdev(values) if len(values) > 1 else float("nan")
            print(f"  {axis} = {fmt(level):>8}: {statistics.mean(values):10.0f}  sd {sd:8.0f}  n {len(values)}")

    # Paired differences along --pair at matched --by and matched other axes.
    if args.pair:
        levels = sorted({r["config"].get(args.pair) for r in rows}, key=str)
        if len(levels) != 2:
            print(f"\n--pair {args.pair} has {len(levels)} levels, need 2")
            return 1
        others = [a for a in axes if a not in (args.pair, args.by)]
        cells: Dict[tuple, Dict[Any, Dict[Any, float]]] = defaultdict(lambda: defaultdict(dict))
        for r in rows:
            v = val(r)
            if v is None:
                continue
            key = tuple(r["config"].get(a) for a in others)
            cells[key][r["config"].get(args.by)][r["config"].get(args.pair)] = v
        hi, lo = levels[1], levels[0]
        print(f"\npaired: {args.pair}={hi} minus {args.pair}={lo}, matched on {args.by}" + (f" and {others}" if others else ""))
        all_diffs = []
        for key, by_group in sorted(cells.items(), key=str):
            diffs = []
            for by_value, pair in sorted(by_group.items(), key=str):
                if hi in pair and lo in pair:
                    d = pair[hi] - pair[lo]
                    diffs.append(d)
                    print(f"  {dict(zip(others, key)) if others else ''} {args.by}={by_value}: {pair[hi]:8.0f} - {pair[lo]:8.0f} = {d:+8.0f} ({100 * d / max(abs(pair[lo]), 1e-9):+.0f}%)")
            if diffs:
                se = statistics.stdev(diffs) / len(diffs) ** 0.5 if len(diffs) > 1 else float("nan")
                print(f"    mean {statistics.mean(diffs):+.0f}  se {se:.0f}  n {len(diffs)}")
                all_diffs.extend(diffs)
        if len(all_diffs) > 1:
            print(f"  overall mean {statistics.mean(all_diffs):+.0f}  se {statistics.stdev(all_diffs) / len(all_diffs) ** 0.5:.0f}  n {len(all_diffs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

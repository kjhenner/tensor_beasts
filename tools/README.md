# Command-line tools

Scripts that operate on the simulation but are not part of the package. Run
them from the repository root, since they load configs by relative path:

```bash
source venv/bin/activate && python tools/sim_bench.py golden
```

| Tool | What it is for |
|---|---|
| `sim_bench.py` | Determinism and throughput. `golden` hashes world state so a refactor can be proved behaviour-preserving; `bench` reports steps per second. Hashes are only comparable within one device, so the baseline records which produced them. |
| `evaluate_policy.py` | Scores the rule-based policy, random, and stay-put on herbivore survival. The baseline every learned result is quoted against. |
| `sim_diagnostics.py` | Per-step ecosystem statistics: populations, energy flows, eating rates. For asking why the ecology did something, rather than whether. |
| `sweep_report.py` | Reads a W&B sweep back as a table, paired differences at matched seeds, and axis marginals. The way every sweep in `conf/sweeps/` is meant to be read. |
| `sweep_rl.py` | The original parallel hyperparameter search, predating the W&B sweeps in `conf/sweeps/`. Its worker budget is sized against CPU memory, which is the wrong constraint on a GPU; prefer the W&B configs. |
| `grid_search.py` | Grid search over simulation parameters, not policy ones. Referenced by nothing else; `grid_search_results.json` is its output from February 2026. |
| `run_genetic_sim.py` | Runs the world with the slot-based genome system and watches it evolve. The genetic path is deliberately kept out of the reinforcement learning work, so nothing else refers to this. |

`train_rl.py` stays at the repository root: it is the project's primary entry
point rather than a tool, and every planning document refers to it there.

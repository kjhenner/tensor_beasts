# Claude Code Instructions

## IMPORTANT: Investigate Before Acting

**ALWAYS explore existing code, tests, and tools before writing new scripts or making changes.**

- Read relevant files to understand the existing architecture
- Check for existing test files, diagnostic tools, or utilities
- Understand the code flow before proposing fixes
- Do NOT create new debug scripts when existing infrastructure might work
- Do NOT make changes to fix problems you haven't verified exist in actual usage

## Running Python

Use the venv for all Python commands:

```bash
source venv/bin/activate && python <script.py>
```

Or for one-off commands:
```bash
venv/bin/python <script.py>
```

## Layout

- `tensor_beasts/` the package. `rl/` holds the learning stack, `entities/` and
  `features/` the simulation.
- `train_rl.py` at the root is the entry point for training and evaluation.
- `tools/` the command-line tools, run from the repository root because they
  load configs by relative path. See `tools/README.md`.
- `conf/` simulation and RL configs.
- `planning/STATE.md` describes what the code does and what the last session
  intended next; `planning/HISTORY.md` lists what was tried and which results
  are void. Neither records a settled decision. Read both before changing
  anything they cover, and rewrite `STATE.md` when the state changes.
- `outputs/` and `wandb/` are gitignored run artifacts.

## Before claiming a change is behaviour-preserving

```bash
venv/bin/python tools/sim_bench.py golden --check baseline_golden.json
```

Hashes are only comparable within one device, so the baseline records which
produced them. When a hash is supposed to move, re-capture it in its own commit
and say why.

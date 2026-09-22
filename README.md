# Tensor Beasts

An ecological simulation that runs entirely in Torch tensors: terrain,
plants, herbivores and predators on one grid, every update a tensor
operation. The world state is already a tensor, so it can be fed to a
neural network directly, and the cost of a step does not depend on how many
animals are alive.

<img src="./assets/img.png" alt="Tensor Beasts screenshot" width="400"/>

The current work asks whether a learned policy for the predator can keep the
predator going better than the hand-written rule does. What the code does
now and what the last session intended next is in `planning/STATE.md`. What
was tried before, and which earlier results are void, is in
`planning/HISTORY.md`. Neither file records a settled decision.

## Setup

Everything assumes a virtualenv at `venv/`, with the dependencies declared in
`pyproject.toml`. Torch is installed first so the CUDA build can be chosen:

```bash
python3 -m venv venv
venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126  # or /cpu
venv/bin/pip install -e .
```

## Running the simulation

```bash
venv/bin/python -m tensor_beasts                                   # conf/basic_config.yaml
venv/bin/python -m tensor_beasts --config_path conf/terrain_config.yaml
venv/bin/python -m tensor_beasts --policy <checkpoint.pt>           # drive one species from a checkpoint
```

In the viewer, `n` toggles the active screen, `+` and `-` zoom, `h` and `p`
re-seed with herbivores and predators.

Configs are strict. Entity names must match class names exactly, and display
renderers must state every field they use. A world below roughly 256x256
loses its predators and is a different problem, not a smaller one.

## Checks

```bash
venv/bin/python -m pytest -q tests
venv/bin/python tools/sim_bench.py golden --check baseline_golden.json
```

The golden check hashes world state after a fixed number of steps and exits
non-zero if simulation behaviour has drifted. Run it before and after any
change meant to be behaviour-preserving. Hashes are only comparable within
one device, and the baseline records which device produced each. When a hash
is supposed to move, re-capture it in its own commit and say why.

## Training

```bash
venv/bin/python train_rl.py --help
venv/bin/python train_rl.py --entity Predator --size 512 --arch rule --metabolic
venv/bin/python train_rl.py --eval-only <checkpoint.pt> --size 512
venv/bin/python tools/search_rule.py --entity Predator --size 512 --device cuda
```

`--device auto` picks a CUDA card when one is available. On a machine with
more than one GPU, a bare `--device cuda` means the card with the most free
memory, since CUDA's index order and `nvidia-smi`'s disagree. Pass
`--device cuda:N` to name one. The trainer estimates its peak memory and
refuses to start a run that will not fit; lower `--minibatch-steps` or
`--eval-seeds` if that happens.

Runs log to Weights & Biases by default, to whichever server
`~/.config/wandb/settings` points at. `--no-wandb` keeps a run local; the
JSONL log under the output directory is written either way. Wandb looks its
API key up by exact host string, so a key stored for `0.0.0.0:8080` is not
found when the base URL says `localhost:8080`, and the error it prints says
only "No API key configured".

The command-line tools in `tools/` are described in `tools/README.md`.

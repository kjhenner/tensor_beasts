# Tensor Beasts
Tensor Beasts is an ecological simulation that runs entirely in Torch tensors.
<img src="./assets/img.png" alt="Tensor Beasts Screenshot" width="400"/>

This is cool because:

1. Performance of the base simulation isn't impacted by simulation state.
    Whatever performance you get when the simulation starts will be maintained.
2. The world state is already tensors, so it can be easily passed through a
    neural network.

## Setup

Everything below assumes a virtualenv at `venv/`:

```bash
source venv/bin/activate
```

The environment is not in the repository, so build one first. With
[Poetry](https://python-poetry.org/docs/), `poetry install` from the repository
root. Or with pip:

```bash
python3 -m venv venv
venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126  # or /cpu
venv/bin/pip install numpy pydantic omegaconf tensordict pytest tqdm rich \
    matplotlib imageio imageio-ffmpeg pygame PyOpenGL gymnasium wandb
venv/bin/pip install -e . --no-deps
```

Training picks up a CUDA device automatically with `--device auto`. On this
project's hardware a 512 world trains about ten times faster on a GPU than on
twenty CPU cores, which is the difference between iterating at 256 and
iterating at the size where the ecology is valid.

A 512 run needs roughly 6 GB of device memory at the default minibatch, and
`train_rl.py` prints its own estimate and the chosen card's free memory before
it starts, refusing outright if the run cannot fit. Lower `--minibatch-steps` if
that happens: the backward pass over full-resolution activations is what uses
the memory, not the stored rollout.

On a machine with more than one GPU, `--device cuda` means *the card with the
most free memory*, not `cuda:0`. An index is not a stable name for a card:
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, which is what `nvidia-smi` prints, and CUDA's
own `FASTEST_FIRST` default disagree about which card is index zero. Pass
`--device cuda:N` to name one explicitly.

## Usage

Run the simulation with:
```bash
python -m tensor_beasts
```

CLI options:
```bash
usage: main.py [-h] [--config_path CONFIG_PATH]

Run the tensor beasts simulation

options:
  -h, --help            show this help message and exit
  --config_path CONFIG_PATH
                        The path to the config file. (default: beast_config.yaml)
```

Once it's running, there are a few commands you can use:

- `n` to toggle the active screen.
- `+` to zoom in.
- `-` to zoom out.
- `h` to re-seed with herbivores.
- `p` to re-seed with predators.

## Configuration

Configs are strict and forward-only. Entity names must match class names exactly:
`Terrain`, `Plant`, `Herbivore`, `Predator`, `DiffusionToy`.

Display renderers must be explicit:
- `fn_name: default` requires `key`
- `fn_name: layered` requires `layers` with `key`, `threshold`, `color_min`, `color_max`, `input_range`
- `fn_name: cross_section` requires `background_color`, `screen_height`, `section_idx`, `section_dim`, `levels`


## Development tools

In `tools/`, all of which take `--help` and are run from the repository root:

```bash
python tools/sim_bench.py golden        # hash world state; proves a change is behaviour-preserving
python tools/sim_bench.py bench         # steps per second by world size and device
python tools/evaluate_policy.py         # score a policy on herbivore survival
python tools/sim_diagnostics.py         # per-step ecosystem stats
```

`tools/sim_bench.py golden --check baseline_golden.json` exits non-zero if simulation
behaviour has drifted. Run it before and after any change meant to be a pure
refactor or optimization. When a hash is *supposed* to move, re-capture the
baseline in its own commit and say why, so behaviour changes stay visible in
review.

## Reinforcement learning

The experiment: **can a learned policy beat the rule-based policy at herbivore
survival?** The rule-based policy is not a strawman, so this is a real question.

```bash
python train_rl.py                          # train, then score against the baseline
python train_rl.py --arch dilated --size 512
python train_rl.py --eval-only outputs/rl/checkpoint.pt
python train_rl.py --metabolic                # learn the metabolic rate as well as the direction
python tools/sweep_rl.py --trials 16              # parallel hyperparameter search
python tools/evaluate_policy.py --size 512        # score the baseline on its own
```

**Every living herbivore is its own agent**, all sharing one set of policy
weights, each with its own reward and its own episode from birth to death. A
shared policy over a spatial observation is a single convolutional forward pass,
so this costs nothing versus treating the world as one controller. The framing
and the reasoning behind it are in `planning/04-reinforcement-learning.md`; the
environment is `tensor_beasts/rl/multiagent.py`.

Algorithms are `ppo`, `vtrace` and `awr`; architectures are `linear`, `conv`,
`residual` and `dilated`. Three flags extend what the learner controls:

```bash
python train_rl.py --imitation-coef 1.0                  # anchor to the rule-based policy, fading out
python train_rl.py --metabolic                           # learn the metabolic rate as well as direction
python train_rl.py --memory-size 4 --recurrent-window 8  # a per-individual memory, trained recurrently
```

The anchor is what first beat the baseline. The metabolic lever and the
memory are newer and their results are recorded honestly, including the
reward artefact that neutered the first metabolic run, in
`planning/04-reinforcement-learning.md`. `linear` is a diagnostic rather than a contender: it
can represent the rule-based policy exactly, so it tells you whether a failure
is in the setup or in the model.

The baseline to beat, five episodes of 600 steps at 512x512, in herbivore-steps
survived:

| Policy | Return | Final population |
|---|---|---|
| Rule-based | 1,082,723 | 4,634 |
| Random | 309,589 | 1,475 |
| Stay put | 60,539 | 0 by step 202 |

A learned policy has beaten it. A `conv` network trained at 256 with the
rule-based policy as a fading anchor beats the baseline at 512 by **11 to 33
percent** across five training seeds, mean 1.20x, on three paired evaluation
seeds over 400 steps; the best checkpoint held 1.155x over 1200 steps. The
same checkpoint with its metabolic rate pinned to basal scores **1.646x**: the
rule-based metabolism burns more than it needs: the rules themselves score
1.25x with their throttle pinned to basal, and the two effects multiply. See
`planning/04-reinforcement-learning.md` for the variance and the caveats.

**Size matters more than anything else here.** Below roughly 256 the predator
population goes extinct and the three-species dynamic degenerates, so a small
world is a different problem rather than a cheap version of this one. Use 256
for iteration and 512 for results.

**The numbers above are stale in two ways.** They were measured against the
integer simulation, before energy and biomass became float32, and they were
collected through an observation taken one update too early. Neither is thought
to change the herbivore conclusion, since a herbivore's food is plants and
plants barely move, but both are reasons to re-measure before quoting. The
timing bug was fatal for the predator, which hunts food that moves: see
`planning/04-reinforcement-learning.md`.

### Sweeps

`conf/sweeps/` holds W&B sweep configs for the predator work, staged so that
each stage's result decides whether the next is worth running:

```bash
wandb sweep --project tensor-beasts-rl conf/sweeps/stage0-release.yaml
wandb agent <sweep-id>
```

`stage0-release` lets the imitation anchor go, which every previous run held on;
`stage1-screen` searches architecture, entropy and learning rate;
`stage2-lineage-reward` replaces the hand-weighted reward with biomass plus a
share of the offspring's. The reasoning behind every axis, and what would
falsify each, is in `planning/06-predator-sweep.md`.

They optimise `eval/ratio_mean_late` rather than the last evaluation, because
the metric's noise floor is 16% across evaluation seeds and one evaluation is a
single sample of it.

### Logging

Runs go to Weights & Biases by default, to whichever server
`~/.config/wandb/settings` points at, which for this project is a local one on
port 8080. `--no-wandb` keeps a run local; the JSONL log under the run's output
directory is written either way, and the run URL is printed at startup.

One trap worth knowing: wandb looks its API key up by exact host string, so a
key stored for `0.0.0.0:8080` is not found if the base URL says
`localhost:8080`, and the error it prints is "No API key configured", which
says nothing about the host. That is why the host defaults to your own wandb
settings rather than to a literal in this repo.

### Watching one individual

Summed metrics say whether a policy is better, never how. `--film-interval`
records two individual-following videos: one sampled from the middle of the
return distribution and one from the top decile, both from lives that began and
ended inside the window so their returns are complete.

```bash
python train_rl.py --film-interval 4000                      # during training
python train_rl.py --eval-only <checkpoint> --film-interval 1  # from a checkpoint
```

Each frame is a crop centred on the followed individual, rendered through the
simulation's own display config, with its recent path behind it and its energy
and biomass as meters across the top. Videos land in `<out>/films/` and are
logged to W&B when `--wandb` is on. Rare on purpose: a film holds one thinned
world snapshot per recorded step.

To watch a learned policy in the interactive viewer instead of reading numbers
about it:

```bash
python -m tensor_beasts --policy outputs/rl/conv_imitation2/checkpoint.pt
python -m tensor_beasts --policy <checkpoint> --deterministic   # its clearest intent
```

Herbivores are then driven by the checkpoint and everything else runs as
normal; the stats printed each frame say which policy is in charge. Use the
config the policy was trained on, which for the checkpoints here is the default
`conf/basic_config.yaml`.

`tensor_beasts/rl/envs/world_environment.py` is a separate, single-controller
Gymnasium environment, kept for off-the-shelf algorithms that expect that API.

## TODO

**UI**
- [ ] Add panning contorls.
- [ ] Add play, pause, and step controls.
- [ ] Add overlay views to put herbivores and predators on top of plants, scents.
- [ ] Parametrize all the constants.
- [ ] Add channel aliases in entity config. This way, for example, I could specify
    that `predator.food` is `herbivore.energy`. This will be useful if there are
    more species in the future.

**Simulation**
- [ ] Make the herbivore population sustain rather than decline. It falls from
    91 to roughly 40 over a few hundred steps at 128x128 even under the
    rule-based policy.
- [ ] Keep predators alive. They go extinct by roughly step 200 on
    `basic_config.yaml`.
- [ ] Add obstacles.
- [ ] Non-linear scent diffusion. It should be hard to accumulate maximum scent,
    but also hard for it to completely dissapate.
- [ ] Add scent trails.
- [ ] Add wind and dynamic wind direction.
- [ ] Add non-uniform terrain. Heightmap? Nutrient map? Water map?
- [ ] Seasons?
- [ ] Rainfall? Watersheds?
- [ ] Add multiple kinds of plants.

**Performance**
- [ ] Refactor things to identify repeated calculations.
- [ ] Can move and eat be batched?
- [ ] Don't update inactive screens.


8/15:
- [ ] Support dynamic screen sizes.
- [ ] Slice view.
- [ ] Control panels for live param changes.

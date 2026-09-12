# Tensor Beasts
Tensor Beasts is an ecological simulation that runs mostly in `uint8` Torch
tensors.
<img src="./assets/img.png" alt="Tensor Beasts Screenshot" width="400"/>

This is cool because:

1. Performance of the base simulation isn't impacted by simulation state.
    Whatever performance you get when the simulation starts will be maintained.
2. The world state is already tensors, so it can be easily passed through a
    neural network.

## Setup

The repository ships a virtualenv. Everything below assumes it:

```bash
source venv/bin/activate
```

To build one from scratch instead, set up
[Poetry](https://python-poetry.org/docs/) and run `poetry install` from the
repository root.

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

Three scripts at the repository root, all of which take `--help`:

```bash
python sim_bench.py golden        # hash world state; proves a change is behaviour-preserving
python sim_bench.py bench         # steps per second by world size and device
python evaluate_policy.py         # score a policy on herbivore survival
python sim_diagnostics.py         # per-step ecosystem stats
```

`sim_bench.py golden --check baseline_golden.json` exits non-zero if simulation
behaviour has drifted. Run it before and after any change meant to be a pure
refactor or optimization. When a hash is *supposed* to move, re-capture the
baseline in its own commit and say why, so behaviour changes stay visible in
review.

## Reinforcement learning

```python
from tensor_beasts.rl.envs import make_env, make_vector_env

env = make_env("conf/base/simulation.yaml", size=(128, 128))
vec = make_vector_env(8, size=(128, 128))   # eight worlds, eight processes
```

The observation, action, reward and termination contract is documented at the
top of `tensor_beasts/rl/envs/world_environment.py`. The current rule-based
baseline, which a learned policy has to beat, is about 25,500 herbivore-steps
over a 600 step episode at 128x128; random scores about 14,200 and standing
still goes extinct. See `planning/03-performance-and-rl-foundation.md`.

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

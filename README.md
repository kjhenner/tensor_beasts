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

Set up [Poetry](https://python-poetry.org/docs/) if you haven't already.

From the root of this repository, run:
```bash
poetry install tensor-beasts
```

## Usage

Run the simulation with:
```bash
poetry run python -m tensor_beasts
```

CLI options:
```bash
usage: main.py [-h] [--size SIZE] [--device DEVICE]

Run the tensor beasts simulation

options:
  -h, --help       show this help message and exit
  --size SIZE      The size of the world. (default: 768)
  --device DEVICE  The device to use. (default: mps)
```
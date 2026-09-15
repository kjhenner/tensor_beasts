# A read of the project, September 2026

Written after building the predator experiment and the individual-following
films. This is the "anything here worth doing that I haven't mentioned" list,
ordered by what it would cost to be wrong about each one. Everything in it was
checked against the code on a Linux box with two NVIDIA cards, not inferred
from the documentation.

## What is in good shape

The engineering here is better than most research code. The planning record is
unusually honest: it contains four corrections where an earlier claim in the
same document is marked wrong and left standing, which is exactly the property
that made the predator bug findable at all. The test suite is 362 tests that
run in about a minute, and several of them pin the *reasoning* rather than the
output, notably the GAE test that parks decoy values in vacated cells and the
`ChannelNorm` test that perturbs one corner and asserts the opposite corner does
not move. The golden-hash harness makes behaviour changes visible in review.
None of that is common and none of it should be traded away for speed.

## The findings, in order

### 1. The learner saw a stale world, and it cost the predator everything

Fixed in this session, recorded in `04-reinforcement-learning.md`. Worth
repeating here because of what it implies for method rather than for the bug.

`MultiAgentWorldEnv.step` built the observation before `World.update` ran, while
the rule-based baseline it was compared against observed inside its own entity
update, after the entities ahead of it in dependency order had already moved. A
predator therefore aimed at where the prey *was*. The same policy scored 0.59x
of itself through the two paths, and drove its population extinct where the
rules recovered.

The method lesson: **the comparison was unfair in a way no amount of staring at
the learning curves would reveal, and it was found by running the baseline
through the learner's own path.** That control is cheap and it belongs in the
test suite for every new entity that gets a learned policy. It is now, for the
predator.

### 2. Three evaluation seeds cannot see the predator

The rule-based predator scores between 0.80 and 1.00 of *itself* over a 400-step
window depending on nothing but the random stream, and the coefficient of
variation across sixteen seeds is 15.9%. The trainer's default of three
evaluation seeds resolves nothing smaller than an 18% difference. Eight seeds
resolve 11%, sixteen resolve 8%.

A longer window makes this worse, not better, because ten times the score
arrives in the second four hundred steps once the population leaves its trough,
so the total is dominated by whether the boom landed inside the window. Seeds
are the only lever. On a GPU that is ten minutes, so the old default is simply
wrong now rather than a compromise.

### 3. The golden baseline is device-dependent, and silently so

`baseline_golden.json` cannot be reproduced on this machine. Every hash differs,
including `conf/toy_config.yaml`, whose hash has been constant across every
deliberate behaviour change since the harness was written and which contains no
animals at all. The toy world is four float32 diffusion grids, so this is
float-reduction order between devices, not drift: the baseline was captured on
the author's Metal machine and cannot be checked anywhere else.

That is a real gap in a tool whose whole purpose is to make behaviour changes
visible. A `git bisect` or a CI run on any non-Metal machine reports drift that
is not there, and would train someone to ignore it. Three options, none of them
free:

- Record a per-device baseline, `baseline_golden.{mps,cuda,cpu}.json`, and have
  `--check` select by the device it ran on. Honest and cheap; admits the hash is
  not portable.
- Hash at reduced precision, rounding to a tolerance before digesting. Keeps one
  file, and blinds the check to genuinely small changes.
- Compare against a tolerance on the tensors themselves rather than a hash,
  which is the real fix and the most work.

The first is what I would do.

### 4. There is no environment, and the README said there was

The README claimed the repository ships a virtualenv. It does not, and nothing
in the project runs without one. Fixed, including the CUDA wheel, because that
turned out to matter more than a setup note usually does: **a 512 world trains
about ten times faster on the 3090 than on twenty CPU cores**, 1.12 seconds a
segment against 10.88. The planning record's advice to iterate at 256 and only
validate at 512 was correct on a Mac and is not correct here. Training directly
at the size where the ecology is valid is now affordable, which removes the
transfer caveat attached to every herbivore result in the document.

`poetry.lock` pins torch 2.4.0 and numpy 1.26.4; the project runs fine on torch
2.14 and numpy 2.5, with identical golden hashes on the same device, so the pins
are stale rather than load-bearing.

### 5. Two tests decided what device the whole suite ran on

Fixed. `test_flow` and `test_flow_gradient` called
`torch.set_default_device('mps')` and never restored it, so on any non-Metal
machine they failed *and took three unrelated tests in `test_world_reset.py`
with them*, which read as a bug in world reset. The suite went from 4 failures
to 0. The pattern to keep is the one `test_networks.py` already used: skip when
the device is absent, restore in a `finally`.

### 6. A debug print ran on every rendered frame

Fixed. `cross_section` ended with `print(output.shape)`, and it is the
configured renderer for both `terrain_config.yaml` and `beast_config.yaml`, so
the viewer printed a tensor shape once per frame for as long as it was open.

### 7. `--help` had been broken since `--foraging-reward` was added

Fixed. Argparse `%`-formats help text, and the string contained "99.4%", so
`python train_rl.py --help` raised `ValueError` instead of printing. Worth
noting as a class: nothing runs `--help` in the test suite, and a smoke test
that runs every script's `--help` costs nothing.

### 8. Roughly 1,700 lines of self-declared dead code

`tensor_beasts/rl/dqn/`, `rl/iql/` and `rl/iqn/` are marked `"""DEPRECATED. Does
not run against the current API."""` at the top of their scripts. Seven of their
eight modules fail to import, and `iqn_trainer.py` imports
`tensor_beasts.display_manager`, a path that has not existed since the display
package moved. Nothing references them but historical prose.
`terminal_display_util.py` is a 49-line orphan with two debug prints and zero
call sites.

I have not deleted them, because the branch is named `iqn-batchiness` and a
previous commit deliberately marked them deprecated rather than removing them,
which reads as a decision. But the uncommitted diff in the working tree is two
tuning edits to `rl/dqn/` yaml files, which is to say edits to code that cannot
run. If the IQN work is genuinely over, deleting the three directories also lets
`torchrl` and `hydra-core` leave `pyproject.toml`, and those two are the reason
a fresh install looks inconsistent with what the project actually imports.

### 9. Checkpoints the README tells you to run do not exist

`python -m tensor_beasts --policy outputs/rl/conv_imitation2/checkpoint.pt` is
in the README, and `outputs/` is gitignored, so that path exists only on the
machine that trained it. Every headline number in the project rests on
checkpoints nobody else can obtain. Either the winning checkpoint should be
committed, it is 92k parameters and would be under a megabyte, or the README
should say plainly that the checkpoints are not distributed and give the command
that reproduces one.

## What I would do next, in order

1. **Re-measure the herbivore claims through the fixed observation path.** They
   were all collected through the stale one. The herbivore control shows no gap,
   because plants update after the herbivore and barely move, so the conclusion
   is very likely to survive. But "very likely to survive" is exactly the phrase
   this project's planning record has twice refused to accept in place of a
   measurement, and re-running the five seeds is under an hour on the GPU now.

2. **Finish the predator experiment on the fixed harness.** It sits at 0.859x
   after pretraining, against 0.591x on the broken one, with the population
   growing rather than collapsing. That is a live experiment, not a failed one.

3. **Commit the winning checkpoints**, or stop telling people to run them.

4. **Make `--check` device-aware** so the golden harness works for anyone who
   is not on the original machine.

5. **Decide on the dead RL trees.** Delete them or say in their docstrings that
   they are kept deliberately as a record. Either is fine; the current state
   costs a reader their time twice, once to find they do not import and once to
   find nothing uses them.

## What I would leave alone

The per-individual framing, the successor-map bookkeeping, and the anchor. They
are the load-bearing ideas, they are tested at the level of their reasoning, and
the planning record for each one is better than most papers. The README's
simulation TODO list is also still accurate, which I checked: at 128 the
predators are extinct by step 200 and herbivores fall to single digits, and at
256 herbivores recover while predators sit near 3 to 8. Those entries describe
the simulation as it actually is.

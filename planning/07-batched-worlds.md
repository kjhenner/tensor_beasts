# A batch dimension for the world

Design record, written before the code, so the approach can be argued with
rather than discovered in a diff.

## Why

Standard PPO runs many environments so each update's batch is drawn across
decorrelated states rather than through one world's timeline. This project runs
one. The repo owner's framing is the right one: otherwise what is learned in a
boom is unlearned in a bust, because every update sees a single point of the
predator-prey cycle and consecutive updates all pull the same direction.

The measurement that justifies the work, from `planning/06`: four independent
512 worlds in separate processes reach 2.55x the throughput of one, and eight
reach 2.91x, so a second world costs about 12% rather than 100%. Peak memory is
0.14 GB per world on a 24 GB card, so contention is the limit, not capacity.

That measurement is a *floor* on what a batch dimension gives rather than an
estimate of it. The simulation is 74% of a collection step, 14.7 ms against the
network's 5.3 ms, and it is many small kernels over a 512x512 grid. That is
launch-bound work: N processes pay N times the launch cost, while a batched
world pays it once for B times the data. This supersedes `planning/03`'s
conclusion that batching was worth only 1.7x, which was measured on CPU and
reasoned from memory bandwidth that does not bind here.

## What makes this tractable

Phase 1 was already done by someone else and is the reason this is a week's
work rather than a month's. `tests/test_rank_agnostic.py` holds 28 tests that
each run a function twice, once on `(H, W)` and once on a `(B, H, W)` stack of
*different* worlds, and assert the batched result equals the stack of the
individual results. Every util and observation kernel is covered:
`get_observation`, `process_observation`, `get_direction_masks`,
`torch_correlate_*`, `flow`, `neighbors`, `pad_matrix`. Those address spatial
axes from the end and already work.

So the work is not "make the kernels batched". It is "find and fix everything
*above* the kernels that assumes exactly two dimensions".

## The design

**`World.size` stays `(H, W)`. A separate `World.batch_shape` holds `(B,)` or
`()`.** Features allocate at `batch_shape + size + trailing`. This is the
choice that matters, and the alternative, folding B into `size`, is worse:
`size[0]` and `size[1]` are read as height and width in a dozen places, from
`torch.randint(0, world.size[0], ...)` in animal and plant seeding to
`main.py`'s `height, width = config.world.size`. Prepending to `size` breaks
every one of them silently, by making them sample from the batch axis.

`batch_shape` of `()` must be bit-identical to today. That is what the golden
hashes check and it is the property that makes the refactor reviewable: an
unbatched world is not a special case of the new code path, it is the same code
path with an empty prefix.

## The three hazards, in order of danger

**1. Silent cross-world coupling through flat indexing.** The worst category,
because it produces plausible wrong numbers rather than a crash.
`compute_gae` flattens `(H, W)` to `H*W` and gathers by successor index. With
`(B, H, W)` a `reshape(-1)` gives `B*H*W` while the successor indices are still
per-world, so world 2's individuals would bootstrap their values from world 0.
The same pattern appears in `_reward` and `_offspring_credit`, which do
`biomass.reshape(-1)[successor]`, and in `Animal._record_transition`, which
builds `torch.arange(H*W).reshape(H, W)`.

The fix is to make every flat index batch-aware: either offset each world's
indices by `b * H * W`, or reshape to `(B, H*W)` and gather along dim 1. The
second is clearer and is what I will use. **Every one of these sites needs a
test that would fail if the worlds were coupled**, and the pattern for such a
test already exists in `test_rank_agnostic.py`: stack *different* worlds and
assert the batched result equals the stack of individual results. Identical
worlds would pass a coupled implementation.

**2. Global reductions that silently span the batch.** `.sum()` with no `dim`
over a grid tensor reduces across B as well, coupling worlds. Audited: the ones
that exist are all in `terrain_features.py`, in the hydrology features
(`SoilWaterVolume`, `SurfaceWaterVolume`, elevation normalization), which
`conf/basic_config.yaml` does not use. `SimpleTerrain` uses `SimpleWater`,
`Oscillator`, `Carrion` and `CarrionScent`, and none of those reduce globally.
So the predator and herbivore configs are clear, and the hydrology configs are
out of scope for the first pass and must be *rejected* rather than silently
mis-simulated when `batch_shape` is non-empty.

**3. Loud failures, which are fine.** `perlin_noise` raises on a
`(B, H, W)` shape rather than corrupting anything, which is the good kind of
break. It needs to generate B independent fields so the worlds differ, and that
is the point: worlds seeded identically would defeat the entire purpose.

## What is out of scope for the first pass

The interactive viewer, which renders one world; the genetic registry; the
hydrology terrain features; and `rl/envs/world_environment.py`, the
single-controller Gymnasium environment. None of these need a batch dimension
for the predator experiment, and each is a place where the refactor could
sprawl. `batch_shape` non-empty should raise a clear error in each rather than
pretend to work.

## How it will be verified

In this order, and each is a gate rather than a nice-to-have.

1. **Golden hashes unchanged at `batch_shape=()`.** The refactor is not
   behaviour-preserving if this moves.
2. **`B` independent worlds equal `B` separate worlds.** Build a batched world
   from B seeds, build B unbatched worlds from the same seeds, step both, and
   assert every feature matches world by world. This is the test that catches
   cross-world coupling, and it must use *different* seeds per world.
3. **A learned policy trains** on a batched world and reaches a comparable
   ratio, since the point is the training loop rather than the simulation.
4. **The measured speedup**, against the 2.55x floor the process experiment
   established. If a batched world does not beat four processes, the refactor
   has not earned itself and should be reverted to the tag.

`git tag pre-batch-dimension` marks the commit to return to.

## Result: the simulation batches, and it is faster than processes

Gates 1 and 2 pass. A batch of one is bit-identical to an unbatched world over
25 steps across predators, herbivores, plants and water; perturbing world 0 of
a batch of three leaves worlds 1 and 2 untouched; and the golden hashes have not
moved. Five tests in `tests/test_batched_worlds.py` hold those properties.

Two silent bugs were in the way, both found by diffing a batch of one against a
single world feature by feature rather than by reading code.

The per-step `random` field was sized from `size` rather than `feature_shape`,
so every world in a batch shared one draw. Plant germination and the ids
offspring inherit are sampled from it, so the worlds' events would have been
correlated while looking independent. And `perlin_noise` reads `size[0]` and
`size[1]` as its grid, so a batched shape made it return the wrong thing; its
one caller swallowed that in `except (IndexError, RuntimeError)` and fell back
to uniform random, so a batched world silently lost its terrain and every world
got the same flat field.

Gate 4, the speedup, measured at 512 on the 3090:

| B | ms/step | World-steps/s | Speedup | Four processes, for comparison |
|---|---|---|---|---|
| 1 | 12.5 | 80.3 | 1.00x | 66.1 |
| 2 | 13.6 | 147.2 | 1.83x | 116.0 |
| 4 | 13.1 | **305.0** | **3.80x** | 168.3 |
| 8 | 23.4 | 342.5 | 4.26x | 192.0 |

**Four worlds cost 5% more wall-clock than one**, 13.1 ms against 12.5 ms, which
is what launch-bound work looks like when it is finally given something to do.
Against the process experiment's 2.55x at four workers, the batched world reaches
3.80x, so the refactor earned itself: 1.81x more throughput than the cheap
approach at the same world count. Memory is 0.36 GB at B=4, so it is not the
constraint.

B=8 is where the card saturates, at 23.4 ms a step for 4.26x, which agrees with
the process experiment finding the knee between four and eight. **Four is the
number to train at.**

Gate 3, a learned policy training on a batched world, is the remaining work: the
trainer and `MultiAgentWorldEnv` still assume one world.

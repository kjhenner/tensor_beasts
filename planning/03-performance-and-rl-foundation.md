# Performance and the RL foundation

Status doc for the work done in the September 2026 session. Written so the
reasoning survives, not just the diffs.

## The goal this work serves

> Get a learned policy to beat the rule-based policy at herbivore survival,
> reproducibly, from one command.

Chosen because it is a finish line that can be *failed*, which is what this
project needs more than it needs more features. It also orders the work: the
environment has to exist, the simulation has to be fast enough that training
is not painful, and the ecology has to reward doing something other than
standing still. Terrain, seasons, wind, multiple plant species and the rest of
the README wishlist stay parked until that baseline exists.

## Ordered plan

1. Commit the outstanding policy and genetic work. **Done.**
2. Make the simulation faster; fix the Metal device problem. **Done.** 1.8x.
3. Rebuild the RL environment against the current API so it cannot rot. **Done.**
4. Give training throughput. **Done, by a different route than planned.** 4.7x
   from process-level vectorization; the batch dimension itself is groundwork
   only.
5. Train, compare against the rule-based policy, then ask whether the ecology
   gives an agent anything worth learning. **Not started.**

## Decisions and why

### Verification is a harness, not a vibe

`sim_bench.py golden` hashes full world state after a fixed number of steps
from a fixed seed. Every performance change in this session was required to
reproduce the previous hash byte for byte, and `--check` exits non-zero on
drift. `baseline_golden.json` is the committed reference.

This paid for itself immediately. Three "obviously safe" optimizations in
`perform_move` changed simulation output, and the harness caught all three
within seconds. Without it they would have been silent.

When a hash *should* move, the baseline is re-captured in its own commit with
the reason stated. A behaviour change is then visible in review rather than
buried inside a performance commit.

### Speed: what was actually slow

Measured, not guessed. Single world, CPU, `conf/basic_config.yaml`:

| World     | Before | After |
|-----------|--------|-------|
| 64 x 64   | 150    | 238   |
| 128 x 128 | 56     | 98    |
| 256 x 256 | 11.8   | 17.1  |

Steps per second. Four changes got that 1.8x:

- Feature config was an OmegaConf `DictConfig`, so every `self.config.x` in the
  update loop re-ran the interpolation grammar through ANTLR. About a quarter
  of step time. Config is now resolved once at construction into a plain
  namespace.
- `torch.argmax` along dim 0 of a `(5, H, W)` stack hits a slow CPU kernel;
  `max().indices` is about thirty times faster for an identical result.
- Four directional clearance correlations became one batched conv.
- `perform_move` accumulates arrivals instead of stacking and reducing.

### Metal was never broken

The crash was a `lru_cache` on the kernel generators keyed only on arguments.
The first call baked in whichever device was default, and any later world on a
different device got a kernel on the wrong one, which surfaced as a `conv2d`
dtype complaint that named nothing relevant. Caches are now keyed by device.

Metal is nonetheless **not** the answer for small worlds. It sits flat near 24
steps per second from 64 wide to 512 wide, which is the signature of launch
overhead, and only beats CPU at 256 and above.

### Throughput: two hypotheses I got wrong, and what actually worked

I predicted single-world stepping was dispatch-bound, and that batching worlds
into a leading `(B, H, W)` dimension would be the big win. **Wrong.** Batching
alone measures about 1.7x on CPU and 1.15x on Metal, because at these sizes the
work is memory-bandwidth bound, not dispatch bound.

I then predicted that batching plus `torch.compile` would reach roughly 10x,
on the strength of this microbenchmark of a representative op mix:

| Configuration           | us/world | vs single eager |
|-------------------------|----------|-----------------|
| single world, eager     | 78.7     | 1.0x            |
| B=16 batched, compiled  | 13.3     | 5.9x            |
| B=64 batched, compiled  | 7.5      | 10.4x           |
| B=256 batched, compiled | 5.6      | 14.0x           |

**Also wrong, and this one is worth remembering.** That microbenchmark is a
single fusable graph. The real `World.update` is not. Dynamo traces it into 44
graphs with 43 breaks, capturing 2263 ops, and the breaks come from
data-dependent control flow (`if not actually_dead.any(): return`) and from
torch calls returning non-tensors. Compiled, the real step is *slower*:

| World     | Eager | Compiled | Ratio |
|-----------|-------|----------|-------|
| 128 x 128 | 103.4 | 57.7     | 0.56x |
| 256 x 256 | 19.0  | 13.5     | 0.71x |
| 512 x 512 | 9.7   | 5.2      | 0.54x |

Steps per second. Each graph break costs a guard check and a re-entry, and the
fused segments are not large enough to pay for it. Compilation would only
become interesting after the data-dependent branches are removed throughout the
simulation, which is a real project rather than a flag.

**What worked was much dumber.** Reinforcement learning needs total environment
steps per second, and independent worlds are embarrassingly parallel. Running
them in separate processes, with torch threads divided between workers:

| Processes | Threads each | Per process | Total env-steps/s |
|-----------|--------------|-------------|-------------------|
| 1         | 8            | 103.6       | 103.6             |
| 2         | 4            | 92.3        | 184.6             |
| 4         | 2            | 71.6        | 286.3             |
| 8         | 1            | 49.1        | 392.7             |
| 12        | 1            | 34.9        | 418.7             |

4.7x, for a few dozen lines, in `tensor_beasts/rl/envs/vector.py`. Combined with
the 1.8x single-world work, throughput went from 56 to 419 world-steps per
second at 128 x 128, which is 7.5x overall.

The thread split is the part that is easy to miss. Torch gives each process
every core by default, so uncapped workers contend and scaling is far worse.

### Where batching still matters

Not on CPU, where processes are cheaper and better. On GPU: Metal sits flat near
24 steps per second from 64 wide to 512 wide, extra processes do not help a
single device, and one small world cannot fill it. At 512 x 512 Metal already
does 5.19 Mcell-steps/s against the CPU's 2.30. Batching is the only way to feed
it properly.

So the groundwork went in, but the batch dimension itself did not.

**Phase 1 is done**: every spatial operation now addresses dimensions from the
end, so the code runs unchanged on `(H, W)` and `(..., H, W)`, proven by 35
tests that compare a batched call against a stack of individual calls. All
golden hashes are unchanged.

**Phase 2, the actual batch dimension, is not done**, and three things block it:

1. `get_direction_masks` strips a singleton batch dimension by rank, for the RL
   path. A real B=1 world batch is indistinguishable from that and would be
   silently eaten. This is a hard blocker and must be fixed first.
2. `_flatten_feature` and `Feature.render` dispatch on `ndim` to tell an
   unchannelled `(H, W)` feature from a channelled `(H, W, C)` one. `(B, H, W)`
   has the same rank as `(H, W, C)`, so this is undecidable from shape alone
   and needs an explicit channel flag on `Feature`.
3. `torch_correlate_3d` is channel-trailing while everything else is
   spatial-trailing, and it is on the hot path in `Scent.diffuse`.

Two semantic decisions also need making, not just shape work: whether B worlds
share one RNG draw, which would make a batched run not bit-identical to B
separate runs, and whether B worlds share one genetic slot registry.

### The RL environment contract

Stated explicitly in the module docstring, because leaving it implicit is how
the previous one rotted.

- **Observation**: `World.observable`, every feature tagged observable,
  concatenated on a channel axis. Global, not egocentric. Reusing the
  simulation's existing notion of observability rather than inventing a second
  one that can diverge.
- **Action**: one direction per cell, in the simulation's own encoding, passed
  through as the tensor the simulation already consumes. No packing layer that
  could silently mis-map cells.
- **Reward**: living herbivore cells after the step, using the same liveness
  test the simulation uses to kill animals. The undiscounted return is
  herbivore-steps survived, which is exactly the quantity the goal names, needs
  no tuning constants, and is directly comparable between a learned policy and
  the rule-based baseline run through the same environment.
- **Termination**: population zero, which is unrecoverable.
- **Truncation**: `max_steps`, since a world otherwise runs forever.

**Known weakness, deliberately accepted for now**: the reward is a single
global scalar while the action space is one choice per cell. Credit assignment
will be hard. The honest alternative is a per-cell reward, which needs a
defensible per-animal survival signal. Revisit at step 5 with evidence from an
actual training run rather than by guessing now.

### Reward bookkeeping was broken, and it mattered

`offspring_count` counted steps, not offspring. After one step of a 64 x 64
world, 4094 of 4096 cells had a non-zero count and 4075 of those held no
animal. Cause: carried-feature functions were applied once per direction, and
`Animal` passed `safe_add(x, 1)`, which defaults to in-place.

This is a good example of why the reward definition should live in one place.
`World.entity_scores` used `offspring_count` as its reward, so reinforcement
learning on this world would have been rewarded for elapsed time. That method
was itself dead code that could never have executed, and has been deleted
rather than repaired.

`perform_move` now requires its carried functions to be pure, says so, and
tests pin the call count directly rather than relying on a caller happening to
be impure.

### Configs

Both non-loading configs were repaired. `beast_config.yaml` matters because it
is the default named in the README, so the documented entry point was broken.

Stale keys were **removed from the yaml** rather than added to
`default_config`, because adding them would mean inventing simulation
behaviour to match a config file.

The scent failure exposed an unwritten convention: `Scent.energy_key` defaults
to the *parent* 3-D shared tensor, and every working config avoids the
resulting broadcast failure only by setting `scent.energy_key` explicitly per
entity. That convention lived in `conf/base/simulation.yaml` and nowhere else.
It is now set explicitly in the repaired configs too, but the underlying
default is still a trap worth removing.

## The baseline to beat

`evaluate_policy.py` scores a policy through the environment's own reward,
termination and truncation rules. Eight episodes, `conf/base/simulation.yaml`,
128 x 128, 600 steps:

| Policy     | Return | Std dev | Episode length | Final population | Move rate |
|------------|--------|---------|----------------|------------------|-----------|
| rule-based | 25462  | 8942    | 600            | 55.6             | 0.280     |
| random     | 14169  | 5331    | 600            | 44.0             | 0.379     |
| stay put   | 3866   | 448     | 168 (extinct)  | 0.0              | 0.021     |

Return is herbivore-steps survived, so higher is better. **This corrects an
earlier worry.** I had read the committed grid search results, which record a
mean movement rate near 0.003, as evidence that herbivores camp on plants and
that the ecology rewards standing still. Measured through the environment that
is not what happens: standing still goes extinct by step 168, moving randomly
is 3.7x better than that, and the rule-based policy is 1.8x better again. There
is real signal here for a learned policy to find.

Two caveats for step 5. The spread is roughly 35% of the mean, so demonstrating
that a learned policy beats 25462 will need many episodes, not a handful. And
at 64 x 64 the same comparison collapses into noise, with random statistically
indistinguishable from rule-based, because the surviving population is only a
few cells; evaluate at 128 or larger.

## Known problems, not fixed
- **Herbivore populations decline hard** even under the rule-based policy,
  from 91 down to roughly 40 over a few hundred steps at 128 x 128. The world
  sustains a small population rather than a thriving one.
- **Predators go extinct** by roughly step 200 on `basic_config.yaml` at
  128 x 128, leaving a herbivore-and-plant world.
- `conf/toy_zoo/single_herbivore.yaml` still does not load. It configures a
  nutrients block on a terrain class with no nutrients feature; repairing it
  means deciding whether that nutrient cycle was ever meant to exist.
- The DQN and IQN training scripts remain broken against the current API.
  Untouched deliberately: the new environment is the supported path, and those
  scripts should be rewritten against it or deleted rather than patched.
- A small leak: an animal that is removed without going through `_handle_death`
  leaves its `offspring_count` behind. Observed as one stale cell over 400
  steps, so it is not accumulating, but it is not right either.

## Next

Finish phase 2 of batching, then step 5: train, compare against the rule-based
baseline through the same environment, and use the result to decide whether the
ecology needs to be made interesting before the learning problem is worth
posing at all.

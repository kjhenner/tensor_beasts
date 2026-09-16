# The next sweep

Written after stage 0 answered its question and after the batch dimension
landed, both of which change what is worth running. Supersedes the staging in
`06-predator-sweep.md`; the per-axis arguments there still stand and are not
repeated.

## What stage 0 settled

Two trials, the imitation floor at zero:

| | Anchored, floor 0.3 | Released, target 0.8 | Released, target 0.95 |
|---|---|---|---|
| ratio against the rules | 1.064 | **1.116** | 1.025 |

**The anchor was suppressing the result, not protecting it.** Releasing it does
not regress, and the looser target does better, which is consistent: less pull
toward the rules is better now that the observation timing is fixed. Two trials
against a 16% noise floor is a sign, not a measurement, but the sign is what
stage 0 existed to establish. `imitation_floor: 0` is now a commitment, and
`imitation_target` stays in the grid at {0.8, 0.95} because stage 0 hints the
two differ and cannot show it.

## The metric changed: smoothed biomass, not a ratio

Stage 0's numbers above are ratios against the rule-based policy, which is what
every result in this project has been quoted in. That is no longer the headline,
and the reason is that the denominator was arbitrary.

The rule-based policy's navigation weights, metabolic sensitivity and log scale
were chosen by hand. Dividing by its score made every result a statement about
those particular constants, and `planning/04` already contains a case of that
going wrong: the throttle finding looked like a discovery about metabolism and
turned out to be integer truncation in the baseline. The ratio moved because
the denominator was broken. Worse, the baseline is not a fixed reference at all,
since predators and herbivores share a world: changing the learned predator
changes the prey population, which changes what the rule-based predator would
have scored. The denominator responded to the numerator.

**The headline is now `score`: the controlled entity's total carried biomass,
exponentially smoothed over the run**, in the units the ecology conserves.
Biomass rather than a population count because the ecology conserves it, and a
count weights a starving animal about to die the same as a thriving one about
to divide. Smoothed over a 100-step window rather than averaged because the
predator-prey cycle swings by a factor of four inside one run, so a plain mean
mostly reports which phase the window caught.

Reported alongside it, as the ecology rather than as a score: mean and final
biomass, mean population, reproductions, lifespan, survived agent-steps. The
rule-based policy is still run and still reported, as one row of context. The
old ratio is still computed so the existing record stays readable, but nothing
optimises it.

Sweeps target `eval/score_mean_late`.

## The thing that must be fixed before any sweep runs

**Evaluation is 80 to 90 percent of a trial's cost and it is the one thing that
does not use the batch dimension.** `Trainer.evaluate` loops
`range(eval_seeds)` and runs one world per seed, twice, once learned and once
rule-based. Measured cost of a single trial at 512:

| | Training | Evaluation | Total |
|---|---|---|---|
| 60 updates, 8 eval seeds | 1 min | 11 min | 12 min |
| 60 updates, 16 eval seeds | 1 min | 21 min | 23 min |

Evaluation seeds are exactly what the batch dimension is for: N independent
worlds from N seeds, stepped together, which is what `--worlds` already does
for training. Batching it turns 8 seeds from eight sequential runs into one
batched run, and the measured simulation speedup says that costs about 1.6x the
time of one rather than 8x.

**This is the highest-value piece of work available**, and not because of the
wall-clock. The metric's 16% noise floor is the reason every result in this
project is hedged, and more evaluation seeds is the only lever that lowers it.
At the current cost, 16 seeds is 21 minutes a trial and nobody will run it; at
a batched cost it is about 3 minutes and 32 seeds becomes reasonable. **Cheap
evaluation seeds convert directly into claims that survive.**

Scope, checked against the code rather than guessed. `Trainer.evaluate` builds
one env with `worlds=eval_seeds` and resets each world to its own seed, then
scores the batch in one pass. `EpisodeTracker` is already per world, so episode
return and length come out right for free. What changes is `_score`, whose four
accumulators sum globally:

    total_reward  += float(batch.reward.sum())
    survived      += float((batch.acted & ~batch.done).sum())
    reproductions += float(batch.reproduced.sum())
    populations.append(float(env.population()))

Each needs `dim=(-2, -1)` so it yields one number per world, and
`env.population()` likewise. The headline ratio then averages over worlds
exactly as it averages over seeds today, and `eval_seeds` becomes the batch
width rather than a loop count. The rule-based baseline runs through the same
path, so it batches with it.

A test worth writing alongside: scoring N seeds batched must equal scoring them
one at a time, to within the chaos tolerance, since a global sum here would
silently blend the worlds and inflate nothing but the confidence.

Do this first. Everything below assumes it.

## The sweep

Three stages, and the first is not a hyperparameter search.

### Stage A: does batching help? 2 trials

The batch dimension was built on the argument that a gradient averaged over
several points of the predator-prey cycle beats one drawn from a single point.
That argument has never been tested. It costs two trials to test and it decides
whether `--worlds` belongs in every later config or none of them.

    worlds in {1, 4}

Same recipe, same seed, **the same number of updates** rather than the same
number of world steps. That is the comparison that isolates the variance
reduction: four worlds see four times the data per update, which is the whole
point, so matching world steps would be matching the wrong thing.

What to look at, in order of what it would actually prove:

- **Gradient noise.** `approx_kl` and `clip_fraction` per update should be
  *lower* at four worlds if the gradient is less noisy. This is the direct
  measurement and it does not need the evaluation at all.
- **The ratio**, which is the thing anyone cares about but sits inside the
  noise floor at two trials.
- **The seed spread**, which is the real claim and needs stage C.

**Falsified if** the KL and clip fraction are unchanged. That would say each
update was never gradient-noise-limited and the decorrelation buys nothing,
which is a genuinely useful negative: it would close the line and return the
wall-clock to more updates instead.

### Stage B: the screen, 12 trials

Unchanged from `06` except that the anchor floor is now a commitment rather
than an axis, and `worlds` is set by stage A.

    arch          in {conv, dilated, residual}
    entropy_coef  in {0.0, 0.002}
    lr            in {3e-4, 1e-3}

committed: `imitation_floor` 0, `imitation_target` from the stage A winner,
`segment_steps` 32 with `worlds` 4 or 96 with `worlds` 1 (see the memory note
below), `gamma` 0.997.

`arch` is still the axis I would bet on: every result so far used a 7-cell
receptive field on three smooth diffused gradient fields, and `dilated` sees 31
cells for fewer parameters.

### Stage C: the lineage reward, 3 trials

Unchanged from `06`. It is a different objective rather than a hyperparameter,
so it runs after the screen picks a configuration, and the zero control is what
says whether the term does anything.

    offspring_credit in {0, 0.5, 1.0}

with `survival_reward` 0, `reproduction_reward` 0 and `foraging_reward` 1.0, so
biomass is the whole reward and the hand-weighting is gone rather than retuned.

### Stage D: confirmation, 5 seeds each

The best two configurations overall. Still not optional, and with batched
evaluation it is now affordable to run it at 16 or 32 evaluation seeds, which
is what would let a 5% claim mean something.

## Memory, which now constrains the grid

Measured against the 3090's roughly 20 GB free, with the world-aware estimator:

| Configuration | Estimated peak |
|---|---|
| `worlds=4 segment=32 minibatch=2` | 8.46 GB |
| `worlds=2 segment=96 minibatch=4` | 10.00 GB |
| `worlds=4 segment=48 minibatch=4` | 15.37 GB |
| `worlds=4 segment=96 minibatch=4` | 52.21 GB, refused |

So `worlds` and `segment_steps` trade against each other and cannot both be
large. The horizon argument in `06` wanted a 96-step segment to span an 81-step
predator life; at four worlds that is unaffordable, and `segment_steps` 32 with
`worlds` 4 is the configuration that fits. **That is a real tension between two
arguments this project has made, and stage A is what resolves it**: if
decorrelation buys nothing, take the long segment; if it buys a lot, take the
worlds.

One known limitation to work around rather than fix: pretraining materialises
its whole segment as a single float32 tensor, 3.88 GB at four worlds, which the
estimator does not model because it is transient. Use fewer
`--pretrain-updates` at four worlds until that accumulates per step instead.

## What I would not do

**Sweep `worlds` as a column in stage B.** It changes what a single update is,
so crossing it with the learning rate and the architecture would confound the
thing being measured with the amount of data it is measured on. Settle it in
stage A and commit it.

**Run stage B before batched evaluation.** Twelve trials at 12 minutes is two
and a half hours, most of it evaluating at 8 seeds, which resolves 11% against
a noise floor of 16%. That is buying a ranking that is mostly noise. The same
twelve trials after batched evaluation can afford 32 seeds and resolve about
6%, which is the difference between a screen that ranks and a screen that
misleads.

## The overnight sweep: 32 trials, four axes

Best configuration: `conv`, `lr` 1e-3, biomass reward, `imitation-target` 0.95,
at **12,601** smoothed predator biomass against the rule-based policy's 4,777.
**2.64x the rules**, on an absolute metric, with the imitation anchor fully
released. The population it sustains is 137 against the rules' 97.

### One cell failed, and it was masquerading as two bad axes

Read naively the marginals said `dilated` was half as good as `conv` and
`lr` 3e-4 half as good as 1e-3, both with enormous error bars. Both were the
same seven runs:

| Cell | Mean score |
|---|---|
| conv, lr 1e-3 | 12,087 |
| dilated, lr 1e-3 | 10,526 |
| conv, lr 3e-4 | 10,130 |
| **dilated, lr 3e-4** | **854** |

Seven of the eight runs in that one cell scored about 50, a hundredfold below
everything else, and dragged two axis averages down with them. The mechanism is
in the diagnostics: dilated pretrains to 0.77 agreement with the rules where
conv reaches 0.88, and at the small learning rate it never recovers. Explained
variance ends at **0.305** against 0.87 in every other cell, so the critic never
learns, the advantages are noise, and the population sits at 48 instead of 140.
At 1e-3 dilated pretrains to 0.82 and works normally.

This is the repo owner's point about pretraining, confirmed: the anchor is
fitted to a rule that compares one cell against its four neighbours, and a
network built to sample sparsely out to 31 cells fits that worse. Dilated is
not incapable; it needs a step size large enough to escape a poor start.

**A marginal is only honest when the cells behind it are unimodal.** Excluding
that one cell, the remaining 24 runs are tight and the axes separate properly:

| Axis | Effect |
|---|---|
| `lr` 1e-3 vs 3e-4 | 11,306 vs 10,130, the only large effect |
| `arch` conv vs dilated | 11,108 vs 10,526 |
| `reward-mode` | biomass 11,270, then 10,659 to 10,883 for the rest |
| `imitation-target` | 10,984 vs 10,844, a non-effect |

### What the reward modes say

Nothing decisive, and that is itself informative. The four modes span 10,659 to
11,270, a 6% range against standard errors of about 450. Plain `biomass` is
nominally first, but it is inside the band.

So **the reward's exact composition is not the binding constraint**, which is
worth knowing before more effort goes into designing rewards. The lineage credit
neither helped nor hurt: 10,883 at 0.5 and 10,659 at 1.0 against 11,270 for no
credit at all. The hypothesis it was built on, that a biomass reward without it
is a hoarding reward, is not supported: reproductions are 487 to 511 across the
top five runs regardless of credit. The predators divide anyway, because
dividing is how the population that carries biomass grows.

Biomass is kept as the default nonetheless: it scores as well as `classic` with
one hand-chosen constant instead of three, and `classic`'s survival term is 90%
of a signal that fires on 98.9% of steps.

# Sweep configs

Run from the repository root:

```bash
wandb sweep --project tensor-beasts-rl conf/sweeps/overnight.yaml
wandb agent <sweep-id>
```

A second agent in another shell shares the grid. The GPU is the bottleneck, so
two agents do not halve the wall-clock, but they usually beat one.

Everything optimises `eval/score_mean_late`: the predator population's total
carried biomass, exponentially smoothed over the run and averaged over the
later evaluations. Absolute, in the units the ecology conserves. The rule-based
policy is still scored and reported alongside, as context and not as a
denominator. Every trial also reports `score_spread`, `score_min` and
`score_max` across its evaluation seeds.

| Config | Trials | Question | Status |
|---|---|---|---|
| `metabolic` | 8 | Does a learned throttle beat the rule's throttle? Paired, cold start | **Ready** |
| `done-metabolic-anchored` | 16 | The same question, first attempt | Done: no answer, see below |
| `done-overnight` | 32 | Reward mode x architecture x learning rate x anchor target | Done |
| `done-stageA-worlds` | 2 | Does training across four worlds help? | Done: yes |
| `done-stage0-release` | 2 | Can the imitation anchor be released? | Done: yes |

The `done-*` configs are records of what ran. Two of their flags,
`--imitation-target` and `--imitation-floor`, no longer exist: the anchor is
now a fade over `--imitation-release-updates` RL updates (planning/09).

## What the first metabolic sweep settled

Nothing about the throttle, and the reason is worth keeping. Late-score means,
with the control at seed 0 only:

| Cell | Control, seed 0 | Metabolic, seed 0 | Metabolic, 4 seeds |
|---|---|---|---|
| biomass / 0.95 | 12,601 | 11,548 | 12,992 |
| biomass-0.5 / 0.95 | 11,704 | 11,399 | 12,737 |
| biomass / 0.8 | 12,303 | 10,477 | 10,318 |
| biomass-0.5 / 0.8 | 12,563 | 10,771 | 11,267 |

Evaluation worlds derive from the training seed, so only the seed-0 columns
are scored on the same worlds, and all four of those pairs are negative. The
one clean effect inside the sweep was the anchor target: 0.95 beat 0.8 by
about 2,000 with a standard error near 400.

Both facts have one cause. The throttle head's std started at 0.6 on a unit
interval where the rule sits near 0.08, and the clamp to [0, 1] rectified
that noise into a throttle averaging 0.22, a quarter more burn than the rule,
in training and in evaluation. The std annealed over the entire run, the
score tracked it (about half the control's at the second evaluation, above
it at the last), and target 0.95 won because its anchor annealed the std
faster: the anchor was a Gaussian likelihood under the learned std, so its
pull scaled as one over std squared and it strengthened itself. By the end
the winning arm's throttle was pinned within 0.02 of the rule's. The paired
`metabolic` sweep starts the std at 0.05, anchors at a fixed scale, and
scores the throttle at its mean. Full account in `planning/09`.

## What the overnight sweep settled

Best configuration `conv / lr 1e-3 / biomass / target 0.95` at **12,601**
smoothed predator biomass against the rule-based policy's 4,777: **2.64x the
rules**, absolute. Not, as first recorded, with the anchor released: the
anchor's weight was proportional to the shortfall from the target, agreement
plateaus near 0.87, and at target 0.95 the weight sat at 0.08 to 0.13 for
the whole run. At 0.8 it hit zero on the first update and re-engaged on
every dip. The controller is gone; see `planning/09`.

Read naively its marginals said `dilated` and `lr 3e-4` were each half as good
as their alternatives. Both were the same seven runs: `dilated` at `lr 3e-4`
collapsed in seven of eight trials, scoring about 50 against 9,000 to 12,600
everywhere else, and dragged two axis averages down. Dilated pretrains to 0.77
agreement where conv reaches 0.88, and at the small step it never recovers:
explained variance ends at 0.305 against 0.87 in every other cell. **A marginal
is only honest when the cells behind it are unimodal.**

Excluding that cell, `lr` 1e-3 over 3e-4 is the only large effect (11,306 vs
10,130); `arch` conv over dilated is modest (11,108 vs 10,526); and the reward
modes span 6% against standard errors near 450, so the reward's composition is
not the binding constraint. The lineage credit neither helped nor hurt, and
reproductions were 487 to 511 across the top runs regardless of it, so the
hoarding the credit was built to prevent does not happen.

## Pairing

Evaluation worlds derive from the training seed (`seed + 10_000`), so two
trials at the same seed are scored on the same worlds and their difference is
the measurement. A control from an earlier sweep only pairs with a trial at
the seed it ran at. The first metabolic sweep paired 16 trials against the
overnight grid's seed 0 and so had one clean pair per cell; `metabolic` runs
both arms at four seeds.

## How to read a sweep

**By axis marginals, not by the winning cell.** A single trial's score has a
standard error of roughly 6% at 16 evaluation seeds, and the spread across
seeds within one trial is 20 to 40 percent. Two cells that differ by less than
that are the same cell. But the grid is a full factorial, so every level of
every axis is averaged over 16 trials, and a main effect is resolvable at a
couple of percent. Group by `reward-mode` first; that is the question the
sweep exists to answer.

The stage A `worlds=4` run is a comparable 33rd point at
`conv / classic / lr 3e-4 / target 0.95`: score 8,849 against the rules' 4,777.

## What the earlier stages settled

**Stage 0.** Released from the imitation anchor, the policy improved rather than
regressing: 1.116x against 1.064x anchored. The floor was suppressing the
result. The floor was removed along with the controller it belonged to; the
anchor now fades to zero on a schedule of updates.

**Stage A.** Over the same 170 updates, four worlds cut median `approx_kl`
from 0.00704 to 0.00483 and `clip_fraction` from 0.090 to 0.066. The scores
tied inside the seed spread, so the gradient noise is the evidence. `worlds: 4`
with `segment-steps: 32` is committed, and resolves the tension with the
96-step horizon argument in `planning/06` toward worlds.

## Memory

`train_rl.py` estimates the peak and refuses to start a run that will not fit.
Measured at 512, `worlds=4 segment=32 minibatch=2`, 16 evaluation seeds:

| Phase | Peak |
|---|---|
| Pretraining | 1.8 GB |
| Collection and update | 8.1 GB |
| Evaluation, 16 seeds | 9.3 GB |

Evaluation is the peak, so `eval-seeds` is what to lower first if a run does
not fit. The 3090 has about 20 GB free with the resident llama-server, so two
agents fit with room to spare.

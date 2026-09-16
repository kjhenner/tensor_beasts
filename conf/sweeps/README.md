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
| `metabolic` | 24 | Can a predator learn its own throttle? | **Ready** |
| `done-overnight` | 32 | Reward mode x architecture x learning rate x anchor target | Done |
| `done-stageA-worlds` | 2 | Does training across four worlds help? | Done: yes |
| `done-stage0-release` | 2 | Can the imitation anchor be released? | Done: yes |

## What the overnight sweep settled

Best configuration `conv / lr 1e-3 / biomass / target 0.95` at **12,601**
smoothed predator biomass against the rule-based policy's 4,777: **2.64x the
rules**, absolute, with the anchor fully released.

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
result. `imitation-floor: 0` is committed.

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

# Sweep configs

Run from the repository root:

```bash
wandb sweep --project tensor-beasts-rl conf/sweeps/stageA-worlds.yaml
wandb agent <sweep-id>
```

All of them optimise `eval/score_mean_late`: the controlled entity's total
carried biomass, exponentially smoothed over the run, averaged over the later
evaluations of that run. Absolute, in the units the ecology conserves, and not
a ratio against the hand-tuned rule-based policy, whose constants would
otherwise be baked into every result. That policy is still scored and reported
alongside, as context.

Every trial also reports `score_spread`, `score_min` and `score_max` across its
evaluation seeds, so each one carries its own error bar rather than needing a
separate seed study.

| Stage | Trials | Question | Status |
|---|---|---|---|
| `stageA-worlds` | 2 | Does training across several worlds help? | **Done**: yes, worlds=4 |
| `stage1-screen` | 8 | Architecture, entropy, learning rate | Ready |
| `stage2-lineage-reward` | 3 | Does crediting offspring biomass help? | Waiting on 1 |
| `done-stage0-release` | 2 | Can the imitation anchor be released? | **Done**: yes |

## Order matters

**Stage A is done.** Over the same 170 updates, four worlds cut gradient noise:
median `approx_kl` 0.00483 against 0.00704 and `clip_fraction` 0.066 against
0.090. The scores tied, 8,849 against 8,676, which is well inside a seed spread
of about 3,500, so the KL is the evidence and the scores are not. Stages 1 and 2
now carry `worlds: 4`, `segment-steps: 32`, `minibatch-steps: 2`.

`residual` is dropped from stage 1's architecture axis: at four worlds its
estimated peak is 23.6 GB against about 20 GB free, so it is refused before it
starts. Eight trials instead of twelve.

**Read stage A on the KL, not the score.** If four worlds genuinely reduce
gradient noise, `approx_kl` and `clip_fraction` fall. That is the direct
measurement and it does not depend on the evaluation at all. Two trials cannot
resolve a score difference.

## Reading any of them

The measured spread across evaluation seeds is 20 to 25 percent of the score.
**Two trials whose scores differ by less than that are the same trial.** The
screen ranks; it does not decide. Anything that matters gets confirmed on
several training seeds afterwards.

## Memory

`train_rl.py` estimates the peak and refuses to start a run that will not fit,
naming the shortfall. It models the larger of the training and evaluation
phases, which do not overlap.

Measured at 512, `worlds=4 segment=32 minibatch=2` with 16 evaluation seeds:

| Phase | Peak |
|---|---|
| Pretraining | 1.76 GB |
| Collection and update | 8.10 GB |
| Evaluation, 16 seeds | 9.31 GB |
| Estimate for the run | 10.74 GB |

Evaluation is the peak, so evaluation seeds are the setting to lower first if a
run does not fit, not `worlds`.

The first stage A attempt died here: pretraining used to hold a whole segment
of observations on the device, 3.88 GB at four worlds, and peaked at 16 GB
while nothing else in the run needed more than 8. It now moves each step off
the device as it is taken and peaks at 1.76 GB, so `pretrain-updates` no longer
needs to be kept artificially small.

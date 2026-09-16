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
| `stageA-worlds` | 2 | Does training across several worlds help? | Ready |
| `stage1-screen` | 12 | Architecture, entropy, learning rate | Waiting on A |
| `stage2-lineage-reward` | 3 | Does crediting offspring biomass help? | Waiting on 1 |
| `done-stage0-release` | 2 | Can the imitation anchor be released? | **Done**: yes |

## Order matters

**Stage A first.** It answers a question the later stages depend on, and it
resolves a real conflict: the horizon argument wants a 96-step segment so credit
spans an 81-step predator life, four worlds want the memory, and at this budget
both do not fit. Stage A says which to keep. Stages 1 and 2 currently carry the
one-world values; if A says four worlds wins, set `worlds: 4`,
`segment-steps: 32` and `minibatch-steps: 2` in both before running them.

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

Estimated peak at 512, checked by `train_rl.py` before a run starts:

| Configuration | Peak |
|---|---|
| `worlds=1 segment=96 minibatch=4` | 5.00 GB |
| `worlds=4 segment=32 minibatch=2` | 8.46 GB |
| `worlds=4 segment=96 minibatch=4` | 52.21 GB, refused |

Evaluation allocates separately: 16 seeds is 5.6 GB at 512, 32 seeds is 11.1 GB.

One limitation the estimate does not model: pretraining materialises its whole
segment as a single float32 tensor, about 3.9 GB at four worlds, because it is
transient rather than resident. Keep `pretrain-updates` small at four worlds.

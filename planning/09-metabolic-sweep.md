# The metabolic sweep, and what it turned out to measure

Written after the first metabolic sweep (16 trials, 16 September 2026) and
the overnight grid before it (32 trials). Both are recorded in
`conf/sweeps/README.md`; this document keeps the reasoning, including the two
claims in `08` that turned out to be wrong. The numbers here were re-derived
from the W&B runs, not taken from the earlier summaries.

## The question and what came back

Does a predator that sets its own metabolic rate beat one whose rate is set
by the rule? The sweep ran the throttle head at the four best cells of the
overnight grid, four seeds each, and paired them against the grid's own runs
as controls. Late-score means:

| Cell | Control, seed 0 | Metabolic, seed 0 | Metabolic, mean of 4 seeds |
|---|---|---|---|
| biomass / target 0.95 | 12,601 | 11,548 | 12,992 |
| biomass-0.5 / 0.95 | 11,704 | 11,399 | 12,737 |
| biomass / 0.8 | 12,303 | 10,477 | 10,318 |
| biomass-0.5 / 0.8 | 12,563 | 10,771 | 11,267 |

Read as intended, the 0.95 cells tie or edge ahead and the 0.8 cells lose by
10 to 16 percent. Inside the sweep, target 0.95 beat 0.8 by 2,070 with a
standard error of about 410 over eight runs a side, which is the one clean
effect. The reward mode did nothing, as in the grid.

## Three things wrong with that reading

**The pairing was not paired.** Evaluation worlds derive from the training
seed (`Trainer.evaluate` seeds them at `seed + 10_000`), so a metabolic trial
at seed 1 was scored on worlds the seed-0 control never saw. Those worlds
differ enough that the rule-based policy itself ranges from 3,867 to 5,578
across the four seed sets. The only same-world comparisons are the seed-0
pairs, and all four are negative on the late score.

**The metabolic runs were still climbing.** Score at each evaluation,
averaged per cell group:

| Runs | step 1792 | 3232 | 4672 | 5760 |
|---|---|---|---|---|
| Control, 0.95 cells | 10,252 | 12,720 | 12,700 | 11,038 |
| Metabolic, 0.95 cells | 6,186 | 11,038 | 12,609 | 14,947 |
| Control, 0.8 cells | 12,438 | 14,317 | 11,291 | 11,693 |
| Metabolic, 0.8 cells | 6,966 | 9,570 | 10,629 | 12,179 |

The late score averages the last three evaluations and so penalises a run
that starts at half the control and climbs the whole way. On the final
evaluation the 0.95 metabolic arm is a third ahead, and even at seed 0 on
the same worlds it is ahead (12,612 vs 11,496 and 13,700 vs 10,580). The
controls, for their part, peak at the third evaluation and lose about 10
percent by the last, in six of eight cells. Not this sweep's question, but
it is in the record now.

**The anchor never released.** The weight was
`coef * max(floor, 1 - conformance / target)`. Conformance plateaus near
0.87 in every conv run, so at target 0.95 the weight sat at 0.08 to 0.13 for
the entire run, and at 0.8 it hit zero on the first update and re-engaged at
0.03 to 0.12 whenever agreement dipped. The overnight grid's "2.64x with the
anchor fully released" was anchored at about a tenth throughout, and the
comment in the sweep config that the anchor "holds through the first updates
and then fades" described neither arm. The intent was always a full release.

## What was actually climbing

Not a learned throttle. The throttle head's log-std was initialised at -0.5,
a std of 0.6 on a unit interval where the rule's throttle sits near 0.08.
Ten pretraining updates barely moved it: 0.45 at the first RL update.
Samples are clamped to [0, 1], and clamping a wide Gaussian centred near
zero rectifies the noise into a bias. A mean of 0.08 with std 0.45 gives a
clamped average of 0.22, which is exactly what the log shows. That is a
quarter more biomass burned per step than the rule, at lower efficiency, in
training and in evaluation, since evaluation sampled the throttle.

| | After pretraining | End, target 0.95 | End, target 0.8 |
|---|---|---|---|
| Sampled throttle mean | 0.22 | 0.08 to 0.09 | 0.10 to 0.12 |
| Policy std | 0.45 | 0.035 | 0.05 to 0.085 |
| Mean distance from the rule | 0.12 to 0.14 | 0.016 to 0.019 | 0.026 to 0.038 |

The std decayed over the entire budget and the score tracked it. Target 0.95
won because its anchor was stronger and annealed the std faster. And the
anchor was a Gaussian log-likelihood under the learned std, so its pull on
the mean scaled as one over std squared: as the anchor shrank the std it
made itself stronger. By the end, at 0.95, the throttle was pinned within
0.02 of the rule's. So the winning arm tested the rule's throttle plus
decaying noise, and the losing arm tested a lightly anchored throttle that
wandered 0.03 hotter and paid for it. Neither tested whether a policy can
use the lever, and the diagnostic the config pointed at could not have told:
`metabolic_std` is the policy's state-independent exploration std, not the
spread of the throttle across the hunt.

## What changed

Four things, in one commit, so the next sweep asks the question.

1. **The anchor is a schedule.** `imitation_release_updates` fades the
   weight linearly from `imitation_coef` to zero over that many RL updates,
   after which it is gone whatever agreement does. Pretraining fits at full
   weight. The conformance controller, its target and its floor are removed;
   conformance is still logged. A resumed run restores the update count so
   it does not re-anchor.
2. **The throttle starts cold.** The head's log-std initialises at -3, a
   std of 0.05, about the rule's own spread across individuals.
3. **The throttle anchor has a fixed scale.** It is the Gaussian likelihood
   at `METABOLIC_ANCHOR_STD = 0.1`, so a disagreement of a tenth of the
   range costs half a nat, and it fits the mean only. The policy's std is
   exploration, owned by PPO and the entropy bonus. The old argument for the
   learned-std likelihood, that a squared error "cannot be traded off
   against the entropy term", was the mechanism by which the anchor pinned
   the throttle.
4. **Evaluation scores the throttle at its mean.** The direction is still
   sampled unless `--eval-deterministic`; the throttle is not, because the
   clamp turns its sampling noise into a bias rather than a variance.

And the diagnostics the question needs: `metabolic_head_mean` and
`metabolic_rule_mean` on the same cells, `metabolic_head_spread` across
acting individuals, and `metabolic_rule_corr` between the two. A flat head
has learned nothing; a head with spread and low correlation is doing
something the rule does not.

## The next sweep

`conf/sweeps/metabolic.yaml`: metabolic on or off, four seeds each, eight
trials. Everything else is committed at a best guess from the two sweeps so
far: conv, lr 1e-3, biomass reward, a 20-update fade. None of these is
expected to moderate the throttle, and the exercise is exploratory rather
than a proof, so a paired difference that is large either way is the
answer, and one inside the seed spread says the lever does not matter at
this budget and the diagnostics say why.

What would be worth a second look afterwards, in order: the budget, since a
cold-started throttle may still need longer than 5,760 steps to find a use
for the lever; the controls' late decline; and whether the entropy bonus
should apply to the throttle's differential entropy at all, since it goes
negative and dominates the joint entropy, though at 0.002 the effect is
small.

## Two simulation faults found on the way, fixed before the sweep runs

An audit of carrion conservation and of reproduction, done because the
sweep's ecology is the measurement, found two faults that every result so far
ran on. Both are fixed in the commit after this document, and the golden
hashes move with them.

**Death did not zero energy.** Energy is stored as a SharedFeature for
tensor-layout reasons, and the death handler skipped SharedFeatures on the
theory that they are persistent fields like scent. A dead animal therefore
kept its energy, and energy is both what lets a cell move and what the
clearance kernel sees, so it went on moving, paying move costs, and blocking
living animals until dissipation drained it: about 35 steps for a herbivore
and over 200 for a predator. At one sampled step a 128-wide world held 28
predator ghosts and no live predators. The handler now zeroes every
per-animal feature but scent. A dead animal's energy is destroyed; its
biomass becomes carrion, as before, exactly once.

**Prey eaten to exactly zero never died.** The handler only cleaned cells
with biomass above zero, as a proxy for presence. A bite that takes the last
unit leaves biomass at exactly zero, so the prey's id, gradient EMA, offspring
count, slot and memory stayed on an empty cell, and the next animal to step
there was summed into them, because movement adds arrivals rather than
overwriting. The handler now cleans every cell that is not alive; on an empty
cell that is a no-op.

Two smaller ones went in alongside: an offspring no longer inherits its
parent's pre-increment offspring count, and a division now costs exactly one
move rather than half of one. The clearance check also treats any cell with
biomass as occupied, not only cells with energy; the two agree for every
living animal in the shipped configs, so this is a guard rather than a change.

Measured with a per-phase ledger at 128 squared over 300 steps, every unit of
biomass across plants, animals and carrion is attributed to a named source or
sink with zero residual, before and after the fix. The remaining sinks are the
intended ones: metabolic inefficiency, dissipation, move cost, carrion decay,
plant death, and the 255 ceilings. Two of those ceilings are worth knowing
about rather than fixing: a predator near full biomass takes food it cannot
store, about 7% of what it eats, and a predator near full energy discards
most of its metabolic yield while still burning the biomass, about three
quarters of it. Both are design consequences of the 0..255 scale.

The overnight grid and the first metabolic sweep predate these fixes, so
their absolute scores are not comparable with anything run after them. The
paired sweep is the first result on the corrected ecology.

# Extinction is not a verdict

Written the night of 16 September 2026, after the paired metabolic sweep on
the corrected ecology (planning/09) went extinct in all eight runs. This
records the general point, the hypotheses it leaves open, and what runs
overnight to test them.

## What the paired sweep showed

Every run, both arms, ended on the extinction guard between world steps
1,568 and 2,720, a quarter to half of its budget. The scores were of a policy
in collapse and the pairs were noise. Under the rules alone the corrected
ecology never goes extinct: four seeds at 512 for 6,000 steps cycle between
roughly 500 and 1,500 predators, with a trough of 205 at the very lowest, on
a period of 300 to 800 steps. The learned predator booms to about 1,250 per
world, inside the rules' range, and then falls through 200 to zero. It dies
in troughs the rules survive.

The rule-based predator now scores 16,000 to 19,000 smoothed biomass where
it scored 4,777 before the death fix, so the old headline of 2.64 times the
rules was measured against a baseline crippled by its own ghosts. On the
corrected ecology the learned policy has not beaten the rules in any
evaluation. At its best, around step 1,800, it matched the rules' population
and reproductions exactly and carried a third less biomass.

## The general point

**The world is not the run.** A training run is a policy's exposure to a
distribution of world states. A world going extinct is one of those states
ending, and the only thing it should cost is that world. Treating it as the
run's end confuses the environment's absorbing state with a training outcome,
and it does so with a timing bias: it punishes exactly the runs that reach a
bust early, which are the runs that would have learned the most from one.
Whether a policy survives busts is the thing we want to know, and a run that
stops at the first bust cannot tell us.

Four things follow.

1. **Extinction is the world's episode boundary.** This environment has no
   episodes by design; the world runs forever and individuals are born and
   die inside it. Extinction is the one place that breaks, and the right
   response is the one every episodic environment has: reset and continue.
   The trainer now resets an extinct world in place, warmed up under the
   rules, and leaves the other worlds alone. The run spends its whole
   budget and extinction becomes a count, `world_resets`, rather than a stop.
2. **No extra penalty is needed.** The individuals that died in the bust
   already entered the loss as the low returns they earned. The learner saw
   the extinction; only the trainer had stopped listening.
3. **Start where the data is.** A fresh world spends its first several
   hundred steps in a startup transient: predators fall to a third of their
   starting count and recover. Training began after 100 rule steps, inside
   that transient, and the four batched worlds, started together, were
   phase-locked and all crashed together, which defeated the argument for
   batching them (planning/07). Training now begins after 1,200 rule steps,
   in the settled cycles, and long enough that the seeds have drifted out of
   phase.
4. **Evaluate the same regime.** A 400-step evaluation from a reset measured
   the transient: the rules' smoothed biomass at step 400 was a fifth of
   their steady-state level, and every score this project has quoted is a
   score on a world that was still settling. Evaluation worlds now run 600
   steps under the rules before either policy is scored on them for 800,
   from the same reset, so the comparison stays paired.

The thing to read from a run is how often worlds die per world-step over
training, and whether that falls. Not whether one world lived.

## Hypotheses about why the learned predator dies in troughs

- **Drift.** Agreement with the rule falls from 0.89 to between 0.5 and 0.7
  once the anchor is gone, and the crash proceeds alongside. A slower release
  holds the policy nearer the rule through the first trough.
- **Step size.** The KL early stop fired on the first epoch of nearly every
  update at lr 1e-3, so the runs got about 1.4 of their 2 epochs and moved
  as far as the stop allowed every time. 1e-3 was chosen on the old ecology.
- **Exposure.** A policy that has met one bust has met one bust. With resets
  it meets several per run, and more with a longer run.
- **The pretrained policy is a weak hunter.** At the first evaluation it
  sustains half the rules' population. A frozen run, zero PPO epochs after
  pretraining, says whether that policy alone survives troughs, which
  separates "the start is bad" from "RL makes it worse".
- **The reward.** A predator rewarded for what it eats hunts its prey out.
  This is the one hypothesis nothing overnight tests, because it is a design
  decision rather than a setting. If the others fail, it is next.

## What runs overnight

In order, unattended, on the 3090, one agent at a time (a second agent was
refused by the memory guard and burned three cells before it was stopped;
those run again as the make-up sweeps):

1. The frozen diagnostic: one run at the committed settings with zero PPO
   epochs, so the pretrained policy is scored and its resets counted.
2. `conf/sweeps/overnight2.yaml`, sweep `ypprc6q9`: metabolic on and off,
   release over 20 or 100 updates, lr 3e-4 or 1e-3, three seeds. 24 runs,
   of which three were burned and are re-run as `vbst3pbj` and `tvt49ilc`.
3. `conf/sweeps/overnight2-long.yaml`, sweep `fzv7yfni`: twice the budget
   at release 100 and lr 3e-4, metabolic on and off, two seeds. 4 runs.

Everything else is committed at a best guess. In the morning:

    venv/bin/python tools/sweep_report.py ypprc6q9,vbst3pbj,tvt49ilc --pair metabolic
    venv/bin/python tools/sweep_report.py ypprc6q9,vbst3pbj,tvt49ilc --pair imitation-release-updates
    venv/bin/python tools/sweep_report.py fzv7yfni --pair metabolic

The driver is `outputs/overnight/run3.sh` and its logs sit beside it;
`touch outputs/overnight/STOP` stops it between phases.

Read `world_resets` first, by axis. Then the paired metabolic differences at
matched seeds. Then the scores against the rules, which are now scores on
the settled ecology and are not comparable with anything before this night.

## First result: the frozen diagnostic

Run at the committed settings with zero PPO epochs, so the policy is the
pretrained one, unchanged, for the whole 5,760 steps.

| | Frozen pretrained policy | Rules, same warmed worlds |
|---|---|---|
| Smoothed biomass, 8 seeds | 119,496 (97,513 to 168,500) | 66,373 |
| Mean population per world | 1,547 | |
| Mean lifespan | 122 steps | |
| World resets in training | 0 | |

The pretrained policy alone is stable across every trough in 5,760 steps and
carries 1.8 times the rules' biomass on the settled ecology. On the old
transient window it had looked like half the rules. So the hypothesis that
"the start is bad" is out: it is RL that made the policy worse and then
killed it, in all eight paired runs. Drift after release, step size and
exposure remain, and they are what overnight2 varies.

Every earlier evaluation, being of a fresh world's first 400 steps, ranked
policies by how they handle a startup transient the training worlds are
never in. That is now a settled point, not a hypothesis.

## Decisions after the overnight grid (17 September)

The grid answered its question the hard way: every cell collapsed, the
anchor's fade only set how long the collapse took, and the frozen policy
beat every RL run. A thorough re-evaluation of the inherited settings
returns to parsimonious options unless a strong justification says
otherwise, and these are the results of that re-evaluation:

- **The RL-time anchor is removed**, in all its forms: the coefficient, the
  conformance controller, the floor, the fade. The rules are the learner's
  initialisation and its yardstick, nothing more. Distance from them is still
  logged (`conformance`, `argmax_agreement`, `metabolic_error`) and pulls
  nothing. Reimplement if a reason ever appears; do not keep it around.
- **Pretraining is offline distillation, run to convergence.** Labelled
  observation grids are sampled from a rule-based run across the cycle,
  augmented by the eight symmetries of the grid, and fitted until agreement
  on a held-out split plateaus; `pretrain_epochs` is a ceiling. The
  simulation is only the sampler. The 0.88 the record kept quoting was where
  ten live updates happened to land, not a property of anything.
- **The entropy bonus is zero.** Thousands of sampled individuals explore
  already, and entropy rose on its own in every run that collapsed.
- **lr 3e-4 and conv are defaults, not findings.** The measurements that
  preferred 1e-3 and conv over dilated were made on the ghost ecology
  through an anchor that never released, on a transient-window metric.

The reward, the metric and the lineage bootstrap are the next phase and are
laid out in planning/11.

# What to sweep for the predator, and why

The learned predator sits at 1.01x to 1.06x across three seeds. This proposes
where the remaining headroom is, argued from the run's own diagnostics rather
than from a list of hyperparameters that are conventionally worth tuning. Every
number below is measured from `outputs/rl/pred512_v3/train_log.jsonl`, the
seed-0 run, 345 training updates.

## What the winning run actually did

Five measurements decide the whole argument.

| Quantity | Value | What it says |
|---|---|---|
| `argmax_agreement`, median | 0.919, ending 0.901 | The policy still behaves like the rules |
| `imitation_weight` at the floor | 98.6% of updates | The anchor never released |
| `conformance` above target | 96.8% of updates | And could not have released |
| `explained_variance` | 0.87 | The critic is excellent |
| `approx_kl`, median | 0.0056 | The policy is barely moving |

Put together: **we trained a policy that imitates the rule-based predator, held
it there with a permanent anchor, and it beat the rules by 5%.** The 5% is
whatever the learner managed to sneak past the anchor. That is not a policy
that has been allowed to find its own strategy, and it is the single largest
piece of unexploited headroom.

The mechanism of the win, from the evaluation:

| | Population | Reproductions | Lifespan |
|---|---|---|---|
| Learned | 100.7 | 301 | 81.2 |
| Rules | 94.6 | 287 | 78.4 |
| Delta | +6.5% | +4.9% | +3.6% |

The score is survived agent-steps, which is the population integrated over the
window, and population is a stock: it rises when births outrun deaths. So
**everything that matters runs through reproduction and lifespan, and both run
through kills**, because a predator reproduces by reaching a biomass threshold
and gains biomass only by eating. Any change that does not eventually produce
more successful hunts cannot move the number.

## The sweep

Seven axes, in the order I would bet on them. Each says what the measurement
is, what the mechanism would be, and what would falsify it.

### 1. The anchor's release: `imitation_floor` and `imitation_target`

**The measurement.** The anchor weight is
`imitation_coef * max(floor, 1 - conformance / target)`. With `target` 0.8 and
measured conformance at 0.913, the computed term is negative on 96.8% of
updates, so the weight is exactly `floor` on 340 of 345. I set that floor to
0.3 to stop the collapse that happened without it, and the collapse turned out
to be the stale-observation bug, not the anchor releasing. **The floor is now
solving a problem that no longer exists.**

**The mechanism.** A permanent cross-entropy pull toward the rules is a
constraint on the policy's asymptote, not just its initialization. The
herbivore work found exactly this and recorded it: matching the rules more
faithfully did not convert into survival, and the plausible reading there was
that the edge comes from *departing* from the rules in particular ways. The
predator now has a working critic, 0.87 explained variance against the
herbivore's 0.2 ceiling, so unlike the herbivore it has a reliable advantage
signal to move on once released.

**Sweep.** `imitation_floor` in {0.0, 0.05, 0.15, 0.3}, crossed with
`imitation_target` in {0.8, 0.95}. A target of 0.95 with a zero floor is the
interesting corner: the anchor stays engaged until agreement is genuinely high,
then lets go completely.

**Falsified if** the zero-floor runs drift below parity the way the herbivore's
soft-distillation runs did. That is a real possibility and it is why the floor
values are swept rather than simply removed.

### 2. The discount horizon: `gamma`

**The measurement.** `gamma` is 0.99, an effective horizon of 100 steps. A
predator's episode is 81 steps at evaluation. So the discount weight on a
reward at the end of a typical life is 0.443: **the policy values the back half
of its own lifetime at less than half weight.**

**The mechanism.** This matters far more for a predator than for a herbivore,
and the asymmetry is the argument. A herbivore eats on 78% of steps, so its
reward arrives continuously and a short horizon loses little. A predator eats
on 2.5% of steps and reproduces on 1.1%: its entire payoff is a handful of
widely spaced events, and the investment that produces them, crossing open
ground toward a scent gradient, pays out tens of steps later. A 100-step
horizon systematically underprices exactly the behaviour we want.

**Sweep.** `gamma` in {0.99, 0.995, 0.997}. At 0.997 a reward 81 steps out
keeps 0.784 weight instead of 0.443.

**Falsified if** the longer horizons raise value loss without moving the ratio,
which would mean the critic cannot support the longer credit assignment. Watch
`explained_variance`: if it falls from 0.87, the horizon is too long for the
data.

**Caveat, stated honestly.** `segment_steps` is 32, so GAE is truncated and
bootstrapped at 32 steps regardless of gamma. Raising gamma without raising the
segment mostly changes how much the bootstrap is trusted rather than extending
real credit assignment. Which is why the next axis is coupled to this one.

### 3. The segment length: `segment_steps`

**The measurement.** Segments are 32 steps; training episode length reports
16.5, which is a truncation artefact of exactly that. Evaluation lifespan is
81. **Credit never propagates across more than 32 steps of an 81-step life.**

**The mechanism.** A kill is preceded by a pursuit. If the pursuit began more
than 32 steps before the kill, no gradient connects them. This is the same
argument as gamma but about a hard cutoff rather than a soft one, and the two
have to move together: raising gamma while the segment stays at 32 buys the
bootstrap's opinion, not the actual return.

**Sweep.** `segment_steps` in {32, 64, 128}, crossed with gamma. Measured peak
memory at 512 with the `conv` network and minibatch 4: 3.46 GB at a segment of
32, 4.23 GB at 64, 5.77 GB at 128. All three fit on either card, so this axis
is cheaper than it looks.

**Falsified if** longer segments raise wall-clock without raising the ratio.
There is a real risk here: longer segments mean fewer updates for the same
budget, and the ecology is non-stationary, so stale data may cost more than
longer credit gains.

### 4. The reward composition: `reproduction_reward` and `foraging_reward`

**The measurement.** Per agent-step, at the current settings:

| Term | Contribution | Share |
|---|---|---|
| Survival (1.0 x 98.9%) | 0.989 | 89.6% |
| Reproduction (10.0 x 1.14%) | 0.114 | 10.4% |
| Foraging (0.02 x eaten) | 0.052 | measured residual |

**Survival is 90% of the reward and is nearly constant**, which is the finding
this project already recorded for the herbivore: a term that fires on 98.9% of
steps carries almost no gradient, it just adds a large mean. Meanwhile
reproduction, the thing that actually drives the score, is 10%.

**The mechanism.** The evaluation metric is population integrated over time, and
the population is a stock driven by births. The reward's weighting is close to
the inverse of what the metric rewards. Raising `reproduction_reward` aligns
the two. This is legitimate reward shaping: what is optimized may change, what
is judged stays survived agent-steps.

**Sweep.** `reproduction_reward` in {10, 30, 60}, and `foraging_reward` in
{0.01, 0.02, 0.05}. The foraging coefficient was set by matching the reward's
coefficient of variation to the herbivore recipe's, which was a reasoned guess
and is explicitly flagged in `04` as wanting a sweep.

**Falsified if** high reproduction reward produces predators that divide at the
threshold and then starve, tanking lifespan. Watch `learned_episode_length`
against `learned_reproductions`: if reproductions rise while lifespan falls
enough to cancel them, the trade is bad.

### 5. Entropy: `entropy_coef`

**The measurement.** Entropy sits at 0.50 against a maximum of ln(5) = 1.609,
and `approx_kl` has a median of 0.0056 against a `target_kl` of 0.02. The policy
is confident and barely moving. I set `entropy_coef` to 0.002, down from the
default 0.01, to stop the first run's entropy from rising, and that rise was
also the observation bug rather than the coefficient.

**The mechanism.** For a predator, exploration is not obviously good: hesitation
in a chase is costly, and the earlier noise test showed 30% random actions cost
only 7% of survival, so the policy is not fragile to noise but also gains
nothing from it. The honest position is that I do not have a first-principles
reason to prefer a value here, which is exactly what a sweep is for.

**Sweep.** `entropy_coef` in {0.0, 0.002, 0.01}. Include zero: with a working
critic and an anchor, the usual argument for an entropy bonus, preventing
premature collapse, is largely already covered.

**Falsified if** zero entropy produces a policy that collapses onto one
direction. Watch the entropy trace, not the ratio.

### 6. Architecture: `arch`

**The measurement.** All three seeds used `conv`, receptive field 7. Predator
perception is three scent fields, each a diffused gradient over many cells, and
`04` already argues that `dilated` is the architecture to bet on for a smooth
gradient field because dilation buys range cheaply. Receptive field 31 against
7.

**The mechanism.** A predator navigating toward a distant prey cluster needs to
read a gradient over tens of cells. A 7-cell window sees a local slope, which is
what the rule-based policy also sees, and that may be part of why the learned
policy cannot do much better than imitate it: **it has been given the same
narrow view the rules have.** This is the axis most likely to let the policy
find something the rules structurally cannot.

**Sweep.** `arch` in {conv, dilated, residual}. Measured: `conv` is 92k
parameters with a receptive field of 7, `dilated` is 76k with a field of 31,
and `residual` is 315k with 19. **`dilated` is both smaller and four times
wider than the network every result so far used.** This is the axis I would run
even if the sweep budget were tiny.

**Falsified if** dilated trains to the same agreement but no better ratio,
which would say range is not the binding constraint.

### 7. Learning rate: `lr`

**The measurement.** Median `approx_kl` is 0.0056 against a `target_kl` of 0.02,
and `epochs_run` is 2 of 2 on nearly every update, so the KL early-stop almost
never fires. **There is roughly 3.5x of unused KL budget.**

**The mechanism.** The run is leaving allowed policy movement on the table. With
`target_kl` as a safety net that is not being hit, a higher learning rate costs
little: if it overshoots, the early stop catches it.

**Sweep.** `lr` in {3e-4, 1e-3}. Low confidence on its own, which is why it is
last, but it is nearly free to include and it interacts with axis 1: a released
anchor needs the policy to be able to move.

## What I would actually run

`sweep_rl.py` exists and its memory guard is sized against CPU RAM, which is now
the wrong constraint; that wants fixing first, or the sweep wants driving by a
small script that runs trials serially on the GPU.

**Stage 1, the two-factor screen, 12 trials.** The anchor and the architecture
are the two axes with a mechanism argument strong enough to bet on, and they are
plausibly interacting: a wider receptive field is only useful if the policy is
allowed to depart from the rules to use it.

    imitation_floor  in {0.0, 0.15, 0.3}
    arch             in {conv, dilated}
    gamma            in {0.99, 0.997}

At 512, 6,000 world steps, 8 evaluation seeds, roughly 12 minutes a trial on the
3090 plus 6 minutes of evaluation. Call it four hours serially.

**Stage 2, the reward, 9 trials**, at the best stage-1 corner:

    reproduction_reward in {10, 30, 60}
    foraging_reward     in {0.01, 0.02, 0.05}

**Stage 3, confirmation.** The best two configurations on five training seeds
each, because the 16% noise floor means a 12-trial screen will produce an
apparent winner by chance alone. **This stage is not optional.** The screen
ranks; only the seeds decide.

## The thing I would not sweep

`survival_reward`. It is 1.0 so that the summed training reward is literally
predator-steps survived, which is the quantity the evaluation reports. Changing
it breaks the correspondence between what is optimized and what is judged for no
gain that could not be had by changing the other two terms instead.

## The honest caveat about all of this

The metric's noise floor is 16% across evaluation seeds, and every effect
proposed here is plausibly a few percent. A 12-trial screen at 8 evaluation
seeds resolves about 11%. **Most of this sweep will return noise, and the
correct reading of a 1.1x trial next to a 1.0x trial is that they are the same
trial.** The sweep's real job is to find an effect large enough to survive
stage 3, and the axis most likely to produce one is the anchor, because that is
the one where the current setting is not a tuning choice at all but a workaround
for a bug that has since been fixed.

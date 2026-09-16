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

Seven items, in the order I would bet on them, revised after the repo owner's
review. Two are now commitments rather than axes, one is a build rather than a
knob, and four are swept. Each says what the measurement is, what the mechanism
would be, and what would falsify it.

| # | Item | Disposition |
|---|---|---|
| 1 | Imitation anchor | **Commit**: fade to zero. `imitation_target` swept |
| 2 | `gamma` | **Commit**: 0.997, reasoning below |
| 3 | `segment_steps` | **Commit**: 96, with one control run |
| 4 | Reward | **Build**: biomass of self and offspring, then sweep the credit |
| 5 | `entropy_coef` | Sweep {0, 0.002, 0.01} |
| 6 | `arch` | Sweep {conv, dilated, residual} |
| 7 | `lr` | Sweep {3e-4, 1e-3} |

### 1. The anchor fades to zero, and is not a sweep axis

**The repo owner's call, and it is the right one.** The anchor is scaffolding
for the start of training, not a term in the objective. Once the policy is at
parity, a regression when the anchor lets go is information: it says the reward
is wrong. Holding the policy near the rules with a permanent floor hides that
information and buys a number we cannot interpret. So the floor goes to zero and
stays there; it is a decision, not a parameter.

**The measurement that makes this urgent.** The anchor weight is
`imitation_coef * max(floor, 1 - conformance / target)`. With `target` 0.8 and
measured conformance at 0.913, the computed term is negative on 96.8% of
updates, so the weight was exactly the floor on 340 of 345. Every result so far
was produced under a permanent pull toward the rules, by a policy that still
agrees with them 90% of the time. **We have never observed what this policy
does when it is free.**

**What this implies for how to read the sweep.** With the floor at zero, a
configuration that regresses below parity is not a failed trial to be discarded.
It is the reward function failing a test it could not previously fail, and the
configuration that regresses *least* is evidence about which reward term is
load-bearing. The anchor's release is therefore the instrument the rest of the
sweep is measured with, which is a second reason not to sweep it: a varying
instrument makes every other axis unreadable.

**Kept as a parameter:** `imitation_target`, which sets when the fade completes,
not whether. At 0.8 the anchor is already released before RL begins, given
pretraining reaches 0.93. Raising it to 0.95 means the anchor holds through the
first updates and then genuinely lets go, which is the behaviour the design
intends and has never actually exhibited. Two values, {0.8, 0.95}, and it is
the only imitation knob in the grid.

**Falsified if** every zero-floor configuration collapses. That outcome is not a
reason to restore the floor; it is the finding that the reward cannot sustain
the policy on its own, and it redirects the work to axis 4.

### 2. The discount horizon: why the commitment below is 0.997

**The measurement.** `gamma` is 0.99, an effective horizon of 100 steps. A
predator's episode is 81 steps at evaluation, so the discount weight on a reward
at the end of a typical life is 0.443: **the policy values the back half of its
own lifetime at less than half weight.**

| gamma | horizon | weight on a reward 81 steps out |
|---|---|---|
| 0.99 | 100 | 0.443 |
| 0.995 | 200 | 0.666 |
| 0.997 | 333 | 0.784 |

**The mechanism, and the asymmetry that makes it a predator problem.** A
herbivore eats on 78% of steps, so its reward arrives continuously and a short
horizon loses little. A predator eats on 2.5% of steps and reproduces on 1.1%:
its entire payoff is a handful of widely spaced events, and the investment that
produces them, crossing open ground toward a scent gradient, pays out tens of
steps later. A 100-step horizon systematically underprices exactly the behaviour
we want.

This is the soft half of the horizon argument; the hard half is the segment
length, and the two are set together in the next section rather than swept
independently.

### 3. The segment length: one commitment, not an axis

**The repo owner's call: commit rather than sweep.** Agreed, and the reasoning
is that this axis has a defensible answer from first principles, so spending
trials on it buys less than spending them elsewhere.

**The commitment: `segment_steps` 96, `gamma` 0.997.**

**Why 96.** A predator lives 81 steps at evaluation. Credit should reach across
a typical life, and a segment shorter than that truncates it: at 32, the
training log's episode length reads 16.5, which is purely the truncation. 96 is
the smallest round number above 81, so a typical life fits inside one segment
without paying for a longer one. 128 would buy only the tail of the lifespan
distribution at a third more memory and a third fewer updates per unit time.

**Why the two move together.** These are the soft and hard versions of the same
horizon and it is incoherent to set them apart. A gamma of 0.997 weights a
reward 81 steps out at 0.784, so the discount no longer discards the back half
of a life; a segment of 96 means the return that gamma discounts is actually
observed rather than bootstrapped. Setting one without the other buys the
critic's opinion instead of the data.

**Measured cost.** Peak memory at 512 with `conv` and minibatch 4: 3.46 GB at a
segment of 32, 4.23 GB at 64, 5.77 GB at 128. 96 interpolates to roughly 5 GB,
which fits on either card.

**The risk, stated plainly.** Longer segments mean fewer policy updates per unit
of world time, and this ecology is non-stationary, so the data ages. If the
committed setting underperforms the 32-step baseline at equal wall-clock, that
is the explanation, and the answer is more world steps rather than a shorter
segment.

**One control run**, at the best stage-1 corner with the old 32 / 0.99 pair, to
check the commitment rather than assume it. One trial, not an axis.

### 4. The reward: expected total biomass of self and offspring

**The repo owner's proposal, and it is the most valuable item here.** The
current reward is three hand-weighted terms standing in for fitness. The
proposal replaces the proxy with the thing itself: an individual's reward is the
biomass it accumulates, plus the biomass its descendants accumulate. That is
lifetime reproductive success measured in the currency the simulation actually
conserves.

**Why the current reward is close to the inverse of the metric.** Measured per
agent-step over the winning run:

| Term | Contribution | Share of reward |
|---|---|---|
| Survival (1.0 x 98.9%) | 0.989 | 89.6% |
| Reproduction (10.0 x 1.14%) | 0.114 | 10.4% |
| Foraging (0.02 x eaten) | 0.052 | measured residual |

Survival is 90% of the reward and fires on 98.9% of steps, so it is very nearly
a constant: it contributes almost no gradient, only a large mean that the
advantage normalizer then removes. Reproduction, which is what the score
integrates, is 10%. The evaluation counts population over time, population is a
stock driven by births, and births come from biomass. **The reward spends its
signal on the one quantity that barely varies.**

**Why biomass is the right currency.** It is the simulation's own conserved
resource: a predator eats biomass, carries biomass, divides when biomass crosses
a threshold, and passes half of it to its offspring. Survival, reproduction and
foraging are three lossy projections of it. Rewarding biomass directly removes
the three weights, and with them the question of what their ratio should be,
which this document was otherwise going to spend nine trials on.

**Why the offspring term is the hard and interesting part.** Without it, biomass
alone is a hoarding reward: an individual maximizes it by eating and never
dividing, and division actively halves it. That is the same failure mode
recorded in `04` when net biomass change punished the metabolic lever. Crediting
an individual with its descendants' biomass makes division an investment rather
than a loss, which is exactly the trade the metric rewards.

**It is implementable, and here is the mechanism.** At division the simulation
leaves the offspring in the parent's origin cell and moves the parent to its
destination, so both endpoints are known at the step reproduction happens. The
existing successor map already follows individuals through time, and
`IndividualTracker` in `rl/film.py` already chains it to reconstruct whole
lives. A lineage credit is the same machinery with one addition: when
`reproduced` is set, record an edge from parent to the offspring's cell, and
propagate the offspring's accumulated biomass back along that edge with a decay.

**The design question to settle before coding it, not after.** Whether the
offspring's contribution is discounted by generation, and by how much. Undiscounted
lineage credit is unbounded in a growing population and makes an early ancestor's
return depend on a hundred descendants it never saw, which is a variance disaster.
A per-generation discount, or crediting only the first generation, bounds it. My
recommendation is **first generation only, at a swept weight**, because it
captures the investment-in-division mechanism with bounded variance, and deeper
lineage can follow once the one-generation version is shown to work.

**Sweep.** `offspring_credit` in {0, 0.5, 1.0}, the fraction of an offspring's
own accumulated biomass credited back to its parent, with 0 as the control that
isolates whether the lineage term does anything. Held fixed alongside:
`survival_reward` 0 and biomass as the base reward, since keeping the old terms
in parallel would leave the comparison unreadable.

**Falsified if** the biomass reward without the lineage term hoards, which is
the prediction, and the lineage term does not fix it. Watch reproductions per
agent-step directly: if it falls below the rules' 0.718 births per step while
biomass per individual rises, the reward is being gamed exactly as predicted.

**Note that this is a real piece of work**, not a config change: it needs a
reward mode in `MultiAgentWorldEnv`, a parent-to-offspring edge in
`TransitionInfo`, and a test that a lineage's credit is conserved. It should be
built and tested before the sweep runs, and it is the reason stage 2 exists as a
separate stage.

### 5. Entropy: `entropy_coef`

I owed a pitch here and gave a shrug. Here is the argument.

**The measurement, decoded.** Entropy sits at 0.505 median against a maximum of
ln(5) = 1.609. That number is abstract until it is inverted: for a distribution
putting `p` on one direction and the rest uniform,

| p(top direction) | Entropy |
|---|---|
| 0.85 | 0.631 |
| 0.90 | 0.464 |
| 0.92 | 0.390 |

So **the policy is committing to a single direction about 88% of the time.** It
is not exploring; it is executing.

**Why that is the wrong default for this animal, and the asymmetry is again the
argument.** A herbivore is surrounded by food: any direction is nearly as good
as any other, sampling costs little, and an entropy bonus is cheap insurance
against premature collapse. A predator's prey is sparse and mobile. Its
information is a scent gradient it must follow across many steps, and a policy
that samples a different direction 12% of the time is not exploring the space of
strategies, it is adding noise to a pursuit. The measured noise test supports
this reading: 30% random actions cost only 7% of survival, so the policy is
robust to noise but gains nothing from it. **Entropy here buys robustness the
predator does not need and pays for it in pursuit coherence.**

**The counter-argument, which is why it stays in the grid.** Axis 1 removes the
anchor. The anchor was the thing preventing collapse onto a degenerate policy,
and with it gone the entropy bonus becomes the only remaining regularizer. The
right value with a released anchor is therefore not knowable from the anchored
run, and that interaction is precisely what makes it worth a column rather than
a commitment.

**Sweep.** `entropy_coef` in {0.0, 0.002, 0.01}, and it must be crossed with the
anchor's release rather than tuned against the old anchored runs.

**Falsified if** entropy at 0.0 falls toward zero and the policy commits to one
direction regardless of observation. Watch the entropy trace, not the ratio: a
run that wins on ratio while entropy collapses has probably found a degenerate
strategy the 400-step evaluation window is too short to punish.

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

Four axes remain after the owner's decisions: two are commitments, one is a
build, and the grid is small enough to read.

**Commitments, not swept.** `imitation_floor` 0, so the anchor fades entirely.
`segment_steps` 96 with `gamma` 0.997, so credit spans a typical 81-step life.

**Stage 0: the release, 2 trials.** Before anything else, run the current best
configuration with the floor at 0 and `imitation_target` at 0.8 and 0.95. This
is the cheapest and most informative experiment available, because it answers
the question every other result depends on: what does this policy do when it is
free? If it holds parity, the anchor was never load-bearing and the rest of the
sweep is a search for gains. If it regresses, the reward is the whole problem
and stage 2 becomes the only work worth doing. Either answer redirects the
effort, which is what a first experiment should do.

**Stage 1: the screen, 12 trials.**

    imitation_target in {0.8, 0.95}
    arch             in {conv, dilated, residual}
    entropy_coef     in {0.0, 0.002}

with `lr` at 3e-4, plus 6 more trials repeating the best three at `lr` 1e-3,
since the learning rate is table stakes and cheap to bolt on rather than cross
fully. Roughly 18 trials at about 20 minutes each with the longer segment: six
hours serially.

**Stage 2: the lineage reward, after building it.** Not a continuation of the
grid, because it changes the objective rather than a hyperparameter. Build the
biomass-plus-offspring reward, test it, then run `offspring_credit` in
{0, 0.5, 1.0} at the best stage-1 configuration. Three trials, and the zero
control is the one that says whether the lineage term does anything.

**Stage 3: confirmation, five seeds each.** The best two configurations overall.
Not optional, and the reason is arithmetic: 18 trials against a 16% noise floor
will produce an apparent winner by chance alone.

**One control run** of the committed segment and gamma against the old 32 / 0.99
pair, at the best stage-1 corner, so the commitment in axis 3 is checked rather
than assumed.

## The thing I would not sweep

`survival_reward`, but for a different reason now. Under the current reward it
is 1.0 so that the summed training reward is literally predator-steps survived,
matching what the evaluation reports. Under the lineage reward of axis 4 it goes
to zero, because biomass replaces it rather than complements it. What must not
happen is a grid that varies it against the other terms: that is the
hand-weighting the lineage reward exists to abolish, and running both is how a
sweep produces a number nobody can interpret.

## The honest caveat about all of this

The metric's noise floor is 16% across evaluation seeds, and most effects
proposed here are plausibly a few percent. **Most of this screen will return
noise, and the correct reading of a 1.1x trial next to a 1.0x trial is that they
are the same trial.** The screen ranks; only stage 3 decides.

Two things escape that caveat, and they are the reason to run this at all.
Stage 0 does not measure a small effect: releasing the anchor either holds
parity or it does not, and the difference will be far larger than 16%. And axis
4 is not a tuning change but a different objective, which is the only kind of
change with a mechanism for a large gain rather than a marginal one.

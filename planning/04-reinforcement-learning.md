# Reinforcement learning on tensor beasts

Design record for the per-individual reinforcement learning setup. Written so
the reasoning survives, including the parts that were wrong.

## The experiment

> Can a learned policy beat the rule-based policy at herbivore survival, on
> `conf/basic_config.yaml`, reproducibly, from one command?

This is a hard and honest target. The rule-based policy is not a strawman: it
is a tuned controller that produces a homeostatic three-species ecology with
boom and bust cycles that persist over thousands of steps.

## What the agent is

**Every living herbivore is its own agent. They all share one set of policy
weights. Each gets its own reward and its own episode, from birth to death.**

This was the central design question and it deserves the space. The
alternative, which the first environment implemented, is a single controller
that receives the whole world and emits one direction per cell, rewarded by
total population. That framing is a single agent with a 16,384-dimensional
action and one scalar reward. It is the least learnable arrangement available.

Three facts made the per-individual framing the obvious choice:

1. **The observation is already per-individual and local.** `Observation`
   carries, for each cell, the four directional neighbour readings of each
   perceived scent plus that individual's own energy, biomass and gradient
   history. Nobody sees the global map.
2. **The rule-based policy is already a shared per-cell function.** It is
   translation-equivariant and, in substance, linear: navigation weights dotted
   against local scent gradients. It is not a global controller either.
3. **It costs nothing.** A shared policy over a spatial observation is one
   convolutional forward pass over the grid, exactly as before. Only the reward
   and trajectory bookkeeping change, and those are precisely the parts that
   determine whether learning works.

The sample throughput is the payoff. At 512 square with several thousand
herbivores alive, every simulation step yields several thousand transitions.

### What this framing does not claim

The agents are not independent. They compete for the same plants and they are
each other's scent field, so every agent's environment shifts as the shared
policy improves. Parameter sharing handles this in practice, but this
non-stationarity is inherent to an ecology and is the most likely source of
training instability. Expect it rather than being surprised by it.

## Identity: why the `id` feature could not be used

Following an individual through time needs identity. The simulation has an `id`
feature, it is carried to the destination cell on every move, and offspring get
a fresh one, so it looks like the answer. It is not: offspring draw their id
from the world's uint8 `random` field, so **only 256 distinct ids exist** across
thousands of animals. Collisions are constant.

Instead `Animal` optionally records a `TransitionInfo` each step: which
individuals acted, the cell each one ended up in, and which reproduced. The
successor is derived from the chosen direction and whether the move actually
succeeded, which the movement code already knows.

Measured over 55,000 agent-steps at 256 square:

| Property | Result |
|---|---|
| Two individuals sharing a successor cell | 0 |
| Successors landing on an occupied cell | 99.98% |

The 0.02% are individuals that died mid-step, which is correct and is handled
by the `done` flag. The mapping is one-to-one in practice, not just in intent.

Tracking is off by default, is stored as an entity attribute rather than in the
world TensorDict, and world state hashes are identical with it on and off. It
is observation, not interference.

## Reward

Survival plus reproduction, per individual:

- `survival_reward` for each step the individual is still alive afterwards.
- `reproduction_reward` when it divides.
- The individual's episode ends when it dies.

The metric being approximated is total herbivore-steps survived, which is what
`evaluate_policy.py` reports for the baseline. Survival reward tracks that
directly. Reproduction reward credits an individual for the future population
it creates, which survival reward alone would attribute entirely to the
offspring. The ratio between the two is the main thing to tune: too low and
reproduction is under-valued, too high and the policy will trade its own life
for a division.

## Advantage estimation has to follow the agent

This is the part most likely to be silently wrong, so it is worth stating
plainly. Generalized advantage estimation assumes you can follow an agent from
one timestep to the next. Here agents move. A grid-shaped recursion that
bootstraps each cell from the same cell at the next step would credit an
individual with the future of whichever animal happened to walk into the cell
it just left.

Carrying the successor map fixes it, at one gather per timestep:

    delta_t[p] = r_t[p] + gamma * V_{t+1}[succ_t[p]] - V_t[p]
    A_t[p]     = delta_t[p] + gamma * lambda * A_{t+1}[succ_t[p]]

The tests hand-compute GAE along a three-step path and park decoy values in the
cells the agent vacates, so a cell-indexed implementation fails them rather
than merely looking plausible.

## Networks

Four fully convolutional actor-critics, all producing one action distribution
and one value per cell.

| Name | Parameters | Receptive field | Role |
|---|---|---|---|
| `linear` | 90 | 1 | Diagnostic, not a contender |
| `conv` | 82k | 7 | Default workhorse |
| `residual` | 305k | 19 | For when the small net saturates |
| `dilated` | 69k | 31 | Wide view, cheap |

`linear` is the important one to run first. The rule-based policy is a linear
function of exactly these channels, so a 1x1 convolution can represent it
exactly. If training cannot bring the linear policy near the baseline, the
problem is the learning setup rather than model capacity, and nothing deeper is
worth running. A negative result there is more valuable than a plausible
training curve from a big network.

`dilated` is the one worth betting on. Scent is a diffused field, so the
informative structure is a smooth gradient over many cells rather than fine
local detail, and dilation buys range cheaply.

### A bug the tests caught

The residual network originally used `GroupNorm`. Group norm pools over height
and width, so perturbing one corner of the map shifts activations at the
opposite corner. That breaks the property the entire framing rests on: an
individual may condition only on what it can perceive. Batch norm would be
worse, mixing statistics across timesteps whose population is booming or
busting.

The replacement, `ChannelNorm`, normalizes across channels at each cell
independently. The test that caught it perturbs one corner and asserts the
opposite corner does not move; it is worth keeping for anything added later.

## The size trap

**The ecology is strongly size-dependent, and small worlds are not just faster,
they are qualitatively different.** Herbivore and predator counts on
`conf/basic_config.yaml`:

| World | Predators at step 240 | Behaviour |
|---|---|---|
| 128 x 128 | 0 | Extinct by step 240; herbivores decline |
| 256 x 256 | 0 | Extinct by step 240; herbivores recover |
| 512 x 512 | 63 | Dips and recovers to 719 by step 600 |

At 512 the three-species dynamic works and stays homeostatic over 3000 steps.
Below 256 it degenerates into a two-species world.

This invalidated my own earlier measurements. The first rule-based baseline I
reported was taken at 128, where predators are extinct for most of the run.
Any result meant to mean something must be measured at 512. Small sizes are for
iteration speed only, and every test that uses one says so in a comment.

## What actually happened when we trained

### The baseline, measured at 512

Five episodes of 600 steps on `conf/basic_config.yaml`, return in
herbivore-steps survived:

| Policy | Return | Spread | Final population |
|---|---|---|---|
| Rule-based | 1,082,723 | 6% | 4,634 |
| Random | 309,589 | 16% | 1,475 |
| Stay put | 60,539 | 3% | 0 by step 202 |

At the native size the baseline is 3.5x better than random and the spread is 6%
of the mean, so the comparison is clean. At 128 the same comparison was 1.8x
with a 35% spread, which is why the earlier measurement was worthless.

### The pipeline is sound: a hand-set linear policy reaches 0.92x

Setting the linear policy's weights by hand, to approximate what the rule-based
controller computes, scores 0.92x of the baseline through the environment's own
evaluation. That is the single most useful number here. It says the
observation carries enough information, the action plumbing works, the reward
bookkeeping works, and the evaluation is fair. Anything that fails from here
fails at optimization, not at setup.

### Value targets needed normalizing, but not for the reason I first claimed

Measured on a real segment, the value term was 100% of the gradient norm: 14.02
against the policy's 0.027, with `max_grad_norm` at 0.5. I concluded the policy
was being starved by clipping and that fixing it would give a 28x larger policy
step.

**That overstated it.** Gradient clipping rescales every gradient by the same
scalar, and Adam largely absorbs a uniform rescaling. The matched ablation, same
seed, same learning rate, tells the real story:

| Step | Entropy off | Entropy on | Explained variance off | on | Grad norm off | on |
|---|---|---|---|---|---|---|
| 640 | 1.6092 | 1.6090 | +0.002 | +0.011 | 19.7 | 0.89 |
| 1280 | 1.6087 | 1.6079 | +0.003 | +0.024 | 19.0 | 0.46 |

The real benefit is critic conditioning: explained variance improves about
eightfold. The policy moves only marginally faster. Normalization is still
clearly worth keeping, and gradients now sit below the clip threshold instead of
28 times above it, but it did not unfreeze the policy, because the policy was
never frozen by clipping in the first place.

### The actual blocker: learning happens, and it makes things worse

At the default learning rate of 3e-4, approximate KL sits near 5e-06 and the
policy barely changes over thousands of steps. Raise it to 1e-2 and the policy
does move, and the result is worse than doing nothing:

| Step | Entropy | Explained variance | Population |
|---|---|---|---|
| 64 | 1.6057 | +0.001 | 110 |
| 768 | 1.5032 | +0.056 | 1,469 |
| 2176 | 1.2212 | +0.144 | 614 |
| 3584 | 1.2079 | +0.214 | 18 |

The critic improves steadily while the policy becomes more deterministic and
the population collapses. This is the finding that matters, and it is a
statement about the problem rather than about the code.

The likely cause is the reward's own structure. Herbivores survive about 99.4%
of steps, so per-step reward is nearly constant at 1.0 with a standard deviation
of 0.076. Almost all of an individual's return is fixed no matter what it does,
and the part that depends on its actions is buried in that. A policy can reduce
immediate risk in ways that are fatal over a longer horizon, and with a critic
that explains only 20% of return variance there is nothing to catch it.

### The first sweep, and what it taught about this machine

Twelve trials at 256 with three workers ran for six hours, completed five, and
then the parent process was killed and its workers were left orphaned at full
CPU. Two lessons.

**The memory budget has to be against memory that is free, not memory that
exists.** The guard sized three workers against half of 17 GB of physical RAM.
The machine already had other things resident, the sweep pushed it 6 GB into
swap, per-trial throughput fell from about 8 world-steps per second to between
0.06 and 0.4, and the parent was eventually killed. Budgeting against available
memory with a larger safety factor now says one worker fits on this machine as
it stands, which is the honest answer: a parallel sweep at 256 needs the small
minibatch and the cheaper architectures, or a bigger box.

**The headline ratio was dividing the wrong thing.** Two salvaged trials
reported ratios of -0.72 and -0.47, which is impossible for a ratio of survival
counts. Adding the foraging reward had made the training reward signed, and the
evaluation's ratio was built on total reward. It is now built on herbivore-steps
survived, with a test that pins it. What is optimized may change; what is
judged must not.

None of the five completed trials beat the baseline. Their survival ratios,
computed after the fix, are not worth tabulating: the runs were starved and the
budget of 4000 world steps was never reached by three of them.

### Anchoring to the rules, and what it uncovered about the rules

The repo owner's suggestion: use the rule-based policy as the reward structure
first and cross-fade to the real rewards as conformance rises. Implemented as a
supervised imitation term in the loss rather than as a reward, since the rule
action is free at every cell and a direct cross-entropy avoids the noisy
advantage path. Weight = `imitation_coef * max(0, 1 - conformance / target)`.

The first run stalled at 0.31 conformance, barely above the 0.20 chance level,
and finding out why took four diagnostics, each of which eliminated a
hypothesis:

| Check | Result | Eliminated |
|---|---|---|
| Rule action sampled twice from one observation | 99.8% self-agreement | "the label is noise" |
| Imitation with every reward zeroed | still 0.34 | "the RL gradient fights it" |
| Plain supervised fit, no PPO | linear 0.46, conv 0.66 | "the training loop is broken" |
| Analytic rule scoring over the env's channels | 1.000 agreement | "the observation lacks what the rule uses" |

The first real bug was mine: `get_observation` log-compresses scent as
`log1p(x * 50)`, so those channels top out near 9.4, and the environment divided
them by 255 as if they were bytes. Every perceived channel the learner saw lived
in roughly [0, 0.04]. Fixed by scaling to the actual ceiling.

That alone did not rescue the linear fit, and the reason is the finding worth
keeping. **The rule's decisions are knife-edge.** Scent is a smooth field, so
the difference between a cell and its neighbour, which is all the rule compares,
is two orders of magnitude smaller than the values. Measured on a settled world:

| Quantity | Value |
|---|---|
| Median relative margin, best vs second-best direction | 0.18% |
| Decisions settled by under 1% | 93% |
| Agreement after perturbing one navigation weight by 1% | 0.949 |
| by 5% | 0.744 |
| by 20% | 0.665 |

So the baseline's behaviour is dominated by hair-thin comparisons, an
approximate model tops out near 0.9 agreement however it is trained, and a
learned policy that drifts a few percent in one weight ratio changes a quarter
of its decisions. That is a large part of why "learning makes it worse" was so
easy to reach.

Two consequences went into the code. The observation now carries explicit
gradient channels, neighbour minus own cell, scaled by 25 so their standard
deviation is about 0.25 and clipped at 4; a small conv then fits the rule to
0.91 held-out, against 0.66 before. And the default conformance target is 0.8,
below the ceiling, so the anchor can actually let go.

Before the fix the anchored run reached 0.58x of the baseline at step 1664,
already up from 0.49x at the start. The post-fix run is the next number.

### Distilling the scoring regime, not the outcome

The repo owner's refinement: the rule works by scoring each action and taking
the best, so match a softmax over the learner's logits to a softmax over the
rule's scores rather than matching the argmax. Given the knife-edge finding
this is the right target. A near-tie becomes a near-uniform distribution that
costs nothing to disagree with, so no gradient is spent on the rule's
coin-flips, while confident decisions still get sharp targets.

The environment now emits the rule's five scores alongside its action, and
PPO's imitation term is the KL from `softmax(scores / temperature)` to the
policy. The temperature decides what counts as confident; measured on a settled
world the absolute score gaps have a median of 0.0076, so at the default of
0.01 a typical decision is a 60% preference and a clear one is sharp.
Conformance for the cross-fade is now the Bhattacharyya coefficient between
the two distributions: 1 exactly when they match, including where the rule
itself is unsure, 0 when they share no support. Argmax agreement is still logged, as a
separate diagnostic, because it is the number the knife-edge analysis was done
in.

The Q-learning connection is real and deliberately not built yet. The rule's
scores are a heuristic action-value function, so the same target that shapes
the policy logits here would regress a Q-head directly, with Bellman updates
refining it from there. In the actor-critic the logits are the natural home
for that prior; a value-based learner with the rule as its initial Q is the
obvious follow-on once the actor-critic path has a result.

### First result that beats the baseline

The hard-anchored run on the fixed observation (`conv`, 256 world, imitation
weight 1.0 cross-fading to zero at 0.8 conformance, foraging reward 0.1, two
epochs, minibatch 4) reached 1.075x the rule-based policy at step 1664 on
paired seeds at 256. Because the network is fully convolutional, the same
checkpoint can be scored at 512 without retraining. Three paired seeds, 400
steps each:

| Policy | Survived agent-steps | Reproductions | Mean population | Episode length |
|---|---|---|---|---|
| Learned | 537,030 | 4,175 | 1,353 | 85.2 |
| Rule-based | 403,426 | 2,575 | 1,015 | 88.3 |

**1.331x on herbivore-steps survived, at the size where the ecology is valid.**

How it wins is more interesting than that it wins. Individual lifespans are
marginally shorter under the learned policy, 85 steps against 88, but it
reproduces 62% more, so the population it sustains is a third larger. It beat
the rules on the metric the goal names by trading a little longevity for a lot
of offspring, which is exactly what a survival-plus-reproduction reward should
select for.

Caveats, in order of weight. The evaluation window is 400 steps against boom
and bust cycles of roughly 3000, so this is a measurement of the early cycle
and a longer window is the next confirmation. Three seeds. The policy was
trained at 256, where predators go extinct, and evaluated at 512, where they
do not; that it transferred is encouraging but a policy trained at 512 may do
better still. And this used hard argmax imitation; the soft distillation
committed alongside it has not yet been run head to head.

### Soft versus hard anchoring, head to head

Same recipe, same seeds, differing only in the imitation target. Argmax
agreement with the rules over training, and the paired evaluation at step 1664
on the 256 world:

| Step | Hard imitation | Soft distillation |
|---|---|---|
| 448 | 0.40 (conformance) | 0.65 |
| 832 | 0.75 | 0.86 |
| 1664 evaluation | **1.075x** | 0.988x |
| 3200 (final agreement) | 0.76 | 0.78 |

Soft distillation learns the rules faster and better, 0.86 agreement by step
832 where hard imitation plateaued near 0.76, which confirms the reasoning:
matching how the rule decides is a cleaner target than matching what it
decided. It did not translate into a better policy on this pair of seeds.

The training log shows the reason, and it is a problem with the cross-fade
rather than with either target. Once the anchor released, reinforcement
learning pulled the policy steadily away from the rules: agreement fell from
0.87 to 0.78, conformance from 0.77 to 0.40, and the anchor re-engaged to 0.43
and oscillated for the rest of the run. The hard-anchored run drifted less,
sitting near its target with the anchor barely engaged. A gate that releases
on conformance alone releases into a gradient that immediately lowers
conformance. Options, untested: a floor on the anchor weight rather than zero,
a release keyed on the *evaluation ratio* rather than on agreement, or simply
a slower fade.

### The result holds over a long window

Same winning checkpoint, same three paired seeds at 512, three times the
window, plus the soft-distillation checkpoint under the original window for an
apples-to-apples comparison:

| Checkpoint | Window | Learned survived | Rule-based survived | Learned lifespan | Rule-based lifespan | Ratio |
|---|---|---|---|---|---|---|
| Hard anchor | 400 | 537,030 | 403,426 | 85.2 | 88.3 | 1.331x |
| Soft anchor | 400 | 430,150 | 403,426 | 82.3 | 88.3 | 1.066x |
| Hard anchor | 1200 | 4,645,223 | 4,022,149 | 136.9 | 123.6 | **1.155x** |

Two corrections to the earlier reading. The advantage narrows as the ecology
settles, from 1.33x to 1.16x, so a full 3000-step cycle would likely narrow it
further; it is a real but modest edge, not a rout. And the mechanism changes
with the window: over 400 steps the learned policy won by reproducing 62%
more with slightly shorter lives, but over 1200 it wins by living longer, 137
steps against 124, and sustaining a 15% larger population, with reproductions
only 5% higher. The early-window story was a window artefact, which is exactly
why the longer evaluation was worth running.

Soft distillation learned the rules better and produced the weaker policy,
1.066x against 1.331x under identical conditions. Adding a floor of 0.2 to
the cross-fade, so the anchor never fully releases, kept argmax agreement
higher through training, 0.84 against 0.78, and recovered some of the gap at
512: 1.138x. The ordering at 512 under identical seeds and window is therefore
hard 1.331x, floored soft 1.138x, soft 1.066x, all above the rules. At 256 on
two seeds the three sat at 1.075x, 0.996x and 0.988x. Matching the rules more
faithfully has not converted into survival; the plausible reading is that the
winning policy's edge comes from departing from the rules in particular ways
and a target that keeps pulling toward the rule's exact scoring works against
those departures. This line is closed for now. What would actually settle it
is variance: the winning recipe on several seeds.

### Taking stock

The stated goal is met, with caveats that are recorded rather than hidden:
three paired seeds, a policy trained at 256 and evaluated at 512, and an
advantage that narrows as the ecology settles. Before it is called solid the
same recipe should run on a handful of seeds, about half an hour each at 256,
to separate the recipe from the draw.

The learned policy controls one of the two levers the rules control. The
external action overrides only the movement direction; the metabolic rate,
the move probability and the gradient history that sets the rate all still
come from the rule-based policy. The rule burns biomass at a basal rate plus a
term in the smoothed scent-gradient history, capped by carried biomass, with
efficiency falling as the rate rises. So the learned herbivore chooses where to
go while its throttle is set by a rule reacting to gradients it may have chosen
to ignore. It cannot conserve on purpose, sprint from a predator it sees, or
trade efficiency for speed.

The proposed next goal follows from that: **a learned policy that controls
both movement and metabolic rate beats the rule-based policy.** A second
output head over a few metabolic levels, the action override extended to carry
it, and the anchor extended to the rule's metabolic choice so the learner
starts from a working throttle. Evaluation stays herbivore-steps survived
against the unchanged rules. Learning the predator as well is the more
exciting experiment and belongs after a two-lever herbivore exists to be
hunted.

### Two levers, first run: parity, and a reward artefact

Same recipe as the winner plus four metabolic levels, paired evaluation at
step 1664 on the 256 world: **1.008x**, against 1.075x and 1.041x for
direction-only on two seeds. The direction head imitated the rules better than
any direction-only run, 0.88 agreement, and the metabolic head imitated them at
0.95, which is the problem: its mean chosen level was 0.086 on a scale of 0 to
3 with entropy 0.14, nearly deterministic on the coldest setting, from step
576 onward.

The learner is not being timid; it is being rational under the reward I gave
it. The foraging term is net biomass change per step, and burning biomass into
energy is exactly what the metabolic lever does, so every unit burned costs
reward directly. Energy never appears in the reward at all, and survival is
judged on biomass, so under this signal hoarding is optimal and the throttle's
only correct setting is off. The anchor agrees, since the rule sits near basal
most of the time. The lever was neutered by the signal, not by the network.

The fix is to reward what foraging actually is, biomass *eaten*, rather than
net change. Then burning is neutral to the foraging term and pays off only
through what energy buys, movement, which is the trade the lever exists to
make.

### Two levers at 512, and the corrected run

The first two-lever checkpoint, the one whose throttle collapsed to basal and
which evaluated at parity on the 256 world, was scored at 512 under the same
three seeds and 400-step window as every other checkpoint:

| Checkpoint | Survived | Reproductions | Population | Lifespan | Ratio |
|---|---|---|---|---|---|
| Direction-only winner | 537,030 | 4,175 | 1,353 | 85.2 | 1.331x |
| Two levers, throttle at basal | 560,502 | 4,613 | 1,413 | 80.8 | **1.389x** |
| Rule-based | 403,426 | 2,575 | 1,015 | 88.3 | 1.000x |

The best 512 number so far comes from a policy whose throttle sits at basal
almost always, against a rule whose mean rate is 2.16 on a basal of 2. So
"run cold" is not the artefact it looked like at 256; at the valid size it is
a better policy than the rule's, and the rule-based metabolism appears to burn
more than it needs. That is an ecological finding as much as a learning one,
and it should be checked the boring way, by evaluating the direction-only
winner with the throttle pinned to basal.

The corrected run, eaten-based reward and the throttle anchor off, evaluated
at 0.940x at step 1664 on the 256 world. Freed of the anchor the metabolic
head stayed near uniform, entropy 1.05 to 1.39 against a maximum of 1.39, and
drifted slowly cold, mean level 1.54 down to 0.64 by the end. Nothing in the
reward taught it to run hot, because nothing in the reward values energy
except through movement it did not learn to need. That run has not been
scored at 512.

Seed variance of the direction-only recipe at 256, step 1664: 1.075x, 1.041x,
0.969x, 0.963x. The 256
edge is small and partly the draw; the 512 results are the ones to trust.

### The claim, with variance: five training seeds at 512

Every checkpoint of the winning recipe, one per training seed, scored at 512
on the same three paired evaluation seeds over 400 steps:

| Training seed | At 256, step 1664 | At 512 | Survived agent-steps |
|---|---|---|---|
| 0 | 1.075x | 1.331x | 537,030 |
| 1 | 1.041x | 1.183x | 477,162 |
| 2 | 0.969x | 1.110x | 447,902 |
| 3 | 0.963x | 1.261x | 508,834 |
| 4 | not evaluated | 1.124x | 453,519 |
| Rule-based | 1.000x | 1.000x | 403,426 |

Mean 1.20x, range 1.11x to 1.33x, five of five above parity. Three of those
checkpoints looked like losses on the 256 world. The 256 evaluation was
measuring noise around parity: a world where the predators are extinct is too
small and too degenerate to show the edge, so the size trap documented for the
ecology applies to evaluation as well as training.

**The claim is therefore: the anchored recipe beats the rule-based policy by
11 to 33 percent on herbivore-steps survived at 512, across five training
seeds.** Training-seed variance is real and the spread is the honest number,
not the best draw. The remaining caveats are the ones already recorded: a
400-step window against long cycles, though the seed-0 checkpoint held 1.155x
over 1200, and policies trained at 256 rather than 512.

## Design: per-individual memory

The repo owner's suggestion: give each individual a small vector it can read
at one step and write for the next, a channel for communicating with its own
future, and groundwork for communication between individuals later. Recorded
here before any code, so the design can be argued with.

### The simulation already has a one-dimensional version

`gradient_ema` is a scalar every animal writes each step, carries with it when
it moves, halves into its offspring on reproduction, and reads back next step.
The policy's `Action` returns the new value and `Animal.update` writes it into
the feature. Memory is that, generalized: a `memory` feature of `K` float32
channels per cell, off when `K` is 0 so the golden hashes cannot move.

Carried on movement exactly like `gradient_ema`, with each channel passed as a
carried-feature slice. On reproduction the offspring receives a copy of the
parent's memory. That is a choice, and a deliberate one: inheritance means a
lineage can carry state across generations, which is the seed of the
coordination idea. The alternative, offspring start blank, is one line to
switch and worth testing as an ablation.

### What the policy sees and writes

The `K` memory channels join the observation as they are; the write head
bounds them with `tanh`, so no scaling is needed. The network gains a memory
head, `(B, K, H, W)` in `[-1, 1]`, and `forward_all` returns it beside the
action logits. The environment writes it back through the same override path
as the metabolic rate, so the viewer drives all three levers from one
checkpoint. A memory channel can be visualised as a colour layer, which is
worth doing early: the fastest way to learn what the agents chose to remember
is to look.

### The learning rule is the hard part, and it comes in two stages

**Stage 1, no recurrence.** The write is a deterministic function of the
current observation and is treated like any other output; no gradient crosses
the step boundary. The agent can still learn to *read* memory, but it cannot
learn what to *write* except by accident: the write head is a fixed projection
of the current observation that the next step happens to find useful or not.
This is cheap, it lands with the plumbing, and it is the control the real
thing has to beat.

**Stage 2, backpropagation through the individual.** For the write to be
learned, the gradient at a step's decision must flow back through the memory it
read to the earlier step that wrote it, and it must follow the individual, not
the cell, because the individual moved in between. That is exactly what
`compute_gae` already does for advantages: a gather through the successor map
at every step of a backward recursion. The recurrent training does the same in
the forward direction. During the update the segment is replayed in time order
rather than as shuffled minibatches: the network computes each step's memory
write from the stored observation, that write is gathered through the
successor map into the memory slots of the next step's observation, and the
loss at every step is summed before the backward pass. Truncated at the
segment boundary, detached there, as ordinary truncated backpropagation through
time.

Three consequences to design around. Memory read at a step must come from the
network's own recomputed write, not from the stored observation, or there is
no path for the gradient; the stored observation supplies everything except
those `K` channels. Individuals born inside the segment start from the copied
parent memory, which is itself a recomputed write, so inheritance is inside the
gradient path too. And PPO's ratio still needs the behaviour policy's
log-probability from collection time, which is stored as now; the recurrent
recomputation only changes how the current policy's log-probability is
produced. Memory costs one extra forward pass per step of the segment during
the update, which is the same cost as one epoch, so the budget is unchanged at
one epoch and doubled at two.

### Stage 1 has landed

The `memory` feature, its carry through movement and copy into offspring, the
observation channels, the write head, the action override, the trainer flag
`--memory-size`, and the viewer, which sets the feature width from the
checkpoint before building its world. Default off; with it off the simulation
is bit-for-bit identical to before over 300 steps at 128, checked against a
worktree of the previous commit.

Two things the implementation turned up. Movement carries features with
`safe_add`, whose uint8 wraparound guard rewrites any cell that went *down*
after an add to 255. No carried float had ever been negative, so it never
fired, but memory lives in [-1, 1]; the guard is now integer-only. And a
memory write has to be masked to living cells, because movement *adds* an
arriving animal's carried features onto its destination, so a stale value left
on an empty cell would be summed into whoever moved there next. The test that
caught it compares what an individual wrote with what its new cell holds.

### Stage 2 has landed: backpropagation through the individual

`--recurrent-window N` replays each segment in time order, feeds every step's
recomputed memory write into the next step's read through the successor map,
and backpropagates through windows of N steps, detaching at window
boundaries. The routing function is tested against the simulation's own carry
and matches it at every cell over real steps; with unchanged weights the
replayed reads reproduce the stored ones up to float16 rounding, and the first
recurrent epoch has unit PPO ratio, so the recomputation sees exactly what the
behaviour policy saw.

The test that justifies the machinery: a synthetic task where the target at
step t is the argmax of five channels the individual observed at step t-1,
carried through a random move. Only a policy that writes what it saw and reads
it back can match it. After the same 120 updates, stage 1 sits near chance
and recurrent training reaches over 0.7 agreement. The learner can learn what
to write.

Cost: one extra forward per step of the segment per epoch and the activations
of a window held at once, so the window is the memory lever in the same way
the minibatch was for the plain update.

### Stage 1 control result

Four memory channels with the fixed, non-recurrent write, same recipe as the
winner, evaluated at 0.942x at step 1664 on the 256 world, inside the band the
recipe's own seeds span there, 0.96x to 1.08x. No benefit and no harm, which
is what a fixed random projection of the observation should give. This is the
number the recurrent run has to beat.

### Stage 2 result at 256: parity with the control

Same recipe, four memory channels, recurrent window 8: 0.945x at step 1664
against the fixed-write control's 0.942x. The write head trained, mean write
magnitude 0.73 and the highest rule agreement of any run at 0.88, so the
machinery works; it did not turn into survival at this budget on this world.
Since the 256 evaluation is noise around parity for every policy, both memory
checkpoints are being scored at 512 before any conclusion is drawn. The
honest expectation is modest: a task the memory could help with has not been
identified, and a memory that helps only shows up where remembering pays.

### Correction: the memory numbers above were measured without memory

The evaluator dropped the memory write. Training wrote and carried memory
correctly, but the scoring loop called the environment without it, so every
memory checkpoint's evaluation, 0.942x and 0.945x above, was of a policy whose
memory channels read zero throughout. Those numbers say nothing about memory
either way. The fix is in, with a test that scores a memory policy and checks
living cells hold non-zero memory afterwards, and both checkpoints are being
re-scored at 512.

### The throttle finding, tested directly

The direction-only winner was scored at 512 twice on each of three seeds: as
trained, with the rule setting its throttle, and with its throttle pinned to
the basal rate.

| Seed | Rule-based | Direction-only | Pinned to basal |
|---|---|---|---|
| 0 | 560,162 | 869,149 (1.55x) | 1,112,931 (1.99x) |
| 1 | 502,892 | 968,322 (1.93x) | 1,211,045 (2.41x) |
| 2 | 593,987 | 914,787 (1.54x) | 1,050,141 (1.77x) |
| Total | | 1.66x | **2.04x** |

Survived agent-steps over 400 steps. The absolute ratios are not comparable to
the standard evaluation's 1.331x, because this quick script counts the
baseline's acting cells rather than its survivors; the within-table comparison
is what matters, and it is clean: pinning the throttle to basal beats the
rule's throttle on every seed, by about 23 percent in total. **The rule-based
metabolism burns more than it needs.** That is the mechanism behind the cold
two-lever checkpoint's 1.389x, and it is a statement about the ecology's
hand-written rules rather than about learning. An apples-to-apples number
through the standard evaluation is the next measurement.

### Memory at 512, measured with memory actually written

| Checkpoint | Survived agent-steps | Ratio |
|---|---|---|
| Fixed write (stage 1) | 494,259 | 1.225x |
| Recurrent (stage 2) | 393,597 | **0.976x** |
| Direction-only, same seed, no memory | 537,030 | 1.331x |

The fixed write sits inside the five-seed band of the plain recipe, which is
what a random projection of the observation should do. The recurrent policy is
the only checkpoint in the recipe family to fall below parity at 512. The
write head trained, the agreement with the rules was the highest of any run,
and the result is worse. The plausible reading is that the policy learned to
rely on a memory whose content is not worth relying on: the learning signal
for *what* to write is the same weak survival signal everything else has,
and a memory trained on it is noise the policy has been taught to trust.
Memory is not a win at this budget. What it would take is a task where
remembering demonstrably pays, and evidence in a memory channel that a person
can read; neither exists yet.

### The throttle finding, apples to apples

Through the standard paired evaluation at 512, the direction-only winner with
its throttle pinned to the basal rate:

| | Survived | Reproductions | Population | Lifespan | Ratio |
|---|---|---|---|---|---|
| Direction-only, rule's throttle | 537,030 | 4,175 | 1,353 | 85.2 | 1.331x |
| Direction-only, throttle pinned to basal | 663,881 | 5,526 | 1,674 | 92.4 | **1.646x** |
| Rule-based | 403,426 | 2,575 | 1,015 | 88.3 | 1.000x |

Pinning the throttle adds 24 percent on top of learned movement and is worth
more than everything the policy learned about where to go. Herbivores live
longer, sustain a two-thirds larger population and reproduce twice as often
as under the rules. The rule-based metabolism burns more than it needs, and
the cold two-lever checkpoint's 1.389x was this effect arriving through the
anchor rather than through learning. Whether the rule-based policy *itself*
improves with its throttle pinned is the next measurement; if it does, this
is a finding about the ecology's tuning rather than about learning.

### The throttle finding decomposes, and it is about the rules

The rule-based policy with its own throttle pinned to basal, scored against
the rule-based policy as is, three seeds at 512: 1.171x, 1.335x, 1.256x,
1.251x in total. The advantage was available to the hand-written rules all
along. Assembling the four measurements:

| Movement | Throttle | Ratio |
|---|---|---|
| Rules | Rules | 1.000x |
| Rules | Pinned to basal | 1.251x |
| Learned | Rules | 1.331x |
| Learned | Pinned to basal | 1.646x |

The two contributions are close to independent and multiplicative: 1.331 times
1.251 is 1.665 against 1.646 measured. Learned movement is worth about a
third; not burning above basal is worth about a quarter; and only the first is
a learning result. The rule's metabolism, basal 2 plus 2.5 times the smoothed
gradient, runs the herbivores hotter than pays off in this ecology. That is a
tuning observation for `conf/basic_config.yaml`, not a claim about learning,
and it is the repo owner's call whether the rules should change: a cooler rule
is a stronger baseline, and every ratio above would shrink against it.

### Correction: the rules were not running hot, the arithmetic was truncating

Cooling the rule's metabolic sensitivity from 2.5 to 0.5 changed herbivore
survival by 1.5 percent; the last step to exactly zero changed it by 21. A
cliff at an exact value is not "too hot". The cause is that energy conversion
truncated to uint8: at exactly the basal rate an animal burned 2 biomass for
2 x 3.5 = 7 energy, while at a rate of 2.05 it burned the same 2 biomass for
int(6.975) = 6. Every setting except one exact value paid a 14 percent tax
for nothing, and the learned throttle "chose cold" because it found the one
tax-free point. The over-burn story above is wrong and is left in place so
the correction can be read against it.

Two remedies, measured rule against rule at 512 on three seeds:

| Change | Herbivore agent-steps | Mean predators |
|---|---|---|
| None | 1,247,564 (1.000x) | 132 |
| Sensitivity 0, pin at basal | 1,510,900 (1.211x) | 142 |
| Round energy instead of truncating | 1,435,007 (1.150x) | 142 |

Rounding is the change made. It is a simulation fix rather than a tuning, it
keeps the sprint response and makes the throttle a real trade-off again, and
the 5 percent it leaves behind is the genuine cost of burning above basal.
Every ratio recorded above this point was measured against the truncating
rules and should be re-measured before being quoted against the new ones.
The golden baseline moves for every config with animals, in its own commit.

### Rounding is not enough: energy goes float

The repo owner's objection to the rounding fix was right: energy is a uint8
grid, so rounding removes the cliff at basal and keeps the staircase. A burn
of 2.0 to 2.5 biomass yields exactly 7, 3.0 yields 10, nothing between exists,
and the amount burned truncates from the other side, so a rate of 2.5 costs 2.
The throttle cannot be a smooth trade-off on integers.

Whether the integers buy anything is measurable. At 512x512 the per-step
accounting ops that touch energy and biomass, metabolism, dissipation, eating,
death and reproduction checks:

| Representation | Accounting ops per step | Share of a full step | Grid memory |
|---|---|---|---|
| uint8 | 2.50 ms | 1.5% | 0.5 MB |
| float32 | 1.18 ms | 0.7% | 2.1 MB |

uint8 is *slower*, because the saturating add-and-fix-up helpers it needs cost
more than its bandwidth saves, and the memory difference is a rounding error
beside the float16 scent field. The integers cost the staircase, the
truncation bug, fourteen saturating-helper call sites, twenty explicit casts
and a workaround for a PyTorch bug comparing uint8 against values above 255,
and give nothing back.

Decision: energy and biomass become float32 on the same 0..255 scale, so every
config, threshold and renderer stays valid, with every truncation, rounding
and saturation hack removed. The rounding commit above is therefore an interim
step and is superseded. Every ratio in this document was measured against the
integer simulation; the first predator run was started against the rounded
integer simulation and will be rerun against float before it is quoted.

### Float landed, and it made life more expensive

Energy, biomass and carrion are float32 on the 0..255 scale; every saturating
helper is deleted and every site clamps explicitly. Faster, as measured: 13.5
against 11.8 steps per second at 256, 6.8 against 5.8 at 512.

It shifts the ecology, and deliberately was not tuned back. uint8 had been
silently forgiving part of every burn: `min(rate, biomass).to(uint8)` charged
`floor(rate)`, so with basal 2 and sensitivity 2.5 animals running at ~2.2 paid
2. Instrumented, that was 0.18 biomass per step per herbivore, 8.1% of its
burn, and 0.20 per predator. Float charges what the config says. Rule against
rule at 512 over 400 steps: herbivore agent-steps 1,240,152 against 1,435,007
for the rounded integer simulation, mean predators 95 against 140. One seed to
800 steps shows the same predator-prey cycle with a deeper predator trough, 11
against roughly 50, and recovery about 200 steps later. Both species persist.
If the old balance is wanted back, fractional `basal_rate` values such as 1.8
are now meaningful; that is a tuning decision for the owner.

**Every ratio in this document was measured against an integer simulation and
is stale.** The findings about the method, the anchor, seed variance and the
size trap stand; the numbers need re-measuring before they are quoted.

### Memory: closed for now, on a clean negative

At its own training size the recurrent checkpoint scores 1.065x against 1.247x
for the fixed-write control, so its failure at 512 was not transfer. The
write is learnable, the synthetic test proved that; trained on the survival
signal at this budget, what it learns to write is noise the policy then
trusts, and the policy is worse for it everywhere. Nothing here says memory
cannot help. It says that a memory needs a task where remembering
demonstrably pays before its training signal is anything but noise, and that
task has not been posed. The plumbing and the recurrent update stay, tested,
for when it is.

### What would count as "big if true"

A learned memory has to beat the same recipe with `K` set to 0 on the same
seeds, at 512, over the long window. The stage-1 control has to be run as well,
because if the fixed random write already helps, the credit belongs to the
extra channels rather than to anything learned. And something should be visible:
a memory channel that tracks time since eating, or distance travelled from a
predator, would be a result a person can look at and believe.

## The predator experiment

The goal, stated by the repo owner: **herbivores on the rule-based policy,
predators learned, and the learned predator beats the rule-based predator.**
This is the experiment the herbivore work was scaffolding for, and it is a
different problem in ways worth stating before any run.

### Re-measuring the baseline, because everything above it is stale

Every ratio recorded before this section was measured against the integer
simulation and marked stale when energy and biomass went float32. The predator
baseline had never been measured against float at all. Rule against rule at
512, herbivores and predators both on their own rules, three paired evaluation
seeds over 400 steps, counting predator agent-steps survived:

| Seed | Predator agent-steps | Reproductions | Mean predators | Minimum | Final |
|---|---|---|---|---|---|
| 10000 | 34,456 | 267 | 87.9 | 9 | 39 |
| 10001 | 37,401 | 302 | 95.3 | 20 | 67 |
| 10002 | 31,448 | 201 | 80.1 | 4 | 5 |
| Total | **103,305** | 770 | 87.8 | | |

**This is the number a learned predator has to beat: 103,305 predator
agent-steps over three seeds.**

The trajectory matters more than the total. Predators fall from roughly 200 at
reset to between 6 and 30 by step 300 while herbivores crash to about 530 and
then boom past 2,000; predators recover only in the last hundred steps, and on
seed 10002 they recover to 5. That is a textbook predator-prey cycle rather
than a bug, but it has two consequences for the experiment. A 400-step window
samples one trough, so the variance between seeds is the cycle's phase as much
as the policy. And a learned predator that hunts harder early can drive its own
prey down and starve, which means a policy can lose by being better at hunting.
A longer window is the honest measurement, and the cheap early one should be
read as a screen rather than a result.

### The predator's reward is sparse where the herbivore's is dense

The winning herbivore recipe used a foraging reward of 0.1 on biomass eaten.
Copying that number to the predator would be a mistake, and the reason is
measurable. At 256, rule-based, over 60 steps after settling:

| | Agent-steps | Ate on | Mean bite | Survived per step |
|---|---|---|---|---|
| Herbivore | 8,939 | 78.3% | 3.0 | 98.76% |
| Predator | 4,784 | **2.5%** | **29.6** | 97.97% |

A herbivore grazes almost every step in small mouthfuls; a predator makes a
kill on one step in forty and eats ten times as much when it does. The same
coefficient multiplies a signal with an entirely different shape, and the
foraging term stops being a dense gradient and becomes a rare spike.

Matching the *variance* rather than the coefficient transfers the recipe
honestly. Reward per acting individual, same measurement:

| Policy | Foraging coefficient | Mean | Std | Coefficient of variation |
|---|---|---|---|---|
| Herbivore | 0.1 | 1.227 | 0.522 | 0.426 |
| Predator | 0.1 | 1.072 | 0.659 | 0.615 |
| Predator | **0.02** | 1.013 | 0.432 | **0.427** |

So the predator runs use a foraging reward of 0.02, chosen because it puts the
reward's dispersion where the herbivore recipe's was, not because it looked
reasonable. Whether variance is the right thing to match is itself a guess, and
it is written down here so a later run that sweeps the coefficient has
something to disagree with.

### Why predators need supervised pretraining

Recorded in the commit that added `--pretrain-updates`: twice, a learned
predator took its population at 512 from 486 to zero inside 200 steps, before
the imitation anchor could pull a near-random policy toward anything that
hunts. Herbivores survive the same near-random start only because there are
thousands of them and the plants do not run away. A few hundred predators that
wander at random all starve together, and a dead population produces no
gradient at all. Pretraining on rule-based rollouts is therefore not an
optimization for the predator, it is what makes the run possible.

## What to try next, in order

1. **A denser, more action-dependent reward.** Energy gained by eating is the
   obvious candidate: it responds immediately to moving well, whereas survival
   barely responds at all. Train on that and keep *evaluating* on
   herbivore-steps survived, which is legitimate and standard.
2. **Sweep the reproduction-to-survival ratio.** Reproduction is rare, around
   0.7% of agent-steps, and is the only part of the current reward with real
   variance. `sweep_rl.py` varies it from 0 to 30.
3. **Fix the critic before blaming the policy.** Explained variance peaks around
   0.2. Until the critic can predict return, every advantage is mostly noise and
   no policy-gradient method will do better than drift.
4. **Longer horizons.** Episode lengths run to a hundred-odd steps while gamma
   is 0.99. Try 0.997 and a longer GAE lambda.

## Open questions

- **Reward ratio.** Survival to reproduction is currently 1 to 10, chosen by
  reasoning rather than measurement. It should be swept.
- **Credit for offspring.** An individual is currently rewarded for dividing but
  inherits nothing from how its offspring fare. A discounted share of offspring
  return would be closer to fitness, at the cost of a much harder assignment
  problem.
- **Genetic variation.** The slot-based genome system is deliberately left out
  of the reinforcement learning work. Mixing gradient learning with evolving
  per-slot parameter variation at the same time makes any improvement
  impossible to attribute. A hybrid is interesting once the gradient path works.
- **Predator policies.** Only herbivores are learned. Training both at once is
  a genuinely adversarial, non-stationary problem and should wait.

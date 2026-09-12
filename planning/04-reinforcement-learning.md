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

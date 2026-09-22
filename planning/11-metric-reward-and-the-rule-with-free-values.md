# The metric, the reward, and the rule with free values

Written 17 September 2026, at the end of a week that removed the ghosts
from the ecology (planning/09), the anchor from RL (planning/10), and the
transient from the evaluation. This is the plan for the next phase and the
record of why. A fresh session should be able to proceed from this file and
the code as committed at `069614e`.

## Where things stand

**The ecology is sound.** Dead animals no longer leave energy behind or
persist when eaten to zero; the biomass ledger balances to zero over 300
steps; under the rules alone, four seeds at 512 for 6,000 steps cycle
between roughly 500 and 1,500 predators with no extinction.

**The trainer no longer stops on extinction.** An extinct world is reset in
place from a fresh warmed world; the run spends its budget and reports
`world_resets`. Training starts after 1,200 rule steps, in the settled
cycles. Evaluation warms its worlds for 600 rule steps before scoring.

**Pretraining is offline distillation** (`tensor_beasts/rl/distill.py`):
labelled grids sampled from a rule-based run, the eight symmetries of the
grid, a fit to a plateau on a held-out split. The conv network reaches 0.93
agreement; the linear one starts as the rule analytically.

**Nothing pulls the policy toward the rules during RL.** Distance from
them is logged and inert. The entropy bonus is zero. Reward modes and their
coefficients still exist and are the next thing to go.

## What the controls showed

Three runs, all from the pretrained policy, all on the settled ecology.

**Frozen conv policy, no RL.** Zero world resets in 5,760 steps and 119,496
smoothed biomass against the rules' 66,373 on the same warmed worlds. The
initialisation is already a better predator than the rule.

**Every RL run collapses.** The overnight grid (`ypprc6q9`, 24 runs over
release speed, learning rate and the throttle) reset worlds 17 to 48 times
per run and ended at 3 to 8 percent of the frozen score. Agreement with the
rule held at 0.9 exactly as long as the anchor had weight and drifted the
moment it did not; reward per agent-step never rose above the frozen
policy's level.

**Gamma 0.** With the reward reduced to "did this move land on prey", 54
updates did not raise it, while agreement drifted from 0.94 to 0.43.

**The linear rule copy.** With the actor a single 1x1 convolution set to the
rule's weights, RL moved the policy almost nothing by every measure it has:
KL near 1e-4 per update, clip fraction zero, three percent of argmax
decisions, a fifth of the entropy. On the same evaluation worlds:

| | Step 32 | Step 2,016 |
|---|---|---|
| Biomass eaten per individual per step | 2.6 | 2.5 |
| Lifespan | 82.3 | 81.3 |
| Biomass carried per individual | 107 | 106 |
| Reproductions per individual | 9.1 | 8.9 |
| Population | 640 | 488 |
| Total biomass | 60,452 | 34,066 |

Every quantity an individual experiences is unchanged. The world sustains a
quarter fewer of them and half the biomass.

**The inference.** A reward that is any function of an individual's own
outcomes, eaten biomass, net stock change, survival, reproductions, is
identical before and after this collapse, because those outcomes are
identical. Its gradient, built from differences in those outcomes between
individuals under the current policy, cannot prefer the first policy to the
second. The changed decisions act through the shared prey field, and the
only quantities that register it are counts: how many predators a patch of
world sustains. This also rules out a lineage-based return as the fix, since
reproductions per individual were unchanged and so were expected
descendants.

## The metric

For a species with survival threshold θ and biomass field b_t(c), stock
B_t = Σ_c b_t(c). With D a bank of warmed start states, π the policy and T
the window,

    M_T(π) = E_{s0 ~ D} [ (1/T) Σ_{t=1..T} B_t ]   under π from s0,

extinction absorbing (B_t = 0 after the population reaches zero, which the
simulation guarantees). T at least 4,000 steps, longer than a collapse.
Report beside it the extinction fraction, the rules' M_T on the same starts,
and the spread over starts. Persistence and growth are one quantity at long
horizons; every earlier tension between them was an artefact of windows
shorter than a collapse.

"Interesting emergent behaviour" is diagnosed alongside, never optimised:
throttle spread and its correlation with local conditions, carrion use in
busts against booms, cycle amplitude and period, spatial clustering.

## The reward, when one is needed

Per individual i acting from c_t(i) to c_{t+1}(i):

    r_t(i) = eat_t(i) − burn_t(i) − loss_t(i)

with loss the reserve that becomes carrion when b_{t+1}(i) < θ, and
division neutral (the halves sum to the whole; the newborn's own eating is
its own r). Then Σ_i r_t(i) = B_{t+1} − B_t exactly, every step. It is the
metric's own increment and has no coefficients.

That reward is still per-individual and so still blind to the collapse
above. The coupling to the population is spatial pooling: with ρ_t the
field holding each individual's r at its successor cell and K_R a kernel
of radius R,

    R_t(i) = Σ_c K_R(c − c_{t+1}(i)) ρ_t(c).

R = 0 is the individual reward; R covering the world is B_{t+1} − B_t for
everyone, correct and hopeless for credit assignment; a few cells pays each
individual for its neighbourhood's stock, which scales with how many
neighbours there are and so registers what the controls saw. Radius is the
one knob, held fixed, and needing it is recorded as a finding about the
ecology. Implementation: a `burned` field on `TransitionInfo`; the loss
where `alive_after` is false; a scatter to successors, one convolution, a
gather back. Snags: newborns eat in their birth step and nobody is paid for
it; herbivore bites are losses the prey did not choose and are not
recoverable per individual yet, so predator-only.

## The actor: the rule with free values

The general linear head has five weights per channel and sits at weights
near a thousand to reproduce a soft target at temperature 0.01, which is why
Adam at 3e-4 froze it. The rule has one weight per perceived feature, shared
across the five directions. The actor for this phase is that:

    score_a(x) = Σ_k w_k · x_{k,a}          (k over perceived features)
    logits_a   = β · score_a
    rate       = basal + s · gradient_ema, clamped as the rule clamps it

About three direction weights, a throttle sensitivity s, and a sharpness β,
all initialised at the rule's own values. Symmetric by construction, so no
augmentation; readable, so drift is a sentence; at natural scale, so the
learning rate means what it says; and β makes "how stochastic should a
shared policy be" a number the policy finds rather than an entropy bonus we
set. Depth in the critic is separate from depth in the actor: the linear
critic reached explained variance 0.43 against conv's 0.8, and a conv critic
does not change what the policy can express.

The ladder, each rung compared against the one below on the same worlds,
with the frozen rule copy the bar throughout: the rule with free values,
then the general linear head, then conv.

## The order of work

1. **The bank and the metric.** At run start, a batched rule-based run of
   eight worlds for 3,000 steps, snapshots every 100 steps after the first
   thousand, held in RAM (about 35 MB each at 512); training starts and
   resets draw from it, evaluation uses a fixed seeded subset of it. M_T
   with T = 4,000, extinction fraction, the rules beside. Two evaluations
   per run plus the final. Delete the evaluation warmup, which the bank
   replaces. A checkpoint written at the end of pretraining, so any drifted
   policy can be compared with its own start on the same worlds.
2. **The rule-parametrised actor**, initialised from the config, with the
   critic architecture chosen separately. This is also the harness for the
   next step, since the policy is a parameter vector.
3. **Direct search on the metric.** Five parameters need no gradient: a
   coordinate sweep or a small evolution strategy, one evaluation per
   parameter setting at about a minute each, a few hundred evaluations. No
   reward, no critic, no credit assignment. The result is the best rule the
   ecology admits and the standard every reward-based learner on the same
   actor must reach.
4. **The reward, validated against that standard.** The stock reward with
   pooling at radius 0 and at a few cells, PPO on the rule-parametrised
   actor, from the same starts. Radius 0 should reproduce the linear
   control; the pooled arm either lands where the search did or the
   reward is wrong. Delete the reward modes and their coefficients with
   this.
5. **Depth**, one rung at a time, only once a reward has been validated.

## One diagnostic that can run before any of it

Today's collapse came with entropy falling under a zero bonus; the
overnight collapses came with entropy rising under a positive one. Whether
a shared policy's stochasticity is itself load-bearing is testable without
training: pretrain the linear network with a checkpoint saved, then evaluate
it sampled and with `--eval-deterministic` on the same worlds. If the argmax
version scores near the rules' 66,000, sharpening is not the harm and the
three percent of changed decisions are. If it scores like the drifted
34,000, stochasticity matters and β is a first-order parameter. Needs the
post-pretraining checkpoint from step 1.

## Guiding principle

A thorough re-evaluation of inherited assumptions returns to the
parsimonious option unless a strong justification says otherwise. Findings
measured on the ghost ecology, through an anchor that never released, or on
the transient window are void, not merely stale.

## What was built (17 September, evening)

Steps 1 to 4 of the order of work are code; nothing has been run at 512
beyond timing checks. The commands are at the end.

**The bank** (`tensor_beasts/rl/bank.py`). One batched rule-based run of
`bank_worlds` worlds for `bank_steps` steps, every per-world leaf of the
world's TensorDict cloned to the CPU every `bank_stride` steps after
`bank_warmup`: 8, 3,000, 100, 1,000 by default, so 160 states from steps
1,000 to 2,900. A state at 512 is 23 MB, and its labelled grid another 23.
Training
worlds, extinction resets, the film world and the evaluation worlds all
start from bank states; evaluation from a fixed seeded subset of
`eval_seeds` of them, loaded afresh for each policy from a reseeded RNG, so
the comparison is paired and every evaluation of a run scores the same
starts. Both warm-ups are gone. Two things not in the plan: the bank run is
also the pretraining sampler, keeping the rule's labelled grid beside each
state, so `pretrain_grids` and `pretrain_stride` are gone too and
distillation fits on exactly the states training starts from; and the
world clock, which is shared across a batch and enters the simulation only
through the 20,000-step water oscillator, is set to the loaded state's step
when a whole batch is loaded and left alone when one slot is.

**The metric.** `score` is `learned_mean_biomass`, the stock carried by
living individuals averaged over `eval_steps` (4,000) from `eval_seeds` (8)
banked starts; `learned_extinct_fraction`, `rule_based_mean_biomass`,
`score_spread`, `score_min` and `score_max` sit beside it. The smoothed
headline and both ratios against the rules are gone. The rules are scored
once per process on the evaluation starts and that number is reused. The
stock counts living individuals only, so extinction is absorbing exactly
and a dead animal's reserve, which becomes carrion at the next step, is
not stock. `pretrained.pt` is written before the first update, whether or
not distillation ran.

**The reward** (`tensor_beasts/rl/multiagent.py`). `TransitionInfo.burned`
was added; the reward is `eaten - burned - loss` with `loss` the reserve at
the successor cell where `alive_after` is false, scattered to successors,
box-summed over `reward_radius` and gathered back. Verified on a real
predator world, batched and unbatched: the sum over acting individuals
plus what newborns ate equals the change in living stock to 1e-4 every
step, and no two individuals shared a successor. The four coefficients,
the reward modes and the offspring credit are deleted. The plan's snag
stands: newborns' eating in their birth step is paid to nobody, and the
herbivore's bites are unaccounted, so the identity is the predator's.

**The actor** (`RulePolicy` in `tensor_beasts/rl/networks.py`, `--arch
rule`). Weights per perceived feature, a log-parametrised sharpness
starting at one over the distillation temperature, and with `--metabolic` a
sensitivity, with the rule's biomass cap applied inside the head so the
throttle error reads zero at the start. Scores are in the rule's own units,
so the values are the config's. Agreement with the rule at the start is
0.95 for both species; the remainder is the rule's random tie-breaking and
its clamp of all-repulsive scores to zero, which the linear control never
had either. The critic is separate: `critic: conv` (default) or `linear`.
The values are logged every update as `rule_*` and printed as a sentence
under each evaluation, and the checkpoint carries them as `rule_values`.

**The search** (`tools/search_rule.py`). Coordinate sweep over the values,
each tried at a set of factors that are pulled toward 1 each round, one
evaluation per setting, every result appended to `search.jsonl` and
skipped on a rerun, `--report` to read it back. The tool scores the rules
once on the same starts.

**Measured on the 3090 at 512, before any run.** The bank builds at 38 ms
per step for eight worlds, so the default bank takes about two minutes and
7.4 GB of host memory, with a GPU peak of 1.7 GB. That two minutes is paid
once per bank, not once per process: a built bank is written to
`outputs/bank` under a key made from the simulation config's text, the
world arguments, the bank parameters and seed, the device type and a digest
of the source files the rule-based step runs through, and the next process
asking for the same bank reads the file. Editing one of those files costs a
rebuild; a stale bank is never read. `--no-bank-cache` builds in-process
and writes nothing. An evaluation of eight
starts over T = 4,000 is about 48 ms per step, so three minutes for the
learned policy and six for the first evaluation in a process, which scores
the rules too; the GPU peak is 3.2 GB. The plan's "about a minute" per
search evaluation was optimistic by three: a few hundred evaluations is a
night, or half that with four starts.

### Commands

The diagnostic, first, since it needs only a checkpoint: pretrain the
linear network and score its start sampled and at the argmax on the same
starts.

    venv/bin/python train_rl.py --entity Predator --size 512 --arch linear --metabolic \
        --pretrain-epochs 200 --steps 0 --worlds 4 --eval-seeds 8 --eval-steps 4000 \
        --out outputs/rl/linear-start --no-wandb
    venv/bin/python train_rl.py --eval-only outputs/rl/linear-start/pretrained.pt --size 512 \
        --eval-seeds 8 --eval-steps 4000 --out outputs/rl/linear-start
    venv/bin/python train_rl.py --eval-only outputs/rl/linear-start/pretrained.pt --size 512 \
        --eval-seeds 8 --eval-steps 4000 --out outputs/rl/linear-start --eval-deterministic

The architecture, entity and heads come from the checkpoint; the size, the
bank and the window do not, since the network is the same at any size.

The search, step 3, a few hundred evaluations at about three minutes each:

    venv/bin/python tools/search_rule.py --entity Predator --size 512 --device cuda \
        --eval-seeds 8 --eval-steps 4000 --rounds 3 --out outputs/search/predator

The reward, step 4, radius 0 against a few cells on the rule actor:

    venv/bin/python train_rl.py --entity Predator --size 512 --arch rule --metabolic \
        --worlds 4 --steps 5760 --segment-steps 32 --minibatch-steps 2 --gamma 0.997 \
        --epochs 2 --target-kl 0.02 --extinction-patience 1 \
        --eval-seeds 8 --eval-steps 4000 --eval-interval 2880 \
        --reward-radius 0 --out outputs/rl/rule-r0
    venv/bin/python train_rl.py ... --reward-radius 3 --out outputs/rl/rule-r3

Radius 0 should reproduce the linear control's near-immobility on the rule
actor; the pooled arm either lands where the search did or the reward is
wrong.

## The diagnostic, and the bug it exposed (21 September)

The linear network was pretrained for 200 epochs and its start scored on
eight banked starts over T = 4,000, sampled and at the argmax, in two
eval-only processes on the 3090:

| | Sampled | Argmax | Rules |
|---|---|---|---|
| M_T | 80,514 | 86,540 | 85,063 |
| Extinct fraction | 0.00 | 0.00 | 0.00 |
| Population | 746.6 | 797.8 | 786.8 |
| Reproductions | 37,615 | 39,795 | 37,661 |
| Lifespan | 78.3 | 79.3 | 82.5 |
| Spread across starts (std) | 19,220 | 19,076 | |

**The reading.** The argmax policy scores at the rules' level and the
sampled one five percent under it. Neither is anywhere near the collapse
the drifted linear policy showed (half the biomass, a quarter fewer
individuals), so a shared policy's stochasticity is not load-bearing and
the sharpness β is not a first-order parameter for persistence. The three
percent of changed decisions in the drifted policy are the harm. The
five-percent gap between sampled and argmax is inside the metric's noise:
with a standard deviation of 19,000 over eight starts the standard error of
the mean is about 6,800, eight percent of the score. The search accepts any
strictly higher score with no margin, and so will follow that noise unless
it evaluates on more starts or demands a margin of a couple of standard
errors.

**The bug.** The pretraining process's own evaluation scored the rules at
97,211 on the same eight starts. That gap was traced by reading, not by
running: the water feature drew its base pattern and phase map at
initialisation and kept them as plain attributes on the feature object,
and rewrote the water leaf from them every step. The bank restores
TensorDict leaves only, so a loaded world kept its banked water for one
step and then ran on the host environment's own landscape, drawn from
whatever the global random stream held when that environment was created.
Every bank load was affected: training starts, extinction resets, the film
world and evaluation. Within one process the learned and rule-based
policies shared a landscape, so the paired comparisons above hold; across
processes nothing was comparable, and no population was ever evaluated on
the terrain it grew on. The old warm-ups grew each population in place, so
nothing measured before the bank was affected.

The fix stores both as leaves under the terrain's prefix, drawn in the same
order, so the values are unchanged: the golden hash over the previous keys
is identical, and the baseline moves only by the two added keys. A test in
`tests/rl/test_bank.py` loads a state into a world with a different
landscape and checks that after one step the water is the banked run's
next water. The rule score at a small size is now the same whether the
training environment holds two worlds or eight, and the same from the
training loop and from `--eval-only`. The numbers in the table were scored
on a landscape that was not the bank's and should be re-scored, though the
reading does not depend on that.

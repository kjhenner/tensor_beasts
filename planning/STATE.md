# State of the work

Written 21 September 2026. This file describes what the code does and what
the last session chose to do next. Nothing in it is settled. Each choice is
recorded with its reason so that it can be reopened on its merits, and a
session that finds a better reason should take the other path and rewrite
this file. What was tried before, and which earlier results are void, is in
`HISTORY.md`. The full narrative record that preceded these two files is in
git at commit `1938f40`, under `planning/`.

## The question

Can a learned policy for one species keep that species going better than
the hand-written rule does? The species under study is the predator. The
herbivore work came first and is summarised in the history.

## What the code does now

**The simulation.** A dead animal leaves its biomass as carrion and nothing
else. A per-phase biomass ledger at 128x128 over 300 steps attributes every
unit to a named source or sink with zero residual. Under the rules alone,
four seeds at 512 for 6,000 steps cycle between roughly 500 and 1,500
predators with no extinction.

**The agents** (`tensor_beasts/rl/multiagent.py`). Every living animal of
the controlled species is an agent. All share one policy, each has its own
reward, and each episode runs from birth to death. The policy is a
convolution over the grid, so the number of agents costs nothing at runtime.

**The bank** (`tensor_beasts/rl/bank.py`). A run's first act is a batched
rule-based run of `--bank-worlds` worlds for `--bank-steps` steps, with a
snapshot every `--bank-stride` steps after `--bank-warmup`. The defaults are
8, 3,000, 100 and 1,000, so 160 states. Training worlds, the replacement for
an extinct world, the film world and the evaluation worlds all start from
bank states. Evaluation uses a fixed seeded subset of them, so every
evaluation in a run scores the same starts. The bank run also supplies the
labelled grids for distillation. A built bank is written to `outputs/bank`
under a key made from the config text, the world arguments, the bank
parameters and seed, the device type, and a digest of the source files the
rule-based step runs through. A stale bank is never read.

**The metric.** The score is the species' biomass carried by living
individuals, averaged over `--eval-steps` world steps from `--eval-seeds`
banked starts. Extinction is absorbing. The defaults are 4,000 and 8. It is
logged as `learned_mean_biomass`, with `learned_extinct_fraction`,
`rule_based_mean_biomass`, `score_spread`, `score_min` and `score_max`
beside it. The rules are scored once per process on the same starts.

**The reward.** Per individual, eaten minus burned minus the reserve lost at
death, then box-summed over `--reward-radius` cells around the individual's
successor cell. At radius 0 the sum over individuals equals the species'
stock change each step, except for what newborns eat in their birth step.
Herbivore bites are not attributed per individual, so the identity holds
for the predator only. There are no other reward terms and no coefficients.

**The actor.** `--arch rule` is the rule with free values: one weight per
perceived feature shared across the five directions, a log-parametrised
sharpness, and with `--metabolic` a throttle sensitivity. It is initialised
at the config's own values and agrees with the rule on 95 percent of argmax
decisions at the start. The remainder is the rule's random tie-breaking and
its clamp of all-repulsive scores. The critic is chosen separately with
`--critic conv` or `--critic linear`. The other actors are `linear`, `conv`,
`residual` and `dilated`. The rule values are logged every update as
`rule_*` and stored in the checkpoint as `rule_values`.

**Pretraining.** Offline distillation (`tensor_beasts/rl/distill.py`) of
labelled grids from the bank run, under the eight symmetries of the grid,
fitted until agreement on a held-out split plateaus. `--pretrain-epochs` is
a ceiling. The conv network reaches 0.93 agreement. The linear network is
set to the rule's weights analytically. `pretrained.pt` is written before
the first update.

**During RL** nothing pulls the policy toward the rules. Distance from them
is logged and inert. The entropy bonus defaults to zero. An extinct world is
replaced from the bank and `world_resets` is counted.

**The search** (`tools/search_rule.py`). A coordinate sweep over the rule's
values, one metric evaluation per setting, every result appended to
`search.jsonl` and skipped on a rerun. `--report` reads a log back.

**Wired but unused in this phase.** A per-individual memory trained
recurrently (`--memory-size`, `--recurrent-window`). Individual-following
films (`--film-interval`). The genetic slot system (`conf/genetic_simulation.yaml`,
`tools/run_genetic_sim.py`), off by default. All three have tests.

## Measured costs

On the 3090 at 512, measured 17 September:

| Operation | Per step | Total | Memory |
|---|---|---|---|
| Bank build, 8 worlds, 3,000 steps | 38 ms | about 2 min, once per bank | 7.4 GB host, 1.7 GB GPU peak |
| Evaluation, 8 starts, 4,000 steps | 48 ms | about 3 min | 3.2 GB GPU peak |
| First evaluation in a process | | about 6 min, scores the rules too | |

## The observation behind the current choices

Three controls on 17 September, all from the pretrained policy on the
settled ecology, measured on the previous smoothed-biomass metric:

- The frozen conv policy, with no RL, scored 119,496 against the rules'
  66,373 on the same warmed worlds, with zero world resets in 5,760 steps.
- Every RL run in the 24-run overnight grid collapsed, resetting worlds 17
  to 48 times and ending at 3 to 8 percent of the frozen score.
- With the actor a single 1x1 convolution set to the rule's weights, RL
  moved the policy almost nothing: KL near 1e-4 per update, three percent
  of argmax decisions changed. Every per-individual quantity was unchanged
  (biomass eaten per step 2.6 to 2.5, lifespan 82.3 to 81.3, reproductions
  per individual 9.1 to 8.9), while the population fell from 640 to 488 and
  total biomass from 60,452 to 34,066.

The last session's reading was that a reward built from an individual's own
outcomes cannot distinguish the two policies, because those outcomes are
identical, and that the change acts through the shared prey field and
registers only in counts. Spatial pooling of the reward is the coupling it
chose to test. This is one reading. It has not been tested, and the
diagnostic below does not confirm it.

## The last measurement, 21 September

The linear network was pretrained for 200 epochs and scored on eight banked
starts over 4,000 steps:

| | Sampled | Argmax | Rules |
|---|---|---|---|
| Score | 80,514 | 86,540 | 85,063 |
| Extinct fraction | 0.00 | 0.00 | 0.00 |
| Population | 746.6 | 797.8 | 786.8 |
| Std across starts | 19,220 | 19,076 | |

Argmax scores at the rules' level and sampled five percent under it. With a
standard deviation of 19,000 over eight starts the standard error of the
mean is about 6,800, eight percent of the score, so that five percent is
inside the noise. The search accepts any strictly higher score, so it will
follow that noise unless it evaluates on more starts or demands a margin.

These numbers were scored on a landscape that was not the bank's. The water
feature kept its base pattern and phase map outside the TensorDict, so a
loaded world ran on the host process's landscape from its second step.
Fixed at commit `ef676bd`. Within one process both policies shared a
landscape, so the paired comparison holds. The absolute numbers should be
re-scored.

## What the last session intended next

1. Re-score the diagnostic above on the corrected landscape.
2. Run the search over the rule's values. The result is the best rule the
   ecology admits and the standard any reward-based learner on the same
   actor must reach.
3. Run PPO on the rule actor with the stock reward at radius 0 and at a few
   cells, from the same starts. The expectation was that radius 0 moves the
   policy as little as the linear control did, and that the pooled arm
   either matches the search or shows the reward is wrong.
4. Only after a reward is validated, add depth: the general linear head,
   then conv.

The commands, as last used:

    venv/bin/python train_rl.py --eval-only outputs/rl/linear-start/pretrained.pt --size 512 \
        --eval-seeds 8 --eval-steps 4000 --out outputs/rl/linear-start
    venv/bin/python train_rl.py --eval-only outputs/rl/linear-start/pretrained.pt --size 512 \
        --eval-seeds 8 --eval-steps 4000 --out outputs/rl/linear-start --eval-deterministic

    venv/bin/python tools/search_rule.py --entity Predator --size 512 --device cuda \
        --eval-seeds 8 --eval-steps 4000 --rounds 3 --out outputs/search/predator

    venv/bin/python train_rl.py --entity Predator --size 512 --arch rule --metabolic \
        --worlds 4 --steps 5760 --segment-steps 32 --minibatch-steps 2 --gamma 0.997 \
        --epochs 2 --target-kl 0.02 --extinction-patience 1 \
        --eval-seeds 8 --eval-steps 4000 --eval-interval 2880 \
        --reward-radius 0 --out outputs/rl/rule-r0

The architecture, entity and heads come from a checkpoint; the size, the
bank and the window do not.

## Open items

- Newborns' eating in their birth step is paid to nobody.
- The search accepts any improvement with no margin against the noise.
- `conf/toy_zoo/single_herbivore.yaml` does not load. It configures a
  nutrients block on a terrain class with no nutrients feature.
- Old W&B sweeps are in project `tensor-beasts-rl` on the local server.
  `tools/sweep_report.py` reads them. Their configs were deleted because none
  of them ran unchanged against the current flags.

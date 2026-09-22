# What was tried

One entry per phase, in order. The full documents are in git at commit
`1938f40` under `planning/`, numbered 00 to 11; each entry names its number.
Results marked **void** were measured under a condition later found to
invalidate them. They are listed so that no session re-quotes them.

**Stabilisation, 00.** Decorator-based registry for entities and features,
validated configs with `${key:...}` references, a golden hash over world
state (`tools/sim_bench.py`). Complete.

**Policy extraction and genetic slots, 01 and 02.** The animal's decision
was moved out of `Animal.update` into a policy object. A slot-based genome
system was built and is off by default. Kept out of the RL work by choice.

**Performance and the RL foundation, 03.** Padding and stride fixes in the
simulation core. The per-individual environment replaced an earlier
single-controller Gymnasium environment, since deleted. Two hypotheses
about throughput were wrong and are recorded there.

**Herbivore RL, 04.** A conv network pretrained on the rules beat the rules
at herbivore survival by 11 to 33 percent across five training seeds at 512.
**Void as numbers:** measured on the integer simulation before energy and
biomass became float32, and through an observation taken one update early.
The stale observation was fatal for the predator, which hunts food that
moves. Value normalisation, the size trap below 256, and the identity
problem with the uint8 `id` feature are recorded there and still apply.

**Review, 05.** Found the stale observation, that three evaluation seeds
cannot resolve the predator, that the golden baseline is device-dependent,
and 1,700 lines of dead RL code. The code was deleted on 21 September.

**Predator sweeps, 06 and 08.** Releasing the imitation anchor improved the
policy. Training across four worlds cut gradient noise. A 32-run grid over
reward mode, architecture, learning rate and anchor target found reward
composition was not the binding constraint, and that one collapsed cell
had confounded two axis marginals. **Void:** all of it ran on the ghost
ecology below, through an anchor that never actually released, on a metric
of the transient window.

**Batched worlds, 07.** The world gained a batch dimension. Four worlds in
one process ran 3.80 times faster than one, against 2.55 for four
processes. Verified behaviour-preserving by the golden hash.

**Metabolic sweep, 09.** The first sweep measured the throttle head's
initial spread being rectified by a clamp, not the throttle. Two simulation
faults were found: a dead animal kept its energy and went on moving and
blocking as a ghost for up to 200 steps, and prey eaten to exactly zero
never died. Both fixed; the golden hashes moved. **Every absolute score
before this fix is not comparable with anything after it.**

**Extinction, 10.** The paired metabolic sweep on the corrected ecology went
extinct in every run. An extinct world is now reset from a fresh one instead
of ending the run. The pretrained predator, frozen, carried 1.8 times the
rules' biomass. A 24-run grid over release speed, learning rate and the
throttle collapsed in every cell. The session then removed the RL-time
anchor in all forms, moved pretraining to offline distillation run to
convergence, set the entropy bonus to zero, and declared learning rate
3e-4 and conv to be defaults rather than findings.

**The metric, the reward and the rule with free values, 11.** The bank,
the biomass-stock metric, the stock reward with spatial pooling, the rule
actor and the coordinate search were built on 17 September. The linear
diagnostic on 21 September found argmax and sampled policies both at the
rules' level, and exposed the water-landscape bug fixed at `ef676bd`. The
current state is in `STATE.md`.

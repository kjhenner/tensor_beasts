"""Offline distillation of the rule-based policy: the learner's initialisation.

The rule's direction is a fixed score over the cell and its four neighbours'
scents, and its throttle is a fixed function of the gradient EMA and biomass.
Every input to both is a channel the network sees, so distilling the rule is
supervised learning on labelled observation grids, and the simulation is only
the sampler that produces them. It used to run in lockstep with the fit, one
segment per update and thrown away after two epochs, a leftover of the days
when imitation and RL shared a loop.

Three pieces:

* :func:`collect_labelled_grids` runs a batched world under the rules and
  keeps every ``stride``-th step's observation with the rule's own labels,
  across the cycle phases the run passes through.
* :func:`augment` applies one of the eight symmetries of the square grid to a
  grid and its labels. The rule is equivariant under them (flip the world
  top to bottom and it swaps up for down), so each stored grid is eight
  labelled grids. Directions are encoded as channels, so a spatial transform
  must permute the up/down/left/right channel groups and the labels with it;
  :func:`channel_permutation` builds that permutation from the channel names.
* :func:`distill` fits a network to the grids to a plateau on a held-out
  split, augmenting the training split.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from tensor_beasts.rl.ppo import NUM_ACTIONS, masked_mean, rule_distillation, throttle_distillation

# Action order the policy head uses, and the vector each direction moves by
# in (row, column).
ACTION_NAMES = ("stay", "up", "down", "left", "right")
DIRECTION_VECTORS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
SYMMETRIES = 8

# Stop once the mean training loss over the latest WINDOW epochs is within a
# fraction MIN_DELTA of the mean over the WINDOW before it. The training
# loss, on the augmented data, is the objective being optimised and falls
# steadily; agreement is a step function of it that barely moves in the first
# epochs, and the validation loss rises for a while under augmentation
# before it falls, so rules on either stopped the fit before it had started.
# Window means because a single epoch is a handful of minibatches.
WINDOW = 5
MIN_DELTA = 0.01


@dataclass
class LabelledGrids:
    """Observation grids with the rule's labels, on the CPU.

    ``observation`` is (N, C, H, W) float16 as the network input is stored
    everywhere else; ``acted`` (N, H, W) bool; ``rule_action`` (N, H, W)
    int64; ``rule_scores`` (N, 5, H, W) float16 or None; ``rule_unit``
    (N, H, W) float32 or None (present when the entity has a throttle head).
    """

    observation: torch.Tensor
    acted: torch.Tensor
    rule_action: torch.Tensor
    rule_scores: Optional[torch.Tensor]
    rule_unit: Optional[torch.Tensor]
    channel_names: List[str]

    def __len__(self) -> int:
        return int(self.observation.shape[0])

    def subset(self, index: torch.Tensor) -> "LabelledGrids":
        pick = lambda t: None if t is None else t[index]
        return LabelledGrids(
            pick(self.observation), pick(self.acted), pick(self.rule_action),
            pick(self.rule_scores), pick(self.rule_unit), self.channel_names,
        )


def collect_labelled_grids(env, grids: int, stride: int, warmup: int) -> LabelledGrids:
    """Run ``env`` under its rules and keep every ``stride``-th step's grids.

    ``env`` should already be reset. Each step of a batched world yields one
    grid per world, so ``grids`` is reached after ``grids / worlds`` samples
    spaced ``stride`` steps apart, following ``warmup`` steps that are not
    sampled. Everything lands on the CPU as it is taken, as float16 where the
    network input is float16 anywhere else, so the sampler's memory is one
    step's batch and not the dataset.
    """
    from tensor_beasts.rl.trainer import policy_input

    for _ in range(warmup):
        env.rule_based_step()
    fields: Dict[str, List[torch.Tensor]] = {k: [] for k in ("observation", "acted", "rule_action", "rule_scores", "rule_unit")}
    taken = 0
    step = 0
    while taken < grids:
        batch = env.rule_based_step()
        step += 1
        if step % stride:
            continue
        worlds = env.num_worlds
        def per_world(t):
            # (B, ...) stays; an unbatched (...) gains the world axis.
            return t if worlds > 1 else t.unsqueeze(0)
        fields["observation"].append(policy_input(per_world(batch.observation)).to("cpu", torch.float16))
        fields["acted"].append(per_world(batch.acted).to("cpu"))
        fields["rule_action"].append(per_world(batch.rule_action).to("cpu", torch.int64))
        if batch.rule_scores is not None:
            fields["rule_scores"].append(per_world(batch.rule_scores).to("cpu", torch.float16))
        if batch.rule_metabolic_unit is not None:
            fields["rule_unit"].append(per_world(batch.rule_metabolic_unit).to("cpu", torch.float32))
        taken += worlds
    cat = lambda name: torch.cat(fields[name])[:grids] if fields[name] else None
    return LabelledGrids(
        cat("observation"), cat("acted"), cat("rule_action"), cat("rule_scores"), cat("rule_unit"),
        list(env.channel_names),
    )


def direction_map(k: int) -> Dict[str, str]:
    """Where each direction goes under symmetry ``k`` of the square grid.

    ``k`` encodes transpose (4), a flip of the row axis (2) and a flip of the
    column axis (1), applied in that order to the grid.
    """
    out = {}
    for name, (dr, dc) in DIRECTION_VECTORS.items():
        if k & 4:
            dr, dc = dc, dr
        if k & 2:
            dr = -dr
        if k & 1:
            dc = -dc
        out[name] = next(n for n, v in DIRECTION_VECTORS.items() if v == (dr, dc))
    return out


def transform_field(t: torch.Tensor, k: int) -> torch.Tensor:
    """Apply symmetry ``k`` to the last two axes of ``t``."""
    if k & 4:
        t = t.transpose(-2, -1)
    if k & 2:
        t = t.flip(-2)
    if k & 1:
        t = t.flip(-1)
    return t


def channel_permutation(channel_names: Sequence[str], k: int) -> torch.Tensor:
    """Index ``perm`` such that ``transformed[c] = original[perm[c]]``.

    A channel named ``.../up`` or ``.../grad_up`` in the transformed grid
    holds what the original called the direction that maps onto up.
    """
    forward = direction_map(k)
    inverse = {new: old for old, new in forward.items()}
    index = {name: i for i, name in enumerate(channel_names)}
    perm = []
    for name in channel_names:
        head, _, tail = name.rpartition("/")
        prefix = "grad_" if tail.startswith("grad_") else ""
        direction = tail[len(prefix):]
        if direction in DIRECTION_VECTORS:
            perm.append(index[f"{head}/{prefix}{inverse[direction]}"])
        else:
            perm.append(index[name])
    # On the CPU explicitly: these index CPU-resident grids whatever the
    # default device is.
    return torch.tensor(perm, device="cpu")


def action_permutation(k: int) -> torch.Tensor:
    """``new_action = perm[old_action]`` under symmetry ``k``; stay is fixed."""
    forward = direction_map(k)
    return torch.tensor([
        ACTION_NAMES.index(forward[name]) if name in forward else i
        for i, name in enumerate(ACTION_NAMES)
    ], device="cpu")


def augment(grids: LabelledGrids, k: int) -> LabelledGrids:
    """The same grids under symmetry ``k``, labels included."""
    if k == 0:
        return grids
    perm = channel_permutation(grids.channel_names, k)
    act = action_permutation(k)
    # Scores are indexed by action, so they permute as the actions do:
    # transformed[a] = original[a'] where act[a'] = a.
    score_perm = torch.argsort(act)
    return LabelledGrids(
        observation=transform_field(grids.observation[:, perm], k),
        acted=transform_field(grids.acted, k),
        rule_action=act[transform_field(grids.rule_action, k)],
        rule_scores=None if grids.rule_scores is None else transform_field(grids.rule_scores[:, score_perm], k),
        rule_unit=None if grids.rule_unit is None else transform_field(grids.rule_unit, k),
        channel_names=grids.channel_names,
    )


def _losses(network, grids: LabelledGrids, temperature: float, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(loss, argmax_agreement, metabolic_error) on one minibatch of grids."""
    observation = grids.observation.to(device).float()
    mask = grids.acted.to(device)
    out = network.forward_all(observation)
    log_probs = F.log_softmax(out["logits"], dim=1)
    scores = None if grids.rule_scores is None else grids.rule_scores.to(device).float()
    loss, _, agreement = rule_distillation(log_probs, grids.rule_action.to(device), scores, temperature, mask)
    error = torch.full((), float("nan"), device=device)
    if "metabolic_mean" in out and grids.rule_unit is not None:
        throttle_loss, error = throttle_distillation(out["metabolic_mean"], grids.rule_unit.to(device), mask)
        loss = loss + throttle_loss
    return loss, agreement, error


@torch.no_grad()
def evaluate(network, grids: LabelledGrids, temperature: float, device, minibatch: int) -> Dict[str, float]:
    """Agreement and throttle error over ``grids``, weighted by acting cells."""
    totals = {"argmax_agreement": 0.0, "metabolic_error": 0.0, "loss": 0.0}
    weight = 0.0
    error_weight = 0.0
    for start in range(0, len(grids), minibatch):
        part = grids.subset(torch.arange(start, min(start + minibatch, len(grids)), device="cpu"))
        count = float(part.acted.sum())
        if count == 0:
            continue
        loss, agreement, error = _losses(network, part, temperature, device)
        totals["loss"] += float(loss) * count
        totals["argmax_agreement"] += float(agreement) * count
        if error == error:
            totals["metabolic_error"] += float(error) * count
            error_weight += count
        weight += count
    if weight == 0:
        return {k: float("nan") for k in totals}
    return {
        "loss": totals["loss"] / weight,
        "argmax_agreement": totals["argmax_agreement"] / weight,
        "metabolic_error": totals["metabolic_error"] / error_weight if error_weight else float("nan"),
    }


def distill(
    network,
    optimizer,
    grids: LabelledGrids,
    *,
    epochs: int,
    minibatch: int,
    temperature: float,
    max_grad_norm: float,
    device,
    validation_fraction: float = 0.125,
    symmetries: bool = True,
    on_epoch: Optional[Callable[[Dict[str, float]], None]] = None,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, float]:
    """Fit ``network`` to the rule on ``grids`` until validation agreement plateaus.

    ``epochs`` is a ceiling. Each epoch is one pass over the training split
    in minibatches of ``minibatch`` grids, each minibatch under a random
    symmetry when ``symmetries`` is on. The validation split is never
    augmented. Returns the last epoch's record, whose ``argmax_agreement`` and
    ``metabolic_error`` are the validation numbers: the honest ones. Without
    a validation split the training loss stands in for it.

    Augmentation makes the fit slower and honest: an unaugmented fit on one
    trajectory reaches high agreement partly by learning that trajectory's
    skew toward whichever directions its scent gradients happened to point,
    which the symmetries remove.
    """
    generator = generator or torch.Generator().manual_seed(0)
    n = len(grids)
    held_out = max(1, int(round(n * validation_fraction))) if n > 1 else 0
    order = torch.randperm(n, generator=generator, device="cpu")
    validation = grids.subset(order[:held_out]) if held_out else None
    training = grids.subset(order[held_out:])

    losses: List[float] = []
    record: Dict[str, float] = {}
    for epoch in range(epochs):
        network.train()
        order = torch.randperm(len(training), generator=generator, device="cpu")
        totals = {"loss": 0.0, "train_agreement": 0.0}
        weight = 0.0
        for start in range(0, len(training), minibatch):
            part = training.subset(order[start:start + minibatch])
            if symmetries:
                part = augment(part, int(torch.randint(SYMMETRIES, (1,), generator=generator, device="cpu")))
            count = float(part.acted.sum())
            if count == 0:
                continue
            loss, agreement, _ = _losses(network, part, temperature, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)
            optimizer.step()
            totals["loss"] += float(loss) * count
            totals["train_agreement"] += float(agreement) * count
            weight += count
        network.eval()
        scored = evaluate(network, validation, temperature, device, minibatch) if validation is not None else {}
        network.train()
        record = {
            "pretrain_epoch": float(epoch + 1),
            "loss": totals["loss"] / max(weight, 1.0),
            "train_agreement": totals["train_agreement"] / max(weight, 1.0),
            "argmax_agreement": scored.get("argmax_agreement", totals["train_agreement"] / max(weight, 1.0)),
            "metabolic_error": scored.get("metabolic_error", float("nan")),
            "validation_loss": scored.get("loss", float("nan")),
        }
        if on_epoch is not None:
            on_epoch(record)
        losses.append(record["loss"])
        if len(losses) >= 2 * WINDOW:
            latest = sum(losses[-WINDOW:]) / WINDOW
            previous = sum(losses[-2 * WINDOW:-WINDOW]) / WINDOW
            if latest > previous * (1.0 - MIN_DELTA):
                record["converged"] = 1.0
                break
    return record

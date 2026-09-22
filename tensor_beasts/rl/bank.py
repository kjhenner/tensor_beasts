"""A bank of warmed start states, and the labelled grids taken at them.

Every world a run touches starts from a state drawn from here: the training
worlds, the world that replaces an extinct one, and the evaluation worlds,
which use a fixed seeded subset so every evaluation in a run, and every run
with the same seed, scores on the same starts. It replaced two
warm-ups, one before training and one before each evaluation, that each ran
a fresh world through its startup transient under the rules.

The bank is one batched rule-based run of ``worlds`` worlds for ``steps``
steps, snapshotted every ``stride`` steps once ``warmup`` steps are past.
The run is also the sampler for pretraining: at each snapshot the rule's own
labelled grid, the observation and the rule's direction, scores and throttle
on it, is kept beside the state, so distillation fits on exactly the states
training starts from. At 512 a state is 23 MB and its grid another 23, so
the default bank of 160 states is about 7.4 GB of host memory and takes
about two minutes to build on a 3090. A built bank is written to disk under
a key made from everything that shapes it and read back by the next process
that asks for the same one; see the cache section at the end of this file.

A snapshot is every per-world leaf of the world's TensorDict, on the CPU.
Restoring one is a slice assignment into a batched world's leaves, which is
what lets one world of a batch be replaced without touching the others. The
world clock is shared across a batch and is set when a whole batch is loaded
and left alone when one slot is; it enters the simulation only through the
water oscillator, whose period is 20,000 steps.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import torch

from tensor_beasts.rl.distill import LabelledGrids


@dataclass
class StartBank:
    """``states[i]`` is one world's leaves on the CPU, taken at world step
    ``steps[i]``; ``grids`` holds the matching labelled grid at the same index."""

    states: List[Dict[tuple, torch.Tensor]]
    steps: List[int]
    grids: LabelledGrids
    seed: int

    def __len__(self) -> int:
        return len(self.states)

    @property
    def nbytes(self) -> int:
        total = sum(v.numel() * v.element_size() for state in self.states for v in state.values())
        for field in (self.grids.observation, self.grids.acted, self.grids.rule_action,
                      self.grids.rule_scores, self.grids.rule_unit):
            if field is not None:
                total += field.numel() * field.element_size()
        return total

    def draw(self, count: int, generator: torch.Generator) -> List[int]:
        """``count`` state indices, without replacement while the bank allows it."""
        if count <= len(self):
            return torch.randperm(len(self), generator=generator, device="cpu")[:count].tolist()
        return torch.randint(len(self), (count,), generator=generator, device="cpu").tolist()

    def load(self, env, indices: Sequence[int]) -> None:
        """Start every world of ``env`` from the named states, one per world.

        The world clock is set to the first state's step, so a batch loaded
        from the same indices always starts identically.
        """
        if len(indices) != env.num_worlds:
            raise ValueError(f"{len(indices)} states for an environment of {env.num_worlds} worlds")
        for world, index in enumerate(indices):
            self.load_into(env, index, world)
        env.world.step = int(self.steps[indices[0]])
        env.entity.last_transition = None

    def load_into(self, env, index: int, world: int = 0) -> None:
        """Restore state ``index`` into slot ``world`` of ``env``, leaving the
        other slots and the world clock alone.

        A leaf the target does not hold yet (some are first written during
        an update, so a fresh world lacks them) is created, so the loaded
        world is the banked one leaf for leaf whatever the target's age.
        """
        state = self.states[index]
        destination = env.world.td
        batched = bool(env.world.batch_shape)
        present = set(destination.keys(True, True))
        for key, value in state.items():
            if key not in present:
                shape = (env.num_worlds, *value.shape) if batched else tuple(value.shape)
                destination.set(key, torch.zeros(shape, dtype=value.dtype, device=env.device))
                present.add(key)
            target = destination.get(key)
            if batched:
                target[world].copy_(value)
            else:
                target.copy_(value)


def snapshot_world(env, world: int) -> Dict[tuple, torch.Tensor]:
    """Every per-world leaf of ``env``'s world ``world``, cloned to the CPU.

    Two leaves are left out: the water oscillator's value, a scalar shared
    across a batch and recomputed from the clock every step, and the
    per-step random field, which every update draws afresh before anything
    reads it.
    """
    td = env.world.td
    batched = bool(env.world.batch_shape)
    out: Dict[tuple, torch.Tensor] = {}
    for key in td.keys(True, True):
        value = td.get(key)
        if not isinstance(value, torch.Tensor) or value.dim() == 0 or key == "random":
            continue
        leaf = value[world] if batched else value
        out[key] = leaf.detach().to("cpu", copy=True)
    return out


def build_bank(
    make_env: Callable[[int], object],
    *,
    worlds: int,
    steps: int,
    warmup: int,
    stride: int,
    seed: int,
    verbose: bool = False,
) -> StartBank:
    """Run ``worlds`` worlds under the rules and bank their states.

    A state is taken after ``s`` rule steps for every ``s`` in
    ``range(warmup, steps, stride)``, from each world of the batch, together
    with the rule's labelled grid at that state. ``make_env(worlds)`` builds
    the environment; it is reset from ``seed`` and discarded afterwards. The
    global RNG is restored, so building the bank leaves the caller's stream
    where it was.
    """
    from tensor_beasts.rl.trainer import policy_input

    if stride <= 0:
        raise ValueError("bank stride must be positive")
    if warmup >= steps:
        raise ValueError(f"a bank of {steps} steps with a warmup of {warmup} holds no states")

    rng_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    env = make_env(worlds)
    env.reset(seed=seed)
    states: List[Dict[tuple, torch.Tensor]] = []
    taken_at: List[int] = []
    fields: Dict[str, List[torch.Tensor]] = {
        k: [] for k in ("observation", "acted", "rule_action", "rule_scores", "rule_unit")
    }
    snapshot_steps = set(range(warmup, steps, stride))
    for s in range(steps):
        take = s in snapshot_steps
        if take:
            for world in range(env.num_worlds):
                states.append(snapshot_world(env, world))
                taken_at.append(s)
        batch = env.rule_based_step()
        if take:
            def per_world(t):
                return t if env.num_worlds > 1 else t.unsqueeze(0)
            fields["observation"].append(policy_input(per_world(batch.observation)).to("cpu", torch.float16))
            fields["acted"].append(per_world(batch.acted).to("cpu"))
            fields["rule_action"].append(per_world(batch.rule_action).to("cpu", torch.int64))
            if batch.rule_scores is not None:
                fields["rule_scores"].append(per_world(batch.rule_scores).to("cpu", torch.float16))
            if batch.rule_metabolic_unit is not None:
                fields["rule_unit"].append(per_world(batch.rule_metabolic_unit).to("cpu", torch.float32))
        if verbose and (s + 1) % 500 == 0:
            print(f"bank: {s + 1}/{steps} rule steps, {len(states)} states", flush=True)

    def cat(name):
        return torch.cat(fields[name]) if fields[name] else None

    grids = LabelledGrids(
        cat("observation"), cat("acted"), cat("rule_action"), cat("rule_scores"), cat("rule_unit"),
        list(env.channel_names),
    )
    del env
    torch.set_rng_state(rng_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return StartBank(states=states, steps=taken_at, grids=grids, seed=seed)


# ----------------------------------------------------------------------
# The cache
# ----------------------------------------------------------------------
#
# A bank is a deterministic function of its inputs, and building one at 512
# takes about two minutes on a 3090, so it is written to disk under a key
# made from every input and read back by the next process that asks for the
# same one. The inputs are the simulation config's text, the environment
# arguments that shape the world (size, entity, throttle, memory), the bank
# parameters and seed, the device type, and a digest of the source files
# the rule-based step and the labelled grid run through. A change to any of
# them makes a new key, so a stale bank is never read; the cost of editing
# one of those files is one rebuild, not a wrong result. The reward radius
# is left out: it enters only the reward computed after a step, never the
# state or the grid.
#
# The device type is in the key because a rule-based run on the CPU and on
# CUDA differ in the last bits, so their banks differ. Two CUDA devices
# share a key; a run cares that its starts are warmed, settled states, not
# that they match another card's to the bit.

BANK_FORMAT = 1

# Inside tensor_beasts/rl, only these take part in producing a bank: the
# environment's rule-based step, the observation round-trip, the grid type
# and this module. Every other file in the package is outside rl/ and is
# digested whole.
_RL_MODULES_IN_THE_BANK = ("bank.py", "distill.py", "multiagent.py", "trainer.py")


def source_digest() -> str:
    """A digest of the package source the bank's contents depend on."""
    import hashlib
    from pathlib import Path

    package = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package)
        if relative.parts[0] == "rl" and not (
            len(relative.parts) == 2 and relative.parts[1] in _RL_MODULES_IN_THE_BANK
        ):
            continue
        digest.update(str(relative).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def bank_key(parts: Dict[str, object]) -> str:
    """The cache key for a bank built from ``parts``, a stable digest of
    their JSON. ``config_path`` is replaced by the file's text, so the key
    follows the config's contents and not its name."""
    import hashlib
    import json
    from pathlib import Path

    parts = dict(parts)
    parts["format"] = BANK_FORMAT
    parts["source"] = source_digest()
    if "config_path" in parts:
        parts["config_text"] = Path(parts.pop("config_path")).read_text()
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def save_bank(bank: StartBank, path) -> None:
    """Write ``bank`` to ``path`` atomically: a partial file from an
    interrupted write, or from another process writing the same bank at the
    same time, is never left under the final name."""
    import os
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grids = bank.grids
    payload = {
        "format": BANK_FORMAT,
        "seed": int(bank.seed),
        "steps": [int(s) for s in bank.steps],
        # Leaf keys are tuples of strings; they are kept as lists so the
        # file loads under weights_only.
        "states": [[(list(k) if isinstance(k, tuple) else [k], v) for k, v in s.items()] for s in bank.states],
        "grids": {
            "observation": grids.observation, "acted": grids.acted, "rule_action": grids.rule_action,
            "rule_scores": grids.rule_scores, "rule_unit": grids.rule_unit,
            "channel_names": list(grids.channel_names),
        },
    }
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_bank(path) -> StartBank:
    """Read a bank written by :func:`save_bank`. The file is memory-mapped,
    so a 7 GB bank opens in the time it takes to read its index."""
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("format") != BANK_FORMAT:
        raise ValueError(f"{path} is bank format {payload.get('format')}, this code reads {BANK_FORMAT}")
    states = [{tuple(k) if len(k) > 1 else k[0]: v for k, v in s} for s in payload["states"]]
    g = payload["grids"]
    grids = LabelledGrids(
        g["observation"], g["acted"], g["rule_action"], g["rule_scores"], g["rule_unit"], list(g["channel_names"]),
    )
    return StartBank(states=states, steps=list(payload["steps"]), grids=grids, seed=int(payload["seed"]))


def cached_bank(
    make_env: Callable[[int], object],
    cache_dir,
    key_parts: Dict[str, object],
    *,
    worlds: int,
    steps: int,
    warmup: int,
    stride: int,
    seed: int,
    verbose: bool = False,
) -> StartBank:
    """The bank :func:`build_bank` would return, read from ``cache_dir`` when
    a file for its key is there and built and written there when not. With
    ``cache_dir`` None the bank is built and nothing is written. ``key_parts``
    holds the inputs beyond the bank parameters that shape the bank: the
    config path, the environment arguments and the device type."""
    from pathlib import Path

    def build():
        if verbose:
            print(f"banking {worlds} worlds for {steps} rule steps, every {stride} after {warmup}", flush=True)
        return build_bank(make_env, worlds=worlds, steps=steps, warmup=warmup, stride=stride, seed=seed, verbose=verbose)

    if cache_dir is None:
        return build()
    parts = dict(key_parts)
    parts.update(bank_worlds=int(worlds), bank_steps=int(steps), bank_warmup=int(warmup), bank_stride=int(stride), bank_seed=int(seed))
    path = Path(cache_dir) / f"bank-{bank_key(parts)}.pt"
    if path.exists():
        if verbose:
            print(f"bank: reading {path}", flush=True)
        return load_bank(path)
    bank = build()
    save_bank(bank, path)
    if verbose:
        print(f"bank: written to {path}", flush=True)
    return bank

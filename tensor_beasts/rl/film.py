"""Watch one individual live its life.

Every number this project reports is a sum over thousands of animals. That is
the right way to decide whether a policy is better, and it is a poor way to
understand *how*. A ratio of 1.3x does not say whether the policy learned to
circle a patch of plants, to run from a predator, or merely to stop wasting
biomass. This module answers that the only way it can be answered: follow one
animal, frame by frame, and look.

How an individual is followed
-----------------------------

The simulation has no usable identity. Its ``id`` feature draws from a uint8
random field, so 256 values are shared among thousands of animals and
collisions are constant; ``planning/04-reinforcement-learning.md`` records why
it could not be used. What does work is the same successor map that the
learner's advantage estimation already relies on: ``TransitionInfo.successor``
gives, for every cell that acted, the flat index of the cell that same
individual occupies once the step is over. Chaining successors follows one
individual exactly as long as it is alive, which is the property measured in
that document at 55,000 agent-steps with no two individuals ever sharing a
successor.

So :class:`IndividualTracker` walks every living individual forward at once,
one gather per step, and remembers each one's path, its reward, and when it
died. Nothing here is per-individual Python: the cost is one gather and a few
scatters per step regardless of how many thousands are alive.

Who gets filmed
---------------

Picking the best individual would produce a highlight reel that says nothing
about the policy, because the luckiest animal in a chaotic ecology looks
impressive under any policy at all. Two are sampled instead, both from
individuals whose lives ended inside the window so their returns are complete:

* one from the **typical** band, sampled near the median return,
* one from the **high** band, sampled from the top decile.

Sampled rather than argmax, so a second film of the same policy is a different
animal, and the pair together shows the spread rather than the peak.

What a frame contains
---------------------

A crop centred on the followed individual, rendered through the simulation's
own display config so the colours mean what they mean in the viewer, scaled up
with nearest-neighbour so single cells are visible, and marked with a reticle
so the followed animal can be told from its neighbours. The world wraps, so the
crop wraps with it via ``torch.roll`` rather than clamping at the edges, which
would make an individual near a border drift out of its own frame.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from tensor_beasts.snapshot import WorldSnapshot

# Colour of the reticle drawn around the followed individual, RGB.
RETICLE_COLOR = (255, 255, 0)
# Colour of the trail showing where it has been. Drawn dimmer than the reticle
# so a long trail does not compete with the animal at the centre of the frame.
TRAIL_COLOR = (150, 80, 0)
# The followed individual's own cell. Brightened towards white so it reads as
# "this one" rather than as one more animal of its species: at one cell per
# animal the species colour alone is not enough to pick it out of a crowd.
SUBJECT_COLOR = (255, 255, 255)
# How far the subject's own colour is pushed towards SUBJECT_COLOR. Not 1.0,
# so the species colour underneath stays readable.
SUBJECT_BLEND = 0.55


@dataclass
class Life:
    """One individual's recorded life.

    Attributes:
        path: Flat cell index at each step, from the step it was first seen to
            the step it was last alive.
        start_step: Index into the recording at which ``path`` begins.
        reward: Total reward accumulated over the life, on the same scale the
            trainer optimizes.
        steps_survived: Number of steps the individual was alive and acting.
        reproductions: Times it divided.
        ate: Total biomass eaten, when the environment reports it.
        died: Whether the individual's death was observed. Lives still running
            when the recording ends are incomplete and are not filmed, because
            their return is censored: a long life that has not finished yet
            would look like a short one.
    """

    path: List[int] = field(default_factory=list)
    start_step: int = 0
    reward: float = 0.0
    steps_survived: int = 0
    reproductions: int = 0
    ate: float = 0.0
    died: bool = False


class IndividualTracker:
    """Follows every living individual through a run, by successor chaining.

    Args:
        size: World shape, ``(H, W)``.
        device: Where the bookkeeping tensors live.
        max_tracked: Cap on simultaneously tracked lives. The tracker assigns
            every individual alive at the start, and every newborn, a slot; a
            512x512 world with several thousand predators needs far fewer slots
            than cells, and the cap keeps memory bounded on a herbivore world
            with tens of thousands alive. Slots are reused once a life ends.
    """

    def __init__(self, size: Tuple[int, int], device: torch.device, max_tracked: int = 20000):
        self.size = size
        self.device = device
        self.max_tracked = max_tracked
        self.step_index = 0
        # Slot id occupying each cell, -1 where nothing tracked lives.
        self.slot_at_cell = torch.full(size, -1, dtype=torch.long, device=device)
        self.lives: Dict[int, Life] = {}
        self._free_slots: List[int] = []
        self._next_slot = 0

    def _claim_slot(self) -> Optional[int]:
        if self._free_slots:
            return self._free_slots.pop()
        if self._next_slot >= self.max_tracked:
            return None
        slot = self._next_slot
        self._next_slot += 1
        return slot

    def _assign(self, cells: torch.Tensor) -> None:
        """Give every cell in ``cells`` (flat indices) a fresh life."""
        flat = self.slot_at_cell.reshape(-1)
        for cell in cells.tolist():
            if flat[cell] >= 0:
                continue
            slot = self._claim_slot()
            if slot is None:
                return
            flat[cell] = slot
            self.lives[slot] = Life(path=[int(cell)], start_step=self.step_index)

    def begin(self, alive: torch.Tensor) -> None:
        """Start tracking everything alive before the first step."""
        self._assign(alive.reshape(-1).nonzero(as_tuple=True)[0])

    def update(self, batch) -> None:
        """Advance every tracked life by one step from an ``AgentBatch``.

        Individuals that acted move to their successor cell; individuals that
        died have their lives closed; cells holding an individual the tracker
        has not seen before, which is how offspring appear, start a new life.
        """
        flat = self.slot_at_cell.reshape(-1)
        acted = batch.acted.reshape(-1)
        done = batch.done.reshape(-1)
        successor = batch.successor.reshape(-1)
        reward = batch.reward.reshape(-1)
        reproduced = batch.reproduced.reshape(-1)
        eaten = None
        if getattr(batch, "eaten", None) is not None:
            eaten = batch.eaten.reshape(-1)

        acting_cells = acted.nonzero(as_tuple=True)[0]
        moved_slots = flat[acting_cells]
        next_flat = torch.full_like(flat, -1)

        for cell, slot in zip(acting_cells.tolist(), moved_slots.tolist()):
            if slot < 0:
                continue
            life = self.lives.get(slot)
            if life is None:
                continue
            life.reward += float(reward[cell])
            if bool(reproduced[cell]):
                life.reproductions += 1
            if bool(done[cell]):
                life.died = True
                self._free_slots.append(slot)
                continue
            life.steps_survived += 1
            destination = int(successor[cell])
            if destination < 0:
                life.died = True
                self._free_slots.append(slot)
                continue
            if eaten is not None:
                life.ate += float(eaten[destination])
            life.path.append(destination)
            next_flat[destination] = slot

        self.slot_at_cell = next_flat.reshape(self.size)
        self.step_index += 1

    def observe_newcomers(self, alive: torch.Tensor) -> None:
        """Start lives for living cells with no slot, i.e. newborns.

        Called after :meth:`update`, once the world's new occupancy is known.
        """
        untracked = alive & (self.slot_at_cell < 0)
        self._assign(untracked.reshape(-1).nonzero(as_tuple=True)[0])

    def completed_lives(self, min_steps: int = 1) -> List[Life]:
        """Every life that ended inside the recording, longest-lived first."""
        out = [life for life in self.lives.values() if life.died and life.steps_survived >= min_steps]
        out.sort(key=lambda life: life.steps_survived, reverse=True)
        return out


def select_bands(
    lives: Sequence[Life],
    generator: Optional[torch.Generator] = None,
    high_quantile: float = 0.9,
) -> Dict[str, Life]:
    """Sample one typical and one high-performing life.

    Returns a mapping with keys ``"typical"`` and ``"high"``, omitting either
    when there are too few completed lives to define it. Selection is on total
    reward, the quantity the policy optimizes.

    The typical band is the middle 20% of the return distribution rather than
    the single median, and the high band is everything at or above
    ``high_quantile``. Both are then sampled uniformly, so repeated films of one
    policy show different animals and a viewer sees the range rather than one
    lucky trajectory.
    """
    if not lives:
        return {}
    returns = torch.tensor([life.reward for life in lives], dtype=torch.float32)
    order = returns.argsort()
    count = len(lives)

    def sample(candidates: List[int]) -> Optional[Life]:
        if not candidates:
            return None
        index = int(torch.randint(len(candidates), (1,), generator=generator).item())
        return lives[candidates[index]]

    low_cut = max(0, int(0.4 * count))
    high_cut = min(count, max(low_cut + 1, int(0.6 * count)))
    typical_pool = [int(order[i]) for i in range(low_cut, high_cut)]

    top_cut = min(count - 1, int(high_quantile * count))
    high_pool = [int(order[i]) for i in range(top_cut, count)]

    out: Dict[str, Life] = {}
    typical = sample(typical_pool)
    high = sample(high_pool)
    if typical is not None:
        out["typical"] = typical
    if high is not None and (typical is None or high is not typical):
        out["high"] = high
    return out


def crop_around(frame: torch.Tensor, center: Tuple[int, int], window: int) -> torch.Tensor:
    """A ``window`` x ``window`` crop of ``frame`` centred on ``center``.

    The world is a torus and animals near an edge are ordinary animals, so the
    crop wraps rather than clamping. Clamping would slide the followed
    individual off the centre of its own frame exactly when it crosses a
    boundary.
    """
    height, width = frame.shape[0], frame.shape[1]
    half = window // 2
    row, col = center
    rolled = torch.roll(frame, shifts=(half - row, half - col), dims=(0, 1))
    return rolled[:window, :window]


def draw_reticle(frame: torch.Tensor, center: int, color=RETICLE_COLOR, radius: int = 3) -> torch.Tensor:
    """Draw corner brackets around ``center`` on a ``(W, W, 3)`` crop.

    Corners rather than a closed box, and set back from the individual rather
    than drawn on it. A closed box at a small radius covers the animal it is
    pointing at, which defeats the purpose: the first version of this drew over
    the followed predator so completely that only its colour in the frame
    buffer proved it was ever there.
    """
    out = frame.clone()
    tint = torch.tensor(color, dtype=out.dtype, device=out.device)
    low, high = center - radius, center + radius
    size = out.shape[0]
    if low < 0 or high >= size:
        return out
    arm = max(2, radius // 2)
    for row, col_start, col_step in ((low, low, 1), (high, low, 1)):
        out[row, col_start:col_start + arm] = tint
        out[row, high - arm + 1:high + 1] = tint
    for col in (low, high):
        out[low:low + arm, col] = tint
        out[high - arm + 1:high + 1, col] = tint
    return out


def draw_bar(
    frame: torch.Tensor,
    row: int,
    fraction: float,
    color,
    height: int = 3,
    margin: int = 4,
) -> torch.Tensor:
    """Draw a horizontal meter across the top of a frame.

    The films are watched without a caption, so the individual's state has to be
    in the picture. A bar is legible at a glance and at any resolution, which
    rendered text is not at these sizes.
    """
    out = frame
    width = out.shape[1] - 2 * margin
    filled = int(max(0.0, min(1.0, fraction)) * width)
    if filled <= 0:
        return out
    tint = torch.tensor(color, dtype=out.dtype, device=out.device)
    out[row:row + height, margin:margin + filled] = tint
    return out


def upscale(frame: torch.Tensor, factor: int) -> torch.Tensor:
    """Nearest-neighbour upscale of an ``(H, W, 3)`` frame.

    Nearest neighbour on purpose: one cell is one animal, and smoothing it into
    its neighbours is exactly the information a viewer is trying to read.
    """
    if factor <= 1:
        return frame
    return frame.repeat_interleave(factor, dim=0).repeat_interleave(factor, dim=1)


def render_snapshot(snapshot: WorldSnapshot, display_config) -> torch.Tensor:
    """One world snapshot as an ``(H, W, 3)`` uint8 tensor.

    Imported lazily so that recording a film does not pull in the display stack
    for callers that never render one.
    """
    from tensor_beasts.display.rendering import dispatch_render

    frame = dispatch_render(snapshot, display_config)
    # default_renderer returns an expanded view, which has no real strides and
    # cannot be handed to a video encoder.
    return frame.contiguous()


def film_life(
    snapshots: Sequence[WorldSnapshot],
    life: Life,
    display_config,
    window: int = 96,
    scale: int = 4,
    trail: int = 12,
    entity_name: Optional[str] = None,
    max_energy: float = 255.0,
    max_biomass: float = 255.0,
) -> List[torch.Tensor]:
    """Render one individual's life as a list of ``(S, S, 3)`` uint8 frames.

    Args:
        snapshots: One world snapshot per recorded step.
        life: The life to follow. Its ``path`` indexes into ``snapshots``
            starting at ``life.start_step``.
        display_config: A single entry from ``config.display.color_displays``.
        window: Side length of the crop in world cells.
        scale: Nearest-neighbour upscale factor.
        trail: How many previous positions to mark, so direction of travel is
            visible in a still frame.
        entity_name: Lower-case entity key, e.g. ``"predator"``. When given, the
            individual's own energy and biomass are drawn as meters, which is
            what turns a dot moving over a field into a story about whether it
            is finding food.
        max_energy: Scale for the energy meter.
        max_biomass: Scale for the biomass meter.
    """
    frames: List[torch.Tensor] = []
    for offset, cell in enumerate(life.path):
        index = life.start_step + offset
        if index >= len(snapshots):
            break
        snapshot = snapshots[index]
        rendered = render_snapshot(snapshot, display_config)
        width = rendered.shape[1]
        row, col = divmod(int(cell), width)

        marked = rendered.clone()
        # The trail is drawn under the animal, oldest first, so the current
        # position is never painted over by its own history.
        tint = torch.tensor(TRAIL_COLOR, dtype=marked.dtype, device=marked.device)
        for previous in life.path[max(0, offset - trail):offset]:
            prow, pcol = divmod(int(previous), width)
            marked[prow, pcol] = tint

        # The subject last, over its own trail, blended rather than replaced so
        # that its species colour is still visible underneath.
        subject = torch.tensor(SUBJECT_COLOR, dtype=torch.float32, device=marked.device)
        own = marked[row, col].to(torch.float32)
        marked[row, col] = (own * (1 - SUBJECT_BLEND) + subject * SUBJECT_BLEND).to(marked.dtype)

        crop = crop_around(marked, (row, col), window)
        crop = upscale(crop, scale)
        centre = (window // 2) * scale + scale // 2
        frame = draw_reticle(crop, centre, radius=max(4, scale * 2))

        if entity_name is not None:
            energy = _read_state(snapshot, entity_name, "energy", cell)
            biomass = _read_state(snapshot, entity_name, "biomass", cell)
            if energy is not None:
                frame = draw_bar(frame, 2, energy / max_energy, (80, 200, 255))
            if biomass is not None:
                frame = draw_bar(frame, 7, biomass / max_biomass, (255, 120, 120))
        frames.append(frame)
    return frames


def _read_state(snapshot: WorldSnapshot, entity_name: str, feature: str, cell: int) -> Optional[float]:
    """One individual's value of one feature, or None when the key is absent."""
    try:
        data = snapshot.get((entity_name, feature))
    except KeyError:
        return None
    return float(data.reshape(-1)[cell])


def write_video(frames: Sequence[torch.Tensor], path: Path, fps: int = 10) -> Optional[Path]:
    """Write frames to ``path`` as mp4, falling back to a GIF.

    Returns the path actually written, or None if no encoder is available and
    no frames could be saved. Encoding is optional infrastructure: a training
    run must not die because a video writer is missing.
    """
    if not frames:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stack = [frame.cpu().numpy() for frame in frames]
    try:
        import imageio.v2 as imageio
    except ImportError:
        return None

    try:
        with imageio.get_writer(path, fps=fps, macro_block_size=None) as writer:
            for frame in stack:
                writer.append_data(frame)
        return path
    except Exception:
        gif = path.with_suffix(".gif")
        try:
            imageio.mimsave(gif, stack, duration=1.0 / max(fps, 1))
            return gif
        except Exception:
            return None

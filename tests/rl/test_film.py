"""Following one individual has to follow the individual, not the cell.

The whole point of a film that follows an animal is that it is the same animal
in every frame. The simulation offers no usable identity, so the tracker chains
the successor map, and the test that matters is the one that would fail if it
chained cells instead: park a decoy in the cell the followed individual just
left and check the path does not jump to it.

Small worlds here; these are plumbing tests, not ecology.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from tensor_beasts.rl.film import (
    IndividualTracker,
    Life,
    crop_around,
    draw_reticle,
    film_life,
    select_bands,
    upscale,
    write_video,
)

SIZE = (8, 8)


@dataclass
class FakeBatch:
    """The fields IndividualTracker reads out of an AgentBatch."""

    acted: torch.Tensor
    done: torch.Tensor
    successor: torch.Tensor
    reward: torch.Tensor
    reproduced: torch.Tensor
    eaten: Optional[torch.Tensor] = None


def batch_of(moves, size=SIZE, dead=(), reproduced_at=(), reward=1.0):
    """Build a batch where each (from_cell, to_cell) pair is one moving animal."""
    acted = torch.zeros(size, dtype=torch.bool)
    done = torch.zeros(size, dtype=torch.bool)
    successor = torch.full(size, -1, dtype=torch.long)
    rewards = torch.zeros(size, dtype=torch.float32)
    repro = torch.zeros(size, dtype=torch.bool)
    flat_acted = acted.reshape(-1)
    flat_done = done.reshape(-1)
    flat_succ = successor.reshape(-1)
    flat_reward = rewards.reshape(-1)
    flat_repro = repro.reshape(-1)
    for source, destination in moves:
        flat_acted[source] = True
        flat_succ[source] = destination
        flat_reward[source] = reward
    for cell in dead:
        flat_acted[cell] = True
        flat_done[cell] = True
    for cell in reproduced_at:
        flat_repro[cell] = True
    return FakeBatch(acted=acted, done=done, successor=successor, reward=rewards, reproduced=repro)


def test_tracker_follows_the_individual_not_the_cell():
    """The decoy test. A cell-indexed tracker fails this; a successor-chained one does not."""
    tracker = IndividualTracker(SIZE, torch.device("cpu"))
    alive = torch.zeros(SIZE, dtype=torch.bool)
    alive.reshape(-1)[0] = True
    tracker.begin(alive)
    slot = int(tracker.slot_at_cell.reshape(-1)[0])

    # The tracked individual walks 0 -> 1 -> 2 -> 3, and at every step another
    # animal moves into the cell it just vacated.
    tracker.update(batch_of([(0, 1), (9, 0)]))
    tracker.update(batch_of([(1, 2), (0, 1)]))
    tracker.update(batch_of([(2, 3), (1, 2)]))

    assert tracker.lives[slot].path == [0, 1, 2, 3]
    assert tracker.lives[slot].steps_survived == 3


def test_death_closes_a_life_and_frees_its_slot():
    tracker = IndividualTracker(SIZE, torch.device("cpu"))
    alive = torch.zeros(SIZE, dtype=torch.bool)
    alive.reshape(-1)[5] = True
    tracker.begin(alive)
    slot = int(tracker.slot_at_cell.reshape(-1)[5])

    tracker.update(batch_of([(5, 6)]))
    tracker.update(batch_of([], dead=(6,)))

    life = tracker.lives[slot]
    assert life.died is True
    assert life in tracker.completed_lives()
    # The slot is reusable, so a long run does not grow without bound.
    assert slot in tracker._free_slots


def test_newborns_start_their_own_lives():
    tracker = IndividualTracker(SIZE, torch.device("cpu"))
    alive = torch.zeros(SIZE, dtype=torch.bool)
    alive.reshape(-1)[0] = True
    tracker.begin(alive)
    assert len(tracker.lives) == 1

    tracker.update(batch_of([(0, 1)], reproduced_at=(0,)))
    born = torch.zeros(SIZE, dtype=torch.bool)
    born.reshape(-1)[1] = True   # the parent
    born.reshape(-1)[2] = True   # the offspring
    tracker.observe_newcomers(born)

    assert len(tracker.lives) == 2
    parent = tracker.lives[0]
    assert parent.reproductions == 1


def test_incomplete_lives_are_not_offered_for_filming():
    """A life still running has a censored return and would look short."""
    tracker = IndividualTracker(SIZE, torch.device("cpu"))
    alive = torch.zeros(SIZE, dtype=torch.bool)
    alive.reshape(-1)[0] = True
    tracker.begin(alive)
    tracker.update(batch_of([(0, 1)]))
    assert tracker.completed_lives() == []


def test_select_bands_picks_a_typical_and_a_high_life():
    lives = []
    for value in range(100):
        life = Life(reward=float(value), died=True, steps_survived=10)
        lives.append(life)
    generator = torch.Generator().manual_seed(0)
    chosen = select_bands(lives, generator=generator)

    assert set(chosen) == {"typical", "high"}
    # The high band is the top decile; the typical band is the middle fifth.
    assert chosen["high"].reward >= 89
    assert 39 <= chosen["typical"].reward <= 60
    assert chosen["high"] is not chosen["typical"]


def test_select_bands_is_empty_without_completed_lives():
    assert select_bands([]) == {}


def test_crop_wraps_at_the_world_edge():
    """The world is a torus, so an animal at the border stays centred."""
    frame = torch.zeros(8, 8, 3, dtype=torch.uint8)
    frame[0, 0] = torch.tensor([255, 0, 0], dtype=torch.uint8)
    crop = crop_around(frame, (0, 0), window=4)
    assert crop.shape == (4, 4, 3)
    # The marked cell lands at the crop's centre, not at a clamped corner.
    assert torch.equal(crop[2, 2], torch.tensor([255, 0, 0], dtype=torch.uint8))


def test_upscale_is_nearest_neighbour():
    frame = torch.tensor([[[1, 2, 3]]], dtype=torch.uint8)
    out = upscale(frame, 3)
    assert out.shape == (3, 3, 3)
    assert torch.equal(out[0, 0], out[2, 2])


def test_reticle_is_open_so_the_animal_stays_visible():
    frame = torch.zeros(9, 9, 3, dtype=torch.uint8)
    marked = draw_reticle(frame, center=4, radius=2)
    # The centre, where the individual is, is untouched.
    assert torch.equal(marked[4, 4], torch.zeros(3, dtype=torch.uint8))
    # The box edge is drawn.
    assert marked[2, 2].sum() > 0


def test_the_followed_individual_is_marked_distinctly():
    """A viewer has to be able to tell the subject from every other animal.

    One cell is one animal, so the species colour alone does not identify the
    one being followed. The subject's cell is blended towards white; this
    checks the centre of the frame is brighter than the same animal would be
    unmarked.
    """
    from tensor_beasts.rl.film import SUBJECT_BLEND

    frame = torch.zeros(8, 8, 3, dtype=torch.uint8)
    frame[1, 1] = torch.tensor([128, 0, 0], dtype=torch.uint8)  # a predator
    snapshot = _one_cell_snapshot(frame)

    life = Life(path=[9], start_step=0, died=True, steps_survived=1)
    frames = film_life([snapshot], life, _passthrough_config(), window=4, scale=1, trail=0)
    centre = frames[0][2, 2]

    assert int(centre.sum()) > 128, "the subject must be brighter than its unmarked species colour"
    assert SUBJECT_BLEND < 1.0, "the species colour must remain visible underneath"


def _one_cell_snapshot(frame):
    from tensor_beasts.snapshot import WorldSnapshot

    return WorldSnapshot(step=0, data={("fake", "rgb"): frame})


class _PassthroughConfig:
    """The attributes dispatch_render reads, with a tuple key.

    OmegaConf turns a tuple into a list, and the snapshot is keyed by tuple, so
    a DictConfig cannot express this. The real configs get their tuple from the
    ``${key:entity,feature}`` resolver at load time.
    """

    fn_name = "default"
    key = ("fake", "rgb")
    input_range = None


def _passthrough_config():
    return _PassthroughConfig()


def test_trainer_records_a_film_and_never_raises(tmp_path):
    """The trainer's film path, end to end, on a world small enough to be fast.

    Not a valid ecology; this checks the plumbing, that a film lands on disk and
    that the method reports statistics rather than throwing into a training run.
    """
    from tensor_beasts.rl.ppo import PPOConfig
    from tensor_beasts.rl.trainer import Trainer, TrainerConfig

    trainer = Trainer(
        TrainerConfig(
            size=96, entity="Predator", arch="conv", arch_kwargs={"hidden_channels": 8},
            bank_worlds=2, bank_steps=24, bank_warmup=20, bank_stride=2, total_world_steps=0, segment_steps=8, eval_interval=0,
            checkpoint_interval=0, device="cpu", output_dir=str(tmp_path),
            film_interval=1, film_steps=60, film_window=24, film_scale=3,
        ),
        PPOConfig(epochs=1, minibatch_steps=2),
    )
    trainer.start_worlds()
    result = trainer.record_film()

    assert "film_error" not in result, result.get("film_error")
    # Either it filmed somebody, or it said why it could not.
    filmed = [k for k in result if k.endswith("_path")]
    assert filmed or "film_skipped" in result
    for key in filmed:
        assert Path(result[key]).exists()


def test_film_life_renders_one_frame_per_step(tmp_path):
    """End to end over a real snapshot and the real display config."""
    from omegaconf import OmegaConf

    from tensor_beasts.config import load_config
    from tensor_beasts.world import World

    config = load_config("conf/basic_config.yaml")
    config.world.size = [32, 32]
    config.world.device = "cpu"
    world = World(config.world)
    world.initialize()

    snapshots = []
    for _ in range(4):
        snapshots.append(world.snapshot())
        world.update()

    display = [d for d in config.display.color_displays if d.title == "layers"][0]
    life = Life(path=[0, 1, 33, 34], start_step=0, died=True, steps_survived=4)
    frames = film_life(snapshots, life, display, window=16, scale=2, trail=3)

    assert len(frames) == 4
    assert frames[0].shape == (32, 32, 3)
    assert frames[0].dtype == torch.uint8

    written = write_video(frames, tmp_path / "life.mp4", fps=5)
    # Encoding is optional; when it works the file must be non-empty.
    if written is not None:
        assert written.exists() and written.stat().st_size > 0

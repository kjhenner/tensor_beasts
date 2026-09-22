"""A batched world must be B independent worlds, not one world in a trench coat.

Two properties, and both matter for different reasons.

``batch_shape`` of ``(1,)`` must be bit-identical to an unbatched world. That is
what makes the refactor reviewable: batching is not a separate code path, it is
the same one with a prefix, and any divergence at B=1 is a batching bug with the
ecology's chaos held out of the way.

Worlds in a batch must not influence each other. The dangerous failures here are
silent: a shared random field correlates their plant germination, a flat index
without a per-world stride reads a neighbour's cells, and a global reduction
couples their totals. None of those crash.
"""

import pytest
import torch

from tensor_beasts.config import load_config
from tensor_beasts.world import World

SIZE = 32
FEATURES = (
    ("Predator", "biomass"),
    ("Herbivore", "biomass"),
    ("SimplePlant", "energy"),
)


@pytest.fixture(autouse=True)
def _cpu_default_device():
    previous = torch.get_default_device()
    torch.set_default_device("cpu")
    try:
        yield
    finally:
        torch.set_default_device(previous)


def build(batch=None, seed=7, size=SIZE):
    config = load_config("conf/basic_config.yaml")
    config.world.size = [size, size]
    config.world.device = "cpu"
    if batch:
        config.world.batch = batch
    torch.manual_seed(seed)
    world = World(config.world)
    world.initialize()
    return world


def copy_into_slot(batched, single, slot=0):
    """Make ``batched``'s world ``slot`` hold exactly ``single``'s state."""
    for key in list(batched.td.keys(True, True)):
        target, source = batched.td.get(key), single.td.get(key)
        if not isinstance(target, torch.Tensor) or target.ndim == 0:
            continue
        if isinstance(source, torch.Tensor) and source.shape == target.shape[1:]:
            target[slot] = source


def test_a_batch_of_one_is_bit_identical_to_an_unbatched_world():
    """The reviewability property. Any divergence here is a batching bug."""
    single, batched = build(), build(batch=1)
    copy_into_slot(batched, single)

    for step in range(25):
        torch.manual_seed(900 + step)
        batched.update()
        torch.manual_seed(900 + step)
        single.update()

    for entity, feature in FEATURES:
        got = getattr(getattr(batched, entity), feature).data[0]
        want = getattr(getattr(single, entity), feature).data
        assert torch.equal(got, want), f"{entity}.{feature} diverged at batch size one"
    assert torch.equal(
        batched.SimpleTerrain.simple_water.data[0], single.SimpleTerrain.simple_water.data
    )


def test_worlds_in_a_batch_have_their_own_terrain():
    """Perlin noise used to fail on a batched shape and be swallowed by a bare
    ``except``, so every world silently fell back to the same uniform random
    field. Worlds that share their terrain are not independent worlds."""
    batched = build(batch=3)
    water = batched.SimpleTerrain.simple_water.data
    assert not torch.equal(water[0], water[1])
    assert not torch.equal(water[1], water[2])


def test_worlds_in_a_batch_have_their_own_randomness():
    """The per-step random field is what plant germination and offspring ids
    draw from. One field shared across the batch correlates them."""
    batched = build(batch=3)
    batched.update()
    field = batched.td.get("random")
    assert field.shape[0] == 3, "the random field must carry the batch dimension"
    assert not torch.equal(field[0], field[1])


def test_worlds_in_a_batch_diverge_from_each_other():
    batched = build(batch=3)
    for _ in range(30):
        batched.update()
    biomass = batched.Herbivore.biomass.data
    assert not torch.equal(biomass[0], biomass[1])
    assert not torch.equal(biomass[1], biomass[2])


def test_perturbing_one_world_leaves_the_others_alone():
    """The direct test for cross-world coupling.

    Two identical batched worlds, one perturbed in slot 0 only. Slot 0 must
    change and the others must not. A shared field or a flat index without a
    per-world stride fails this.
    """
    reference, perturbed = build(batch=3), build(batch=3)
    assert torch.equal(
        reference.Herbivore.biomass.data, perturbed.Herbivore.biomass.data
    ), "the two worlds must start identical for this to prove anything"

    perturbed.Herbivore.biomass.data[0] = 0.0

    for step in range(12):
        torch.manual_seed(31 + step)
        reference.update()
        torch.manual_seed(31 + step)
        perturbed.update()

    for entity, feature in FEATURES:
        got = getattr(getattr(perturbed, entity), feature).data
        want = getattr(getattr(reference, entity), feature).data
        assert not torch.equal(got[0], want[0]), (
            f"{entity}.{feature} world 0 was perturbed and must differ"
        )
        for other in (1, 2):
            assert torch.equal(got[other], want[other]), (
                f"{entity}.{feature} world {other} changed when only world 0 was "
                "perturbed, so the worlds are coupled"
            )


def test_the_rl_environment_exposes_a_batched_contract():
    """Every field the learner exchanges gains the batch axis, in the right place.

    The channel axis is the one to get wrong. Each observation channel is
    (H, W) unbatched and (B, H, W) batched, so stacking at dim 0 would give
    (C, B, H, W) and hand a convolution the batch as its channels. The same
    applies to the rule's five action scores.
    """
    from tensor_beasts.rl.multiagent import MultiAgentWorldEnv

    size, worlds = 32, 3
    env = MultiAgentWorldEnv(
        size=(size, size), device="cpu", entity_name="Predator", worlds=worlds
    )
    observation, acted = env.reset(seed=0)

    assert env.num_worlds == worlds
    assert env.field_shape == (worlds, size, size)
    assert observation.shape == (worlds, env.observation_channels, size, size)
    assert acted.shape == (worlds, size, size)

    for _ in range(5):
        env.rule_based_step()
    batch = env.rule_based_step()

    for field in ("acted", "reward", "done", "successor", "reproduced", "rule_action"):
        assert getattr(batch, field).shape == (worlds, size, size), field
    # Five action scores, in the channel position a policy's logits occupy.
    assert batch.rule_scores.shape == (worlds, 5, size, size)


def test_one_world_keeps_the_unbatched_shapes():
    """The batched path must not impose a batch axis on ordinary runs."""
    from tensor_beasts.rl.multiagent import MultiAgentWorldEnv

    size = 32
    env = MultiAgentWorldEnv(size=(size, size), device="cpu", entity_name="Predator")
    observation, acted = env.reset(seed=0)

    assert env.num_worlds == 1
    assert env.field_shape == (size, size)
    assert observation.shape == (env.observation_channels, size, size)
    assert acted.shape == (size, size)

    for _ in range(5):
        env.rule_based_step()
    assert env.rule_based_step().rule_scores.shape == (5, size, size)


def test_the_trainer_collects_and_updates_across_worlds():
    """The point of the whole refactor: one update, several ecologies.

    The unit ratio on the first epoch is the correctness check this codebase
    relies on everywhere. It only holds if the log-probabilities recomputed in
    the update come from exactly the observation collection saw, so a folded
    (T * B) minibatch that mixed up time and worlds would break it.
    """
    import tempfile

    from tensor_beasts.rl.ppo import PPOConfig
    from tensor_beasts.rl.trainer import Trainer, TrainerConfig

    worlds, steps = 3, 4
    trainer = Trainer(
        TrainerConfig(
            size=64, entity="Predator", arch="conv", arch_kwargs={"hidden_channels": 8},
            worlds=worlds, bank_worlds=2, bank_steps=9, bank_warmup=5, bank_stride=2, total_world_steps=0, segment_steps=steps,
            eval_interval=0, checkpoint_interval=0, device="cpu",
            output_dir=tempfile.mkdtemp(), wandb=False,
        ),
        PPOConfig(epochs=1, minibatch_steps=2),
    )
    trainer.start_worlds()

    rollout, _ = trainer.collect(steps)
    assert rollout.observation.shape == (
        steps, worlds, trainer.observation_channels, 64, 64
    )
    assert rollout.reward.shape == (steps, worlds, 64, 64)

    diagnostics = trainer.algorithm.update(trainer.network, rollout, trainer.optimizer)
    assert diagnostics["approx_kl"] == pytest.approx(0.0, abs=1e-6), (
        "the first epoch must have unit ratio; the update is not seeing what "
        "collection saw"
    )


def test_one_world_trains_exactly_as_before():
    """worlds=1 must not acquire a batch axis anywhere in the rollout."""
    import tempfile

    from tensor_beasts.rl.ppo import PPOConfig
    from tensor_beasts.rl.trainer import Trainer, TrainerConfig

    steps = 4
    trainer = Trainer(
        TrainerConfig(
            size=64, entity="Predator", arch="conv", arch_kwargs={"hidden_channels": 8},
            bank_worlds=2, bank_steps=9, bank_warmup=5, bank_stride=2, total_world_steps=0, segment_steps=steps, eval_interval=0,
            checkpoint_interval=0, device="cpu", output_dir=tempfile.mkdtemp(), wandb=False,
        ),
        PPOConfig(epochs=1, minibatch_steps=2),
    )
    trainer.start_worlds()

    rollout, _ = trainer.collect(steps)
    assert rollout.observation.shape == (steps, trainer.observation_channels, 64, 64)
    assert rollout.reward.shape == (steps, 64, 64)
    trainer.algorithm.update(trainer.network, rollout, trainer.optimizer)


def test_evaluation_scores_each_seed_separately():
    """Evaluation seeds are the batch axis, and must not blend together.

    Each world is an independent seed, so its numbers have to survive to the
    caller. Summing over the batch before reporting would hide the spread
    between seeds, which is exactly the quantity every claim in this project is
    hedged against, and would do it silently.
    """
    import tempfile

    from tensor_beasts.rl.ppo import PPOConfig
    from tensor_beasts.rl.trainer import Trainer, TrainerConfig

    seeds = 4
    trainer = Trainer(
        TrainerConfig(
            size=96, entity="Predator", arch="conv", arch_kwargs={"hidden_channels": 8},
            bank_worlds=2, bank_steps=14, bank_warmup=10, bank_stride=2, total_world_steps=0, eval_steps=40, eval_seeds=seeds,
            eval_interval=0, checkpoint_interval=0, device="cpu",
            output_dir=tempfile.mkdtemp(), wandb=False,
        ),
        PPOConfig(),
    )

    # The evaluation world holds every seed at once.
    assert trainer._eval_env(seeds).num_worlds == seeds

    summary = trainer.evaluate()

    # The headline is a mean over seeds, and the spread survives beside it.
    assert "score" in summary
    assert summary["score_max"] >= summary["score"] >= summary["score_min"]
    assert summary["score_spread"] > 0.0, (
        "four independent seeds produced identical scores, so they are not "
        "independent"
    )


def test_scoring_reports_one_number_per_world():
    """_score's accumulators must sum over the grid but not over the batch."""
    import tempfile

    from tensor_beasts.rl.ppo import PPOConfig
    from tensor_beasts.rl.trainer import Trainer, TrainerConfig

    worlds = 3
    trainer = Trainer(
        TrainerConfig(
            size=96, entity="Predator", arch="conv", arch_kwargs={"hidden_channels": 8},
            bank_worlds=2, bank_steps=14, bank_warmup=10, bank_stride=2, total_world_steps=0, eval_steps=30, eval_seeds=worlds,
            eval_interval=0, checkpoint_interval=0, device="cpu",
            output_dir=tempfile.mkdtemp(), wandb=False,
        ),
        PPOConfig(),
    )
    env = trainer._eval_env(worlds)
    env.reset(seed=0)

    result = trainer._score(env, 30, "rule_based")

    for field in ("mean_biomass", "extinct", "mean_population", "reproductions"):
        values = getattr(result, field)
        assert len(values) == worlds, f"{field} collapsed the batch"

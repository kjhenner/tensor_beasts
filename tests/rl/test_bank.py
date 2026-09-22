"""The start bank: every world a run touches starts from a banked state.

Small worlds for speed; bookkeeping, not ecology.
"""

import hashlib

import torch

from tensor_beasts.rl.bank import build_bank, snapshot_world
from tensor_beasts.rl.multiagent import MultiAgentWorldEnv

SIZE = (48, 48)


def make_env(worlds, entity="Predator", metabolic=True):
    return MultiAgentWorldEnv(size=SIZE, device="cpu", entity_name=entity, worlds=worlds, metabolic=metabolic)


def slice_hash(env, world):
    digest = hashlib.sha256()
    state = snapshot_world(env, world)
    for key in sorted(state, key=str):
        digest.update(str(key).encode())
        digest.update(state[key].contiguous().numpy().tobytes())
    return digest.hexdigest()


def test_bank_holds_one_state_and_one_grid_per_world_per_snapshot():
    bank = build_bank(make_env, worlds=2, steps=10, warmup=4, stride=3, seed=0)
    # Snapshots after 4 and 7 rule steps, from each of two worlds.
    assert len(bank) == 4
    assert bank.steps == [4, 4, 7, 7]
    assert len(bank.grids) == 4
    assert bank.grids.observation.shape[0] == 4 and bank.grids.observation.dtype == torch.float16
    assert bank.grids.rule_unit is not None, "the throttle label is kept when the entity has a throttle"
    assert bank.grids.acted.any()
    assert bank.nbytes > 0
    assert all(v.device.type == "cpu" for state in bank.states for v in state.values())


def test_the_grid_is_the_rules_label_on_the_banked_state():
    """Load a banked state, take one rule step, and the batch must be what the
    bank recorded beside that state."""
    bank = build_bank(make_env, worlds=1, steps=8, warmup=5, stride=1, seed=3)
    env = make_env(1)
    for index in range(len(bank)):
        torch.manual_seed(0)
        bank.load(env, [index])
        batch = env.rule_based_step()
        assert torch.equal(batch.acted, bank.grids.acted[index])
        assert torch.equal(batch.observation.to(torch.float16), bank.grids.observation[index])


def test_loading_a_state_is_exact_and_repeatable():
    bank = build_bank(make_env, worlds=2, steps=6, warmup=3, stride=1, seed=1)
    env = make_env(2)
    bank.load(env, [1, 0])
    first = [slice_hash(env, 0), slice_hash(env, 1)]
    # World 0 got state 1 and world 1 got state 0, exactly.
    reference = make_env(1)
    bank.load(reference, [1])
    assert slice_hash(reference, 0) == first[0]
    bank.load(reference, [0])
    assert slice_hash(reference, 0) == first[1]
    # Running on and reloading brings both worlds back.
    for _ in range(3):
        env.rule_based_step()
    bank.load(env, [1, 0])
    assert [slice_hash(env, 0), slice_hash(env, 1)] == first
    assert env.world.step == bank.steps[1]


def test_loading_one_slot_leaves_the_others_and_the_clock_alone():
    bank = build_bank(make_env, worlds=1, steps=6, warmup=3, stride=1, seed=2)
    env = make_env(3)
    env.reset(seed=9)
    for _ in range(4):
        env.rule_based_step()
    before = [slice_hash(env, w) for w in range(3)]
    step = env.world.step
    bank.load_into(env, 2, world=1)
    after = [slice_hash(env, w) for w in range(3)]
    assert after[0] == before[0] and after[2] == before[2]
    assert after[1] != before[1]
    assert env.world.step == step


def test_draws_come_from_the_generator_alone():
    bank = build_bank(make_env, worlds=2, steps=6, warmup=2, stride=1, seed=0)
    a = bank.draw(3, torch.Generator().manual_seed(5))
    b = bank.draw(3, torch.Generator().manual_seed(5))
    assert a == b and len(set(a)) == 3, "without replacement while the bank allows"
    many = bank.draw(20, torch.Generator().manual_seed(5))
    assert len(many) == 20 and max(many) < len(bank)


def test_building_the_bank_leaves_the_rng_stream_alone():
    torch.manual_seed(11)
    state = torch.get_rng_state()
    build_bank(make_env, worlds=1, steps=4, warmup=2, stride=1, seed=0)
    assert torch.equal(torch.get_rng_state(), state)


def bank_hash(bank):
    digest = hashlib.sha256()
    for state, step in zip(bank.states, bank.steps):
        digest.update(str(step).encode())
        for key in sorted(state, key=str):
            digest.update(str(key).encode())
            digest.update(state[key].contiguous().numpy().tobytes())
    for field in (bank.grids.observation, bank.grids.acted, bank.grids.rule_action,
                  bank.grids.rule_scores, bank.grids.rule_unit):
        if field is not None:
            digest.update(field.contiguous().numpy().tobytes())
    return digest.hexdigest()


def test_a_saved_bank_reads_back_exactly(tmp_path):
    from tensor_beasts.rl.bank import load_bank, save_bank

    bank = build_bank(make_env, worlds=2, steps=8, warmup=4, stride=2, seed=5)
    save_bank(bank, tmp_path / "bank.pt")
    loaded = load_bank(tmp_path / "bank.pt")
    assert bank_hash(loaded) == bank_hash(bank)
    assert loaded.steps == bank.steps and loaded.seed == bank.seed
    assert loaded.grids.channel_names == bank.grids.channel_names
    assert set(loaded.states[0]) == set(bank.states[0]), "leaf keys survive the round trip as the same tuples"
    assert not any(str(p).endswith(".tmp") for p in tmp_path.iterdir())


def test_the_cache_builds_once_and_reads_after(tmp_path, monkeypatch):
    from tensor_beasts.rl import bank as bank_module

    builds = []
    real_build = bank_module.build_bank

    def counting_build(*args, **kwargs):
        builds.append(kwargs["seed"])
        return real_build(*args, **kwargs)

    monkeypatch.setattr(bank_module, "build_bank", counting_build)
    parts = {"size": 48, "entity": "Predator", "metabolic": True, "memory_size": 0, "device": "cpu"}
    kw = dict(worlds=1, steps=8, warmup=4, stride=2, seed=7)
    first = bank_module.cached_bank(make_env, tmp_path, parts, **kw)
    second = bank_module.cached_bank(make_env, tmp_path, parts, **kw)
    assert builds == [7], "the second request is served from the file"
    assert bank_hash(second) == bank_hash(first)
    assert len(list(tmp_path.glob("bank-*.pt"))) == 1
    # A change to any input that shapes the bank is a different file.
    bank_module.cached_bank(make_env, tmp_path, parts, **{**kw, "seed": 8})
    bank_module.cached_bank(make_env, tmp_path, {**parts, "device": "cuda"}, **kw)
    assert builds == [7, 8, 7]
    assert len(list(tmp_path.glob("bank-*.pt"))) == 3
    # No directory: built every time, nothing written.
    bank_module.cached_bank(make_env, None, parts, **kw)
    assert builds == [7, 8, 7, 7]
    assert len(list(tmp_path.glob("bank-*.pt"))) == 3


def test_the_key_follows_the_config_text_and_the_source(tmp_path, monkeypatch):
    from tensor_beasts.rl import bank as bank_module

    config = tmp_path / "sim.yaml"
    config.write_text("a: 1\n")
    parts = {"config_path": str(config), "size": 48}
    key = bank_module.bank_key(parts)
    assert bank_module.bank_key(parts) == key, "the key is stable"
    renamed = tmp_path / "other.yaml"
    renamed.write_text("a: 1\n")
    assert bank_module.bank_key({**parts, "config_path": str(renamed)}) == key, "the name of the config is not in the key"
    config.write_text("a: 2\n")
    assert bank_module.bank_key(parts) != key, "its text is"
    config.write_text("a: 1\n")
    monkeypatch.setattr(bank_module, "source_digest", lambda: "edited")
    assert bank_module.bank_key(parts) != key, "and so is the source of the simulation"


def test_a_loaded_world_keeps_the_landscape_it_was_banked_on():
    """The water is rewritten every step from a base pattern and a phase map
    drawn when the World was made. Both must travel with the state, or a
    loaded world runs on the host environment's landscape from its second
    step on, which is what happened before they were TensorDict leaves."""
    bank = build_bank(make_env, worlds=1, steps=8, warmup=5, stride=1, seed=11)
    torch.manual_seed(99)  # a host whose own landscape differs from the banked one
    env = make_env(1)
    water = ("simpleterrain", "simple_water")
    assert not torch.equal(env.world.td.get(water), bank.states[0][water])
    bank.load(env, [0])
    env.rule_based_step()
    assert torch.equal(env.world.td.get(water), bank.states[1][water]), \
        "after one step the water is the banked run's next water, not the host's"

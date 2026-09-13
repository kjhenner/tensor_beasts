"""Sweep plumbing, tested without running any training.

The expensive part of a sweep is obviously untestable here, but the cheap part
is where the mistakes live: routing a hyperparameter into the wrong config
object silently gives every trial the default value, and the sweep then looks
like it explored a dimension it never touched.
"""

import random

import pytest

import sweep_rl
from tensor_beasts.rl.ppo import PPOConfig
from tensor_beasts.rl.trainer import TrainerConfig


def test_every_searched_parameter_routes_to_a_real_config_field():
    """A typo here means the sweep silently varies nothing."""
    trainer_fields = set(TrainerConfig.__dataclass_fields__)
    ppo_fields = set(PPOConfig.__dataclass_fields__)

    for space_name, space in (("SEARCH_SPACE", sweep_rl.SEARCH_SPACE), ("GRID_SPACE", sweep_rl.GRID_SPACE)):
        split = sweep_rl._split({key: values[0] for key, values in space.items()})
        for key in split["trainer"]:
            assert key in trainer_fields, f"{space_name}: {key} is not a TrainerConfig field"
        for key in split["ppo"]:
            assert key in ppo_fields, f"{space_name}: {key} is not a PPOConfig field"


def test_split_sends_each_parameter_to_exactly_one_config():
    params = {key: values[0] for key, values in sweep_rl.SEARCH_SPACE.items()}
    split = sweep_rl._split(params)
    assert set(split["trainer"]) | set(split["ppo"]) == set(params)
    assert not (set(split["trainer"]) & set(split["ppo"]))


def test_split_configs_actually_construct():
    params = {key: values[0] for key, values in sweep_rl.SEARCH_SPACE.items()}
    split = sweep_rl._split(params)
    TrainerConfig(size=64, **split["trainer"])
    PPOConfig(**split["ppo"])


def test_random_search_is_reproducible_from_the_seed():
    first = [sweep_rl.sample_params(sweep_rl.SEARCH_SPACE, random.Random(7)) for _ in range(3)]
    second = [sweep_rl.sample_params(sweep_rl.SEARCH_SPACE, random.Random(7)) for _ in range(3)]
    assert first == second


def test_random_search_actually_varies():
    rng = random.Random(0)
    samples = [sweep_rl.sample_params(sweep_rl.SEARCH_SPACE, rng) for _ in range(40)]
    varied = {key for key in sweep_rl.SEARCH_SPACE if len({str(s[key]) for s in samples}) > 1}
    assert varied == set(sweep_rl.SEARCH_SPACE), "every dimension should be explored across 40 draws"


def test_grid_is_the_full_product():
    expected = 1
    for values in sweep_rl.GRID_SPACE.values():
        expected *= len(values)
    combos = sweep_rl.grid_params(sweep_rl.GRID_SPACE)
    assert len(combos) == expected
    assert len({tuple(sorted(c.items())) for c in combos}) == expected, "no duplicates"


def test_report_handles_failures_and_an_empty_sweep(capsys):
    sweep_rl.report([])
    sweep_rl.report(
        [
            {"index": 0, "params": {"arch": "conv"}, "ratio": None, "error": "RuntimeError: boom"},
            {
                "index": 1,
                "params": {"arch": "dilated"},
                "ratio": 1.2,
                "learned_return": 120.0,
                "rule_based_return": 100.0,
                "seconds": 60.0,
            },
        ]
    )
    printed = capsys.readouterr().out
    assert "1 trials completed, 1 failed" in printed
    assert "boom" in printed
    assert "beat the rule-based policy" in printed


def test_report_says_so_when_nothing_beats_the_baseline(capsys):
    sweep_rl.report(
        [
            {
                "index": 0,
                "params": {"arch": "conv"},
                "ratio": 0.8,
                "learned_return": 80.0,
                "rule_based_return": 100.0,
                "seconds": 30.0,
            }
        ]
    )
    printed = capsys.readouterr().out
    assert "Nothing beat it" in printed

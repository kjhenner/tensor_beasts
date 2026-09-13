"""
Tests for the policy system.

Verifies that RuleBasedPolicy produces identical outputs to the
original process_observation and metabolic rate calculations.
"""

import pytest
import torch
from types import SimpleNamespace

from tensor_beasts.policy.base import Action
from tensor_beasts.observations import Observation
from tensor_beasts.policy.rule_based import RuleBasedPolicy
from tensor_beasts.observations import get_observation, process_observation


class MockConfig:
    """Mock config that mimics OmegaConf/Pydantic validated config."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture
def sample_config():
    """Create a sample animal config for testing."""
    return MockConfig(
        navigation_weights={
            ("plant", "scent"): 1.0,
            ("predator", "scent"): -0.5,
        },
        basal_rate=2,
        metabolic_sensitivity=2.0,
        max_metabolic_rate=6,
        survival_threshold=32,
        gradient_ema_alpha=0.1,
    )


@pytest.fixture
def sample_tensors():
    """Create sample tensors for testing."""
    torch.manual_seed(42)
    h, w = 16, 16

    # Create directional tensors (4, H, W)
    plant_directional = torch.rand(4, h, w) * 10
    predator_directional = torch.rand(4, h, w) * 5

    # Create current tensors (H, W)
    plant_current = torch.rand(h, w) * 10
    predator_current = torch.rand(h, w) * 5

    # Internal state
    energy = torch.randint(0, 256, (h, w), dtype=torch.uint8)
    biomass = torch.randint(50, 256, (h, w), dtype=torch.uint8)
    gradient_ema = torch.rand(h, w) * 3

    return {
        "directional": {
            ("plant", "scent"): plant_directional,
            ("predator", "scent"): predator_directional,
        },
        "current": {
            ("plant", "scent"): plant_current,
            ("predator", "scent"): predator_current,
        },
        "energy": energy,
        "biomass": biomass,
        "gradient_ema": gradient_ema,
    }


class TestObservation:
    """Tests for Observation construction."""

    def test_observation_structure(self, sample_tensors):
        """Test Observation has all required fields."""
        obs = Observation(
            directional=sample_tensors["directional"],
            current=sample_tensors["current"],
            energy=sample_tensors["energy"],
            biomass=sample_tensors["biomass"],
            gradient_ema=sample_tensors["gradient_ema"],
            alive_mask=sample_tensors["biomass"] > 32,
            step=100,
        )

        assert obs.directional is sample_tensors["directional"]
        assert obs.current is sample_tensors["current"]
        assert torch.equal(obs.energy, sample_tensors["energy"])
        assert torch.equal(obs.biomass, sample_tensors["biomass"])
        assert torch.equal(obs.gradient_ema, sample_tensors["gradient_ema"])
        assert obs.step == 100


class TestRuleBasedPolicy:
    """Tests for RuleBasedPolicy."""

    def test_init_extracts_parameters(self, sample_config):
        """Test that policy correctly extracts config parameters."""
        policy = RuleBasedPolicy(sample_config)

        assert policy.navigation_weights == {
            ("plant", "scent"): 1.0,
            ("predator", "scent"): -0.5,
        }
        assert policy.basal_rate == 2
        assert policy.metabolic_sensitivity == 2.0
        assert policy.max_metabolic_rate == 6
        assert policy.survival_threshold == 32

    def test_call_returns_action(self, sample_config, sample_tensors):
        """Test that policy call returns Action with correct shapes."""
        policy = RuleBasedPolicy(sample_config)

        obs = Observation(
            directional=sample_tensors["directional"],
            current=sample_tensors["current"],
            energy=sample_tensors["energy"],
            biomass=sample_tensors["biomass"],
            gradient_ema=sample_tensors["gradient_ema"],
            alive_mask=sample_tensors["biomass"] > 32,
        )

        output = policy(obs)

        assert isinstance(output, Action)
        assert output.move_direction.shape == (16, 16)
        assert output.move_probability.shape == (16, 16)
        assert output.metabolic_rate.shape == (16, 16)
        assert output.gradient_ema.shape == (16, 16)

    def test_move_probability_from_energy(self, sample_config, sample_tensors):
        """Test that move probability is energy / 255."""
        policy = RuleBasedPolicy(sample_config)

        obs = Observation(
            directional=sample_tensors["directional"],
            current=sample_tensors["current"],
            energy=sample_tensors["energy"],
            biomass=sample_tensors["biomass"],
            gradient_ema=sample_tensors["gradient_ema"],
            alive_mask=sample_tensors["biomass"] > 32,
        )

        output = policy(obs)

        expected_prob = sample_tensors["energy"].float() / 255.0
        torch.testing.assert_close(output.move_probability, expected_prob)

    def test_metabolic_rate_bounded_by_biomass(self, sample_config):
        """Test that metabolic rate is capped by biomass-dependent max."""
        policy = RuleBasedPolicy(sample_config)

        # Create observation with low biomass (near survival threshold)
        low_biomass = torch.full((4, 4), 40, dtype=torch.uint8)  # Just above threshold=32
        high_gradient_ema = torch.full((4, 4), 10.0)  # Would give high rate

        obs = Observation(
            directional={
                ("plant", "scent"): torch.zeros(4, 4, 4),
                ("predator", "scent"): torch.zeros(4, 4, 4),
            },
            current={
                ("plant", "scent"): torch.zeros(4, 4),
                ("predator", "scent"): torch.zeros(4, 4),
            },
            energy=torch.full((4, 4), 100, dtype=torch.uint8),
            biomass=low_biomass,
            gradient_ema=high_gradient_ema,
            alive_mask=low_biomass > 32,
        )

        output = policy(obs)

        # At biomass=40, threshold=32, range=223
        # biomass_fraction = (40-32)/223 ≈ 0.036
        # effective_max = 2 + (6-2)*0.036 ≈ 2.14
        # rate = 2 + 10*2 = 22 -> clamped to ~2.14
        assert (output.metabolic_rate < 3).all(), "Rate should be capped by low biomass"

    def test_direction_uses_signed_weights(self, sample_config):
        """Test that direction calculation uses signed weights correctly."""
        policy = RuleBasedPolicy(sample_config)

        # Create scenario where predator scent is in one direction
        # and plant scent in another
        h, w = 4, 4
        directional = {
            ("plant", "scent"): torch.zeros(4, h, w),
            ("predator", "scent"): torch.zeros(4, h, w),
        }
        current = {
            ("plant", "scent"): torch.zeros(h, w),
            ("predator", "scent"): torch.zeros(h, w),
        }

        # Put strong plant scent to the right (direction 4)
        directional[("plant", "scent")][3] = 10.0  # right direction

        # Put strong predator scent to the left (direction 3)
        # With negative weight -0.5, this should repel (make left less attractive)
        directional[("predator", "scent")][2] = 10.0  # left direction

        obs = Observation(
            directional=directional,
            current=current,
            energy=torch.full((h, w), 100, dtype=torch.uint8),
            biomass=torch.full((h, w), 100, dtype=torch.uint8),
            gradient_ema=torch.zeros(h, w),
            alive_mask=torch.ones(h, w, dtype=torch.bool),
        )

        output = policy(obs)

        # Should move right (4) toward plant, away from predator
        # The predator scent to the left gets negative weight, so left becomes less attractive
        assert (output.move_direction == 4).all(), "Should move toward plant (right)"


class TestPolicyParameterAccess:
    """Tests for get_parameters / set_parameters."""

    def test_get_parameters_returns_tensors(self, sample_config):
        """Test that get_parameters returns dict of tensors."""
        policy = RuleBasedPolicy(sample_config)
        params = policy.get_parameters()

        assert "navigation_weights" in params
        assert "basal_rate" in params
        assert "metabolic_sensitivity" in params
        assert "max_metabolic_rate" in params

        assert isinstance(params["navigation_weights"], torch.Tensor)
        assert params["navigation_weights"].shape == (2,)  # 2 weights

    def test_set_parameters_modifies_policy(self, sample_config):
        """Test that set_parameters changes policy behavior."""
        policy = RuleBasedPolicy(sample_config)

        # Get original parameters
        original_params = policy.get_parameters()

        # Modify parameters
        new_params = {
            "navigation_weights": torch.tensor([2.0, -1.0]),
            "navigation_weight_keys": original_params["navigation_weight_keys"],
            "basal_rate": torch.tensor(5.0),
            "metabolic_sensitivity": torch.tensor(3.0),
            "max_metabolic_rate": torch.tensor(10.0),
        }
        policy.set_parameters(new_params)

        # Verify changes
        assert policy.basal_rate == 5
        assert policy.metabolic_sensitivity == 3.0
        assert policy.max_metabolic_rate == 10

    def test_parameter_roundtrip(self, sample_config, sample_tensors):
        """Test that get/set parameters preserves behavior."""
        policy1 = RuleBasedPolicy(sample_config)
        policy2 = RuleBasedPolicy(sample_config)

        # Get parameters from policy1
        params = policy1.get_parameters()

        # Set on policy2
        policy2.set_parameters(params)

        # Create observation
        obs = Observation(
            directional=sample_tensors["directional"],
            current=sample_tensors["current"],
            energy=sample_tensors["energy"],
            biomass=sample_tensors["biomass"],
            gradient_ema=sample_tensors["gradient_ema"],
            alive_mask=sample_tensors["biomass"] > 32,
        )

        # Both should produce same output
        output1 = policy1(obs)
        output2 = policy2(obs)

        torch.testing.assert_close(output1.move_direction, output2.move_direction)
        torch.testing.assert_close(output1.move_probability, output2.move_probability)
        torch.testing.assert_close(output1.metabolic_rate, output2.metabolic_rate)
        torch.testing.assert_close(output1.gradient_ema, output2.gradient_ema)


class TestProcessObservationMatch:
    """Tests verifying RuleBasedPolicy matches original process_observation."""

    def test_gradient_ema_update(self, sample_config, sample_tensors):
        """Test that policy correctly computes gradient EMA."""
        # Create mock observation for process_observation
        from dataclasses import dataclass

        @dataclass
        class MockObs:
            directional: dict
            current: dict
            energy: torch.Tensor
            biomass: torch.Tensor

        mock_obs = MockObs(
            directional=sample_tensors["directional"],
            current=sample_tensors["current"],
            energy=sample_tensors["energy"],
            biomass=sample_tensors["biomass"],
        )

        # Get gradient_strength from original process_observation
        weights = {
            ("plant", "scent"): 1.0,
            ("predator", "scent"): -0.5,
        }
        original_gradient, original_direction = process_observation(mock_obs, weights)

        # Get result from policy
        policy = RuleBasedPolicy(sample_config)
        alive_mask = sample_tensors["biomass"] > 32
        obs = Observation(
            directional=sample_tensors["directional"],
            current=sample_tensors["current"],
            energy=sample_tensors["energy"],
            biomass=sample_tensors["biomass"],
            gradient_ema=sample_tensors["gradient_ema"],
            alive_mask=alive_mask,
        )
        output = policy(obs)

        # Verify gradient EMA is computed correctly
        alpha = sample_config.gradient_ema_alpha
        expected_ema = torch.where(
            alive_mask,
            alpha * original_gradient + (1 - alpha) * sample_tensors["gradient_ema"],
            sample_tensors["gradient_ema"]
        )
        torch.testing.assert_close(
            output.gradient_ema,
            expected_ema,
            msg="Policy gradient_ema should match EMA formula"
        )

    def test_direction_distribution_matches(self, sample_config, sample_tensors):
        """Test that policy direction has same distribution as process_observation."""
        # Due to random tie-breaking, exact values may differ
        # But the direction should be correct when there's a clear winner

        # Create observation with clear directional preference
        h, w = 8, 8
        directional = {
            ("plant", "scent"): torch.zeros(4, h, w),
            ("predator", "scent"): torch.zeros(4, h, w),
        }
        current = {
            ("plant", "scent"): torch.zeros(h, w),
            ("predator", "scent"): torch.zeros(h, w),
        }

        # Strong plant scent up (direction 1)
        directional[("plant", "scent")][0] = 100.0  # up

        from dataclasses import dataclass

        @dataclass
        class MockObs:
            directional: dict
            current: dict
            energy: torch.Tensor
            biomass: torch.Tensor

        mock_obs = MockObs(
            directional=directional,
            current=current,
            energy=torch.full((h, w), 100, dtype=torch.uint8),
            biomass=torch.full((h, w), 100, dtype=torch.uint8),
        )

        weights = {
            ("plant", "scent"): 1.0,
            ("predator", "scent"): -0.5,
        }
        _, original_direction = process_observation(mock_obs, weights)

        policy = RuleBasedPolicy(sample_config)
        obs = Observation(
            directional=directional,
            current=current,
            energy=torch.full((h, w), 100, dtype=torch.uint8),
            biomass=torch.full((h, w), 100, dtype=torch.uint8),
            gradient_ema=torch.zeros(h, w),
            alive_mask=torch.ones(h, w, dtype=torch.bool),
        )
        output = policy(obs)

        # With clear winner (up), both should choose up (1)
        assert (original_direction == 1).all()
        assert (output.move_direction == 1).all()

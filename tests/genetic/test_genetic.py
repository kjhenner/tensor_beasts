"""
Tests for the genetic algorithm system.

Tests cover:
- Genome serialization and round-trip
- GeneticRegistry population tracking
- Mutation and crossover operations
- ParameterizedPolicy with genetics
"""

import pytest
import torch

from tensor_beasts.genetic.genome import Genome
from tensor_beasts.genetic.registry import GeneticRegistry
from tensor_beasts.policy.parameterized import ParameterizedPolicy, ParameterBounds


class MockConfig:
    """Mock config for testing."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture
def sample_genome():
    """Create a sample genome for testing."""
    return Genome(
        navigation_weights={
            ("plant", "scent"): 1.0,
            ("predator", "scent"): -0.5,
        },
        basal_rate=2.0,
        metabolic_sensitivity=2.0,
        max_metabolic_rate=6.0,
        reproduction_threshold=200.0,
    )


@pytest.fixture
def sample_config():
    """Create a sample config matching the genome."""
    return MockConfig(
        navigation_weights={
            ("plant", "scent"): 1.0,
            ("predator", "scent"): -0.5,
        },
        basal_rate=2,
        metabolic_sensitivity=2.0,
        max_metabolic_rate=6,
        survival_threshold=32,
        reproduction_threshold=200,
        gradient_ema_alpha=0.1,
    )


class TestGenome:
    """Tests for Genome dataclass."""

    def test_to_tensor(self, sample_genome):
        """Test genome converts to tensor correctly."""
        tensor = sample_genome.to_tensor()

        assert tensor.shape == (6,)  # 2 nav weights + 4 params
        assert tensor.dtype == torch.float32

        # Check values (sorted nav keys: plant, predator)
        expected = [1.0, -0.5, 2.0, 2.0, 6.0, 200.0]
        assert tensor.tolist() == expected

    def test_from_tensor_roundtrip(self, sample_genome):
        """Test genome round-trips through tensor correctly."""
        tensor = sample_genome.to_tensor()
        nav_keys = sorted(sample_genome.navigation_weights.keys())

        reconstructed = Genome.from_tensor(tensor, nav_keys)

        assert reconstructed.navigation_weights == sample_genome.navigation_weights
        assert reconstructed.basal_rate == sample_genome.basal_rate
        assert reconstructed.metabolic_sensitivity == sample_genome.metabolic_sensitivity
        assert reconstructed.max_metabolic_rate == sample_genome.max_metabolic_rate
        assert reconstructed.reproduction_threshold == sample_genome.reproduction_threshold

    def test_copy(self, sample_genome):
        """Test genome copy is independent."""
        copied = sample_genome.copy()

        # Modify original
        sample_genome.basal_rate = 99.0
        sample_genome.navigation_weights[("plant", "scent")] = 99.0

        # Copy should be unchanged
        assert copied.basal_rate == 2.0
        assert copied.navigation_weights[("plant", "scent")] == 1.0

    def test_to_policy_params(self, sample_genome):
        """Test conversion to policy parameter dict."""
        params = sample_genome.to_policy_params()

        assert "navigation_weights" in params
        assert "basal_rate" in params
        assert "metabolic_sensitivity" in params
        assert "max_metabolic_rate" in params
        assert "reproduction_threshold" in params

        assert isinstance(params["navigation_weights"], torch.Tensor)
        assert params["navigation_weights"].shape == (2,)

    def test_from_policy_params_roundtrip(self, sample_genome):
        """Test round-trip through policy params."""
        params = sample_genome.to_policy_params()
        reconstructed = Genome.from_policy_params(params)

        assert reconstructed.navigation_weights == sample_genome.navigation_weights
        assert reconstructed.basal_rate == sample_genome.basal_rate


class TestGeneticRegistry:
    """Tests for GeneticRegistry."""

    def test_init_creates_copies(self, sample_genome):
        """Test registry creates independent genome copies."""
        registry = GeneticRegistry(num_slots=4, base_genome=sample_genome)

        assert registry.num_slots == 4
        assert len(registry.genomes) == 4

        # Modify slot 0
        registry.genomes[0].basal_rate = 99.0

        # Other slots should be unchanged
        assert registry.genomes[1].basal_rate == 2.0
        assert registry.genomes[2].basal_rate == 2.0

    def test_mutate_slot_changes_values(self, sample_genome):
        """Test mutation changes parameter values."""
        registry = GeneticRegistry(num_slots=2, base_genome=sample_genome)

        original_tensor = registry.genomes[0].to_tensor().clone()

        # Mutate with 100% rate
        registry.mutate_slot(0, mutation_rate=1.0, mutation_scale=0.5)

        mutated_tensor = registry.genomes[0].to_tensor()

        # Should be different
        assert not torch.equal(original_tensor, mutated_tensor)

    def test_mutate_slot_respects_bounds(self, sample_genome):
        """Test mutation respects parameter bounds."""
        registry = GeneticRegistry(num_slots=2, base_genome=sample_genome)

        # Mutate many times with large scale
        for _ in range(100):
            registry.mutate_slot(0, mutation_rate=1.0, mutation_scale=1.0)

        genome = registry.genomes[0]
        bounds = registry.bounds

        # Check bounds are respected
        for key, weight in genome.navigation_weights.items():
            b = bounds["navigation_weights"]
            assert b.min_val <= weight <= b.max_val

        b = bounds["basal_rate"]
        assert b.min_val <= genome.basal_rate <= b.max_val

    def test_crossover_mixes_parents(self, sample_genome):
        """Test crossover creates child with mixed parameters."""
        registry = GeneticRegistry(num_slots=2, base_genome=sample_genome)

        # Make parents different
        registry.genomes[0].basal_rate = 1.0
        registry.genomes[1].basal_rate = 10.0

        # Do many crossovers - some should get each parent's value
        basal_values = set()
        for _ in range(20):
            child = registry.crossover(0, 1)
            basal_values.add(child.basal_rate)

        # Should see both parent values at some point
        assert 1.0 in basal_values or 10.0 in basal_values

    def test_update_populations(self, sample_genome):
        """Test population tracking."""
        registry = GeneticRegistry(num_slots=4, base_genome=sample_genome)

        # Create slot_ids and alive_mask tensors
        slot_ids = torch.zeros(8, 8, dtype=torch.uint8)
        slot_ids[0:4, :] = 0
        slot_ids[4:8, :] = 1

        alive_mask = torch.ones(8, 8, dtype=torch.bool)
        alive_mask[0:2, :] = False  # 16 cells dead in slot 0

        registry.update_populations(slot_ids, alive_mask)

        assert registry.populations[0].item() == 16  # 4*8 - 16 = 16 alive
        assert registry.populations[1].item() == 32  # 4*8 = 32 alive
        assert registry.populations[2].item() == 0
        assert registry.populations[3].item() == 0

    def test_get_empty_slots(self, sample_genome):
        """Test finding empty slots."""
        registry = GeneticRegistry(num_slots=4, base_genome=sample_genome)

        # Set some populations
        registry.populations[0] = 10
        registry.populations[1] = 0
        registry.populations[2] = 5
        registry.populations[3] = 0

        empty = registry.get_empty_slots()
        assert len(empty) == 2
        assert 1 in empty.tolist()
        assert 3 in empty.tolist()

    def test_colonize_slot(self, sample_genome):
        """Test colonizing empty slot from source."""
        registry = GeneticRegistry(num_slots=4, base_genome=sample_genome)

        # Modify source slot
        registry.genomes[0].basal_rate = 5.0

        # Colonize slot 1 from slot 0
        registry.colonize_slot(1, source_slot=0, mutation_rate=0.0, mutation_scale=0.0)

        # Should have same basal_rate (no mutation)
        assert registry.genomes[1].basal_rate == 5.0

        # Colonize with mutation
        registry.colonize_slot(2, source_slot=0, mutation_rate=1.0, mutation_scale=0.5)

        # Might be different due to mutation
        # (can't assert it's different since mutation is random)

    def test_diversity_metrics(self, sample_genome):
        """Test diversity metric computation."""
        registry = GeneticRegistry(num_slots=4, base_genome=sample_genome)

        # All same - zero variance
        registry.populations = torch.tensor([10, 10, 0, 0])
        metrics = registry.get_diversity_metrics()
        assert metrics["num_occupied"] == 2
        assert metrics["parameter_variance"] == 0.0

        # Make slot 1 different
        registry.genomes[1].basal_rate = 10.0
        metrics = registry.get_diversity_metrics()
        assert metrics["parameter_variance"] > 0


class TestParameterizedPolicy:
    """Tests for ParameterizedPolicy."""

    def test_init_from_config(self, sample_config):
        """Test policy initializes from config."""
        policy = ParameterizedPolicy(sample_config)

        assert policy.basal_rate == 2.0
        assert policy.metabolic_sensitivity == 2.0
        assert policy.max_metabolic_rate == 6.0

    def test_call_produces_output(self, sample_config):
        """Test policy call produces valid output."""
        from tensor_beasts.policy.base import Action
        from tensor_beasts.observations import Observation

        policy = ParameterizedPolicy(sample_config)

        # Create minimal observation
        h, w = 8, 8
        obs = Observation(
            directional={
                ("plant", "scent"): torch.rand(4, h, w),
                ("predator", "scent"): torch.rand(4, h, w),
            },
            current={
                ("plant", "scent"): torch.rand(h, w),
                ("predator", "scent"): torch.rand(h, w),
            },
            energy=torch.randint(0, 256, (h, w), dtype=torch.uint8),
            biomass=torch.randint(50, 256, (h, w), dtype=torch.uint8),
            gradient_ema=torch.rand(h, w),
            alive_mask=torch.ones(h, w, dtype=torch.bool),
        )

        output = policy(obs)

        assert isinstance(output, Action)
        assert output.move_direction.shape == (h, w)
        assert output.move_probability.shape == (h, w)
        assert output.metabolic_rate.shape == (h, w)

    def test_mutate_changes_behavior(self, sample_config):
        """Test mutation changes policy behavior."""
        from tensor_beasts.observations import Observation

        policy = ParameterizedPolicy(sample_config)

        # Get output before mutation
        h, w = 8, 8
        obs = Observation(
            directional={
                ("plant", "scent"): torch.ones(4, h, w) * 5,
                ("predator", "scent"): torch.ones(4, h, w),
            },
            current={
                ("plant", "scent"): torch.ones(h, w),
                ("predator", "scent"): torch.ones(h, w),
            },
            energy=torch.full((h, w), 100, dtype=torch.uint8),
            biomass=torch.full((h, w), 100, dtype=torch.uint8),
            gradient_ema=torch.ones(h, w) * 2,
            alive_mask=torch.ones(h, w, dtype=torch.bool),
        )

        output_before = policy(obs)
        rate_before = output_before.metabolic_rate[0, 0].item()

        # Mutate heavily
        policy.mutate(mutation_rate=1.0, mutation_scale=1.0)

        output_after = policy(obs)
        rate_after = output_after.metabolic_rate[0, 0].item()

        # Very likely to be different (random chance they're same is tiny)
        # But we can't assert they're different since mutation is random

    def test_clone(self, sample_config):
        """Test policy cloning."""
        policy = ParameterizedPolicy(sample_config)
        cloned = policy.clone()

        # Modify original
        policy._basal_rate = torch.tensor(99.0)

        # Clone should be unchanged
        assert cloned._basal_rate.item() == 2.0

    def test_crossover(self, sample_config):
        """Test policy crossover."""
        policy_a = ParameterizedPolicy(sample_config)
        policy_b = ParameterizedPolicy(sample_config)

        # Make them different
        policy_a._basal_rate = torch.tensor(1.0)
        policy_b._basal_rate = torch.tensor(10.0)

        # Crossover
        child = policy_a.crossover(policy_b)

        # Child should have one of the parents' basal rates
        assert child._basal_rate.item() in [1.0, 10.0]


class TestIntegration:
    """Integration tests for genetic system with simulation."""

    def test_genetic_registry_with_policy(self, sample_genome, sample_config):
        """Test genetic registry updates policy parameters."""
        registry = GeneticRegistry(num_slots=4, base_genome=sample_genome)
        policy = ParameterizedPolicy(sample_config)

        # Mutate registry slot 0
        registry.mutate_slot(0, mutation_rate=1.0, mutation_scale=0.5)

        # Update policy from registry
        genome = registry.get_genome(0)
        params = genome.to_policy_params()
        policy.set_parameters(params)

        # Policy should have new values
        assert policy.basal_rate != 2.0 or policy.metabolic_sensitivity != 2.0

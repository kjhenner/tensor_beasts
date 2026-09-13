"""
Genetic algorithm module for evolutionary dynamics.

This module provides:
- Genome: Storage for evolvable parameters
- GeneticRegistry: Management of per-slot genomes
- Mutation and crossover operators
"""

from tensor_beasts.genetic.genome import Genome
from tensor_beasts.genetic.registry import GeneticRegistry

__all__ = [
    "Genome",
    "GeneticRegistry",
]

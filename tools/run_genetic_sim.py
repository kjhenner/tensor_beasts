#!/usr/bin/env python
"""
Run simulation with genetic algorithm and monitor evolution.

Usage:
    source venv/bin/activate && python tools/run_genetic_sim.py
"""

import torch
from tensor_beasts.world import World
from tensor_beasts.config.load import load_config
from omegaconf import OmegaConf


def main():
    # Load base config
    config = load_config('conf/base/simulation.yaml')

    # Enable genetics - use full size for stable populations
    config.world.size = [256, 256]

    # More herbivores, higher reproduction for population stability
    config.world.entities.Herbivore.init_prob = 0.02
    config.world.entities.Herbivore.reproduction_threshold = 180  # Easier reproduction
    config.world.entities.Herbivore.genetics = OmegaConf.create({
        'enabled': True,
        'num_slots': 8,
        'mutation_probability': 0.15,
        'mutation_rate': 0.2,
        'mutation_scale': 0.15,
        'log_interval': 100,  # Log every 100 steps
    })

    # Fewer predators to let herbivore population grow
    config.world.entities.Predator.init_prob = 0.0005
    config.world.entities.Predator.genetics = OmegaConf.create({
        'enabled': True,
        'num_slots': 4,
        'mutation_probability': 0.1,
        'mutation_rate': 0.2,
        'mutation_scale': 0.15,
        'log_interval': 100,  # Log every 100 steps
    })

    # Create world
    world = World(config.world)
    world.initialize()

    # Workaround: SharedFeature initialization zeros energy - re-init plants
    plant = world.entity_dict.get('SimplePlant')
    spawn_mask = torch.rand(plant.energy.data.shape) < plant.config.init_prob
    plant.energy.data[:] = spawn_mask.to(torch.uint8) * plant.config.initial_energy

    herbivore = world.entity_dict.get('Herbivore')
    predator = world.entity_dict.get('Predator')

    print("=== Genetic Simulation Started ===")
    print(f"World size: {config.world.size}")
    print(f"Herbivore slots: {herbivore.genetic_registry.num_slots}")
    print(f"Predator slots: {predator.genetic_registry.num_slots}")

    # Run simulation - logging happens automatically via log_interval
    num_steps = 1000
    for step in range(num_steps):
        world.update()

    print("\n=== Simulation Complete ===")


if __name__ == "__main__":
    main()

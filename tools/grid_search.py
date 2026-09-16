"""
Grid search for simulation parameters.

Goal: Find parameters that produce stable/cyclic herbivore populations
with active movement (no camping on plants).
"""

import json
import time
from dataclasses import dataclass
from itertools import product
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Any

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from tensor_beasts.config import load_config
from tensor_beasts.world import World


# Parameter grid to explore (reduced for faster iteration)
PARAM_GRID = {
    # Herbivore metabolism
    "herbivore.basal_rate": [1, 2],
    "herbivore.max_metabolic_rate": [4, 6],
    "herbivore.gradient_ema_alpha": [0.1, 0.2],
    # Herbivore survival
    "herbivore.eat_max": [3, 5],
    "herbivore.dissipation_rate": [0.02, 0.04],
    "herbivore.reproduction_threshold": [200, 250],
    # Plant
    "plant.nutrient_consumption_rate": [0.3, 0.5],
}

# Simulation settings
WORLD_SIZE = 128
SIMULATION_STEPS = 2000
BURN_IN_STEPS = 200
SAMPLE_INTERVAL = 10


@dataclass
class TrialMetrics:
    """Metrics from a single trial."""
    extinction: bool
    min_pop: int
    max_pop: int
    mean_pop: float
    std_pop: float
    cv: float  # coefficient of variation
    mean_movement_rate: float  # fraction of herbivores that moved
    mean_gradient_ema: float  # average gradient following
    score: float


def create_config(params: Dict[str, Any]) -> Any:
    """Create a config with the given parameters."""
    # Start from basic_config but override size
    config = load_config("basic_config.yaml")

    # Override world size
    config.world.size = [WORLD_SIZE, WORLD_SIZE]

    # Apply parameter overrides
    for key, value in params.items():
        parts = key.split(".")
        if parts[0] == "herbivore":
            OmegaConf.update(config, f"world.entities.Herbivore.{parts[1]}", value)
        elif parts[0] == "plant":
            OmegaConf.update(config, f"world.entities.SimplePlant.{parts[1]}", value)
        elif parts[0] == "terrain":
            OmegaConf.update(config, f"world.entities.SimpleTerrain.{parts[1]}", value)

    return config


def run_trial(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single trial with given parameters."""
    try:
        config = create_config(params)
        world = World(config.world)

        # Track population and movement over time
        pop_history = []
        movement_rates = []
        gradient_emas = []

        prev_positions = None

        for step in range(SIMULATION_STEPS):
            world.update()

            if step >= BURN_IN_STEPS and step % SAMPLE_INTERVAL == 0:
                herbivore_mask = world.Herbivore.biomass.data > 0
                pop = herbivore_mask.sum().item()
                pop_history.append(pop)

                # Track movement: compare positions to previous sample
                current_positions = herbivore_mask.clone()
                if prev_positions is not None and pop > 0:
                    # Cells that changed (either gained or lost herbivore)
                    changed = (current_positions != prev_positions).sum().item()
                    # Rough movement rate: changed cells / (2 * population)
                    # Factor of 2 because each move creates 2 changes (leave + arrive)
                    movement_rate = min(1.0, changed / (2 * max(pop, 1)))
                    movement_rates.append(movement_rate)
                prev_positions = current_positions

                # Track gradient EMA (indicates active pursuit)
                if pop > 0:
                    ema = world.Herbivore.gradient_ema.data[herbivore_mask].mean().item()
                    gradient_emas.append(ema)

                # Early termination on extinction
                if pop == 0:
                    break

        # Compute metrics
        metrics = compute_metrics(pop_history, movement_rates, gradient_emas)

        return {
            "params": params,
            "metrics": metrics.__dict__,
            "pop_history": pop_history[-20:],  # Last 20 samples for debugging
        }

    except Exception as e:
        return {
            "params": params,
            "metrics": {"error": str(e), "score": 0},
            "pop_history": [],
        }


def compute_metrics(
    pop_history: List[int],
    movement_rates: List[float],
    gradient_emas: List[float]
) -> TrialMetrics:
    """Compute fitness metrics from trial data."""

    if not pop_history or pop_history[-1] == 0:
        return TrialMetrics(
            extinction=True,
            min_pop=0,
            max_pop=max(pop_history) if pop_history else 0,
            mean_pop=0,
            std_pop=0,
            cv=float('inf'),
            mean_movement_rate=0,
            mean_gradient_ema=0,
            score=0,
        )

    min_pop = min(pop_history)
    max_pop = max(pop_history)
    mean_pop = sum(pop_history) / len(pop_history)

    # Standard deviation
    variance = sum((p - mean_pop) ** 2 for p in pop_history) / len(pop_history)
    std_pop = variance ** 0.5

    # Coefficient of variation (lower = more stable)
    cv = std_pop / mean_pop if mean_pop > 0 else float('inf')

    # Movement metrics
    mean_movement = sum(movement_rates) / len(movement_rates) if movement_rates else 0
    mean_ema = sum(gradient_emas) / len(gradient_emas) if gradient_emas else 0

    # Compute score
    # Reward: population size, stability (low cv), movement
    # Penalize: near-extinction (low min_pop), no movement

    if min_pop < 5:
        # Near extinction penalty
        score = mean_pop * 0.1
    else:
        stability_factor = 1 / (1 + cv)  # Higher when cv is low
        movement_factor = 0.5 + 0.5 * min(mean_movement * 2, 1)  # Bonus for movement
        score = mean_pop * stability_factor * movement_factor

    return TrialMetrics(
        extinction=False,
        min_pop=min_pop,
        max_pop=max_pop,
        mean_pop=mean_pop,
        std_pop=std_pop,
        cv=cv,
        mean_movement_rate=mean_movement,
        mean_gradient_ema=mean_ema,
        score=score,
    )


def generate_param_combinations() -> List[Dict[str, Any]]:
    """Generate all parameter combinations from the grid."""
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())

    combinations = []
    for combo in product(*values):
        params = dict(zip(keys, combo))
        combinations.append(params)

    return combinations


def main():
    print("Grid Search for Simulation Parameters")
    print("=" * 50)

    combinations = generate_param_combinations()
    print(f"Total combinations: {len(combinations)}")
    print(f"World size: {WORLD_SIZE}x{WORLD_SIZE}")
    print(f"Steps: {SIMULATION_STEPS} (burn-in: {BURN_IN_STEPS})")
    print()

    # Run trials
    num_workers = max(1, cpu_count() - 1)
    print(f"Running with {num_workers} workers...")

    start_time = time.time()

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(run_trial, combinations),
            total=len(combinations),
            desc="Trials"
        ))

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/len(combinations):.2f}s per trial)")
    print()

    # Sort by score
    results.sort(key=lambda x: x["metrics"].get("score", 0), reverse=True)

    # Save full results
    output_path = Path("tools/grid_search_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to {output_path}")
    print()

    # Print top results
    print("Top 10 Configurations:")
    print("-" * 80)
    for i, r in enumerate(results[:10]):
        m = r["metrics"]
        if "error" in m:
            print(f"{i+1}. ERROR: {m['error']}")
            continue
        print(f"{i+1}. Score: {m['score']:.1f}")
        print(f"   Pop: {m['mean_pop']:.0f} (min={m['min_pop']}, max={m['max_pop']}, cv={m['cv']:.2f})")
        print(f"   Movement: {m['mean_movement_rate']:.2f}, Gradient EMA: {m['mean_gradient_ema']:.1f}")
        print(f"   Params: {r['params']}")
        print()

    # Print extinction rate
    extinctions = sum(1 for r in results if r["metrics"].get("extinction", True))
    print(f"Extinction rate: {extinctions}/{len(results)} ({100*extinctions/len(results):.1f}%)")

    # Print worst results (for debugging)
    print()
    print("Bottom 5 (excluding extinctions):")
    print("-" * 80)
    non_extinct = [r for r in results if not r["metrics"].get("extinction", True)]
    for r in non_extinct[-5:]:
        m = r["metrics"]
        print(f"   Score: {m['score']:.1f}, Pop: {m['mean_pop']:.0f}, Movement: {m['mean_movement_rate']:.2f}")
        print(f"   Params: {r['params']}")


if __name__ == "__main__":
    main()

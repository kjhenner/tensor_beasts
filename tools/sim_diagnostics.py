"""
Diagnostic script for simulation analysis.

Runs a simulation and logs detailed stats at regular intervals to understand
ecosystem dynamics (herbivore/plant populations, energy flows, movement, etc.)
"""

import argparse
from dataclasses import dataclass
from typing import Optional

import torch
from omegaconf import OmegaConf

from tensor_beasts.config import load_config
from tensor_beasts.world import World


@dataclass
class StepStats:
    step: int
    # Herbivore stats
    herb_count: int
    herb_total_biomass: int
    herb_avg_biomass: float
    herb_total_energy: int
    herb_avg_energy: float
    herb_reproductions: int
    herb_avg_gradient_ema: float
    herb_on_food: int  # herbivores currently on plant cells
    herb_eating_pct: float  # percentage of herbivores on food
    # Predator stats
    pred_count: int
    pred_total_biomass: int
    pred_avg_biomass: float
    pred_total_energy: int
    pred_avg_energy: float
    pred_reproductions: int
    pred_on_food: int
    pred_eating_pct: float
    pred_on_carrion: int  # predators on carrion cells
    pred_avg_gradient_ema: float
    # Plant stats
    plant_count: int
    plant_total_energy: int
    plant_avg_energy: float
    # Terrain stats
    total_nutrients: float
    total_carrion: float  # carrion available for scavenging


def collect_stats(world: World, step: int, prev_herb_count: int, prev_pred_count: int) -> StepStats:
    """Collect stats from current world state."""
    # Herbivore stats
    herb_biomass = world.Herbivore.biomass.data
    herb_energy = world.Herbivore.energy.data
    herb_mask = herb_biomass > 0
    herb_count = herb_mask.sum().item()

    # Plant stats (need this early for eating calculation)
    plant_energy = world.SimplePlant.energy.data
    plant_mask = plant_energy > 0
    plant_count = plant_mask.sum().item()
    plant_total_energy = plant_energy.sum().item()
    plant_avg_energy = plant_energy[plant_mask].float().mean().item() if plant_count > 0 else 0

    if herb_count > 0:
        herb_total_biomass = herb_biomass.sum().item()
        herb_avg_biomass = herb_biomass[herb_mask].float().mean().item()
        herb_total_energy = herb_energy[herb_mask].sum().item()
        herb_avg_energy = herb_energy[herb_mask].float().mean().item()
        herb_avg_gradient_ema = world.Herbivore.gradient_ema.data[herb_mask].float().mean().item()
        # Count herbivores on food
        herb_on_food = (herb_mask & plant_mask).sum().item()
        herb_eating_pct = 100.0 * herb_on_food / herb_count
    else:
        herb_total_biomass = 0
        herb_avg_biomass = 0
        herb_total_energy = 0
        herb_avg_energy = 0
        herb_avg_gradient_ema = 0
        herb_on_food = 0
        herb_eating_pct = 0

    # Estimate reproductions (new herbivores since last step)
    herb_reproductions = max(0, herb_count - prev_herb_count) if prev_herb_count > 0 else 0

    # Predator stats (if present)
    pred_count = 0
    pred_total_biomass = 0
    pred_avg_biomass = 0
    pred_total_energy = 0
    pred_avg_energy = 0
    pred_reproductions = 0
    pred_on_food = 0
    pred_eating_pct = 0
    pred_on_carrion = 0
    pred_avg_gradient_ema = 0
    pred_mask = None

    if hasattr(world, 'Predator'):
        pred_biomass = world.Predator.biomass.data
        pred_energy = world.Predator.energy.data
        pred_mask = pred_biomass > 0
        pred_count = pred_mask.sum().item()

        if pred_count > 0:
            pred_total_biomass = pred_biomass.sum().item()
            pred_avg_biomass = pred_biomass[pred_mask].float().mean().item()
            pred_total_energy = pred_energy[pred_mask].sum().item()
            pred_avg_energy = pred_energy[pred_mask].float().mean().item()
            pred_avg_gradient_ema = world.Predator.gradient_ema.data[pred_mask].float().mean().item()
            # Predators eat herbivores
            pred_on_food = (pred_mask & herb_mask).sum().item()
            pred_eating_pct = 100.0 * pred_on_food / pred_count
        else:
            pred_avg_gradient_ema = 0

        pred_reproductions = max(0, pred_count - prev_pred_count) if prev_pred_count > 0 else 0

    # Terrain stats
    total_nutrients = 0
    total_carrion = 0
    terrain = world.SimpleTerrain
    if hasattr(terrain, 'nutrients'):
        total_nutrients = terrain.nutrients.data.sum().item()
    if hasattr(terrain, 'carrion'):
        carrion_data = terrain.carrion.data
        total_carrion = carrion_data.sum().item()
        carrion_mask = carrion_data > 0
        # Count predators on carrion
        if pred_count > 0:
            pred_on_carrion = (pred_mask & carrion_mask).sum().item()
        else:
            pred_on_carrion = 0
    else:
        pred_on_carrion = 0

    return StepStats(
        step=step,
        herb_count=herb_count,
        herb_total_biomass=herb_total_biomass,
        herb_avg_biomass=herb_avg_biomass,
        herb_total_energy=herb_total_energy,
        herb_avg_energy=herb_avg_energy,
        herb_reproductions=herb_reproductions,
        herb_avg_gradient_ema=herb_avg_gradient_ema,
        herb_on_food=herb_on_food,
        herb_eating_pct=herb_eating_pct,
        pred_count=pred_count,
        pred_total_biomass=pred_total_biomass,
        pred_avg_biomass=pred_avg_biomass,
        pred_total_energy=pred_total_energy,
        pred_avg_energy=pred_avg_energy,
        pred_reproductions=pred_reproductions,
        pred_on_food=pred_on_food,
        pred_eating_pct=pred_eating_pct,
        pred_on_carrion=pred_on_carrion,
        pred_avg_gradient_ema=pred_avg_gradient_ema,
        plant_count=plant_count,
        plant_total_energy=plant_total_energy,
        plant_avg_energy=plant_avg_energy,
        total_nutrients=total_nutrients,
        total_carrion=total_carrion,
    )


def print_stats(stats: StepStats):
    """Print stats in a readable format."""
    print(f"\n{'='*60}")
    print(f"Step {stats.step}")
    print(f"{'='*60}")
    print(f"HERBIVORES: {stats.herb_count}")
    print(f"  Biomass:  total={stats.herb_total_biomass:,}, avg={stats.herb_avg_biomass:.1f}")
    print(f"  Energy:   total={stats.herb_total_energy:,}, avg={stats.herb_avg_energy:.1f}")
    print(f"  Gradient EMA: {stats.herb_avg_gradient_ema:.2f}")
    print(f"  On food: {stats.herb_on_food} ({stats.herb_eating_pct:.1f}%)")
    print(f"  Reproductions (est): {stats.herb_reproductions}")
    if stats.pred_count > 0 or stats.herb_count == 0:
        print(f"PREDATORS: {stats.pred_count}")
        print(f"  Biomass:  total={stats.pred_total_biomass:,}, avg={stats.pred_avg_biomass:.1f}")
        print(f"  Energy:   total={stats.pred_total_energy:,}, avg={stats.pred_avg_energy:.1f}")
        print(f"  Gradient EMA: {stats.pred_avg_gradient_ema:.2f}")
        print(f"  On prey: {stats.pred_on_food} ({stats.pred_eating_pct:.1f}%), On carrion: {stats.pred_on_carrion}")
        print(f"  Reproductions (est): {stats.pred_reproductions}")
    print(f"PLANTS: {stats.plant_count}")
    print(f"  Energy:   total={stats.plant_total_energy:,}, avg={stats.plant_avg_energy:.1f}")
    print(f"TERRAIN:")
    print(f"  Nutrients: {stats.total_nutrients:,.0f}")
    print(f"  Carrion: {stats.total_carrion:,.0f}")


def run_diagnostics(
    config_path: str = "basic_config.yaml",
    steps: int = 2000,
    log_interval: int = 100,
    world_size: Optional[int] = None,
):
    """Run simulation with diagnostic logging."""
    print(f"Loading config: {config_path}")
    config = load_config(config_path)

    if world_size:
        config.world.size = [world_size, world_size]

    print(f"World size: {config.world.size}")
    print(f"Running for {steps} steps, logging every {log_interval}")

    world = World(config.world)

    all_stats = []
    prev_herb_count = 0
    prev_pred_count = 0

    # Initial stats
    stats = collect_stats(world, 0, prev_herb_count, prev_pred_count)
    print_stats(stats)
    all_stats.append(stats)
    prev_herb_count = stats.herb_count
    prev_pred_count = stats.pred_count

    for step in range(1, steps + 1):
        world.update()

        if step % log_interval == 0:
            stats = collect_stats(world, step, prev_herb_count, prev_pred_count)
            print_stats(stats)
            all_stats.append(stats)
            prev_herb_count = stats.herb_count
            prev_pred_count = stats.pred_count

            # Early termination on extinction
            if stats.herb_count == 0 and stats.pred_count == 0:
                print("\n*** TOTAL EXTINCTION ***")
                break

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    herb_counts = [s.herb_count for s in all_stats]
    pred_counts = [s.pred_count for s in all_stats]
    plant_counts = [s.plant_count for s in all_stats]

    print(f"Herbivore population: {herb_counts[0]} -> {herb_counts[-1]}")
    print(f"  Min: {min(herb_counts)}, Max: {max(herb_counts)}")
    if max(pred_counts) > 0:
        print(f"Predator population: {pred_counts[0]} -> {pred_counts[-1]}")
        print(f"  Min: {min(pred_counts)}, Max: {max(pred_counts)}")
    print(f"Plant population: {plant_counts[0]} -> {plant_counts[-1]}")
    print(f"  Min: {min(plant_counts)}, Max: {max(plant_counts)}")

    if herb_counts[-1] == 0:
        # Find when extinction happened
        for i, s in enumerate(all_stats):
            if s.herb_count == 0:
                print(f"Herbivore extinction at step {s.step}")
                break
    if max(pred_counts) > 0 and pred_counts[-1] == 0:
        for i, s in enumerate(all_stats):
            if s.pred_count == 0 and i > 0:
                print(f"Predator extinction at step {s.step}")
                break

    return all_stats


def main():
    parser = argparse.ArgumentParser(description="Run simulation diagnostics")
    parser.add_argument("--config", default="basic_config.yaml", help="Config file path")
    parser.add_argument("--steps", type=int, default=2000, help="Number of steps to run")
    parser.add_argument("--interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--size", type=int, default=None, help="Override world size")
    args = parser.parse_args()

    run_diagnostics(
        config_path=args.config,
        steps=args.steps,
        log_interval=args.interval,
        world_size=args.size,
    )


if __name__ == "__main__":
    main()

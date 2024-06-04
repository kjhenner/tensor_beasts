import numpy as np
import pytest
from tensor_beasts.state_updates_numpy import move, eat, germinate, grow


@pytest.fixture
def sample_data():
    return {
        "entity_energy": np.array([
            [5, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],
            [3, 0, 0, 0, 0],
        ]),
        "divide_threshold": 16,
        "food_energy": np.array([
            [20, 0, 20, 0, 0],
            [0, 20, 0, 0, 0],
            [0, 0, 20, 0, 0],
            [20, 20, 0, 0, 0],
            [0, 20, 0, 0, 0],
        ]),
        "rand_array_high": np.ones((5, 5), dtype=np.uint8) * 255,
        "rand_array_low": np.zeros((5, 5), dtype=np.uint8),
        "herbivore_energy": np.array([
            [5, 0, 3, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [3, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ]),
        "plant_energy": np.array([
            [20, 0, 3, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [3, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ]),
        "seeds": np.array([
            [1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ]),
        "germination_odds": 1,
        "plant_growth_odds": 1,
        "crowding": 3,
        "crowding_odds": 5
    }


def test_move(sample_data):

    entity_energy = sample_data['entity_energy']
    initial_entity_energy = entity_energy.copy()
    divide_threshold = sample_data['divide_threshold']
    food_energy = sample_data['food_energy']
    rand_array = sample_data['rand_array_high']

    move(entity_energy, divide_threshold, food_energy, rand_array)

    print()
    print(initial_entity_energy)
    print(entity_energy)

    assert np.any(initial_entity_energy != entity_energy)  # Ensure changes


def test_eat(sample_data):
    herbivore_energy = sample_data['herbivore_energy']
    plant_energy = sample_data['plant_energy']
    eat_max = 2  # max amount a herbivore can eat

    initial_plant_energy = plant_energy.copy()
    initial_herbivore_energy = herbivore_energy.copy()

    eat(herbivore_energy, plant_energy, eat_max)

    assert np.any(herbivore_energy != initial_herbivore_energy)  # Ensure herbivores got some energy
    assert np.any(plant_energy != initial_plant_energy)  # Ensure plants lost energy


def test_germinate(sample_data):
    seeds = sample_data['seeds']
    plant_energy = sample_data['plant_energy']
    germination_odds = sample_data['germination_odds']
    rand_array = sample_data['rand_array_high']

    initial_plant_energy = plant_energy.copy()
    initial_seeds = seeds.copy()

    germinate(seeds, plant_energy, germination_odds, rand_array)

    assert np.any(plant_energy != initial_plant_energy)  # Ensure plant energy changes
    assert np.any(seeds != initial_seeds)  # Ensure seeds changes


def test_grow(sample_data):
    plant_energy = sample_data['plant_energy']
    plant_growth_odds = 256  # Always grow
    crowding = sample_data['crowding']
    crowding_odds = 256  # Always grow
    rand_array = sample_data['rand_array_high']

    initial_plant_energy = plant_energy.copy()

    grow(plant_energy, plant_growth_odds, crowding, crowding_odds, rand_array)

    assert np.any(plant_energy != initial_plant_energy)  # Ensure plant energy changes
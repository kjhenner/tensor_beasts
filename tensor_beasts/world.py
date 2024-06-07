from functools import lru_cache
from typing import Optional, Dict, List, Union, Callable
import torch

from tensor_beasts.util import (
    torch_correlate_2d, torch_correlate_3d, safe_sub, safe_add, timing,
    pad_matrix, get_direction_matrix, directional_kernel_set, generate_diffusion_kernel, generate_plant_crowding_kernel,
    safe_sum
)


class Feature:
    def __init__(self, group: Optional[str] = None):
        self.group = group


class FeatureGroup:
    pass


class BaseEntity:
    pass


class World:
    def __init__(self, size: int, config: Optional[Dict] = None, scalars: Optional[Dict] = None):
        self.width, self.height = size, size

        self.config = {
            "entities": {
                "Predator": {
                    "features": [
                        {"name": "energy", "group": "energy"},
                        {"name": "scent", "group": "scent"},
                        {"name": "offspring_count", "group": None}
                    ]
                },
                "Plant": {
                    "features": [
                        {"name": "energy", "group": "energy"},
                        {"name": "scent", "group": "scent"},
                        {"name": "seed", "group": None},
                        {"name": "crowding", "group": None}
                    ]
                },
                "Herbivore": {
                    "features": [
                        {"name": "energy", "group": "energy"},
                        {"name": "scent", "group": "scent"},
                        {"name": "offspring_count", "group": None}
                    ]
                },
                "Obstacle": {
                    "features": [
                        {"name": "mask", "group": None}
                    ]
                }
            }
        }
        if config is not None:
            self.config.update(config)

        self.scalars = {
            "plant_init_odds": 255,
            "herbivore_init_odds": 255,
            "plant_growth_odds": 255,
            "plant_germination_odds": 255,
            "plant_crowding_odds": 25,
            "plant_seed_odds": 255,
            "herbivore_eat_max": 16,
            "predator_eat_max": 255
        }
        if scalars is not None:
            self.scalars.update(scalars)

        self.entities = {}
        self.feature_groups = {}
        self.total_features = 0
        self.world_tensor = None

        self._initialize_entities()
        self._initialize_world_tensor()
        self._assign_features()

        for name, value in self.scalars.items():
            setattr(self, name, value)

        self.plant.energy[:] = (torch.randint(0, self.plant_init_odds, (self.width, self.height), dtype=torch.uint8) == 0)

        self.herbivore.energy[:] = (
            (torch.randint(0, self.herbivore_init_odds, (self.width, self.height), dtype=torch.uint8) == 0)
        ) * 240

        self.predator.energy[:] = (
            (torch.randint(0, self.herbivore_init_odds, (self.width, self.height), dtype=torch.uint8) == 0)
        ) * 240

        # self.obstacle.mask[:] = (torch.randint(0, 256, (self.width, self.height), dtype=torch.uint8) == 0)
        # self.obstacle.mask[:] = generate_maze(size) * generate_maze(size)
        self.obstacle.mask[:] = torch.zeros((self.width, self.height), dtype=torch.uint8)

    def _initialize_entities(self):
        """Initialize entities based on the configuration."""
        for entity_name, entity_info in self.config["entities"].items():
            entity_class = type(entity_name, (BaseEntity,), {})
            entity_instance = entity_class()
            self.entities[entity_name.lower()] = entity_instance

        for group_name in self.config.get("groups", []):
            self.feature_groups[group_name] = FeatureGroup()

    def _initialize_world_tensor(self):
        """Calculate the total depth and initialize the world tensor."""
        self.total_features = sum(
            len(entity_info["features"]) for entity_info in self.config["entities"].values()
        )
        self.world_tensor = torch.zeros((self.width, self.height, self.total_features), dtype=torch.uint8)

    def _assign_features(self):
        """Assign features and slices to entities and groups based on the config."""
        idx = 0
        group_feature_map = {}

        # Collect features by group
        for entity_name, entity_info in self.config["entities"].items():
            for feature in entity_info["features"]:
                group = feature["group"]
                if group not in group_feature_map:
                    group_feature_map[group] = []
                group_feature_map[group].append((entity_name.lower(), feature["name"]))

        # Assign features to entities and groups
        for group, features in group_feature_map.items():
            group_start_idx = idx

            for entity_name, feature_name in features:
                entity = self.entities[entity_name]
                feature_slice = self.world_tensor[:, :, idx]
                setattr(entity, feature_name, feature_slice)
                idx += 1

            if group:
                group_slice = self.world_tensor[:, :, group_start_idx:idx]
                setattr(self, group, group_slice)

    def __getattr__(self, name):
        """Provide convenient access to entities and feature groups."""
        if name in self.entities:
            return self.entities[name]
        elif name in self.feature_groups:
            return self.feature_groups[name]
        else:
            raise AttributeError(f"'World' object has no attribute '{name}'")

    def update(self, step):
        rand_array = torch.randint(0, 255, (self.width, self.height), dtype=torch.uint8)

        plant_mask = self.plant.energy.bool()

        if step % 2 == 0:
            # self.plant.crowding = torch_correlate_2d(safe_add(plant_mask, self.obstacle.mask.type(torch.bool), inplace=False), self.plant_crowding_kernel, mode='constant')
            self.plant.crowding = torch_correlate_2d(plant_mask, generate_plant_crowding_kernel(), mode='constant')
            self.grow(self.plant.energy, self.plant_growth_odds, self.plant.crowding, self.plant_crowding_odds, rand_array)

            self.plant.seed |= self.plant.crowding > (rand_array % self.plant_seed_odds)

            self.germinate(self.plant.seed, self.plant.energy, self.plant_germination_odds, rand_array)

        self.diffuse_scent(self.energy, self.scent, mask=self.obstacle.mask)

        self.move(
            entity_energy=self.herbivore.energy,
            target_energy=self.plant.scent,
            opposite_energy=[self.herbivore.scent, self.predator.scent],
            carried_features_self=[self.herbivore.offspring_count],
            carried_feature_fns_self=[lambda x: safe_add(x, 1, inplace=False)],
        )
        safe_sub(self.herbivore.energy, 2)

        self.move(
            entity_energy=self.predator.energy,
            target_energy=self.herbivore.scent,
            opposite_energy=self.predator.scent,
            carried_features_self=[self.predator.offspring_count],
            carried_feature_fns_self=[lambda x: safe_add(x, 1, inplace=False)],
        )
        safe_sub(self.predator.energy, 1)

        self.eat(self.herbivore.energy, self.plant.energy, self.herbivore_eat_max)
        self.eat(self.predator.energy, self.herbivore.energy, self.predator_eat_max)

        self.herbivore.offspring_count *= self.herbivore.energy > 0
        self.predator.offspring_count *= self.predator.energy > 0

        return {
            'seed_count': float(torch.sum(self.plant.seed)),
            'plant_mass': float(torch.sum(self.plant.energy)),
            'herbivore_mass': float(torch.sum(self.herbivore.energy)),
            'predator_mass': float(torch.sum(self.predator.energy)),
            'herbivore_offspring_count': float(torch.sum(self.herbivore.offspring_count)),
        }

    @staticmethod
    @timing
    def diffuse_scent(entity_energy, entity_scent, mask=None, diffusion_steps=10):
        for _ in range(diffusion_steps):
            entity_scent[:] = torch_correlate_3d(entity_scent.type(torch.float32), generate_diffusion_kernel().type(torch.float32)).type(torch.uint8)
        safe_add(entity_scent[:], entity_energy[:])
        if mask is not None:
            mask = torch.stack((mask, mask, mask), dim=-1) == 0
            entity_scent *= mask

    @staticmethod
    @timing
    def move(
        entity_energy: torch.Tensor,
        target_energy: Union[torch.Tensor, List[torch.Tensor]],
        opposite_energy: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        clearance_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        clearance_kernel_size: Optional[int] = 5,
        divide_threshold: Optional[int] = 250,
        divide_fn_self: Optional[Callable] = lambda x: x // 2,
        divide_fn_offspring: Optional[Callable] = lambda x: x // 4,
        carried_features_self: Optional[List[torch.Tensor]] = None,
        carried_features_offspring: Optional[List[torch.Tensor]] = None,
        carried_feature_fns_self: Optional[List[Callable]] = None,
        carried_feature_fns_offspring: Optional[List[Callable]] = None,
    ):
        """
        Move entities and reproduce entities. Movement is based on the target energy tensor and optional opposite energy tensor.

        :param entity_energy: Energy tensor of the entity to move. This is transferred to the selected position.
            This is also used for clearance calculations, masking, and reproduction.
        :param target_energy: Movement will favor moving towards higher values in this tensor.
        :param opposite_energy: Movement will avoid moving towards higher values in this tensor.
        :param clearance_mask: Mask tensor for clearance calculations.
        :param clearance_kernel_size: Size of the kernel used for clearance calculations.
        :param divide_threshold: Energy threshold for reproduction.
        :param divide_fn_self: Function to apply to the entity energy on reproduction.
        :param divide_fn_offspring: Function to apply to the offspring energy on reproduction.
        :param carried_features_self: List of features that will be moved along with the entity energy.
        :param carried_features_offspring: List of features that will be copied from the parent to the offspring.
        :param carried_feature_fns_self: List of functions to apply to the carried features of the parent entity.
        :param carried_feature_fns_offspring: List of functions to apply to the carried features of the offspring entity.
        :return: None
        """
        if carried_features_self is not None:
            if carried_feature_fns_self is None:
                carried_feature_fns_self = [lambda x: x] * len(carried_features_self)
            else:
                assert len(carried_features_self) == len(carried_feature_fns_self)

        if carried_features_offspring is not None:
            if carried_feature_fns_offspring is None:
                carried_feature_fns_offspring = [lambda x: x] * len(carried_features_offspring)
            else:
                assert len(carried_features_offspring) == len(carried_feature_fns_offspring)

        if isinstance(target_energy, list):
            target_energy = safe_sum(target_energy)

        if opposite_energy is not None:
            if isinstance(opposite_energy, list):
                opposite_energy = safe_sum(opposite_energy)
            target_energy = safe_sub(target_energy, opposite_energy, inplace=False)

        if clearance_mask is not None:
            if isinstance(clearance_mask, list):
                clearance_mask = safe_sum(clearance_mask)
            else:
                clearance_mask = clearance_mask.clone()
            clearance_mask[entity_energy > 0] = 1
        else:
            clearance_mask = entity_energy > 0

        directions = get_direction_matrix(target_energy)

        # This is 1 where an entity is present and intends to move in that direction
        direction_masks = {d: ((directions == d) * (entity_energy > 0)).type(torch.uint8) for d in range(1, 5)}

        clearance_kernels = directional_kernel_set(clearance_kernel_size)
        # TODO: Batch this!
        for d in range(1, 5):
            direction_masks[d] *= ~(torch.tensor(
                torch_correlate_2d(
                    clearance_mask.type(torch.float32),
                    clearance_kernels[d].type(torch.float32),
                    mode='constant',
                    cval=1
                )
            ).type(torch.bool))

        # One where an entity is present in the current state and will move, and zero where it will not
        move_origin_mask = torch.sum(torch.stack(list(direction_masks.values())), dim=0).type(torch.bool)

        # One where an entity will move away and leave an offspring, and zero where it will not
        offspring_mask = move_origin_mask * (entity_energy > divide_threshold)

        # One where an entity will move away and leave no offspring, and zero where it will not
        vacated_mask = move_origin_mask * (entity_energy <= divide_threshold)

        # After this operation, each feature will be the union of its current state and the state after movement
        for feature, fn in [(entity_energy, divide_fn_self)] + list(zip(carried_features_self or [], carried_feature_fns_self or [])):
            safe_add(feature, torch.sum(torch.stack([
                # TODO: Batch? Which is faster, pad or roll?
                pad_matrix(
                    torch.where(
                        ((direction_masks[d] * entity_energy) > divide_threshold).type(torch.bool),
                        direction_masks[d] * fn(feature),
                        direction_masks[d] * feature
                    ),
                    d
                )
                for d in range(1, 5)
            ]), dim=0))

        # After this operation, each origin position where an offspring will be left will be adjusted by the divisor
        # Currently, there is a bug here.
        for feature, fn in [(entity_energy, divide_fn_offspring)] + list(zip(carried_features_offspring or [], carried_feature_fns_offspring or [])):
            feature[:] = torch.where(
                offspring_mask,
                fn(feature),
                feature
            )

        # After this operation, each origin position where no offspring will be left will be zeroed
        for feature in [entity_energy] + (carried_features_self or []) + (carried_features_offspring or []):
            feature[:] *= ~vacated_mask


    @staticmethod
    @timing
    def eat(eater_energy, eaten_energy, eat_max, eat_efficiency_loss=4):
        old_eaten_energy = eaten_energy.clone()
        safe_sub(eaten_energy, (eater_energy > 0).type(torch.uint8) * eat_max)
        delta = old_eaten_energy - eaten_energy
        safe_add(eater_energy, delta // eat_efficiency_loss)

    @staticmethod
    @timing
    def germinate(seeds, plant_energy, germination_odds, rand_tensor):
        germination_rand = rand_tensor % germination_odds
        seed_germination = (
            seeds & ~(plant_energy > 0) & (germination_rand == 0)
        )
        safe_add(plant_energy, seed_germination)
        safe_sub(seeds, seed_germination)

    @staticmethod
    @timing
    def grow(plant_energy, plant_growth_odds, crowding, crowding_odds, rand_tensor):
        growth_rand = rand_tensor % plant_growth_odds
        growth = plant_energy <= growth_rand
        plant_crowding_mask = (rand_tensor % crowding_odds) >= crowding
        safe_add(plant_energy, (plant_energy > 0) * growth * plant_crowding_mask)

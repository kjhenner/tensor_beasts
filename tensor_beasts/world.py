from typing import Optional, Dict
import torch

from tensor_beasts.state_updates import grow, germinate, diffuse_scent, move, eat
from tensor_beasts.util import torch_correlate_2d, safe_sub, safe_add


class Feature:
    def __init__(self, group: Optional[str] = None):
        self.group = group


class FeatureGroup:
    pass


class BaseEntity:
    pass


class World:
    def __init__(self, size: int, config: Dict, scalars: Dict):
        self.width, self.height = size, size
        self.config = config

        self.entities = {}
        self.feature_groups = {}
        self.total_features = 0
        self.world_tensor = None

        self._initialize_entities()
        self._initialize_world_tensor()
        self._assign_features()

        for name, value in scalars.items():
            setattr(self, name, value)

        self.plant.energy[:] = (torch.randint(0, self.plant_init_odds, (self.width, self.height), dtype=torch.uint8) == 0)

        self.herbivore.energy[:] = (
            (torch.randint(0, self.herbivore_init_odds, (self.width, self.height), dtype=torch.uint8) == 0)
        ) * 255

        self.predator.energy[:] = (
            (torch.randint(0, self.herbivore_init_odds, (self.width, self.height), dtype=torch.uint8) == 0)
        ) * 255

        self.plant_crowding_kernel = torch.tensor([
            [0, 1, 1, 1, 0],
            [1, 1, 2, 1, 1],
            [1, 2, 0, 2, 1],
            [1, 1, 2, 1, 1],
            [0, 1, 1, 1, 0],
        ], dtype=torch.uint8)

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
            plant_crowding = torch_correlate_2d(plant_mask, self.plant_crowding_kernel, mode='constant')
            grow(self.plant.energy, self.plant_growth_odds, plant_crowding, self.plant_crowding_odds, rand_array)
            self.seed.energy |= plant_crowding > (rand_array % self.plant_seed_odds)
            germinate(self.seed.energy, self.plant.energy, self.plant_germination_odds, rand_array)

        diffuse_scent(self.energy, self.scent)

        move(self.herbivore.energy, 250, self.plant.scent, safe_add(self.herbivore.scent, self.predator.scent, inplace=False))

        safe_sub(self.herbivore.energy, 2)

        move(self.predator.energy, 250, self.herbivore.scent, self.predator.scent)
        safe_sub(self.predator.energy, 1)

        eat(self.herbivore.energy, self.plant.energy, self.herbivore_eat_max)
        eat(self.predator.energy, self.herbivore.energy, self.predator_eat_max)

        return {
            'seed_count': float(torch.sum(self.seed.energy)),
            'plant_mass': float(torch.sum(self.plant.energy)),
            'herbivore_mass': float(torch.sum(self.herbivore.energy)),
            'predator_mass': float(torch.sum(self.predator.energy)),
        }

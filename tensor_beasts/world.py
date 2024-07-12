import dataclasses
from collections import defaultdict
from typing import Optional, Dict, List, Union, Callable, Any
from omegaconf import DictConfig
import torch

from tensor_beasts.util import (
    torch_correlate_2d, torch_correlate_3d, safe_sub, safe_add, timing,
    pad_matrix, get_direction_matrix, directional_kernel_set, generate_diffusion_kernel, generate_plant_crowding_kernel,
    safe_sum, perlin_noise
)


@dataclasses.dataclass
class Feature:
    group: Optional[str] = None
    tags: Optional[List[str]] = None
    tensor: Optional[torch.Tensor] = None


class FeatureGroup:
    pass


class BaseEntity:

    def __init__(self, entity_info: Dict[str, Any]):
        self.features = {}
        for feature_info in entity_info["features"]:
            self.features[feature_info["name"]] = Feature(
                group=feature_info.get("group"),
                tags=feature_info.get("tags", []),
            )

    def __getattr__(self, item):
        if item not in self.features:
            raise AttributeError(f"Feature {item} not found in entity.")
        return self.features[item].tensor

    def tagged(self, tag: str) -> List[torch.Tensor]:
        return [feature.tensor for feature in self.features.values() if tag in feature.tags]


class World:
    def __init__(
        self,
        config: Optional[DictConfig] = None,
    ):
        """Initialize the world.

        The world is a tensor of unsigned 8-bit integers with shape (H, W, C) where the size of C is the total number of
        features defined in the config.

        On initialization of a World instance, the entity and feature structure defined in the config are mapped to
        class attributes that access the corresponding feature channels in the world tensor.

        For example, if the predator energy feature is the Nth feature channel, `world.predator.energy` is equivalent to
        `world_tensor[:, :, N]`.

        Assigning a feature to a group will ensure that the feature's index along the channel dimension is adjacent to
        other features in the same group. This is useful for operations that batch process features in the same group.
        A feature can only belong to one group.

        For example, if the features in the `energy` group are assigned to the nth to mth channels, `world.energy` is
        equivalent to `world_tensor[:, :, N:M]`.

        Assigning a feature with one or more tags will add that feature's name to the corresponding tag list.
        """
        self.width, self.height = config.size, config.size
        self.total_features = 0

        # TODO: Use Hydra for config?
        self.entity_config = {
            "entities": {
                "Predator": {
                    "features": [
                        {"name": "energy", "group": "energy", "tags": ["observable"]},
                        {"name": "scent", "group": "scent", "tags": ["observable"]},
                        {"name": "offspring_count", "tags": ["clear_on_death"]},
                        {"name": "id_0", "group": "predator_id", "tags": ["clear_on_death"]},
                        {"name": "id_1", "group": "predator_id", "tags": ["clear_on_death"]}
                    ]
                },
                "Plant": {
                    "features": [
                        {"name": "energy", "group": "energy", "tags": ["observable"]},
                        {"name": "scent", "group": "scent", "tags": ["observable"]},
                        {"name": "fertility_map"},
                        {"name": "seed"},
                        {"name": "crowding"}
                    ]
                },
                "Herbivore": {
                    "features": [
                        {"name": "energy", "group": "energy", "tags": ["observable"]},
                        {"name": "scent", "group": "scent", "tags": ["observable"]},
                        {"name": "offspring_count", "tags": ["clear_on_death"]},
                        {"name": "id_0", "group": "herbivore_id", "tags": ["clear_on_death"]},
                        {"name": "id_1", "group": "herbivore_id", "tags": ["clear_on_death"]},
                    ]
                },
                "Obstacle": {
                    "features": [
                        {"name": "mask", "tags": ["observable"]}
                    ]
                }
            }
        }
        if config.get("entity_cfg") is not None:
            self.entity_config.update(config.entity_cfg)

        self.scalars = {
            "plant_init_odds": 16,
            "plant_init_energy": 32,
            "plant_only_steps": 32,
            "plant_growth_step_modulo": 2,
            "herbivore_init_odds": 255,
            "plant_growth_odds": 255,
            "predator_init_odds": 511,
            "plant_germination_odds": 255,
            "plant_crowding_odds": 25,
            "plant_seed_odds": 255,
            "herbivore_eat_max": 16,
            "predator_eat_max": 255,
            "herbivore_energy_loss": 1,
            "predator_energy_loss": 1,
            "single_herbivore_init": 0
        }
        if config.get("scalars") is not None:
            self.scalars.update(config.scalars)

        self.entities = {}
        self.feature_groups = {}
        self.feature_tags = defaultdict(list)
        self.world_tensor = None

        self._initialize_entities()
        self._initialize_world_tensor()
        self._assign_features()

        for name, value in self.scalars.items():
            setattr(self, name, value)

        self.plant.energy[:] = (torch.randint(0, self.plant_init_odds, (self.width, self.height), dtype=torch.uint8) == 0) * self.plant_init_energy
        # Set plant energy to a split gradient, going from 0 to 255 starting at the

        self.initialize_herbivore()
        self.initialize_predator()

        self.plant.fertility_map[:] = ((perlin_noise((self.width, self.height), (8, 8)) + 3) * 63).type(torch.uint8)

        # self.obstacle.mask[:] = (torch.randint(0, 256, (self.width, self.height), dtype=torch.uint8) == 0)
        # self.obstacle.mask[:] = generate_maze(size) * generate_maze(size)
        # Set obstacles at edges of world:
        self.obstacle.mask[:] = torch.zeros((self.width, self.height), dtype=torch.uint8)
        self.obstacle.mask[0, :] = 1
        self.obstacle.mask[:, 0] = 1
        self.obstacle.mask[self.width - 1, :] = 1
        self.obstacle.mask[:, self.height - 1] = 1

        self.step = 0

        for _ in range(self.plant_only_steps):
            self.update(plant_only=True)


    @staticmethod
    def update_id(ids: torch.Tensor):
        """If ids[0] is 255, increment ids[1]."""
        ids[:, :, 1][ids[:, :, 0] == 255] += 1  # No-good extra assignment. Assigned in the caller too.
        return ids[:, :, 1]

    def initialize_herbivore(self):
        """Initialize herbivore."""
        if self.single_herbivore_init:
            self.herbivore.energy[self.width // 2, self.height // 2] = 240
            self.herbivore.id_0[self.width // 2, self.height // 2] = 0
            self.herbivore.id_1[self.width // 2, self.height // 2] = 0
            return
        else:
            self.herbivore.energy[:] = (
                (torch.randint(0, self.herbivore_init_odds, (self.width, self.height), dtype=torch.uint8) == 0)
            ) * 240

            self.herbivore.id_0[:] = torch.randint(0, 256, (self.width, self.height), dtype=torch.uint8)
            self.herbivore.id_1[:] = torch.randint(0, 256, (self.width, self.height), dtype=torch.uint8)

    def initialize_predator(self):
        """Initialize predator."""
        self.predator.energy[:] = (
            (torch.randint(0, self.predator_init_odds, (self.width, self.height), dtype=torch.uint16) == 0)
        ) * 240

        self.predator.id_0[:] = torch.randint(0, 256, (self.width, self.height), dtype=torch.uint8)
        self.predator.id_1[:] = torch.randint(0, 256, (self.width, self.height), dtype=torch.uint8)

    def _initialize_entities(self):
        """Initialize entities based on the configuration."""
        for entity_name, entity_info in self.entity_config["entities"].items():
            entity_class = type(entity_name, (BaseEntity,), {})
            entity_instance = entity_class(entity_info)
            self.entities[entity_name.lower()] = entity_instance

    def _initialize_world_tensor(self):
        """Calculate the total depth and initialize the world tensor."""
        self.total_features = sum(
            len(entity_info["features"]) for entity_info in self.entity_config["entities"].values()
        )
        self.world_tensor = torch.zeros((self.width, self.height, self.total_features), dtype=torch.uint8)

    def _assign_features(self):
        """Assign features and slices to entities and groups based on the config."""
        idx = 0
        group_feature_map = defaultdict(list)
        ungrouped_features = []

        # Collect features groups and tags
        for entity_name, entity_info in self.entity_config["entities"].items():
            for feature in entity_info["features"]:
                if "group" in feature:
                    group_feature_map[feature["group"]].append(
                        (entity_name.lower(), feature)
                    )
                else:
                    ungrouped_features.append((entity_name.lower(), feature))

        # Start by assigning grouped features in contiguous slices
        for group, features in group_feature_map.items():
            group_start_idx = idx

            for entity_name, feature in features:
                entity = self.entities[entity_name]
                entity.features[feature["name"]].tensor = self.world_tensor[:, :, idx]
                for tag in feature.get("tags", []):
                    self.feature_tags[tag].append(self.world_tensor[:, :, idx])
                idx += 1

                group_slice = self.world_tensor[:, :, group_start_idx:idx]
                setattr(self, group, group_slice)

        for entity_name, feature in ungrouped_features:
            entity = self.entities[entity_name]
            entity.features[feature["name"]].tensor = self.world_tensor[:, :, idx]
            idx += 1

    def __getattr__(self, name):
        """Provide convenient access to entities and feature groups."""
        if name in self.entities:
            return self.entities[name]
        elif name in self.feature_tags:
            return torch.stack(self.feature_tags[name], dim=-1)
        else:
            raise AttributeError(f"'World' object has no attribute '{name}'")

    def update(self, action: Optional[torch.Tensor] = None, plant_only=False):

        # For now, hard code actions to herbivore
        if action is None:
            action_dict = {}
        else:
            action_dict = {
                "herbivore_move": torch.tensor(action, dtype=torch.uint8),
            }

        rand_array = torch.randint(0, 255, (self.width, self.height), dtype=torch.uint8)

        plant_mask = self.plant.energy.bool()

        if self.step % self.plant_growth_step_modulo == 0 or plant_only:
            self.plant.crowding = torch_correlate_2d(plant_mask, generate_plant_crowding_kernel(), mode='constant')
            self.grow(
                self.plant.energy,
                self.plant_growth_odds,
                self.plant.crowding,
                self.plant_crowding_odds,
                self.plant.fertility_map,
                rand_array
            )

            self.plant.seed |= self.plant.crowding > (rand_array % self.plant_seed_odds)

            self.germinate(self.plant.seed, self.plant.energy, self.plant_germination_odds, rand_array)

        if not plant_only:
            # We don't actually care that much about id collisions, so we can just use a random id
            random_fn = lambda x: torch.randint(0, 256, (self.width, self.height), dtype=torch.uint8)
            self.move(
                entity_energy=self.herbivore.energy,
                target_energy=self.plant.scent,
                target_energy_weights=[0.5],
                opposite_energy=[self.herbivore.scent, self.predator.scent],
                opposite_energy_weights=[0.1, 1],
                carried_features_self=[self.herbivore.offspring_count, self.herbivore.id_0, self.herbivore.id_1],
                carried_feature_fns_self=[lambda x: safe_add(x, 1, inplace=False), lambda x: x, lambda x: x],
                carried_features_offspring=[self.herbivore.id_1, self.herbivore.id_0],
                carried_feature_fns_offspring=[random_fn, random_fn],
                agent_action=action_dict.get("herbivore_move", None)
            )
            safe_sub(self.herbivore.energy, self.herbivore_energy_loss)

            self.move(
                entity_energy=self.predator.energy,
                target_energy=self.herbivore.scent,
                target_energy_weights=[1],
                opposite_energy=self.predator.scent,
                opposite_energy_weights=[1],
                carried_features_self=[self.predator.offspring_count, self.predator.id_0, self.predator.id_1],
                carried_feature_fns_self=[lambda x: safe_add(x, 1, inplace=False), lambda x: x, lambda x: x],
                carried_features_offspring=[self.herbivore.id_1, self.herbivore.id_0],
                carried_feature_fns_offspring=[random_fn, random_fn],
                agent_action=action_dict.get("predator_move", None)

            )
            safe_sub(self.predator.energy, self.predator_energy_loss)

            self.eat(self.herbivore.energy, self.plant.energy, self.herbivore_eat_max)
            self.eat(self.predator.energy, self.herbivore.energy, self.predator_eat_max)

        self.diffuse_scent(self.energy, self.scent, mask=self.obstacle.mask)

        for entity in self.entities.values():
            for cleared_feature in entity.tagged('clear_on_death'):
                cleared_feature *= entity.energy > 0

        self.step += 1
        self.log_scores(self.herbivore)

    @staticmethod
    @timing
    def diffuse_scent(entity_energy, entity_scent, mask=None, diffusion_steps=2):
        for _ in range(diffusion_steps):
            scent = entity_scent.type(torch.float32) ** 5
            entity_scent[:] = torch.pow(torch_correlate_3d(scent, generate_diffusion_kernel().type(torch.float32)), 1.0 / 5.0).type(torch.uint8)
        safe_sub(entity_scent[:], 1)
        safe_add(entity_scent[:], entity_energy[:] // 4)
        if mask is not None:
            mask = torch.stack((mask, mask, mask), dim=-1) == 0
            entity_scent *= mask

    @staticmethod
    def get_direction_masks(
        directions,
        entity_energy,
        clearance_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        clearance_kernel_size: Optional[int] = 5,
    ):
        # If we get batched directions, we need to squeeze the batch dimension
        if len(directions.shape) == 3:
            directions = directions.squeeze(0)

        direction_masks = {d: ((directions == d) * (entity_energy > 0)).type(torch.uint8) for d in range(1, 5)}

        if clearance_mask is not None:
            if isinstance(clearance_mask, list):
                clearance_mask = safe_sum(clearance_mask)
            else:
                clearance_mask = clearance_mask.clone()
            clearance_mask[entity_energy > 0] = 1
        else:
            clearance_mask = entity_energy > 0

        clearance_kernels = directional_kernel_set(clearance_kernel_size)
        for d in range(1, 5):
            direction_masks[d] *= ~(
                torch_correlate_2d(
                    clearance_mask.type(torch.float32),
                    clearance_kernels[d].type(torch.float32),
                    mode='constant',
                    cval=1
                ).detach().type(torch.bool))
        return direction_masks

    @staticmethod
    @timing
    def prepare_move(
        entity_energy: torch.Tensor,
        target_energy: Union[torch.Tensor, List[torch.Tensor]],
        target_energy_weights: List[float],
        opposite_energy: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        opposite_energy_weights: Optional[List[float]] = None,
        clearance_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        clearance_kernel_size: Optional[int] = 5,
    ):
        if target_energy is not None:
            if not isinstance(target_energy, list):
                target_energy = [target_energy]

            if target_energy_weights is None:
                target_energy_weights = [1] * len(target_energy)
            else:
                assert len(target_energy) == len(target_energy_weights)

            target_energy = safe_sum([(target * wt).type(torch.uint8) for target, wt in zip(target_energy, target_energy_weights)])

        if opposite_energy is not None:
            if not isinstance(opposite_energy, list):
                opposite_energy = [opposite_energy]

            if opposite_energy_weights is None:
                opposite_energy_weights = [1] * len(opposite_energy)
            else:
                assert len(opposite_energy) == len(opposite_energy_weights)
            opposite_energy = safe_sum([(opposite * wt).type(torch.uint8) for opposite, wt in zip(opposite_energy, opposite_energy_weights)])
            target_energy = safe_sub(target_energy, opposite_energy, inplace=False)

        directions = get_direction_matrix(target_energy)

        direction_masks = World.get_direction_masks(directions, entity_energy, clearance_mask, clearance_kernel_size)

        return direction_masks


    @timing
    def prepare_move_nn(
        self,
        entity_energy: torch.Tensor,
    ):
        # reshape to (N, C, W, H)
        tensor = self.world_tensor[:, :, :4]
        directions, tau = self.model.forward(tensor.permute(2, 0, 1).unsqueeze(0), self.num_quantiles)
        directions = torch.argmax(directions[0, :, :], dim=0)

        direction_masks = World.get_direction_masks(directions, entity_energy, self.obstacle.mask, 5)

        return direction_masks


    @staticmethod
    @timing
    def perform_move(
        entity_energy: torch.Tensor,
        direction_masks: dict,
        divide_threshold: Optional[int] = 250,
        divide_fn_self: Optional[Callable] = lambda x: x // 2,
        divide_fn_offspring: Optional[Callable] = lambda x: x // 4,
        carried_features_self: Optional[List[torch.Tensor]] = None,
        carried_feature_fns_self: Optional[List[Callable]] = None,
        carried_features_offspring: Optional[List[torch.Tensor]] = None,
        carried_feature_fns_offspring: Optional[List[Callable]] = None,
    ):
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

        move_origin_mask = torch.sum(torch.stack(list(direction_masks.values())), dim=0).type(torch.bool)
        offspring_mask = move_origin_mask * (entity_energy > divide_threshold)
        vacated_mask = move_origin_mask * (entity_energy <= divide_threshold)

        for feature, fn in [(entity_energy, divide_fn_self)] + list(zip(carried_features_self or [], carried_feature_fns_self or [])):
            safe_add(feature, torch.sum(torch.stack([
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

        # After this operation, each origin position where an offspring will be left will be adjusted by corresponding
        # feature functions
        for feature, fn in [(entity_energy, divide_fn_offspring)] + list(zip(carried_features_offspring or [], carried_feature_fns_offspring or [])):
            feature[:] = torch.where(
                offspring_mask,
                fn(feature),
                feature
            )

        # After this operation, each origin position where no offspring will be left will be zeroed
        for feature in [entity_energy] + (carried_features_self or []) + (carried_features_offspring or []):
            feature[:] *= ~vacated_mask

    @timing
    def move(
        self,
        entity_energy: torch.Tensor,
        target_energy: Union[torch.Tensor, List[torch.Tensor]],
        target_energy_weights: List[float],
        opposite_energy: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        opposite_energy_weights: Optional[List[float]] = None,
        clearance_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        clearance_kernel_size: Optional[int] = 5,
        divide_threshold: Optional[int] = 250,
        divide_fn_self: Optional[Callable] = lambda x: x // 2,
        divide_fn_offspring: Optional[Callable] = lambda x: x // 4,
        carried_features_self: Optional[List[torch.Tensor]] = None,
        carried_features_offspring: Optional[List[torch.Tensor]] = None,
        carried_feature_fns_self: Optional[List[Callable]] = None,
        carried_feature_fns_offspring: Optional[List[Callable]] = None,
        agent_action: Optional[torch.Tensor] = None,
    ):
        """
        Move entities and reproduce entities. Movement is based on the target energy tensor and optional opposite energy tensor.

        :param entity_energy: Energy tensor of the entity to move. This is transferred to the selected position.
            This is also used for clearance calculations, masking, and reproduction.
        :param target_energy: Movement will favor moving towards higher values in this tensor.
        :param target_energy_weights: Weights for the target energy tensor.
        :param opposite_energy: Movement will avoid moving towards higher values in this tensor.
        :param opposite_energy_weights: Weights for the opposite energy tensor.
        :param clearance_mask: Mask tensor for clearance calculations.
        :param clearance_kernel_size: Size of the kernel used for clearance calculations.
        :param divide_threshold: Energy threshold for reproduction.
        :param divide_fn_self: Function to apply to the entity energy on reproduction.
        :param divide_fn_offspring: Function to apply to the offspring energy on reproduction.
        :param carried_features_self: List of features that will be moved along with the entity energy.
        :param carried_features_offspring: List of features that will be copied from the parent to the offspring.
        :param carried_feature_fns_self: List of functions to apply to the carried features of the parent entity.
        :param carried_feature_fns_offspring: List of functions to apply to the carried features of the offspring entity.
        :param agent_action: Action tensor of the agent.
        :return: None
        """
        if agent_action is not None:
            direction_masks = World.get_direction_masks(agent_action, entity_energy, self.obstacle.mask, 5)
        else:
            direction_masks = self.prepare_move(
                entity_energy,
                target_energy,
                target_energy_weights,
                opposite_energy,
                opposite_energy_weights,
                clearance_mask,
                clearance_kernel_size
            )

        self.perform_move(
            entity_energy,
            direction_masks,
            divide_threshold,
            divide_fn_self,
            divide_fn_offspring,
            carried_features_self,
            carried_feature_fns_self,
            carried_features_offspring,
            carried_feature_fns_offspring
        )

    @staticmethod
    @timing
    def eat(eater_energy, eaten_energy, eat_max, eat_efficiency_loss=2):
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
    def grow(plant_energy, plant_growth_odds, crowding, crowding_odds, fertility_map, rand_tensor):
        growth_rand = rand_tensor % plant_growth_odds
        growth = safe_add(plant_energy, fertility_map, inplace=False) <= growth_rand
        plant_crowding_mask = (rand_tensor % crowding_odds) >= crowding
        safe_add(plant_energy, (plant_energy > 0) * growth * plant_crowding_mask)

    @timing
    def log_scores(self, entity: BaseEntity):
        scores = entity.offspring_count.type(torch.float32) * 255 + entity.energy
        top_score_idx = torch.argmax(scores)
        converted_ids = (entity.id_0.type(torch.float32) * 255) + entity.id_1
        top_score_id = converted_ids.flatten()[top_score_idx]
        top_score = scores.flatten()[top_score_idx]

    @timing
    def entity_scores(self, entity: Union[BaseEntity | str], reward_mode: str = "default"):
        if isinstance(entity, str):
            entity = self.entities[entity.lower()]

        energy_ratio = entity.energy.type(torch.float32) / self.scalars.get(f"{entity.__class__.__name__.lower()}_init_energy", 240)
        offspring_ratio = entity.offspring_count.type(torch.float32) / self.scalars.get(f"{entity.__class__.__name__.lower()}_offspring_target", 10)
        # step_multiplier = math.sqrt(float(self.step))
        step_multiplier = 1

        # print(f"energy_ratio: {energy_ratio}")
        # print(f"offspring_ratio: {offspring_ratio}")
        # print(f"step: {self.step}")
        # print(f"step_multiplier: {step_multiplier}")

        if reward_mode == "default":
            # Balanced reward considering both energy and offspring
            reward = (energy_ratio + offspring_ratio) / 2
        elif reward_mode == "energy_focused":
            # Reward heavily weighted towards energy
            reward = energy_ratio * 0.8 + offspring_ratio * 0.2
        elif reward_mode == "offspring_focused":
            # Reward heavily weighted towards offspring
            reward = energy_ratio * 0.2 + offspring_ratio * 0.8
        elif reward_mode == "survival":
            # Reward for surviving (energy > 0) and normalized offspring count
            reward = ((entity.energy > 0).type(torch.float32) + offspring_ratio) / 2
        else:
            raise ValueError(f"Unknown reward mode: {reward_mode}")

        # print(f"reward: {reward}")

        # Sum the reward across all entities
        total_reward = torch.sum(reward)
        # print(f"total_reward: {total_reward}")

        # print(f"adjusted_reward: {total_reward / (self.width * self.height)}")
        # print(f"step_multiplier: {step_multiplier}")
        # Normalize the total reward to [-1, 1]
        normalized_reward = torch.clamp(total_reward / (self.width * self.height) * 2 - 1, -1, 1) * step_multiplier

        # print(f"normalized_reward: {normalized_reward}")

        # return normalized_reward
        return total_reward


    def collect_info(self):
        return {
            'seed_count': float(torch.sum(self.plant.seed)),
            'plant_mass': float(torch.sum(self.plant.energy)),
            'herbivore_mass': float(torch.sum(self.herbivore.energy)),
            'predator_mass': float(torch.sum(self.predator.energy)),
            'herbivore_offspring_count': float(torch.sum(self.herbivore.offspring_count)),
        }

    def rgb_array(self):
        # Create the obstacle mask (True where obstacles are)
        obstacle_mask = self.obstacle.mask.unsqueeze(-1).repeat(1, 1, 3)

        # Create energy display
        energy = self.energy.clone()
        # Set obstacles to white (255) and keep other values as they are
        energy = torch.where(obstacle_mask, torch.ones_like(energy) * 255, energy)

        # Create scent display
        scent = self.scent.clone()
        # Normalize scent values to 0-255 range for better visibility
        scent = (scent - scent.min()) / (scent.max() - scent.min() + 1e-8) * 255
        # Set obstacles to white (255) and keep other values as they are
        scent = torch.where(obstacle_mask, torch.ones_like(scent) * 255, scent)

        # Arrange the energy and scent channels beside each other
        output = torch.cat((energy, scent), dim=1)

        # Upscale the output
        output = output.repeat_interleave(3, dim=0).repeat_interleave(3, dim=1)

        return output.to(torch.uint8)  # Ensure output is in correct dtype for display

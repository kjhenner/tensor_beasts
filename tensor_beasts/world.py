import abc
from dataclasses import dataclass
from typing import List, Dict, Optional, Type, Tuple, Union, Callable, Iterable
import torch
from tensordict import TensorDict
from omegaconf import DictConfig, ListConfig

from tensor_beasts.util import (
    safe_add, safe_sub, torch_correlate_2d, perlin_noise,
    get_direction_matrix, pad_matrix, directional_kernel_set, torch_correlate_3d, generate_diffusion_kernel, safe_sum
)


@dataclass
class FeatureDefinition:
    name: str
    dtype: torch.dtype
    shape: Tuple[int, ...] = ()
    shared: bool = False
    observable: bool = False
    use_world_size: bool = True


class Entity(abc.ABC):
    features: List[FeatureDefinition] = []

    def __init__(self, world: 'World', config: DictConfig):
        self.world = world
        self.config = config
        self.data_manager = world.data_manager

    @classmethod
    def get_feature_definitions(cls) -> List[FeatureDefinition]:
        return cls.features

    def get_feature(self, feature_name: str) -> torch.Tensor:
        return self.data_manager.get_feature(self.__class__.__name__.lower(), feature_name)

    def set_feature(self, feature_name: str, value: torch.Tensor) -> None:
        self.data_manager.set_feature(self.__class__.__name__.lower(), feature_name, value)

    @abc.abstractmethod
    def update(self, step: int):
        pass


class EntityDataManager:
    def __init__(self, entity_classes: List[Type[Entity]], world_size: Tuple[int]):
        self.entity_classes = {cls.__name__.lower(): cls for cls in entity_classes}

        self.entity_data = TensorDict({}, batch_size=[])
        self.shared_features = TensorDict({}, batch_size=[])
        self.shared_feature_indices = {}

        self._initialize_features(world_size)

        self.main_td = TensorDict({
            "entities": self.entity_data,
            "shared_features": self.shared_features,
        }, batch_size=[])

    def _initialize_features(self, world_size: Tuple[int]):
        shared_feature_counts = {}
        shared_feature_defs = {}

        for entity_name, cls in self.entity_classes.items():
            for f in cls.get_feature_definitions():
                if f.shared:
                    if f.name not in shared_feature_counts:
                        shared_feature_counts[f.name] = 1
                        shared_feature_defs[f.name] = f
                    else:
                        shared_feature_counts[f.name] += 1
                        if not self._features_match(shared_feature_defs[f.name], f):
                            raise ValueError(f"Mismatched definitions for shared feature {f.name}")
                else:
                    shape = (*world_size, *f.shape) if f.use_world_size else f.shape
                    self.entity_data[entity_name, f.name] = torch.zeros(*shape, dtype=f.dtype)

        for feature_name, count in shared_feature_counts.items():
            f = shared_feature_defs[feature_name]
            shape = (*world_size, *f.shape) if f.use_world_size else f.shape
            shape = shape + (count,)
            self.shared_features[feature_name] = torch.zeros(*shape, dtype=f.dtype)
            self.shared_feature_indices[feature_name] = {}

            current_index = 0
            for entity_name, cls in self.entity_classes.items():
                if any(feat.name == feature_name and feat.shared for feat in cls.get_feature_definitions()):
                    self.shared_feature_indices[feature_name][entity_name.lower()] = current_index
                    current_index += 1

    @staticmethod
    def _features_match(f1: FeatureDefinition, f2: FeatureDefinition) -> bool:
        return all(
            getattr(f1, attr) == getattr(f2, attr) for attr in ["shape", "dtype", "use_world_size", "observable"]
        )

    def get_feature(self, entity_name: str, feature_name: str) -> torch.Tensor:
        if feature_name in self.shared_features:
            index = self.shared_feature_indices[feature_name][entity_name]
            return self.shared_features[feature_name][..., index]
        elif (entity_name, feature_name) in self.entity_data:
            return self.entity_data[entity_name, feature_name]
        else:
            raise KeyError(f"Feature {feature_name} not found for entity {entity_name}")

    def set_feature(self, entity_name: str, feature_name: str, value: torch.Tensor):
        if feature_name in self.shared_features:
            index = self.shared_feature_indices[feature_name][entity_name]
            self.shared_features[feature_name][..., index] = value
        elif (entity_name, feature_name) in self.entity_data:
            self.entity_data[entity_name, feature_name] = value
        else:
            raise KeyError(f"Feature {feature_name} not found for entity {entity_name}")


class World:
    def __init__(self, entity_classes: List[Type[Entity]], config: DictConfig):
        self.size: Tuple[int, ...] = tuple(config.size)
        self.data_manager = EntityDataManager(entity_classes, config.size)
        self.entities = [cls(self, config["entities"][cls.__name__.lower()]) for cls in entity_classes]
        self.step = 0

    def custom_diffusion(self, scent, energy, energy_contribution=20.0, diffusion_steps=2, max_value=255):
        kernel = generate_diffusion_kernel(size=9, sigma=0.99)
        scent_float = scent.type(torch.float32)

        for _ in range(diffusion_steps):
            diffused = torch_correlate_3d(scent_float, kernel)
            scent_float = diffused.clamp(0, max_value)

        # Energy contribution (non-linear)
        energy_float = energy.type(torch.float32)
        energy_effect = torch.where(
            energy_float > 0,
            torch.log1p(energy_float) / torch.log1p(torch.tensor(max_value, dtype=torch.float32)),
            torch.zeros_like(energy_float)
        )
        scent_float = scent_float + energy_effect * energy_contribution

        return scent_float.round().clamp(0, max_value).type(torch.uint8)

    def diffuse_scent(self):
        scent = self.data_manager.main_td.get(("shared_features", "scent")).clone()
        energy = self.data_manager.main_td.get(("shared_features", "energy")).clone()
        scent = self.custom_diffusion(scent, energy)
        self.data_manager.main_td.set(("shared_features", "scent"), scent)

    def update(self):
        self.data_manager.main_td.set("random", torch.randint(0, 256, self.size, dtype=torch.uint8))
        self.diffuse_scent()
        for entity in self.entities:
            entity.update(self.step)
        self.step += 1


class Plant(Entity):
    features = [
        FeatureDefinition("energy", torch.uint8, shared=True, observable=True),
        FeatureDefinition("scent", torch.uint8, shared=True, observable=True),
        FeatureDefinition("fertility_map", torch.float32),
        FeatureDefinition("seed", torch.uint8),
        FeatureDefinition("crowding", torch.float32),
    ]

    def __init__(
        self,
        world: World,
        config: DictConfig,
    ):
        super().__init__(world, config)
        self.initial_energy = config.initial_energy
        self.init_prob = config.init_prob
        self.growth_prob = config.growth_prob
        self.germination_prob = config.germination_prob
        self.seed_prob = config.seed_prob
        self.initialize()

    def initialize(self):
        energy = self.get_feature("energy")
        fertility_map = (perlin_noise(energy.shape, (5, 5)))
        self.set_feature("fertility_map", fertility_map)
        seed = self.get_feature("seed")
        self.set_feature("seed", torch.zeros(seed.shape, dtype=seed.dtype))
        self.set_feature(
            "energy",
            ~ torch.randint(
                0,
                int(self.init_prob * 255),
                energy.shape,
                dtype=torch.uint8
            ).type(torch.bool) * self.initial_energy
        )

    def update_crowding(self):
        energy = self.get_feature("energy")
        kernel = generate_diffusion_kernel(size=5)
        crowding = torch.clamp(
            torch_correlate_2d(energy.bool().type(torch.float32), kernel, mode="constant"),
            0,
            1
        )
        self.set_feature(
            "crowding",
            crowding
        )

    def grow(self):
        energy = self.get_feature("energy")
        fertility_map = self.get_feature("fertility_map")
        crowding = self.get_feature("crowding")
        rand = self.data_manager.main_td.get("random")

        # Calculate growth probability based on fertility
        growth_prob = self.growth_prob * fertility_map ** 2
        # Combine growth probability with crowding factor
        combined_prob = growth_prob * (1 - crowding)

        # Generate growth mask
        growth = rand < (combined_prob * 255)

        # Apply growth to existing plants
        self.set_feature(
            "energy",
            safe_add(energy, (energy > 0) * growth, inplace=False)
        )

    def seed(self):
        crowding = self.get_feature("crowding")
        seed = self.get_feature("seed")
        rand = self.data_manager.main_td.get("random")

        self.set_feature(
            "seed",
            seed | (rand < self.seed_prob * crowding ** 2 * 255).type(seed.dtype)
        )

    def germinate(self):
        crowding = self.get_feature("crowding")
        energy = self.get_feature("energy")
        seed = self.get_feature("seed")
        rand = self.data_manager.main_td.get("random")

        seed_germination = (
            seed & ~(energy > 0) & (rand < ((1 - crowding) ** 2 * self.germination_prob * 255))
        ).type(torch.uint8)
        safe_add(energy, seed_germination)
        safe_sub(seed, seed_germination)

    def update(self, step: int):
        self.update_crowding()
        self.grow()
        self.seed()
        self.germinate()


class Animal(Entity):
    features = [
        FeatureDefinition("energy", torch.uint8, shared=True, observable=True),
        FeatureDefinition("scent", torch.uint8, shared=True, observable=True),
        FeatureDefinition("id", torch.int32),
        FeatureDefinition("offspring_count", torch.uint8),
    ]

    def __init__(
        self,
        world,
        config
    ):
        super().__init__(world, config)
        self.initial_energy = config.initial_energy
        self.init_odds = config.init_odds
        self.eat_max = config.eat_max
        self.energy_loss = config.energy_loss
        self.divide_threshold = config.divide_threshold
        self.toy_init = config.toy_init
        self.target_key = config.target_key
        self.target_weights = config.target_weights
        self.food_key = config.food_key
        self.opposite_key = config.opposite_key
        self.opposite_weights = config.opposite_weights
        self.initialize()

    def initialize(self):
        energy = self.get_feature("energy")
        if self.toy_init:
            energy[self.world.size[0] // 2, self.world.size[1] // 2] = self.initial_energy
        else:
            energy[:] = (
                torch.randint(
                    0,
                    self.init_odds,
                    energy.shape,
                    dtype=torch.uint8
                ) == 0
            ) * self.initial_energy

    def update(self, step: int):

        energy = self.get_feature("energy")
        safe_sub(energy, self.energy_loss)

        random = self.data_manager.main_td.get("random")
        random_fn = lambda x: random

        if isinstance(self.target_key, ListConfig):
            target_scent = [
                self.data_manager.get_feature(key, "scent") for key in self.target_key
            ]
        else:
            target_scent = self.data_manager.get_feature(self.target_key, "scent")

        if isinstance(self.opposite_key, ListConfig):
            print(self.opposite_key)
            opposite_scent = [
                self.data_manager.get_feature(key, "scent") for key in self.opposite_key
            ]
        else:
            opposite_scent = self.data_manager.get_feature(self.opposite_key, "scent")

        food_energy = self.data_manager.get_feature(self.food_key, "energy")

        self.move(
            target=target_scent,
            target_weights=self.target_weights,
            opposite_scent=opposite_scent,
            opposite_scent_weights=self.opposite_weights,
            carried_features_self=("offspring_count", "id"),
            carried_feature_fns_self=(lambda x: safe_add(x, 1), lambda x: x),
            carried_features_offspring=("id",),
            carried_feature_fns_offspring=(random_fn,),
            agent_action=None
        )
        self.eat(food_energy)

    @staticmethod
    def get_direction_masks(
        directions: torch.Tensor,
        entity_energy: torch.Tensor,
        clearance_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        clearance_kernel_size: Optional[int] = 5,
    ) -> Dict[int, torch.Tensor]:
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

    def prepare_move(
        self,
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

        direction_masks = self.get_direction_masks(directions, entity_energy, clearance_mask, clearance_kernel_size)
        # for k, v in direction_masks.items():
        #     print(f"Direction {k}: {v.sum()}")

        return direction_masks

    def perform_move(
        self,
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

    def move(
        self,
        target: Union[torch.Tensor, List[torch.Tensor]],
        target_weights: List[float],
        opposite_scent: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        opposite_scent_weights: Optional[List[float]] = None,
        clearance_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        clearance_kernel_size: Optional[int] = 5,
        divide_threshold: Optional[int] = 250,
        divide_fn_self: Optional[Callable] = lambda x: x // 2,
        divide_fn_offspring: Optional[Callable] = lambda x: x // 4,
        carried_features_self: Optional[Tuple[str, ...]] = None,
        carried_features_offspring: Optional[Tuple[str, ...]] = None,
        carried_feature_fns_self: Optional[Tuple[Callable, ...]] = None,
        carried_feature_fns_offspring: Optional[Tuple[Callable, ...]] = None,
        agent_action: Optional[torch.Tensor] = None,
    ):
        entity_energy = self.get_feature("energy")

        if agent_action is not None:
            # I.e. if we already have an action selected by the RL model.
            direction_masks = self.get_direction_masks(
                agent_action,
                entity_energy,
                self.world.data_manager.get_feature("obstacle", "mask"),
                5
            )
        else:
            direction_masks = self.prepare_move(
                entity_energy,
                target,
                target_weights,
                opposite_scent,
                opposite_scent_weights,
                clearance_mask,
                clearance_kernel_size
            )

        carried_features_self_tensors = [self.get_feature(f) for f in (carried_features_self or [])]
        carried_features_offspring_tensors = [self.get_feature(f) for f in (carried_features_offspring or [])]

        self.perform_move(
            entity_energy=entity_energy,
            direction_masks=direction_masks,
            divide_threshold=divide_threshold,
            divide_fn_self=divide_fn_self,
            divide_fn_offspring=divide_fn_offspring,
            carried_features_self=carried_features_self_tensors,
            carried_feature_fns_self=carried_feature_fns_self,
            carried_features_offspring=carried_features_offspring_tensors,
            carried_feature_fns_offspring=carried_feature_fns_offspring
        )

        # Update the features in the data manager
        self.set_feature("energy", entity_energy)
        for feature, tensor in zip(carried_features_self or [], carried_features_self_tensors):
            self.set_feature(feature, tensor)
        for feature, tensor in zip(carried_features_offspring or [], carried_features_offspring_tensors):
            self.set_feature(feature, tensor)

    def eat(self, food_energy: torch.Tensor):
        energy = self.get_feature("energy")
        old_food_energy = food_energy.clone()
        safe_sub(food_energy, (energy > 0).type(torch.uint8) * self.eat_max)
        delta = old_food_energy - food_energy
        safe_add(energy, delta // 2)


class Herbivore(Animal):
    pass


class Predator(Animal):
    pass

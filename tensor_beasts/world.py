from typing import List, Set, Type, Tuple, Optional, Union, Dict, Callable
import re
import torch
from tensordict import TensorDict
from omegaconf import DictConfig, OmegaConf

from tensor_beasts.entities.entity import Entity, DependencyCycleError
from tensor_beasts.features.feature import SharedFeature
from tensor_beasts.observations import build_observation
from tensor_beasts.registry import get_entity_class, entity_name_to_registry_name
from tensor_beasts.snapshot import WorldSnapshot


class World:
    def __init__(self, config: DictConfig):
        self.size: Tuple[int, ...] = tuple(config.size)
        self.config = config
        self.td = TensorDict({}, batch_size=[])
        self.entity_dict: Dict[str, Entity] = {}
        self.shared_features_dict: Dict[str, SharedFeature] = {}

        # Pre-populate shared feature registry with global config
        # This must happen BEFORE entities are created so children inherit these values
        self._init_shared_feature_registry(config)

        # Create all entities first (but don't initialize yet)
        for entity_name, entity_config in config.entities.items():
            entity_class = get_entity_class(entity_name)
            self.entity_dict[entity_name] = entity_class(self, entity_config)

        # Compute entity initialization order based on cross-entity dependencies
        # Cache this since it never changes after initialization
        self._entity_order = self._get_entity_initialization_order()

        # Initialize entities in dependency order
        for entity_name in self._entity_order:
            entity = self.entity_dict[entity_name]
            entity.initialize()
            # Register shared features (keyed by shared_name to avoid duplicates)
            for feature_name, feature_class in entity.__features__.items():
                if issubclass(feature_class, SharedFeature):
                    # Use shared_name if defined, else feature name
                    shared_name = getattr(feature_class, 'shared_name', None) or feature_class.name
                    if shared_name not in self.shared_features_dict:
                        self.shared_features_dict[shared_name] = feature_class(
                            self.td,
                            is_parent=True,
                            shape_prefix=self.size,
                            key_prefix=("shared_features",)
                        )

        self.step = 0

    def _init_shared_feature_registry(self, config: DictConfig):
        """
        Pre-populate the shared feature registry with global config values.

        This must be called BEFORE entities are created so that SharedFeature
        children inherit these values instead of the first child setting them.
        """
        # Get or create the registry on the TensorDict
        if not hasattr(self.td, '_shared_feature_registry'):
            self.td._shared_feature_registry = {}
        registry = self.td._shared_feature_registry

        # Check for shared_features config
        shared_features_config = getattr(config, 'shared_features', None)
        if shared_features_config is None:
            return

        # Pre-populate scent shared config if specified
        scent_config = getattr(shared_features_config, 'scent', None)
        if scent_config is not None:
            # Initialize the scent registry entry with the global shared config
            registry['scent'] = {
                'count': 0,
                'shared_key_prefix': 'shared_features',
                'shared_key': ('shared_features', 'scent'),
                'shared_config': {
                    'diffusion_steps': scent_config.diffusion_steps,
                    'kernel_size': scent_config.kernel_size,
                    'kernel_sigma': scent_config.kernel_sigma,
                    'max_decay': scent_config.max_decay,
                    'decay_half': scent_config.decay_half,
                },
            }

    def _build_entity_dependency_graph(self) -> Dict[str, Set[str]]:
        """
        Build a dependency graph of entities based on cross-entity feature dependencies.

        Returns:
            Dict mapping entity_name -> set of entity names it depends on
        """
        graph: Dict[str, Set[str]] = {name: set() for name in self.entity_dict}

        for entity_name, entity in self.entity_dict.items():
            entity_name_lower = entity_name.lower()

            for feature_name, feature_class in entity.__features__.items():
                depends_on = getattr(feature_class, 'depends_on', {})

                for dep_name, dep_target in depends_on.items():
                    if isinstance(dep_target, tuple) and len(dep_target) == 2:
                        # Cross-entity dependency: ("other_entity", "feature")
                        other_entity_lower, _ = dep_target
                        # Convert to registry name (PascalCase)
                        other_entity = entity_name_to_registry_name(other_entity_lower)

                        if other_entity in self.entity_dict and other_entity != entity_name:
                            graph[entity_name].add(other_entity)

                    elif isinstance(dep_target, str):
                        # Check if this is actually referencing another entity via config
                        # Look at the feature's default_config for ${key:...} references
                        default_config = getattr(feature_class, 'default_config', None)
                        if default_config:
                            config_dict = OmegaConf.to_container(default_config, resolve=False)
                            for key, value in config_dict.items():
                                if isinstance(value, str) and '${key:' in value:
                                    # Parse ${key:entity,feature}
                                    import re
                                    match = re.search(r'\$\{key:([^,}]+),([^}]+)\}', value)
                                    if match:
                                        ref_entity_lower = match.group(1).strip()
                                        ref_entity = entity_name_to_registry_name(ref_entity_lower)
                                        if ref_entity in self.entity_dict and ref_entity != entity_name:
                                            graph[entity_name].add(ref_entity)

        return graph

    def _get_entity_initialization_order(self) -> List[str]:
        """
        Get the order in which entities should be initialized.

        Returns:
            List of entity names in initialization order (dependencies first)
        """
        graph = self._build_entity_dependency_graph()
        return Entity._topological_sort(graph)

    def __getattr__(self, item):
        # Try exact match first, then case-insensitive match for entities
        if item in self.entity_dict:
            return self.entity_dict[item]

        # Try lowercase -> PascalCase conversion for entity names
        # This allows world.terrain to access entity_dict["Terrain"]
        from tensor_beasts.registry import entity_name_to_registry_name
        pascal_name = entity_name_to_registry_name(item)
        if pascal_name in self.entity_dict:
            return self.entity_dict[pascal_name]

        # Try shared features
        if item in self.shared_features_dict:
            return self.shared_features_dict[item]

        raise AttributeError(f"Entity or SharedFeature '{item}' not found")

    @property
    def observable(self):
        features = []
        for entity in self.entity_dict.values():
            features.extend(entity.features())
        return build_observation(features)

    def initialize(self):
        """Initialize all entities in dependency order."""
        for entity_name in self._entity_order:
            self.entity_dict[entity_name].initialize()

    def snapshot(self) -> WorldSnapshot:
        data = {}
        for key in self.td.keys(True, True):
            value = self.td.get(key)
            if isinstance(value, torch.Tensor):
                data[key] = value.clone()
        return WorldSnapshot(step=self.step, data=data)

    def reset(self):
        self.td.clear()
        self.initialize()
        self.step = 0

    def update(self, action_td: Optional[TensorDict] = None):
        """Update all entities in dependency order."""
        self.td.set("random", torch.randint(0, 256, self.size, dtype=torch.uint8))

        # Update entities (includes emission for SharedFeatures)
        for entity_name in self._entity_order:
            entity = self.entity_dict[entity_name]
            entity.update(action=action_td.get(entity_name, None) if action_td is not None else None)

        # Run shared diffusion on parent SharedFeatures (batched across all slices)
        for shared_feature in self.shared_features_dict.values():
            if hasattr(shared_feature, 'diffuse'):
                shared_feature.diffuse(self.step)

        self.step += 1

    def inspect(self, x: int, y: int):
        output = f"World at ({x}, {y})\n"
        for entity in self.entity_dict.values():
            output += entity.inspect(x, y)
        print(output)

    def spawn_entity(self, entity_name: str, x: Optional[int] = None, y: Optional[int] = None):
        """Spawn a single entity at a random or specified position."""
        entity_name = entity_name_to_registry_name(entity_name)
        if entity_name not in self.entity_dict:
            print(f"Entity '{entity_name}' not found")
            return

        entity = self.entity_dict[entity_name]

        # Get random position if not specified
        if x is None:
            x = torch.randint(0, self.size[1], (1,)).item()
        if y is None:
            y = torch.randint(0, self.size[0], (1,)).item()

        # Set initial values based on entity config
        config = entity.config
        if hasattr(entity, 'energy'):
            entity.energy.data[y, x] = config.get('initial_energy', 50)
        if hasattr(entity, 'biomass'):
            entity.biomass.data[y, x] = config.get('initial_biomass', 150)

        print(f"Spawned {entity_name} at ({x}, {y})")

    def initialize_herbivore(self):
        """Spawn a herbivore at a random position."""
        self.spawn_entity("Herbivore")

    def initialize_predator(self):
        """Spawn a predator at a random position."""
        self.spawn_entity("Predator")

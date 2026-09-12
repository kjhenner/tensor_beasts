import abc
from typing import Optional, Dict, List, Set, Type, ClassVar, Any
import torch
from omegaconf import DictConfig, OmegaConf

from tensor_beasts.config.namespace import ConfigNamespace, to_namespace
from tensor_beasts.features.feature import Feature, SharedFeature


class DependencyCycleError(Exception):
    """Raised when a circular dependency is detected in feature dependencies."""
    pass


class EntityMeta(abc.ABCMeta):
    def __new__(mcs, name, bases, namespace):
        features: Dict[str, Type[Feature]] = {}
        for base in bases:
            if hasattr(base, '__features__'):
                features.update(base.__features__)

        annotations = namespace.get('__annotations__', {})

        for key, value in annotations.items():
            if isinstance(value, type) and issubclass(value, Feature):
                features[key] = value

        namespace['__features__'] = features
        return super().__new__(mcs, name, bases, namespace)


class Entity(abc.ABC, metaclass=EntityMeta):
    __features__: ClassVar[Dict[str, Type[Feature]]]
    default_config = DictConfig({})

    # Cache for computed orderings
    _init_order: Optional[List[str]] = None
    _update_order: Optional[List[str]] = None

    def __init__(self, world: 'World', config: DictConfig):
        self.world = world
        self.td = world.td

        # Build merged default config from class hierarchy (parent configs first)
        merged_default = {}
        for cls in reversed(self.__class__.__mro__):
            if hasattr(cls, 'default_config') and cls.default_config is not None:
                parent_config = OmegaConf.to_container(cls.default_config, resolve=False)
                merged_default.update(parent_config)

        # Apply instance config on top
        merged_default.update(OmegaConf.to_container(config, resolve=False))

        # Validate and transform through Pydantic
        # This converts "entity:feature" strings to tuple keys
        from tensor_beasts.config.load import _get_entity_config_model
        config_model = _get_entity_config_model(self.__class__.__name__)
        validated = config_model.model_validate(merged_default)

        # Get dict for feature configs (features still use OmegaConf)
        # Use exclude_unset to not override feature defaults with Pydantic defaults
        validated_dict = validated.model_dump(exclude_unset=True)

        # But we need the full config with defaults for entity-level access
        full_dict = validated.model_dump()

        # Convert to ConfigNamespace for attribute access with .get() support
        self.config = self._dict_to_namespace(full_dict)

        for feature_name, feature_class in self.__features__.items():
            # Pass raw dict to features (they use OmegaConf.merge internally)
            feature_config = validated_dict.get(feature_name, None)
            feature = feature_class(
                td=self.world.td,
                key_prefix=(self.__class__.__name__.lower(),),
                shape_prefix=self.world.config.size,
                config=feature_config
            )
            setattr(self, feature_name, feature)

        # Compute initialization and update orders
        self._init_order = None
        self._update_order = None

    def _dict_to_namespace(self, d: Dict) -> ConfigNamespace:
        """Recursively convert a dict to ConfigNamespace."""
        return to_namespace(d)

    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """
        Build a dependency graph for features within this entity.

        Returns:
            Dict mapping feature_name -> set of feature names it depends on
        """
        graph: Dict[str, Set[str]] = {}
        feature_names = set(self.__features__.keys())

        for feature_name, feature_class in self.__features__.items():
            deps = set()
            depends_on = getattr(feature_class, 'depends_on', {})

            for dep_name, dep_target in depends_on.items():
                # dep_target can be:
                # - "feature_name" (local feature in same entity)
                # - ("entity", "feature") tuple (cross-entity, handled at World level)
                if isinstance(dep_target, str):
                    # Local dependency
                    if dep_target in feature_names:
                        deps.add(dep_target)
                # Cross-entity dependencies are ignored here (handled by World)

            graph[feature_name] = deps

        return graph

    @staticmethod
    def _topological_sort(graph: Dict[str, Set[str]]) -> List[str]:
        """
        Perform topological sort using Kahn's algorithm.

        Args:
            graph: Dict mapping node -> set of nodes it depends on

        Returns:
            List of nodes in dependency order (dependencies come first)

        Raises:
            DependencyCycleError: If a cycle is detected
        """
        # Calculate in-degree for each node
        in_degree: Dict[str, int] = {node: 0 for node in graph}
        for node, deps in graph.items():
            for dep in deps:
                if dep in in_degree:
                    # dep has an edge TO node, so node's in-degree increases
                    pass
                # We need reverse graph: who depends on whom

        # Build reverse graph: node -> nodes that depend on it
        reverse_graph: Dict[str, Set[str]] = {node: set() for node in graph}
        for node, deps in graph.items():
            for dep in deps:
                if dep in reverse_graph:
                    reverse_graph[dep].add(node)

        # Calculate in-degree (number of dependencies)
        in_degree = {node: len(deps) for node, deps in graph.items()}

        # Start with nodes that have no dependencies
        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            # Sort for deterministic order
            queue.sort()
            node = queue.pop(0)
            result.append(node)

            # For each node that depends on this one, reduce its in-degree
            for dependent in reverse_graph[node]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(graph):
            # Find the cycle for error message
            remaining = set(graph.keys()) - set(result)
            raise DependencyCycleError(
                f"Circular dependency detected among features: {remaining}"
            )

        return result

    def _get_initialization_order(self) -> List[str]:
        """
        Get the order in which features should be initialized.

        Returns:
            List of feature names in initialization order
        """
        if self._init_order is None:
            graph = self._build_dependency_graph()
            self._init_order = self._topological_sort(graph)
        return self._init_order

    def _get_update_order(self) -> List[str]:
        """
        Get the order in which features should be updated.

        For now, this is the same as initialization order.
        Could be different if some features don't need updating.

        Returns:
            List of feature names in update order
        """
        if self._update_order is None:
            # Use same order as initialization for now
            self._update_order = self._get_initialization_order()
        return self._update_order

    def inspect(self, x: int, y: int):
        output = f"Entity: {self.__class__.__name__}"
        for feature_name in self.__features__.keys():
            output += "\n\t" + self.__getattribute__(feature_name).inspect(x, y)
        return output

    def get_feature(self, name: str) -> Feature:
        if name not in self.__features__:
            available = ", ".join(sorted(self.__features__.keys()))
            raise ValueError(f"Unknown feature '{name}'. Known features: {available}")
        return getattr(self, name)

    def features(self):
        return [getattr(self, name) for name in self.__features__.keys()]

    def initialize(self):
        """
        Initialize all features in dependency order.

        Subclasses can override this if they need custom initialization logic,
        but should call super().initialize() or use _get_initialization_order().
        """
        for feature_name in self._get_initialization_order():
            feature = getattr(self, feature_name)
            feature.initialize_data()

    def update(self, action: Optional[torch.Tensor] = None):
        """
        Update all features in dependency order.

        Subclasses can override this if they need custom update logic,
        but should call super().update() or use _get_update_order().
        """
        for feature_name in self._get_update_order():
            feature = getattr(self, feature_name)
            feature.update(self.world.step)

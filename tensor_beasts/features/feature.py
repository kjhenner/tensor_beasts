import abc
from typing import Dict, Tuple, Set, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table
from tensordict import TensorDict, NestedKey

from tensor_beasts.config.namespace import ConfigNamespace, resolve_to_namespace


class Feature(abc.ABC):
    name: str
    shape: Tuple[int, ...] = None
    dtype: torch.dtype = None
    default_tags: Set[str] = None
    default_config: DictConfig = None
    depends_on: Dict[str, str] = {}  # Maps dependency name -> feature name or (entity, feature) tuple

    def __init__(
        self,
        td: TensorDict,
        key_prefix: NestedKey,
        shape_prefix: Optional[Tuple[int, ...]] = tuple(),
        additional_tags: Optional[Tuple[str, ...]] = None,
        config: Optional[DictConfig] = None
    ):
        # Build merged default config from class hierarchy (parent configs first)
        # This allows subclasses to inherit and override parent defaults
        merged_default = {}
        for cls in reversed(self.__class__.__mro__):
            if hasattr(cls, 'default_config') and cls.default_config is not None:
                parent_config = OmegaConf.to_container(cls.default_config, resolve=False)
                merged_default.update(parent_config)

        # Merge with instance config, then resolve once. Interpolations are
        # resolved here rather than on every access in the update loop.
        merged = OmegaConf.merge(OmegaConf.create(merged_default), config or {})
        self.config: ConfigNamespace = resolve_to_namespace(merged)
        self.td = td
        if key_prefix not in td:
            td[key_prefix] = TensorDict({}, batch_size=[])
        self.shape = tuple(shape_prefix + tuple(self.shape or ()))
        self.key = (*key_prefix, self.name) if isinstance(key_prefix, tuple) else (key_prefix, self.name)
        self.tags = set(additional_tags or ())
        self.tags.update(self.default_tags or set())

    def render(self):
        if self.data.ndim == 2:
            return self.data.unsqueeze(-1).expand(-1, -1, 3)
        else:
            return self.data

    def inspect(self, x: int, y: int):
        if self.data.ndim == 0:
            return f"{self.name}: {self.data.item():.4f}"
        elif self.data.ndim == 2:
            return f"{self.name}: {self.data[y, x]}"
        else:
            data = self.data[y, x]
            grid = [
            f"{data[0]:.4f} {data[1]:.4f} {data[2]:.4f}",
            f"{data[3]:.4f}        {data[4]:.4f}",
            f"{data[5]:.4f} {data[6]:.4f} {data[7]:.4f}"
            ]
            return f"{self.name}:\n" + "\n".join(grid)

    def inspect_3d(self, data):
        if data.ndim != 3:
            raise ValueError("This function expects a 3D tensor.")

        rows, cols, depth = data.shape
        console = Console()
        table = Table(padding=0, collapse_padding=True)
        table.show_header = False
        table.show_lines = True

        # Add columns for each element in the 3x3 grid
        for _ in range(cols * 3):
            table.add_column(justify="center", width=8)

        for y in range(rows):
            row_data = []
            for x in range(cols):
                cell_data = data[y, x]
                if depth == 1:
                    row_data.extend([f"{cell_data[0]:.4f}", "", ""])
                else:
                    for i in range(9):
                        if i < 4:
                            row_data.append(f"{cell_data[i]:.4f}")
                        elif i == 4:
                            row_data.append("")
                        else:
                            row_data.append(f"{cell_data[i-1]:.4f}")

            # Add three rows for each y
            for i in range(3):
                table.add_row(*row_data[i::3])
            table.add_section()

        console.print(table)

    def update(self, step: int):
        pass

    def zero_init(self):
        self.data = torch.zeros(self.shape, dtype=self.dtype)

    def initialize_data(self, *args, **kwargs):
        self.zero_init()

    @property
    def data(self):
        return self.td.get(self.key)

    @data.setter
    def data(self, value):
        self.td.set(self.key, value, inplace=True)


class SharedFeature(Feature, abc.ABC):
    """
    A feature that shares a backing tensor across multiple entities.

    Multiple entities can have SharedFeatures of the same family (determined by
    `shared_name`). Each entity gets its own 2D slice of a shared 3D tensor.

    The registry is scoped to the TensorDict, so multiple Worlds can coexist
    without interference.

    Class attributes:
        name: The feature name used for entity-specific TensorDict keys
        shared_name: The name used for the shared tensor (defaults to name).
                     Override this in subclasses to share a tensor with a
                     different feature class (e.g., CarrionScent shares with Scent)
        shared_config_keys: List of config keys that must be identical across
                           all instances sharing a tensor (e.g., diffusion params)

    Example:
        class Scent(SharedFeature):
            name = "scent"
            shared_config_keys = ["kernel_size", "kernel_sigma", "diffusion_steps"]
            # shared_name defaults to "scent"

        class CarrionScent(Scent):
            name = "carrion_scent"
            shared_name = "scent"  # Share tensor with Scent family
            # Inherits shared_config_keys - diffusion params must match Scent
    """
    # Registry key in TensorDict metadata
    _REGISTRY_KEY = ("__meta__", "shared_feature_registry")

    # Instance attributes (set in __init__)
    _is_parent: bool = False
    _shared_key: Tuple = None
    idx: int = None

    # Override in subclass to share tensor with another feature family
    shared_name: str = None  # Defaults to self.name if not set

    # Config keys that must be identical across all instances sharing a tensor
    shared_config_keys: Tuple[str, ...] = ()

    def __init__(
        self,
        td: TensorDict,
        is_parent: bool = False,
        key_prefix: NestedKey = None,
        shape_prefix: Tuple[int, ...] = tuple(),
        additional_tags: Tuple[str, ...] = None,
        config: Optional[DictConfig] = None,
        shared_key_prefix: NestedKey = "shared_features"
    ):
        super().__init__(td, key_prefix, shape_prefix, additional_tags, config)
        if shared_key_prefix not in td:
            td[shared_key_prefix] = TensorDict({}, batch_size=[])

        # Get or create the registry scoped to this TensorDict
        registry = self._get_registry(td)

        # Determine the shared tensor name (defaults to feature name)
        effective_shared_name = self.shared_name or self.name

        # Initialize registry entry for this shared_name if needed
        if effective_shared_name not in registry:
            registry[effective_shared_name] = {
                "count": 0,
                "shared_key_prefix": shared_key_prefix,
                "shared_key": (*shared_key_prefix, effective_shared_name) if isinstance(shared_key_prefix, tuple) else (shared_key_prefix, effective_shared_name),
                "shared_config": None,  # Will be set by first instance
            }

        reg = registry[effective_shared_name]

        # Validate consistent shared_key_prefix
        assert reg["shared_key_prefix"] == shared_key_prefix, \
            f"SharedFeature '{effective_shared_name}' shared_key_prefix must be consistent"

        # Handle shared config (first instance sets it, others must match)
        if self.shared_config_keys and not is_parent:
            my_shared_config = {k: getattr(self.config, k, None) for k in self.shared_config_keys}
            if reg["shared_config"] is None:
                # First instance - set the shared config
                reg["shared_config"] = my_shared_config
            else:
                # Subsequent instance - use the shared config (override local)
                for key, shared_val in reg["shared_config"].items():
                    local_val = my_shared_config.get(key)
                    if local_val != shared_val:
                        # Use shared value, could log warning here
                        if hasattr(self.config, key):
                            setattr(self.config, key, shared_val)

        self._is_parent = is_parent
        self._shared_key = reg["shared_key"]
        self._effective_shared_name = effective_shared_name
        self._registry_ref = registry  # Keep reference for later access

        if not self._is_parent:
            self.idx = reg["count"]
            reg["count"] += 1

    @classmethod
    def _get_registry(cls, td: TensorDict) -> Dict:
        """Get or create the SharedFeature registry scoped to this TensorDict."""
        # Store registry in a regular dict attached to the TensorDict
        # We use object attribute to avoid TensorDict key restrictions
        if not hasattr(td, '_shared_feature_registry'):
            td._shared_feature_registry = {}
        return td._shared_feature_registry

    def _get_shared_count(self) -> int:
        """Get the current count of instances sharing this tensor."""
        return self._registry_ref[self._effective_shared_name]["count"]

    def zero_init(self):
        # self.shape stays the 2-D slice shape. It used to have the slice count
        # appended to it here, which made this method non-idempotent: calling it
        # twice produced (H, W, C, C) and broke World.reset(), and in fact broke
        # a first initialize() too, since zero_init runs more than once there.
        count = self._get_shared_count()
        shared_shape = self.shape + (count,)
        if self._shared_key not in self.td:
            self.td[self._shared_key] = torch.zeros(shared_shape, dtype=self.dtype)
        if self.key not in self.td:
            self.td[self.key] = self.td[self._shared_key][:, :, self.idx]
        self.data = torch.zeros_like(self.td[self.key])

    @property
    def data(self):
        if self._is_parent:
            return self.td[self._shared_key]
        return self.td[self.key]

    @data.setter
    def data(self, value):
        if self._is_parent:
            self.td.set(self._shared_key, value, inplace=True)
        else:
            self.td.set(self.key, value, inplace=True)

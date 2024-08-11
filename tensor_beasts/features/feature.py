import abc
from typing import Tuple, Set, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table
from tensordict import TensorDict, NestedKey

OmegaConf.register_new_resolver(
    "key",
    lambda *args: tuple(args)
)


class Feature(abc.ABC):
    name: str
    shape: Tuple[int, ...] = None
    dtype: torch.dtype = None
    default_tags: Set[str] = None
    default_config: DictConfig = None

    def __init__(
        self,
        td: TensorDict,
        key_prefix: NestedKey,
        shape_prefix: Optional[Tuple[int, ...]] = tuple(),
        additional_tags: Optional[Tuple[str, ...]] = None,
        config: Optional[DictConfig] = None
    ):
        self.config = self.default_config
        if config:
            self.config.update(config)
        self.td = td
        if key_prefix not in td:
            td[key_prefix] = TensorDict({}, batch_size=[])
        self.shape = tuple(shape_prefix + tuple(self.shape or ()))
        self.key = (*key_prefix, self.name) if isinstance(key_prefix, tuple) else (key_prefix, self.name)
        self.tags = set(additional_tags or ()).update(self.default_tags or set())

    def render(self):
        if self.data.ndim == 2:
            return self.data.unsqueeze(-1).expand(-1, -1, 3)
        else:
            return self.data

    def inspect(self, x: int, y: int):
        if self.data.ndim == 2:
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
        print(f"Initializing {self.name} with args: {args} and kwargs: {kwargs}")
        self.zero_init()

    @property
    def data(self):
        return self.td.get(self.key)

    @data.setter
    def data(self, value):
        self.td.set(self.key, value, inplace=True)


class SharedFeature(Feature, abc.ABC):
    _count = 0
    _key_prefix = None
    _is_parent = False

    def __init__(
        self,
        td: TensorDict,
        is_parent: bool = False,
        key_prefix: NestedKey = "shared_features",
        shape_prefix: Tuple[int, ...] = tuple(),
        additional_tags: Tuple[str, ...] = None,
        config: Optional[DictConfig] = None
    ):
        super().__init__(td, key_prefix, shape_prefix, additional_tags, config)
        self._is_parent = is_parent
        if not self._is_parent:
            self.idx = type(self)._count
            type(self)._count += 1
        if not self._key_prefix:
            self._key_prefix = key_prefix
        else:
            assert self._key_prefix == key_prefix, "SharedFeature key_prefix must be the same for all instances."

    def zero_init(self):
        if self.key not in self.td:
            self.td[self.key] = torch.zeros(self.shape, dtype=self.dtype)
        self.data = torch.zeros(self.shape[:-1], dtype=self.dtype)

    @property
    def data(self):
        if self._is_parent:
            return self.td[self.key]
        return self.td[self.key][self.idx]

    @data.setter
    def data(self, value):
        if self._is_parent:
            self.td[self.key] = value
        else:
            self.td[self.key][self.idx] = value

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass(frozen=True)
class WorldSnapshot:
    step: int
    data: Dict[tuple, torch.Tensor]

    def get(self, key) -> torch.Tensor:
        if key not in self.data:
            raise KeyError(f"Snapshot key not found: {key}")
        return self.data[key]

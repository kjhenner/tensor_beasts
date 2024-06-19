from typing import Optional, Dict, Tuple

import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np

from tensor_beasts.world import World


class TensorBeastsEnv(gym.Env):
    def __init__(
        self,
        size: int,
        config: Optional[Dict] = None,
        scalars: Optional[Dict] = None,
    ):
        super(TensorBeastsEnv, self).__init__()

        # Initialize the World
        self.world = World(
            size=size,
            config=config,
            scalars=scalars,
        )

        self.obs_shape = self.world.observable.shape

        # Define the observation space:
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=self.obs_shape,
            dtype=np.uint8
        )

        # Define the action space:
        self.action_space = spaces.Box(
            low=0,
            high=6,
            shape=(self.world.width, self.world.height),
            dtype=np.uint8
        )

    def reset(self, *args) -> Tuple[torch.Tensor, Dict]:
        # Reset the world state and return the initial observation
        self.world.__init__(size=self.world.width)  # Reinitialize the world
        info = {}
        return self.world.observable, info

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool, bool, Dict]:
        action = action.reshape(self.world.width, self.world.height)
        done, info = self.world.update(action)
        reward: torch.Tensor = torch.sum(self.world.entity_scores('herbivore'))
        return self.world.observable, reward, done, False, info

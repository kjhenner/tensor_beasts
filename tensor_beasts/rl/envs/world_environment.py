from typing import Optional, Dict, Tuple

import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np

from tensor_beasts.world import World


class TensorBeastsEnv(gym.Env):

    def __init__(
        self,
        device: str = "cpu",
        world: Optional[World] = None
    ):
        super(TensorBeastsEnv, self).__init__()
        if world is None:
            self.world = World(size=128)
        else:
            self.world = world

        self.obs_shape = self.world.observable.shape

        # Define the observation space:

        # Define a compound observation space
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=0,
                high=255,
                shape=self.obs_shape,
                dtype=np.uint8
            ),
            "rgb_array": spaces.Box(
                low=0,
                high=255,
                shape=(self.obs_shape[1], self.obs_shape[0], 3),
                dtype=np.uint8
            ),
            "info": spaces.Dict({
                k: spaces.Box(low=0, high=np.infty, shape=(1,), dtype=np.int32) for k in self.world.collect_info().keys()
             })
        })

        # Define the action space:
        self.action_space = spaces.Box(
            low=0,
            high=6,
            shape=(self.world.width * self.world.height,),
            dtype=np.uint8
        )

    def reset(self, *args, **kwargs) -> Tuple[Dict, Dict]:
        # Reset the world state and return the initial observation
        self.world.__init__(size=self.world.width)  # Reinitialize the world
        # TorchRL expects info to be packed with the observation rather than as a separate return value
        observation = {
            "observation": self.world.observable,
            # "rgb_array": self.world.energy.clone,
            # "info": self.world.collect_info()
        }
        return observation, {}

    def step(self, action: torch.Tensor) -> Tuple[Dict, torch.Tensor, bool, bool, Dict]:
        action = action.reshape(self.world.width, self.world.height)
        self.world.update(action)
        reward: torch.Tensor = torch.sum(self.world.entity_scores('herbivore'))
        # TorchRL expects info to be packed with the observation rather than as a separate return value
        observation = {
            "observation": self.world.observable,
            "rgb_array": self.world.energy.clone(),
            "info": self.world.collect_info()
        }
        done = not bool(torch.sum(self.world.herbivore.energy))
        return observation, reward, done, False, {}

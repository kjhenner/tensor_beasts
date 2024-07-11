from typing import Optional, Dict, Tuple

import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
from gymnasium.core import RenderFrame
from omegaconf import DictConfig

from tensor_beasts.world import World


class TensorBeastsEnv(gym.Env):

    def __init__(
        self,
        device: str = "cpu",
        world: Optional[World] = None,
        num_actions: int = 5,
        world_cfg: DictConfig = None,
    ):
        super(TensorBeastsEnv, self).__init__()
        self.world_cfg = world_cfg
        if world is None:
            self.world = World(config=world_cfg)
        else:
            self.world = world

        self.obs_shape = self.world.observable.shape
        self.num_actions = num_actions

        # Define the observation space:

        # Define a compound observation space
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=0,
                high=255,
                shape=self.obs_shape,
                dtype=np.uint8
            ),
            "mask": spaces.Box(
                low=0,
                high=1,
                shape=self.obs_shape,
                dtype=bool
            ),
            "rgb_array": spaces.Box(
                low=0,
                high=255,
                shape=(self.obs_shape[1] * 3, self.obs_shape[0] * 3, 3),
                dtype=np.uint8
            ),
            # "info": spaces.Dict({
            #     k: spaces.Box(low=0, high=np.infty, shape=(1,), dtype=np.int32) for k in self.world.collect_info().keys()
            #  })
        })

        self.reward_range = (-np.inf, np.inf)

        # Define the action space:
        # self.action_space = spaces.Box(
        #     low=0,
        #     high=num_actions,
        #     shape=(self.world.width * self.world.height,),
        #     dtype=np.float32
        # )
        self.action_space = spaces.MultiDiscrete(
            nvec=[num_actions] * (self.world.width * self.world.height),
            dtype=np.int32
        )

    def reset(self, *args, **kwargs) -> Tuple[Dict, Dict]:
        # Reset the world state and return the initial observation
        self.world.__init__(self.world_cfg)  # Reinitialize the world
        # TorchRL expects info to be packed with the observation rather than as a separate return value
        terminated = not bool(torch.sum(self.world.herbivore.energy))
        assert not terminated, "The world should not be terminated after a reset"
        observation = {
            "observation": self.world.observable.clone(),
            "mask": self.world.herbivore.energy > 0,
            "rgb_array": self.world.rgb_array(),
            # "info": self.world.collect_info()
        }
        return observation, {}

    def step(self, action: np.ndarray) -> Tuple[Dict, torch.Tensor, bool, bool, Dict]:
        if action.shape[-1] == self.num_actions:
            action = action.argmax(-1)

        action = action.reshape(self.world.width, self.world.height)
        self.world.update(action)
        reward = self.world.entity_scores('herbivore', reward_mode=self.world_cfg.get("reward_mode", "default"))
        # TorchRL expects info to be packed with the observation rather than as a separate return value
        observation = {
            "observation": self.world.observable.clone(),
            "mask": self.world.herbivore.energy > 0,
            "rgb_array": self.world.rgb_array(),
            # "info": self.world.collect_info()
        }
        terminated = not bool(torch.sum(self.world.herbivore.energy))
        # observation, reward, terminated, truncated, info
        return observation, reward, terminated, False, {}

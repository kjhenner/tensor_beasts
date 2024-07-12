# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn
import torch.optim
from torchrl.data import CompositeSpec
from torchrl.envs import (
    CatFrames,
    DoubleToFloat,
    EndOfLifeTransform,
    GrayScale,
    GymEnv,
    NoopResetEnv,
    Resize,
    RewardSum,
    SignTransform,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
    VecNorm,
)

from torchrl.modules import MLP, QValueActor
from torchrl.record import VideoRecorder


# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def make_env(device, env_cfg):
    env = GymEnv(
        env_cfg.env_name,
        from_pixels=False,
        pixels_only=False,
        device=device,
        world_cfg=env_cfg.world_cfg,
    )
    env = TransformedEnv(env, RewardSum())
    env.append_transform(StepCounter(max_steps=4500))
    return env


# ====================================================================
# Model utils
# --------------------------------------------------------------------

class ConvNet(torch.nn.Module):
    def __init__(self, feature_size, embed_size, num_actions, kernel_size=5):
        super(ConvNet, self).__init__()
        self.feature_size = feature_size
        self.embed_size = embed_size
        self.num_actions = num_actions
        self.convolutions = torch.nn.Sequential(
            torch.nn.Conv2d(feature_size, embed_size * 4, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Conv2d(embed_size * 4, embed_size * 2, kernel_size=1, stride=1),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Conv2d(embed_size * 2, embed_size, kernel_size=1, stride=1),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Conv2d(embed_size, num_actions, kernel_size=1, stride=1),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if len(input.shape) == 3:
            input = input.permute(2, 0, 1)
            output = self.convolutions(input.type(torch.float32))
            output = output.permute(1, 2, 0)
            return output.view(input.shape[-1] * input.shape[-2], self.num_actions)
        else:
            input = input.permute(0, 3, 1, 2)
            output = self.convolutions(input.type(torch.float32))
            output = output.permute(0, 2, 3, 1)
            return output.view(input.shape[0], input.shape[-1] * input.shape[-2], self.num_actions)





def make_dqn_modules(proof_environment):
    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape

    H, W, C = input_shape  # For example: (128, 128, 6)
    embedding_size = 32
    num_actions = proof_environment.unwrapped.num_actions
    action_spec = proof_environment.specs["input_spec", "full_action_spec", "action"]

    # Create the CNN model
    cnn = ConvNet(
        feature_size=C,
        embed_size=embedding_size,
        num_actions=num_actions
    )
    qvalue_module = QValueActor(
        module=cnn,
        spec=CompositeSpec(action=action_spec),
        in_keys=["observation"],
    )
    return qvalue_module


def make_dqn_model(cfg):
    proof_environment = make_env(cfg.device, cfg.env)
    qvalue_module = make_dqn_modules(proof_environment)
    del proof_environment
    return qvalue_module


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------


def eval_model(actor, test_env, num_episodes=3):
    test_rewards = torch.zeros(num_episodes, dtype=torch.float32)
    for i in range(num_episodes):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        test_env.apply(dump_video)
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards[i] = reward.sum()
    del td_test
    return test_rewards.mean()


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import functools

import psutil
import torch.nn
from torch import nn
import torch.optim
from tensordict.nn import InteractionType, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    CompositeSpec,
    LazyMemmapStorage,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.datasets.d4rl import D4RLExperienceReplay
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.envs import (
    CatTensors,
    Compose,
    DMControlEnv,
    DoubleToFloat,
    EnvCreator,
    InitTracker,
    ParallelEnv,
    RewardSum,
    TransformedEnv,
)

from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    MLP,
    OneHotCategorical,
    ProbabilisticActor,
    SafeModule,
    TanhNormal,
    ValueOperator,
)
from torchrl.objectives import DiscreteIQLLoss, HardUpdate, IQLLoss, SoftUpdate
from torchrl.record import VideoRecorder

from torchrl.trainers.helpers.models import ACTIVATIONS



# ====================================================================
# Environment utils
# -----------------


def env_maker(cfg, device="cpu", from_pixels=False):
    lib = cfg.env.backend
    if lib in ("gym", "gymnasium"):
        with set_gym_backend(lib):
            return GymEnv(
                cfg.env.name, device=device, from_pixels=from_pixels, pixels_only=False
            )
    elif lib == "dm_control":
        env = DMControlEnv(
            cfg.env.name, cfg.env.task, from_pixels=from_pixels, pixels_only=False
        )
        return TransformedEnv(
            env, CatTensors(in_keys=env.observation_spec.keys(), out_key="observation")
        )
    else:
        raise NotImplementedError(f"Unknown lib {lib}.")


def apply_env_transforms(
    env,
):
    transformed_env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            DoubleToFloat(),
            RewardSum(),
        ),
    )
    return transformed_env


def make_environment(cfg, train_num_envs=1, eval_num_envs=1, logger=None):
    """Make environments for training and evaluation."""
    maker = functools.partial(env_maker, cfg)
    # parallel_env = ParallelEnv(
    #     train_num_envs,
    #     EnvCreator(maker),
    #     serial_for_single=False,
    # )
    # parallel_env.set_seed(cfg.env.seed)
    #
    # train_env = apply_env_transforms(parallel_env)

    train_env = apply_env_transforms(maker())
    train_env.set_seed(cfg.env.seed)

    # maker = functools.partial(env_maker, cfg, from_pixels=cfg.logger.video)
    # eval_env = TransformedEnv(
    #     ParallelEnv(
    #         eval_num_envs,
    #         EnvCreator(maker),
    #         serial_for_single=False,
    #     ),
    #     train_env.transform.clone(),
    # )
    eval_maker = functools.partial(env_maker, cfg, from_pixels=cfg.logger.video)
    eval_env = TransformedEnv(eval_maker(), train_env.transform.clone())

    if cfg.logger.video:
        eval_env.insert_transform(
            0, VideoRecorder(logger, tag="rendered", in_keys=["pixels"])
        )
    return train_env, eval_env


# ====================================================================
# Collector and replay buffer
# ---------------------------


def make_collector(cfg, train_env, actor_model_explore):
    """Make collector."""
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        frames_per_batch=cfg.collector.frames_per_batch,
        init_random_frames=cfg.collector.init_random_frames,
        max_frames_per_traj=cfg.collector.max_frames_per_traj,
        total_frames=cfg.collector.total_frames,
        device=cfg.collector.device,
    )
    collector.set_seed(cfg.env.seed)
    return collector


def make_replay_buffer(
    batch_size,
    prb=False,
    buffer_size=1000000,
    scratch_dir=None,
    device="cpu",
    prefetch=3,
):
    if prb:
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.5,
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=scratch_dir,
                device=device,
            ),
            batch_size=batch_size,
        )
    else:
        replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=scratch_dir,
                device=device,
            ),
            batch_size=batch_size,
        )
    return replay_buffer


def make_offline_replay_buffer(rb_cfg):
    data = D4RLExperienceReplay(
        dataset_id=rb_cfg.dataset,
        split_trajs=False,
        batch_size=rb_cfg.batch_size,
        sampler=SamplerWithoutReplacement(drop_last=False),
        prefetch=4,
        direct_download=True,
    )

    data.append_transform(DoubleToFloat())

    return data


# ====================================================================
# Model
# -----
#
# We give one version of the model for learning from pixels, and one for state.
# TorchRL comes in handy at this point, as the high-level interactions with
# these models is unchanged, regardless of the modality.
#

class ConvEncoder(nn.Module):
    def __init__(self, feature_size, embed_size, kernel_size=5):
        super(ConvEncoder, self).__init__()
        self.feature_size = feature_size
        self.embed_size = embed_size
        self.convolutions = nn.Sequential(
            nn.Conv2d(feature_size, embed_size, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Flatten(start_dim=-2, end_dim=-1),
        )
        self._initialize_weights()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if len(input.shape) == 3:
            input = input.permute(2, 0, 1)
            output = self.convolutions(input.type(torch.float32))
            return output.permute(1, 0)
        else:
            input = input.permute(0, 3, 1, 2)
            output = self.convolutions(input.type(torch.float32))
            return output.permute(0, 2, 1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def make_conv_iql_model(cfg, train_env, eval_env, device):
    """Make convolutional IQL agent."""

    in_keys = ["observation"]
    action_spec = train_env.action_spec
    if train_env.batch_size:
        action_spec = action_spec[(0, ) * len(train_env.batch_size)]

    feature_size, embed_size = cfg.model.feature_size, cfg.model.embed_size
    kernel_size = cfg.model.kernel_size
    # num_actions = action_spec.shape[-1]
    num_actions = 6

    # Actor Network
    # actor_net = ConvEncoder(feature_size, embed_size, kernel_size, label="actor")

    encoder = ConvEncoder(feature_size, embed_size, kernel_size)
    actor_net = nn.Sequential(
        encoder,
        nn.Linear(embed_size, num_actions),
    )
    actor_module = SafeModule(module=actor_net, in_keys=in_keys, out_keys=["logits"])
    actor = ProbabilisticActor(
        spec=CompositeSpec(action=eval_env.action_spec),
        module=actor_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        distribution_kwargs={},
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False
    )

    # Critic Network (Q-Network)
    critic_net = nn.Sequential(
        encoder,
        nn.Linear(embed_size, 1),
    )
    qvalue_module = TensorDictModule(module=critic_net, in_keys=in_keys, out_keys=["state_action_value"])

    # Value Network (V-Network)
    value_net = nn.Sequential(
        encoder,
        nn.Linear(embed_size, 1)
    )
    value_module = TensorDictModule(module=value_net, in_keys=in_keys, out_keys=["state_value"])

    model = torch.nn.ModuleList([actor, qvalue_module, value_module]).to(device)
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = eval_env.reset()
        td = td.to(device)
        for net in model:
            net(td)
    del td
    eval_env.close()

    return model

# ====================================================================
# IQL Loss
# ---------


def make_loss(loss_cfg, model):
    loss_module = IQLLoss(
        model[0],
        model[1],
        value_network=model[2],
        loss_function=loss_cfg.loss_function,
        temperature=loss_cfg.temperature,
        expectile=loss_cfg.expectile,
    )
    loss_module.make_value_estimator(gamma=loss_cfg.gamma)
    target_net_updater = SoftUpdate(loss_module, tau=loss_cfg.tau)

    return loss_module, target_net_updater


def make_discrete_loss(loss_cfg, model):
    loss_module = DiscreteIQLLoss(
        model[0],
        model[1],
        value_network=model[2],
        loss_function=loss_cfg.loss_function,
        temperature=loss_cfg.temperature,
        expectile=loss_cfg.expectile,
    )
    loss_module.make_value_estimator(gamma=loss_cfg.gamma)
    target_net_updater = HardUpdate(
        loss_module, value_network_update_interval=loss_cfg.hard_update_interval
    )

    return loss_module, target_net_updater


def make_iql_optimizer(optim_cfg, loss_module):
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())
    value_params = list(loss_module.value_network_params.flatten_keys().values())

    optimizer_actor = torch.optim.Adam(
        actor_params,
        lr=optim_cfg.lr,
        weight_decay=optim_cfg.weight_decay,
    )
    optimizer_critic = torch.optim.Adam(
        critic_params,
        lr=optim_cfg.lr,
        weight_decay=optim_cfg.weight_decay,
    )
    optimizer_value = torch.optim.Adam(
        value_params,
        lr=optim_cfg.lr,
        weight_decay=optim_cfg.weight_decay,
    )
    return optimizer_actor, optimizer_critic, optimizer_value


# ====================================================================
# General utils
# ---------


def log_metrics(logger, metrics, step):
    if logger is not None:
        for metric_name, metric_value in metrics.items():
            logger.log_scalar(metric_name, metric_value, step)


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()


def print_memory_usage():
    # Get RAM usage
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024 ** 2  # Convert to MB

    # Get VRAM usage if using MPS
    if torch.backends.mps.is_available():
        # Current memory allocated by tensors
        vram_allocated = torch.mps.current_allocated_memory() / 1024 ** 2  # Convert to MB
        # Total memory allocated by Metal driver for the process
        vram_driver = torch.mps.driver_allocated_memory() / 1024 ** 2  # Convert to MB
    else:
        vram_allocated, vram_driver = 0, 0

    print(f'RAM usage: {ram_usage:.2f} MB')
    print(f'VRAM (allocated by tensors): {vram_allocated:.2f} MB')
    print(f'VRAM (allocated by driver): {vram_driver:.2f} MB')

from typing import Optional, Union, Sequence

import torch
from torch.nn import functional as F
import torchrl.envs
from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential, dispatch
from torchrl.collectors import SyncDataCollector
from torchrl.envs import GymEnv, Transform, TransformedEnv, TensorDictPrimer, VecGymEnvTransform, ParallelEnv
from torchrl.envs.transforms import UnsqueezeTransform, PermuteTransform, Compose
from torchrl.data import UnboundedDiscreteTensorSpec, UnboundedContinuousTensorSpec, TensorSpec, LazyTensorStorage, ReplayBuffer
from torchrl.modules import QValueModule
from torchrl.objectives import DistributionalDQNLoss

from tensor_beasts.rl.iqn_agent import IQN
from tensor_beasts.rl.iqn_loss import IqnLoss


def quantile_huber_loss(input, target, quantiles, delta=1.0):
    """Compute the quantile Huber loss.

    Args:
        input (Tensor): predicted quantiles, shape (batch_size, num_quantiles)
        target (Tensor): target quantiles, shape (batch_size, num_quantiles)
        quantiles (Tensor): quantile fractions, shape (batch_size, num_quantiles)
        delta (float): huber loss parameter, default 1.0

    Returns:
        Tensor: computed quantile Huber loss
    """
    td_errors = target - input

    huber_loss = F.huber_loss(input, target, reduction='none', delta=delta)

    # Compute the quantile huber loss as described
    quantile_loss = (quantiles - (td_errors.detach() < 0).float()).abs() * huber_loss
    return quantile_loss.mean()


if __name__ == "__main__":
    torch.set_default_device(torch.device("mps"))
    embed_size = 16  # Embedding size
    num_actions = 5  # Number of possible actions
    window_size = 3  # Attention window size
    iqn_embedding_dimension = 8  # IQN embedding dimension
    num_quantiles = 16  # Number of quantiles

    transforms = Compose(
        # (W, H, C) -> (C, W, H)
        PermuteTransform(dims=[-1, -3, -2], in_keys=["observation"], out_keys=["observation"]),
        # UnsqueezeTransform(unsqueeze_dim=-4, in_keys=["observation"], out_keys=["observation"]),
        TensorDictPrimer(tau=UnboundedContinuousTensorSpec(torch.Size([num_quantiles, 1])), default_value=None)
    )

    # Use SerialEnv?
    env = TransformedEnv(
        GymEnv("tensor-beasts-v0", size=128, device="mps"),
        transform=transforms
    )

    iqn_module = IQN(
        feature_size=env.unwrapped.obs_shape[-1],  # Index refers to the pre-permuted observation shape
        embed_size=embed_size,
        window_size=window_size,
        iqn_embedding_dimension=iqn_embedding_dimension,
        num_actions=num_actions,
        num_quantiles=num_quantiles
    )
    iqn_value_net = TensorDictModule(
        module=iqn_module,
        in_keys=["observation"],
        out_keys=["action", "quantile", "tau"]
    )

    policy = TensorDictSequential(
        iqn_value_net,
    )

    # collector = SyncDataCollector(env, policy, frames_per_batch=200, total_frames=-1)

    # rb = ReplayBuffer(storage=LazyTensorStorage(max_size=10000, device=torch.device("mps")))

    loss_fn = IqnLoss(iqn_value_net)

    num_epochs = 100

    rollout = env.rollout(max_steps=10, policy=policy)
    loss_value = loss_fn(rollout)
    print(loss_value)



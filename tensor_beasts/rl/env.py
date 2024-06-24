import time
from typing import Optional, Union, Sequence

import torch
from torch.nn import functional as F
from tensordict.nn import TensorDictModule, TensorDictSequential, dispatch
from torchrl.collectors import SyncDataCollector
from torchrl.envs import (
    GymEnv, Transform, TransformedEnv, TensorDictPrimer, VecGymEnvTransform, ParallelEnv,
    BatchSizeTransform, SqueezeTransform
)
from torchrl.envs.transforms import UnsqueezeTransform, PermuteTransform, Compose
from torchrl.data import UnboundedDiscreteTensorSpec, UnboundedContinuousTensorSpec, TensorSpec, LazyTensorStorage, ReplayBuffer
from torchrl.objectives import SoftUpdate
from torchrl.record import CSVLogger
from torchrl._utils import logger as torchrl_logger

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
    frames_per_batch = 10

    transforms = Compose(
        # (W, H, C) -> (C, W, H)
        PermuteTransform(dims=[-1, -3, -2], in_keys=["observation"], out_keys=["observation"]),
        # UnsqueezeTransform(unsqueeze_dim=-4, in_keys=["observation"], out_keys=["observation"]),
        # TensorDictPrimer(tau=UnboundedContinuousTensorSpec(torch.Size([num_quantiles, 1])), default_value=None),
        # BatchSizeTransform(batch_size=torch.Size([frames_per_batch]))
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

    loss = IqnLoss(iqn_value_net)
    optim = torch.optim.Adam(iqn_value_net.parameters(), lr=1e-3)
    updater = SoftUpdate(loss, eps=0.99)

    path = "./logs"
    logger = CSVLogger(exp_name="iqn", log_dir=path)

    init_rand_steps = 10
    optim_steps = 10

    # Gross... surely there's a better way to handle dimensionality difference in data collector?
    collector_post_process = Compose(
        SqueezeTransform(
            squeeze_dim=-2,
            in_keys=["action"],
            out_keys=["action"]
        ),
        SqueezeTransform(
            squeeze_dim=-3,
            in_keys=["tau"],
            out_keys=["tau"]
        ),
        SqueezeTransform(
            squeeze_dim=-4,
            in_keys=["quantile"],
            out_keys=["quantile"]
        )
    )

    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=-1,
        init_random_frames=init_rand_steps,
        postproc=collector_post_process
    )
    rb = ReplayBuffer(
        storage=LazyTensorStorage(max_size=1000, device=torch.device("mps")),
        batch_size=frames_per_batch
    )

    total_count = 0
    total_episodes = 0
    t0 = time.time()
    for i, data in enumerate(collector):
        print(f"Data: {data}")
        rb.extend(data)
        max_length = -1
        if len(rb) > init_rand_steps:
            sample = rb.sample(batch_size=frames_per_batch)
            loss_vals = loss(sample)
            loss_vals["loss"].backward()
            optim.step()
            optim.zero_grad()
            updater.step()
            if i % 10 == 0:
                torchrl_logger.info(f"Max num steps: {max_length}, rb length {len(rb)}")
            total_count += data.numel()
            total_episodes += data["next", "done"].sum()
        if max_length > 200:
            break
    t1 = time.time()
    torchrl_logger.info(
        f"solved after {total_count} steps, {total_episodes} episodes and in {t1 - t0}s."
    )

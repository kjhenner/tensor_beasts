import time

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.envs import (
    GymEnv, TransformedEnv, SqueezeTransform
)
from torchrl.envs.transforms import PermuteTransform, Compose
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.modules import EGreedyModule
from torchrl.objectives import SoftUpdate
from torchrl._utils import logger as torchrl_logger

from tensor_beasts.display_manager import DisplayManager
from tensor_beasts.rl.iqn.iqn_agent import IQN
from tensor_beasts.rl.iqn.iqn_loss import IqnLoss
from tensor_beasts.world import World


class IQNTrainer:
    def __init__(
        self,
        world_size: int = 128,
        embed_size: int = 16,
        num_actions: int = 5,
        iqn_embedding_dimension: int = 8,
        num_quantiles: int = 16,
        frames_per_batch: int = 10,
    ):
        self.embed_size = embed_size
        self.num_actions = num_actions
        self.iqn_embedding_dimension = iqn_embedding_dimension
        self.num_quantiles = num_quantiles
        self.frames_per_batch = frames_per_batch

        permute_transform = PermuteTransform(dims=[-1, -3, -2], in_keys=["observation"], out_keys=["observation"])

        # Initialize the World
        self.world = World(
            size=world_size,
        )

        self.display_manager = DisplayManager(world_size, world_size)

        env = TransformedEnv(
            GymEnv("tensor-beasts-v0", world=self.world, device="mps"),
            transform=permute_transform
        )
        iqn_module = IQN(
            feature_size=env.unwrapped.obs_shape[-1],  # Index refers to the pre-permuted observation shape
            embed_size=embed_size,
            iqn_embedding_dimension=iqn_embedding_dimension,
            num_actions=num_actions,
            num_quantiles=num_quantiles
        )
        iqn_value_net = TensorDictModule(
            module=iqn_module,
            in_keys=["observation"],
            out_keys=["action", "quantile", "tau"]
        )

        self.exploration_module = EGreedyModule(
                env.action_spec, annealing_num_steps=10_000, eps_init=1.0, eps_end=0.01
        )
        policy = TensorDictSequential(
            iqn_value_net,
            self.exploration_module
        )

        self.loss = IqnLoss(iqn_value_net)
        self.optim = torch.optim.Adam(iqn_value_net.parameters(), lr=1e-6)
        self.updater = SoftUpdate(self.loss, eps=0.99)
        self.init_rand_steps = 10
        self.optim_steps = 10

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

        self.collector = SyncDataCollector(
            env,
            policy,
            frames_per_batch=frames_per_batch,
            total_frames=-1,
            init_random_frames=self.init_rand_steps,
            postproc=collector_post_process
        )
        self.rb = ReplayBuffer(
            storage=LazyTensorStorage(max_size=1000, device=torch.device("mps")),
            batch_size=frames_per_batch
        )

    def train(self):
        total_count = 0
        total_episodes = 0
        t0 = time.time()
        for i, data in enumerate(self.collector):
            self.rb.extend(data)
            self.display_manager.add_screens_to_buffer(data["rgb_array"])
            self.display_manager.update_from_buffer()
            if len(self.rb) > self.init_rand_steps:
                sample = self.rb.sample(batch_size=self.frames_per_batch)
                loss_vals = self.loss(sample)
                print(f"Loss: {loss_vals['loss']}")
                loss_vals["loss"].backward()
                self.optim.step()
                self.optim.zero_grad()
                self.exploration_module.step(data.numel())
                self.updater.step()
                total_count += data.numel()
                total_episodes += data["next", "done"].sum()
        t1 = time.time()
        torchrl_logger.info(
            f"solved after {total_count} steps, {total_episodes} episodes and in {t1 - t0}s."
        )


if __name__ == "__main__":
    torch.set_default_device(torch.device("mps"))
    embed_size = 16  # Embedding size
    num_actions = 5  # Number of possible actions
    iqn_embedding_dimension = 8  # IQN embedding dimension
    num_quantiles = 16  # Number of quantiles
    frames_per_batch = 10

    trainer = IQNTrainer(
        embed_size=embed_size,
        num_actions=num_actions,
        iqn_embedding_dimension=iqn_embedding_dimension,
        num_quantiles=num_quantiles,
        frames_per_batch=frames_per_batch
    )
    trainer.train()

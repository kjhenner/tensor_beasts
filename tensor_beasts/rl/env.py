from typing import Optional, Union, Sequence

import torch
import torchrl.envs
from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential, dispatch
from torchrl.envs import GymEnv, Transform, TransformedEnv, TensorDictPrimer, VecGymEnvTransform, ParallelEnv
from torchrl.envs.transforms import UnsqueezeTransform, PermuteTransform, Compose
from torchrl.data import UnboundedDiscreteTensorSpec, UnboundedContinuousTensorSpec, TensorSpec
from torchrl.modules import QValueModule

from tensor_beasts.rl.iqn_agent import IQN


class QValueModule2D(QValueModule):
    """Q-Value Module that allows 2D actions output."""

    def __init__(
        self,
        action_space: Optional[str] = None,
        action_value_key: Optional[Union[str, Sequence[str]]] = None,
        action_mask_key: Optional[Union[str, Sequence[str]]] = None,
        out_keys: Optional[Sequence[Union[str, Sequence[str]]]] = None,
        var_nums: Optional[int] = None,
        spec: Optional[TensorSpec] = None,
        safe: bool = False,
    ):
        super().__init__(
            action_space=action_space,
            action_value_key=action_value_key,
            action_mask_key=action_mask_key,
            out_keys=out_keys,
            var_nums=var_nums,
            spec=spec,
            safe=safe,
        )

    @staticmethod
    def _categorical(value: torch.Tensor) -> torch.Tensor:
        print("Categorical")
        print(value.shape)
        print(torch.argmax(value, dim=-3).to(torch.long).shape)
        return torch.argmax(value, dim=-3).to(torch.long)


if __name__ == "__main__":
    torch.set_default_device(torch.device("mps"))
    embed_size = 4  # Embedding size
    num_heads = 1  # Number of attention heads
    num_actions = 5  # Number of possible actions
    window_size = 3  # Attention window size
    iqn_embedding_dimension = 8  # IQN embedding dimension
    num_quantiles = 16  # Number of quantiles

    transforms = Compose(
        # (W, H, C) -> (C, W, H)
        PermuteTransform(dims=[-1, -3, -2], in_keys=["observation"], out_keys=["observation"]),
        # (C, W, H) -> (N, C, W, H)
        UnsqueezeTransform(unsqueeze_dim=-4, in_keys=["observation"], out_keys=["observation"]),
        TensorDictPrimer(tau=UnboundedContinuousTensorSpec(torch.Size([num_quantiles, 1])), default_value=None)
    )

    # Use SerialEnv?
    env = TransformedEnv(
        GymEnv("tensor-beasts-v0", size=128, device="mps"),
        transform=transforms
    )

    module = IQN(
        feature_size=env.unwrapped.obs_shape[-1],  # Index refers to the pre-permuted observation shape
        embed_size=embed_size,
        num_heads=num_heads,
        window_size=window_size,
        iqn_embedding_dimension=iqn_embedding_dimension,
        num_actions=num_actions,
        num_quantiles=num_quantiles
    )
    value_net = TensorDictModule(
        module=module,
        in_keys=["observation", "tau"],
        out_keys=["action", "tau"]
    )

    policy = TensorDictSequential(
        value_net,
        QValueModule(
            action_space='mult_one_hot',
            action_value_key="action",
            out_keys=["action", "action_values", "chosen_action_values"]
        )
    )
    reset = env.reset()
    print(reset)
    reset_with_action = env.rand_action(reset)
    print(reset_with_action)
    # print(reset_with_action)
    # stepped = env.step(reset_with_action)
    # print(stepped)
    rollout = env.rollout(10, policy=policy)
    print(rollout)

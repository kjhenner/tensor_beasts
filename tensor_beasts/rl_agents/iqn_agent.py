# From https://github.com/pb4git/trackmania_rl_public/blob/main/trackmania_rl/agents/iqn.py
from typing import Optional, Tuple
import math
from tensor_beasts.rl_agents.nn_utils import init_kaiming

import torch


class Agent(torch.nn.Module):
    def __init__(
        self,
        float_inputs_dim,
        float_hidden_dim,
        conv_head_output_dim,
        dense_hidden_dimension,
        iqn_embedding_dimension,
        n_actions,
        float_inputs_mean,
        float_inputs_std,
    ):
        super().__init__()
        self.img_head = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, 4), stride=2),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=2),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Flatten(),
        )
        self.float_feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(float_inputs_dim, float_hidden_dim),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(float_hidden_dim, float_hidden_dim),
            torch.nn.LeakyReLU(inplace=True),
        )

        dense_input_dimension = conv_head_output_dim + float_hidden_dim

        self.A_head = torch.nn.Sequential(
            torch.nn.Linear(dense_input_dimension, dense_hidden_dimension // 2),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(dense_hidden_dimension // 2, n_actions),
        )
        self.V_head = torch.nn.Sequential(
            torch.nn.Linear(dense_input_dimension, dense_hidden_dimension // 2),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(dense_hidden_dimension // 2, 1),
        )

        self.iqn_fc = torch.nn.Linear(
            iqn_embedding_dimension, dense_input_dimension
        )  # There is no word in the paper on how to init this layer?
        self.lrelu = torch.nn.LeakyReLU()
        self.initialize_weights()

        self.iqn_embedding_dimension = iqn_embedding_dimension
        self.n_actions = n_actions

        self.float_inputs_mean = torch.tensor(float_inputs_mean, dtype=torch.float32).to("cuda")
        self.float_inputs_std = torch.tensor(float_inputs_std, dtype=torch.float32).to("cuda")

    def initialize_weights(self):
        for m in self.img_head:
            if isinstance(m, torch.nn.Conv2d):
                init_kaiming(m)
        for m in self.float_feature_extractor:
            if isinstance(m, torch.nn.Linear):
                init_kaiming(m)
        # This was uninitialized in Agade's code
        init_kaiming(self.iqn_fc)
        # A_head and V_head are NoisyLinear, already initialized

    def forward(
        self, img, float_inputs, num_quantiles: int, tau: Optional[torch.Tensor] = None, use_fp32: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = img.shape[0]
        img_outputs = self.img_head((img.to(torch.float32 if use_fp32 else torch.float16) - 128) / 128)  # PERF
        float_outputs = self.float_feature_extractor((float_inputs - self.float_inputs_mean) / self.float_inputs_std)
        # (batch_size, dense_input_dimension) OK
        concat = torch.cat((img_outputs, float_outputs), 1)
        if tau is None:
            tau = torch.rand(
                size=(batch_size * num_quantiles, 1), device="cuda", dtype=torch.float32
            )  # (batch_size * num_quantiles, 1) (random numbers)
        quantile_net = tau.expand(
            [-1, self.iqn_embedding_dimension]
        )  # (batch_size*num_quantiles, iqn_embedding_dimension) (still random numbers)
        quantile_net = torch.cos(
            torch.arange(1, self.iqn_embedding_dimension + 1, 1, device="cuda") * math.pi * quantile_net
        )  # (batch_size*num_quantiles, iqn_embedding_dimension)
        # (8 or 32 initial random numbers, expanded with cos to iqn_embedding_dimension)
        # (batch_size*num_quantiles, dense_input_dimension)
        quantile_net = self.iqn_fc(quantile_net)
        # (batch_size*num_quantiles, dense_input_dimension)
        quantile_net = self.lrelu(quantile_net)
        # (batch_size*num_quantiles, dense_input_dimension)
        concat = concat.repeat(num_quantiles, 1)
        # (batch_size*num_quantiles, dense_input_dimension)
        concat = concat * quantile_net

        A = self.A_head(concat)  # (batch_size*num_quantiles, n_actions)
        V = self.V_head(concat)  # (batch_size*num_quantiles, 1) #need to check this

        Q = V + A - A.mean(dim=-1).unsqueeze(-1)

        return Q, tau

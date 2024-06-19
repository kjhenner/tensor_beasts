import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(
        self,
        feature_size,
        embed_size,
        kernel_size=5,
    ):
        super(ConvEncoder, self).__init__()
        self.feature_size = feature_size
        self.embed_size = embed_size

        self.convolutions = nn.Sequential(
            nn.Conv2d(feature_size, embed_size, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Flatten(start_dim=-2, end_dim=-1),
        )

        # Initialize weights
        self._initialize_weights()

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:

        # Reshape input to (N, C, W*H)
        output = self.convolutions(input.type(torch.float32))
        return output.permute(0, 2, 1)

    def _initialize_weights(self):
        # Kaiming initialization for linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class IQN(nn.Module):
    def __init__(
        self,
        feature_size,
        embed_size,
        window_size,
        iqn_embedding_dimension,
        num_actions,
        num_quantiles,
    ):
        super(IQN, self).__init__()
        self.feature_size = feature_size
        self.embed_size = embed_size
        self.window_size = window_size
        self.iqn_embedding_dimension = iqn_embedding_dimension
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles

        self.encoder = ConvEncoder(
            feature_size, embed_size
        )

        # Get action logits from hidden layer
        self.A_head = nn.Sequential(
            nn.Linear(embed_size, num_actions),
            nn.LeakyReLU(negative_slope=0.01),
        )

        # Get value predictions from hidden layer
        self.V_head = nn.Sequential(
            nn.Linear(embed_size, 1),
            nn.LeakyReLU(negative_slope=0.01),
        )

        # Non-linear activation
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  # default negative slope

        # IQN linear layer
        self.iqn_fc = nn.Linear(iqn_embedding_dimension, embed_size)
        
        # Initialize weights
        self._initialize_weights()

    def forward(
        self,
        observation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = observation.device
        print(observation.shape)

        # Add batch dimension if necessary
        if len(observation.shape) == 3:
            observation = observation.unsqueeze(0)

        # world_state is in (N, C, W, H) format
        N, C, W, H = observation.shape

        encoder_output = self.encoder(observation)  # (N, W*H, embed_size)

        # Quantile network processing with IQN
        tau = torch.rand(size=(N * self.num_quantiles, 1), device=device, dtype=torch.float32)

        quantile_net = tau.expand([-1, self.iqn_embedding_dimension])
        quantile_net = torch.cos(torch.arange(1, self.iqn_embedding_dimension + 1, 1, device=device) * math.pi * quantile_net)
        quantile_net = self.iqn_fc(quantile_net)
        quantile_net = self.leaky_relu(quantile_net)
        # (N*num_quantiles, embed_size)

        encoder_output = encoder_output.repeat(self.num_quantiles, 1, 1)
        encoder_output = encoder_output * quantile_net.unsqueeze(1)

        # Compute action logits using A_head
        action_logits = self.A_head(encoder_output)  # (N*num_quantiles, W*H, num_actions)

        # Compute value predictions using V_head
        value_predictions = self.V_head(encoder_output)  # (N*num_quantiles, W*H, 1)

        Q = value_predictions + action_logits - action_logits.mean(dim=-1, keepdim=True)

        # Reshape Q to (N, num_quantiles, W*H, num_actions)
        Q = Q.view(N, self.num_quantiles, W*H, self.num_actions)

        # Compute the mean over the quantiles
        # (N, num_quantiles, W*H, num_actions) -> (N, W*H, num_actions)
        actions = Q.mean(dim=1).argmax(dim=-1)

        tau = tau.view(N, self.num_quantiles).unsqueeze(-1)

        return actions, Q, tau

    def _initialize_weights(self):
        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.kaiming_normal_(self.iqn_fc.weight, nonlinearity='leaky_relu')

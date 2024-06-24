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

        # Get advantage predictions from hidden layer
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

        # Add batch dimension if necessary
        if len(observation.shape) == 3:
            observation = observation.unsqueeze(0)

        # world_state is in (N, C, W, H) format
        N, C, W, H = observation.shape

        # Embed our observation
        obs_embeddings = self.encoder(observation)  # (N, W*H, embed_size)

        # The tau tensor represents the actual quantile values.
        # If tau were uniformly distributed, the values for tau if num quantiles = 8 would be:
        # [0.125, 0.375, 0.625, 0.875, 0.125, 0.375, 0.625, 0.875]
        # This uniform distribution is a reasonable intuitive mental model for tau.
        # In practice, however, tau is randomly sampled from a uniform distribution.
        tau = torch.rand(size=(N * self.num_quantiles, 1), device=device, dtype=torch.float32)

        # Use the cosine fn to generate embeddings for the quantile values.
        # (Cosine fn just maps floats to an embedding space the NN can work with better than a single value input.)
        quantile_embeddings = tau.expand([-1, self.iqn_embedding_dimension])
        quantile_embeddings = torch.cos(torch.arange(1, self.iqn_embedding_dimension + 1, 1, device=device) * math.pi * quantile_embeddings)

        # Pass these quantile embeddings through a linear layer with leaky relu activation to get hidden layer
        # representation of the quantile values.
        quantile_hidden = self.leaky_relu(self.iqn_fc(quantile_embeddings))
        # (N*num_quantiles, embed_size)

        # Repeat the observation embeddings for each quantile
        # TODO: Validate--is the shape here correct?
        # (N, W*H, embed_size) -> (N*num_quantiles, W*H, embed_size)
        obs_embeddings = obs_embeddings.repeat(self.num_quantiles, 1, 1)
        # Multiply the observation embeddings by the quantile hidden layer representation
        # (N*num_quantiles, W*H, embed_size) * (N*num_quantiles, 1, embed_size) unsqueeze to broadcast on dim 1
        # TODO: Is multiplication here the correct operation?
        obs_embeddings = obs_embeddings * quantile_hidden.unsqueeze(1)

        # Compute advantage using A_head
        advantage = self.A_head(obs_embeddings)  # (N*num_quantiles, W*H, num_actions)

        # Compute value predictions using V_head
        value = self.V_head(obs_embeddings)  # (N*num_quantiles, W*H, 1)

        quantiles = value + advantage - advantage.mean(dim=-1, keepdim=True)

        # Reshape quantiles to (N, num_quantiles, W*H, num_actions)
        quantiles = quantiles.view(N, self.num_quantiles, W*H, self.num_actions)

        # Compute the mean over the quantiles
        # (N, num_quantiles, W*H, num_actions) -> (N, W*H, num_actions)
        actions = quantiles.mean(dim=1).argmax(dim=-1)

        tau = tau.view(N, self.num_quantiles).unsqueeze(-1)

        return actions, quantiles, tau

    def _initialize_weights(self):
        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.kaiming_normal_(self.iqn_fc.weight, nonlinearity='leaky_relu')

# Based very loosely on https://github.com/pb4git/trackmania_rl_public/blob/main/trackmania_rl/agents/iqn.py
from typing import Optional, Tuple
import math
from tensor_beasts.rl_agents.nn_utils import init_kaiming

import torch
from torch import nn


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, width, height):
        # Create a grid of (x, y) coordinates
        x = torch.arange(width, dtype=torch.float32, device='cuda').unsqueeze(1)
        y = torch.arange(height, dtype=torch.float32, device='cuda').unsqueeze(0)

        div_term = torch.exp(torch.arange(0, self.embed_dim, 2, device='cuda', dtype=torch.float32) *
                             -(math.log(10000.0) / self.embed_dim))

        pe = torch.zeros(width, height, self.embed_dim, device='cuda')
        pe[:, :, 0::2] = torch.sin(x * div_term)[:, None, :]
        pe[:, :, 1::2] = torch.cos(y * div_term)[None, :, :]

        return pe.permute(2, 0, 1).unsqueeze(0)


class AttentionMechanism(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        window_size,
        iqn_embedding_dimension,
        num_actions
    ):
        super(AttentionMechanism, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_actions = num_actions

        # Positional Embedding (using sinusoidal positional embeddings)
        self.positional_embedding = SinusoidalPositionalEmbedding(embed_dim)

        # MultiheadAttention layer
        self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads)

        # Action head to map attention outputs to action logits
        self.A_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(embed_dim, num_actions)
        )

        # Value head to map attention outputs to value predictions
        self.V_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(embed_dim, 1)  # Predicting a single value, not multiple actions
        )

        # Non-linear activation
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  # default negative slope

        self.iqn_fc = torch.nn.Linear(
            iqn_embedding_dimension, embed_dim
        )

        self.initialize_weights()

        self.iqn_embedding_dimension = iqn_embedding_dimension
        self.num_actions = num_actions

    def forward(
        self,
        world_state: torch.Tensor,
        num_quantiles: int,
        tau: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # world_state is in (N, C, W, H) format
        N, C, W, H = world_state.shape

        # Add positional embeddings
        positional_embeddings = self.positional_embedding(W, H).expand(N, -1, -1, -1)
        world_state = world_state + positional_embeddings

        # Apply Leaky ReLU after adding positional embeddings
        world_state = self.leaky_relu(world_state)

        # Permute and reshape to (W*H, N, C) format
        world_state = world_state.permute(2, 3, 0, 1).reshape(W * H, N, C)  # (W*H, N, C)

        mask = self.create_local_attention_mask(W, H).to(world_state.device)

        attn_output, _ = self.multihead_attention(world_state, world_state, world_state, attn_mask=mask)

        attn_output = self.leaky_relu(attn_output)

        # Reshape the attention output to (N, C, W, H)
        attn_output = attn_output.reshape(W, H, N, C).permute(2, 3, 0, 1)  # (N, C, W, H)

        if tau is None:
            tau = torch.rand(
                size=(N * num_quantiles, 1), device=world_state.device, dtype=torch.float32
            )
        quantile_net = tau.expand(
            [-1, self.iqn_embedding_dimension]
        )
        quantile_net = torch.cos(
            torch.arange(1, self.iqn_embedding_dimension + 1, 1, device=world_state.device) * math.pi * quantile_net
        )
        # (8 or 32 initial random numbers, expanded with cos to iqn_embedding_dimension)

        # (batch_size*num_quantiles, embed_dim)
        quantile_net = self.iqn_fc(quantile_net)

        # (batch_size*num_quantiles, embed_dim)
        quantile_net = self.leaky_relu(quantile_net)

        # (batch_size*num_quantiles, embed_dim)
        attn_output = attn_output.repeat(num_quantiles, 1)

        # (batch_size*num_quantiles, embed_dim)
        attn_output *= quantile_net

        # Compute action logits using A_head
        action_logits = self.A_head(attn_output)  # (N, num_actions, W, H)

        # Compute value predictions using V_head
        value_predictions = self.V_head(attn_output)  # (N, 1, W, H)

        Q = value_predictions + action_logits - action_logits.mean(dim=-1).unsqueeze(-1)

        return Q, tau

    def create_local_attention_mask(self, width, height):
        range_ = self.window_size // 2
        mask = torch.full((width * height, width * height), float('-inf'))

        coordinates = torch.arange(width * height).reshape(width, height)
        for i in range(width):
            for j in range(height):
                local_indices = coordinates[max(i - range_, 0):min(i + range_ + 1, width),
                                            max(j - range_, 0):min(j + range_ + 1, height)].flatten()
                mask[i * height + j, local_indices] = 0
        return mask

    def _initialize_weights(self):
        # Kaiming initialization for linear layers in A_head and V_head
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, height, width):
        """
        :param height: height of the positions
        :param width: width of the positions
        :return: self.embed_dim*height*width position matrix
        """
        if self.embed_dim % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(self.embed_dim))
        pe = torch.zeros(self.embed_dim, height, width)
        # Each dimension use half of d_model
        d_model = int(self.embed_dim / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe


class AttentionEncoder(nn.Module):
    def __init__(
        self,
        feature_size,
        embed_size,
        num_heads,
        window_size,
    ):
        super(AttentionEncoder, self).__init__()
        self.feature_size = feature_size
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.window_size = window_size

        # Positional Embedding (using sinusoidal positional embeddings)
        self.positional_embedding = SinusoidalPositionalEmbedding(feature_size)

        # MultiheadAttention layer
        self.multihead_attention = nn.MultiheadAttention(embed_size, num_heads, batch_first=True)

        self.lin_Q = nn.Sequential(
            nn.Conv2d(feature_size, embed_size, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.lin_K = nn.Sequential(
            nn.Conv2d(feature_size, embed_size, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.lin_V = nn.Sequential(
            nn.Conv2d(feature_size, embed_size, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.01)
        )

        # Non-linear activation
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  # default negative slope

        # Initialize weights
        self._initialize_weights()

    def forward(
        self,
        world_state: torch.Tensor,
    ) -> torch.Tensor:
        device = world_state.device

        # world_state is in (N, C, W, H) format
        N, C, W, H = world_state.shape

        # Add positional embeddings
        positional_embeddings = self.positional_embedding(W, H).to(device).expand(N, -1, -1, -1)
        world_state = world_state + positional_embeddings

        # Apply Leaky ReLU after adding positional embeddings
        world_state = self.leaky_relu(world_state)

        # Create the attention mask
        mask = self.create_local_attention_mask(W, H).to(device)

        # Self-attention with the mask
        attn_output, _ = self.multihead_attention(
            self.lin_Q(world_state).permute(0, 2, 3, 1).reshape(N, W*H, self.embed_size),
            self.lin_K(world_state).permute(0, 2, 3, 1).reshape(N, W*H, self.embed_size),
            self.lin_V(world_state).permute(0, 2, 3, 1).reshape(N, W*H, self.embed_size),
            attn_mask=mask
        )

        # attn_output = self.compute_local_attention(world_state, W, H)

        # Apply Leaky ReLU after attention output
        attn_output = self.leaky_relu(attn_output)

        # # Reshape the attention output to (N, C, W, H)
        attn_output = attn_output.reshape(W, H, N, self.embed_size).permute(2, 3, 0, 1)  # (N, C, W, H)

        return attn_output

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

    def compute_local_attention(self, world_state, width, height):
        range_ = self.window_size // 2
        N, C, _, _ = world_state.shape
        attn_output = torch.zeros_like(world_state)  # (N, C, W*H)

        # Reshape for easier slicing
        world_state = world_state.view(N, C, height, width)  # (N, C, H, W)

        for i in range(0, height):
            for j in range(0, width):
                i_start = max(0, i - range_)
                i_end = min(height, i + range_ + 1)
                j_start = max(0, j - range_)
                j_end = min(width, j + range_ + 1)

                local_patch = world_state[:, :, i_start:i_end, j_start:j_end].contiguous()
                local_patch = local_patch.view(N, C, -1)  # Flatten spatial dimensions into one

                attention_output, _ = self.multihead_attention(
                    self.lin_Q(local_patch), self.lin_K(local_patch), self.lin_V(local_patch)
                )

                attn_output[:, :, i * width + j] = attention_output[:, :, (i - i_start) * (j_end - j_start) + (j - j_start)]
        return attn_output

    def _initialize_weights(self):
        # Kaiming initialization for linear layers in A_head, V_head, and iqn_fc
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


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
        )

        # Initialize weights
        self._initialize_weights()

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:

        # Reshape input to (N, C, W*H)
        output = self.convolutions(input.type(torch.float32))
        return output.reshape(*output.shape[:-2], output.shape[-2] * output.shape[-1])

    def _initialize_weights(self):
        # Kaiming initialization for linear layers in A_head, V_head, and iqn_fc
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class IQN(nn.Module):
    def __init__(
        self,
        feature_size,
        embed_size,
        num_heads,
        window_size,
        iqn_embedding_dimension,
        num_actions,
        num_quantiles,
    ):
        super(IQN, self).__init__()
        self.feature_size = feature_size
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.iqn_embedding_dimension = iqn_embedding_dimension
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles

        # self.encoder = AttentionEncoder(
        #     feature_size, embed_size, num_heads, window_size
        # )

        self.encoder = ConvEncoder(
            feature_size, embed_size
        )

        # Action head to map attention outputs to action logits
        self.A_head = nn.Sequential(
            nn.Conv2d(embed_size, num_actions, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.01),
        )

        # Value head to map attention outputs to value predictions
        self.V_head = nn.Sequential(
            nn.Conv2d(embed_size, 1, kernel_size=1),
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
        world_state: torch.Tensor,
        tau: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = world_state.device

        # world_state is in (N, C, W, H) format
        N, C, W, H = world_state.shape

        encoder_output = self.encoder(world_state)  # (N, C, W, H)

        # Quantile network processing with IQN
        if tau is None:
            tau = torch.rand(size=(N * self.num_quantiles, 1), device=device, dtype=torch.float32)

        quantile_net = tau.expand([-1, self.iqn_embedding_dimension])
        quantile_net = torch.cos(torch.arange(1, self.iqn_embedding_dimension + 1, 1, device=device) * math.pi * quantile_net)
        quantile_net = self.iqn_fc(quantile_net)
        quantile_net = self.leaky_relu(quantile_net)

        encoder_output = encoder_output.repeat(self.num_quantiles, 1, 1, 1)
        encoder_output = encoder_output * quantile_net.view(self.num_quantiles, self.embed_size).unsqueeze(-1).unsqueeze(-1)

        # Compute action logits using A_head
        action_logits = self.A_head(encoder_output)  # (N*num_quantiles, num_actions, W, H)

        # Compute value predictions using V_head
        value_predictions = self.V_head(encoder_output)  # (N*num_quantiles, 1, W, H)

        Q = value_predictions + action_logits - action_logits.mean(dim=-1, keepdim=True)

        # Reshape Q to (N, num_quantiles, num_actions, W, H)
        Q = Q.view(N, self.num_quantiles, self.num_actions, W, H)

        return Q, tau

    def _initialize_weights(self):
        # Kaiming initialization for linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# Testing the module
if __name__ == "__main__":
    # Define the default device
    torch.set_default_device(torch.device("mps"))
    # Define some dimensions and parameters
    N = 1  # Batch size
    C = 16  # Feature size
    W = 128  # Width of world
    H = 128  # Height of world
    embed_size = 32  # Embedding size
    num_heads = 2  # Number of attention heads
    num_actions = 7  # Number of possible actions
    window_size = 6  # Attention window size
    iqn_embedding_dimension = 32  # IQN embedding dimension
    num_quantiles = 8  # Number of quantiles

    # Create example input
    world_state = torch.randn(N, C, W, H, device="mps")

    # Initialize model
    model = IQN(
        feature_size=C,
        embed_size=embed_size,
        num_heads=num_heads,
        window_size=window_size,
        iqn_embedding_dimension=iqn_embedding_dimension,
        num_actions=num_actions
    )
    
    # Compute Q values
    Q, tau = model(world_state, num_quantiles)

    # Print output shapes
    print(f"Q values shape: {Q.shape}")  # Should be (N*num_quantiles, num_actions, W, H)
    assert Q.shape == (N*num_quantiles, num_actions, W, H)
    print(f"Tau shape: {tau.shape}")  # Should be (N*num_quantiles, 1)
    assert tau.shape == (N*num_quantiles, 1)

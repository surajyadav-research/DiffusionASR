"""
Simple Prefix Tuning Implementation
Based on "Prefix-Tuning: Optimizing Continuous Prompts for Generation"

This implementation uses embedding-based prefix tuning that prepends
learnable embeddings to the input, which is compatible with all transformers versions.
"""

import torch
from torch import nn


class SimplePrefixTuning(nn.Module):
    """
    A simplified prefix tuning implementation that learns virtual token embeddings
    prepended to the input sequence for parameter-efficient fine-tuning.

    This uses the embedding-space approach which is simpler and more compatible
    than the KV-cache approach.
    """

    def __init__(
        self,
        num_virtual_tokens=20,
        hidden_size=None,
        prefix_projection=False,
        prefix_projection_hidden_size=512,
        prefix_dropout=0.0,
        **kwargs  # Ignore num_layers, num_attention_heads (not needed for embedding approach)
    ):
        """
        Args:
            num_virtual_tokens: Number of virtual tokens to prepend
            hidden_size: Hidden dimension of the model
            prefix_projection: Whether to use MLP reparameterization
            prefix_projection_hidden_size: Hidden size for MLP if prefix_projection=True
            prefix_dropout: Dropout probability
        """
        super().__init__()

        self.num_virtual_tokens = num_virtual_tokens
        self.hidden_size = hidden_size
        self.prefix_projection = prefix_projection
        self.prefix_dropout = prefix_dropout

        # Use embedding-based approach: learn continuous embeddings in input space
        if prefix_projection:
            # Use MLP reparameterization (more parameter efficient)
            # Embedding layer for virtual token positions
            self.embedding = nn.Embedding(num_virtual_tokens, hidden_size)

            # MLP to transform embeddings
            self.trans = nn.Sequential(
                nn.Linear(hidden_size, prefix_projection_hidden_size),
                nn.Tanh(),
                nn.Linear(prefix_projection_hidden_size, hidden_size),
            )
        else:
            # Directly learn prefix embeddings
            self.prefix_embeddings = nn.Parameter(
                torch.randn(num_virtual_tokens, hidden_size)
            )

        self.dropout = nn.Dropout(prefix_dropout)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        if self.prefix_projection:
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
            for module in self.trans.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        else:
            nn.init.normal_(self.prefix_embeddings, mean=0.0, std=0.02)

    def get_prefix_embeddings(self, batch_size):
        """
        Generate prefix embeddings to prepend to input

        Args:
            batch_size: Batch size

        Returns:
            prefix_embeds: Tensor of shape [batch_size, num_virtual_tokens, hidden_size]
        """
        if self.prefix_projection:
            # Generate through MLP
            # [num_virtual_tokens]
            input_tokens = (
                torch.arange(self.num_virtual_tokens)
                .long()
                .to(self.embedding.weight.device)
            )
            # [num_virtual_tokens, hidden_size]
            prefix_embeds = self.embedding(input_tokens)
            # [num_virtual_tokens, hidden_size]
            prefix_embeds = self.trans(prefix_embeds)
        else:
            # Use directly learned embeddings
            prefix_embeds = self.prefix_embeddings

        # Apply dropout
        prefix_embeds = self.dropout(prefix_embeds)

        # Expand to batch size: [batch_size, num_virtual_tokens, hidden_size]
        prefix_embeds = prefix_embeds.unsqueeze(0).expand(batch_size, -1, -1)

        return prefix_embeds

    def forward(self, batch_size):
        """
        Forward pass to get prefix embeddings

        Args:
            batch_size: Batch size

        Returns:
            prefix_embeds: Prefix embeddings [batch_size, num_virtual_tokens, hidden_size]
        """
        return self.get_prefix_embeddings(batch_size)

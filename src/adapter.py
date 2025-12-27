"""
Adapter Module

To Extend:
    1. Make a new class subclassing `Adapter` with the following methods:
        - `__init__`: Initialize the adapter
        - `forward`: Forward pass
        - `project`: Project the output of the adapter to the LLM input dimension
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from transformers import Blip2QFormerConfig, Blip2QFormerModel

ADAPTER_REGISTRY = {}


def register_adapter(name):
    """
    Decorator to register adapter classes in the ADAPTER_REGISTRY dictionary.

    Args:
        name (str): The name of the adapter.
    """

    def wrapper(cls):
        ADAPTER_REGISTRY[name] = cls
        return cls

    return wrapper


def get_adapter_class(name):
    """Get adapter class by name from registry"""
    if name not in ADAPTER_REGISTRY:
        available_adapters = list(ADAPTER_REGISTRY.keys())
        raise ValueError(
            f"Unknown adapter class: {name}. Available options: {available_adapters}"
        )
    return ADAPTER_REGISTRY[name]


class Adapter(ABC, nn.Module):
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @staticmethod
    def downsample_speech_features(x, k):
        """
        Downsample speech features by factor k
        Input: [bs, L, speech_enc_dim]
        Output: [bs, L // k, speech_enc_dim * k]
        """
        batch_size, seq_len, dim = x.size()
        # print(f"Encoder output shape: {x.shape}")

        # Discard frames that don't fit evenly into downsampling
        num_frames_to_discard = seq_len % k
        # print(f"Num frames to discard: {num_frames_to_discard}")
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        # print(f"Encoder output shape after discarding frames: {x.shape}")

        seq_len = x.size(1)
        x = x.contiguous()
        # print(f"Encoder output shape after contiguous: {x.shape}")

        # Reshape to downsample: combine k consecutive frames
        x = x.view(batch_size, seq_len // k, dim * k)

        return x

    @abstractmethod
    def project(self, x):
        pass


@register_adapter("linear-projector")
class LinearProjector(Adapter):
    def __init__(
        self,
        downsampling_rate=5,
        speech_encoder_dim=1024,
        llm_dim=2048,
        path_to_adapter_weights=None,
    ):
        """Projection Layer for the encoder output to the LLM input dimension.

        Args:
            downsampling_rate (int): Downsampling rate of the encoder output as discussed in the paper
            speech_ (int): Dimension of the encoder output
            llm_dim (int): Dimension of the LLM's input dimension. NOTE: this is not the vocab size,
                 the embedding layer in the llm maps vocab_size -> llm_dim. so this is not a very large number.
        """
        super().__init__()
        self.k = downsampling_rate
        self.speech_encoder_dim = speech_encoder_dim
        self.llm_dim = llm_dim
        self.linear1 = nn.Linear(self.speech_encoder_dim * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, llm_dim)

        if path_to_adapter_weights is not None:
            self.load_state_dict(torch.load(path_to_adapter_weights)["model"])

    def forward(self, x):
        x = self.downsample_speech_features(x, self.k)
        x = x.contiguous()
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

    def project(self, x):
        return self.forward(x)


@register_adapter("base-transformer-projector")
class BaseTransformer(Adapter):
    def __init__(
        self,
        speech_encoder_dim=1024,
        llm_dim=2048,
        downsampling_rate=5,
        num_layers=4,
        intermediate_size=3072,
        num_heads=12,
        hidden_size=768,
        dropout=0.1,
        path_to_adapter_weights=None,
    ):
        """Projection layer for the speech encoder output to the LLM input dimension.
        Reference: https://arxiv.org/abs/2409.17044
        NOTE: Code is most likely not same as the paper. Needed to project the speech encoder output to the intermediate size,
            apply the transformer encoder, and then project to the LLM input dimension.

        Args:
            speech_encoder_dim (int): Dimension of the speech encoder output
            llm_dim (int): Dimension of the LLM's input dimension. NOTE: this is not the vocab size,
                    the embedding layer in the llm maps vocab_size -> llm_dim. so this is not a very large number.
            downsampling_rate (int): Downsampling rate of the encoder output as discussed in the paper
            num_layers (int): Number of layers in the transformer encoder
            intermediate_size (int): Dimension of the intermediate size
            num_heads (int): Number of heads in the transformer encoder
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.k = downsampling_rate  # downsampling factor
        self.speech_encoder_dim = speech_encoder_dim
        self.llm_dim = llm_dim

        # Input projection: from speech encoder dim to transformer hidden size
        # After downsampling, input dim becomes speech_enc_dim * k
        self.input_projection = nn.Linear(speech_encoder_dim * self.k, hidden_size)

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        # Stack encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers
        )

        # Output projection: from transformer hidden size to LLM embedding dim
        self.output_projection = nn.Linear(hidden_size, llm_dim)

        # Layer norms
        self.input_layer_norm = nn.LayerNorm(hidden_size)
        self.output_layer_norm = nn.LayerNorm(llm_dim)

        self.dropout = nn.Dropout(dropout)

        if path_to_adapter_weights is not None:
            self.load_state_dict(torch.load(path_to_adapter_weights)["model"])

    def forward(self, x):
        """
        Args:
            x: [bs, L, speech_enc_dim] - output from speech encoder

        Returns:
            output: [bs, L // k, llm_embed_dim] - ready for LLM embedding layer
        """
        # Downsample speech features
        x = self.downsample_speech_features(x, self.k)  # [bs, L//k, speech_enc_dim * k]

        # Project to transformer hidden size
        x = self.input_projection(x)  # [bs, L//k, hidden_size]
        x = self.input_layer_norm(x)
        x = self.dropout(x)

        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # [bs, L//k, hidden_size]

        # Project to LLM embedding dimension
        output = self.output_projection(x)  # [bs, L//k, llm_embed_dim]
        output = self.output_layer_norm(output)

        return output

    def project(self, x):
        return self.forward(x)


@register_adapter("conv-based-transformer-projector")
class ConvBasedTransformer(Adapter):
    def __init__(
        self,
        speech_encoder_dim=1024,
        llm_dim=2048,
        downsampling_rate=5,
        num_layers=4,
        intermediate_size=3072,
        num_heads=12,
        hidden_size=768,
        dropout=0.1,
        path_to_adapter_weights=None,
    ):
        """Projection layer for the speech encoder output to the LLM input dimension.
        Uses a transformer encoder with 1D convolutional layers after the second layer for additional compression.
        Final compression factor is 4 (2 from initial downsampling, 2 from conv layers).

        Args:
            speech_encoder_dim (int): Dimension of the speech encoder output
            llm_dim (int): Dimension of the LLM's input dimension
            downsampling_rate (int): Initial downsampling rate of the encoder output
            num_layers (int): Number of layers in the transformer encoder
            intermediate_size (int): Dimension of the intermediate size
            num_heads (int): Number of heads in the transformer encoder
            hidden_size (int): Hidden size of the transformer
            dropout (float): Dropout rate
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.k = downsampling_rate  # initial downsampling factor
        self.speech_encoder_dim = speech_encoder_dim
        self.llm_dim = llm_dim

        # Input projection: from speech encoder dim to transformer hidden size
        self.input_projection = nn.Linear(speech_encoder_dim * self.k, hidden_size)

        # Split transformer layers to insert conv layers after second layer
        self.transformer_layers_before_conv = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=intermediate_size,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(2)
            ]
        )

        # 1D Convolutional layers with stride 2
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
                for _ in range(2)
            ]
        )

        # Remaining transformer layers
        self.transformer_layers_after_conv = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=intermediate_size,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_layers - 2)
            ]
        )

        # Output projection: from transformer hidden size to LLM embedding dim
        self.output_projection = nn.Linear(hidden_size, llm_dim)

        # Layer norms
        self.input_layer_norm = nn.LayerNorm(hidden_size)
        self.output_layer_norm = nn.LayerNorm(llm_dim)

        self.dropout = nn.Dropout(dropout)

        if path_to_adapter_weights is not None:
            self.load_state_dict(torch.load(path_to_adapter_weights)["model"])

    def forward(self, x):
        """
        Args:
            x: [bs, L, speech_enc_dim] - output from speech encoder

        Returns:
            output: [bs, L // (k*4), llm_embed_dim] - ready for LLM embedding layer
                   where k is initial downsampling and 4 is final compression factor
        """
        # Initial downsampling
        x = self.downsample_speech_features(x, self.k)  # [bs, L//k, speech_enc_dim * k]

        # Project to transformer hidden size
        x = self.input_projection(x)  # [bs, L//k, hidden_size]
        x = self.input_layer_norm(x)
        x = self.dropout(x)

        # First two transformer layers
        for layer in self.transformer_layers_before_conv:
            x = layer(x)  # [bs, L//k, hidden_size]

        # Apply 1D convolutions with stride 2
        # Transpose for conv1d: [bs, L//k, hidden_size] -> [bs, hidden_size, L//k]
        x = x.transpose(1, 2)
        for conv in self.conv_layers:
            x = conv(x)  # Each conv reduces sequence length by factor of 2
        # Transpose back: [bs, hidden_size, L//(k*4)] -> [bs, L//(k*4), hidden_size]
        x = x.transpose(1, 2)

        # Remaining transformer layers
        for layer in self.transformer_layers_after_conv:
            x = layer(x)  # [bs, L//(k*4), hidden_size]

        # Project to LLM embedding dimension
        output = self.output_projection(x)  # [bs, L//(k*4), llm_embed_dim]
        output = self.output_layer_norm(output)

        return output

    def project(self, x):
        return self.forward(x)


@register_adapter("qformer-projector")
class QFormerProjector(Adapter):
    def __init__(
        self,
        speech_encoder_dim=1024,
        llm_dim=2048,
        downsampling_rate=5,
        qformer_layers=6,
        query_len=64,
        path_to_adapter_weights=None,
    ):
        """Projection layer for the speech encoder output to the LLM input dimension using a Q-Former.

        Args:
            speech_encoder_dim (int): Dimension of the speech encoder output
            llm_dim (int): Dimension of the LLM's input dimension.
            downsampling_rate (int): Downsampling rate of the encoder output.
            qformer_layers (int): Number of layers in the Q-Former.
            query_len (int): Number of query tokens for the Q-Former.
            path_to_adapter_weights (str, optional): Path to pre-trained adapter weights.
        """
        super().__init__()
        self.k = downsampling_rate
        self.speech_encoder_dim = speech_encoder_dim
        self.llm_dim = llm_dim

        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = self.speech_encoder_dim * self.k
        configuration.num_hidden_layers = qformer_layers

        self.query_len = query_len
        self.query = nn.Parameter(
            torch.zeros(1, self.query_len, configuration.hidden_size)
        )
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)

        self.linear = nn.Linear(configuration.hidden_size, self.llm_dim)
        self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)

        if path_to_adapter_weights is not None:
            self.load_state_dict(torch.load(path_to_adapter_weights)["model"])

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: [bs, L, speech_enc_dim] - output from speech encoder
            attention_mask: [bs, L] - attention mask for the speech encoder output

        Returns:
            output: [bs, query_len, llm_dim] - ready for LLM input
        """
        x = self.downsample_speech_features(x, self.k)

        if attention_mask is not None:
            # downsample attention mask
            num_frames_to_discard = attention_mask.shape[1] % self.k
            if num_frames_to_discard > 0:
                attention_mask = attention_mask[:, :-num_frames_to_discard]

            attention_mask = attention_mask.view(
                attention_mask.shape[0], attention_mask.shape[1] // self.k, self.k
            )
            attention_mask = torch.any(attention_mask, dim=2)

        query = self.query.expand(x.shape[0], -1, -1)

        if attention_mask is None:
            attention_mask = torch.ones(x.shape[0], x.shape[1], device=x.device).long()

        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=attention_mask,
            return_dict=True,
        )

        query_proj = self.norm(self.linear(query_output.last_hidden_state))

        return query_proj

    def project(self, x):
        return self.forward(x)


@register_adapter("qformer-small-projector")
class QFormerSmallProjector(Adapter):
    def __init__(
        self,
        speech_encoder_dim=1024,
        llm_dim=2048,
        downsampling_rate=5,
        qformer_layers=2,
        query_len=64,
        path_to_adapter_weights=None,
    ):
        """Projection layer for the speech encoder output to the LLM input dimension using a Q-Former.

        Args:
            speech_encoder_dim (int): Dimension of the speech encoder output
            llm_dim (int): Dimension of the LLM's input dimension.
            downsampling_rate (int): Downsampling rate of the encoder output.
            qformer_layers (int): Number of layers in the Q-Former.
            query_len (int): Number of query tokens for the Q-Former.
            path_to_adapter_weights (str, optional): Path to pre-trained adapter weights.
        """
        super().__init__()
        self.k = downsampling_rate
        self.speech_encoder_dim = speech_encoder_dim
        self.llm_dim = llm_dim

        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = self.speech_encoder_dim * self.k
        configuration.num_hidden_layers = qformer_layers

        self.query_len = query_len
        self.query = nn.Parameter(
            torch.zeros(1, self.query_len, configuration.hidden_size)
        )
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)

        self.linear = nn.Linear(configuration.hidden_size, self.llm_dim)
        self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)

        if path_to_adapter_weights is not None:
            self.load_state_dict(torch.load(path_to_adapter_weights)["model"])

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: [bs, L, speech_enc_dim] - output from speech encoder
            attention_mask: [bs, L] - attention mask for the speech encoder output

        Returns:
            output: [bs, query_len, llm_dim] - ready for LLM input
        """
        x = self.downsample_speech_features(x, self.k)

        if attention_mask is not None:
            # downsample attention mask
            num_frames_to_discard = attention_mask.shape[1] % self.k
            if num_frames_to_discard > 0:
                attention_mask = attention_mask[:, :-num_frames_to_discard]

            attention_mask = attention_mask.view(
                attention_mask.shape[0], attention_mask.shape[1] // self.k, self.k
            )
            attention_mask = torch.any(attention_mask, dim=2)

        query = self.query.expand(x.shape[0], -1, -1)

        if attention_mask is None:
            attention_mask = torch.ones(x.shape[0], x.shape[1], device=x.device).long()

        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=attention_mask,
            return_dict=True,
        )

        query_proj = self.norm(self.linear(query_output.last_hidden_state))

        return query_proj

    def project(self, x):
        return self.forward(x)


@register_adapter("qformer-xsmall-projector")
class QFormerXSmallProjector(Adapter):
    def __init__(
        self,
        speech_encoder_dim=1024,
        llm_dim=2048,
        downsampling_rate=5,
        qformer_layers=2,
        query_len=32,
        path_to_adapter_weights=None,
    ):
        """Projection layer for the speech encoder output to the LLM input dimension using a Q-Former.

        Args:
            speech_encoder_dim (int): Dimension of the speech encoder output
            llm_dim (int): Dimension of the LLM's input dimension.
            downsampling_rate (int): Downsampling rate of the encoder output.
            qformer_layers (int): Number of layers in the Q-Former.
            query_len (int): Number of query tokens for the Q-Former.
            path_to_adapter_weights (str, optional): Path to pre-trained adapter weights.
        """
        super().__init__()
        self.k = downsampling_rate
        self.speech_encoder_dim = speech_encoder_dim
        self.llm_dim = llm_dim

        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = self.speech_encoder_dim * self.k
        configuration.num_hidden_layers = qformer_layers
        configuration.hidden_size = 384
        configuration.num_attention_heads = 8

        self.query_len = query_len
        self.query = nn.Parameter(
            torch.zeros(1, self.query_len, configuration.hidden_size)
        )
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)

        self.linear = nn.Linear(configuration.hidden_size, self.llm_dim)
        self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)

        if path_to_adapter_weights is not None:
            self.load_state_dict(torch.load(path_to_adapter_weights)["model"])

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: [bs, L, speech_enc_dim] - output from speech encoder
            attention_mask: [bs, L] - attention mask for the speech encoder output

        Returns:
            output: [bs, query_len, llm_dim] - ready for LLM input
        """
        x = self.downsample_speech_features(x, self.k)

        if attention_mask is not None:
            # downsample attention mask
            num_frames_to_discard = attention_mask.shape[1] % self.k
            if num_frames_to_discard > 0:
                attention_mask = attention_mask[:, :-num_frames_to_discard]

            attention_mask = attention_mask.view(
                attention_mask.shape[0], attention_mask.shape[1] // self.k, self.k
            )
            attention_mask = torch.any(attention_mask, dim=2)

        query = self.query.expand(x.shape[0], -1, -1)

        if attention_mask is None:
            attention_mask = torch.ones(x.shape[0], x.shape[1], device=x.device).long()

        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=attention_mask,
            return_dict=True,
        )

        query_proj = self.norm(self.linear(query_output.last_hidden_state))

        return query_proj

    def project(self, x):
        return self.forward(x)


@register_adapter("linear-large-projector")
class LinearLargeProjector(Adapter):
    def __init__(
        self,
        downsampling_rate=5,
        speech_encoder_dim=1024,
        llm_dim=2048,
        path_to_adapter_weights=None,
    ):
        """Projection Layer for the encoder output to the LLM input dimension.

        Args:
            downsampling_rate (int): Downsampling rate of the encoder output as discussed in the paper
            speech_ (int): Dimension of the encoder output
            llm_dim (int): Dimension of the LLM's input dimension. NOTE: this is not the vocab size,
                 the embedding layer in the llm maps vocab_size -> llm_dim. so this is not a very large number.
        """
        super().__init__()
        self.k = downsampling_rate
        self.speech_encoder_dim = speech_encoder_dim
        self.llm_dim = llm_dim
        self.linear1 = nn.Linear(self.speech_encoder_dim * self.k, 2048)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(2048, 1024)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(1024, 2048)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(2048, llm_dim)

        if path_to_adapter_weights is not None:
            self.load_state_dict(torch.load(path_to_adapter_weights)["model"])

    def forward(self, x):
        x = self.downsample_speech_features(x, self.k)
        x = x.contiguous()
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        return x

    def project(self, x):
        return self.forward(x)


@register_adapter("linear-large-projector-ln")
class LinearLargeProjectorLN(Adapter):
    def __init__(
        self,
        downsampling_rate=5,
        speech_encoder_dim=1024,
        llm_dim=2048,
        path_to_adapter_weights=None,
    ):
        """Projection Layer for the encoder output to the LLM input dimension.

        Args:
            downsampling_rate (int): Downsampling rate of the encoder output as discussed in the paper
            speech_ (int): Dimension of the encoder output
            llm_dim (int): Dimension of the LLM's input dimension. NOTE: this is not the vocab size,
                 the embedding layer in the llm maps vocab_size -> llm_dim. so this is not a very large number.
        """
        super().__init__()
        self.k = downsampling_rate
        self.speech_encoder_dim = speech_encoder_dim
        self.llm_dim = llm_dim
        self.linear1 = nn.Linear(self.speech_encoder_dim * self.k, 2048)
        self.relu1 = nn.SiLU()
        self.linear2 = nn.Linear(2048, 1024)
        self.ln2 = nn.LayerNorm(1024)
        self.relu2 = nn.SiLU()
        self.linear3 = nn.Linear(1024, 2048)
        self.ln3 = nn.LayerNorm(2048)
        self.relu3 = nn.SiLU()
        self.linear4 = nn.Linear(2048, llm_dim)

        if path_to_adapter_weights is not None:
            self.load_state_dict(torch.load(path_to_adapter_weights)["model"])

    def forward(self, x):
        x = self.downsample_speech_features(x, self.k)
        x = x.contiguous()
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.ln3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        return x

    def project(self, x):
        return self.forward(x)


@register_adapter("linear-small-projector-ln")
class LinearSmallProjectorLN(Adapter):
    def __init__(
        self,
        downsampling_rate=5,
        speech_encoder_dim=1024,
        llm_dim=2048,
        path_to_adapter_weights=None,
    ):
        """Projection Layer for the encoder output to the LLM input dimension.

        Args:
            downsampling_rate (int): Downsampling rate of the encoder output as discussed in the paper
            speech_ (int): Dimension of the encoder output
            llm_dim (int): Dimension of the LLM's input dimension. NOTE: this is not the vocab size,
                 the embedding layer in the llm maps vocab_size -> llm_dim. so this is not a very large number.
        """
        super().__init__()
        self.k = downsampling_rate
        self.speech_encoder_dim = speech_encoder_dim
        self.llm_dim = llm_dim
        self.linear1 = nn.Linear(self.speech_encoder_dim * self.k, 256)
        self.ln1 = nn.LayerNorm(256)
        self.relu1 = nn.SiLU()
        self.linear2 = nn.Linear(256, llm_dim)

        if path_to_adapter_weights is not None:
            self.load_state_dict(torch.load(path_to_adapter_weights)["model"])

    def forward(self, x):
        x = self.downsample_speech_features(x, self.k)
        x = x.contiguous()
        x = self.linear1(x)
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x

    def project(self, x):
        return self.forward(x)


# TODO: add a linear-base-projector-ln-2 layer
@register_adapter("linear-base-projector-ln")
class LinearBaseProjectorLN(Adapter):
    def __init__(
        self,
        downsampling_rate=5,
        speech_encoder_dim=1024,
        llm_dim=2048,
        path_to_adapter_weights=None,
    ):
        """Projection Layer for the encoder output to the LLM input dimension.

        Args:
            downsampling_rate (int): Downsampling rate of the encoder output as discussed in the paper
            speech_ (int): Dimension of the encoder output
            llm_dim (int): Dimension of the LLM's input dimension. NOTE: this is not the vocab size,
                 the embedding layer in the llm maps vocab_size -> llm_dim. so this is not a very large number.
        """
        super().__init__()
        self.k = downsampling_rate
        self.speech_encoder_dim = speech_encoder_dim
        self.llm_dim = llm_dim
        self.linear1 = nn.Linear(self.speech_encoder_dim * self.k, 768)
        self.ln1 = nn.LayerNorm(768)
        self.relu1 = nn.SiLU()
        self.linear2 = nn.Linear(768, 256)
        self.ln2 = nn.LayerNorm(256)
        self.relu2 = nn.SiLU()
        self.linear3 = nn.Linear(256, 768)
        self.ln3 = nn.LayerNorm(768)
        self.relu3 = nn.SiLU()
        self.linear4 = nn.Linear(768, llm_dim)

        if path_to_adapter_weights is not None:
            self.load_state_dict(torch.load(path_to_adapter_weights)["model"])

    def forward(self, x):
        x = self.downsample_speech_features(x, self.k)
        x = x.contiguous()
        x = self.linear1(x)
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.ln3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        return x

    def project(self, x):
        return self.forward(x)


# TODO: add a linear-xlarge-projector-ln layer
@register_adapter("linear-xlarge-projector-ln")
class LinearXLargeProjectorLN(Adapter):
    def __init__(
        self,
        downsampling_rate=5,
        speech_encoder_dim=1024,
        llm_dim=2048,
        path_to_adapter_weights=None,
    ):
        """Projection Layer for the encoder output to the LLM input dimension.

        Args:
            downsampling_rate (int): Downsampling rate of the encoder output as discussed in the paper
            speech_ (int): Dimension of the encoder output
            llm_dim (int): Dimension of the LLM's input dimension. NOTE: this is not the vocab size,
                 the embedding layer in the llm maps vocab_size -> llm_dim. so this is not a very large number.
        """
        super().__init__()
        self.k = downsampling_rate
        self.speech_encoder_dim = speech_encoder_dim
        self.llm_dim = llm_dim
        self.linear1 = nn.Linear(self.speech_encoder_dim * self.k, 3072)
        self.ln1 = nn.LayerNorm(3072)
        self.relu1 = nn.SiLU()
        self.linear2 = nn.Linear(3072, 2048)
        self.ln2 = nn.LayerNorm(2048)
        self.relu2 = nn.SiLU()
        self.linear3 = nn.Linear(2048, 3072)
        self.ln3 = nn.LayerNorm(3072)
        self.relu3 = nn.SiLU()
        self.linear4 = nn.Linear(3072, llm_dim)

        if path_to_adapter_weights is not None:
            self.load_state_dict(torch.load(path_to_adapter_weights)["model"])

    def forward(self, x):
        x = self.downsample_speech_features(x, self.k)
        x = x.contiguous()
        x = self.linear1(x)
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.ln3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        return x

    def project(self, x):
        return self.forward(x)


@register_adapter("linear-large-projector-dual-ln")
class LinearLargeProjectorDualLN(Adapter):
    def __init__(
        self,
        downsampling_rate=5,
        speech_encoder_dim=1024,
        llm_dim=2048,
        path_to_adapter_weights=None,
    ):
        super().__init__()
        self.k = downsampling_rate
        self.speech_encoder_dim = speech_encoder_dim
        self.llm_dim = llm_dim

        self.pre_ln = nn.LayerNorm(speech_encoder_dim * downsampling_rate)
        self.linear1 = nn.Linear(speech_encoder_dim * downsampling_rate, 2048)
        self.silu1 = nn.SiLU()
        self.linear2 = nn.Linear(2048, 1024)
        self.silu2 = nn.SiLU()
        self.linear3 = nn.Linear(1024, 2048)
        self.silu3 = nn.SiLU()
        self.linear4 = nn.Linear(2048, llm_dim)
        self.post_ln = nn.LayerNorm(llm_dim)

        if path_to_adapter_weights is not None:
            self.load_state_dict(torch.load(path_to_adapter_weights)["model"])

    def forward(self, x):
        x = self.downsample_speech_features(x, self.k)
        x = x.contiguous()
        x = self.pre_ln(x)

        x = self.linear1(x)
        x = self.silu1(x)
        x = self.linear2(x)
        x = self.silu2(x)
        x = self.linear3(x)
        x = self.silu3(x)
        x = self.linear4(x)

        x = self.post_ln(x)
        return x

    def project(self, x):
        return self.forward(x)


@register_adapter("convolutional-projector")
class ConvolutionalProjector(Adapter):
    def __init__(
        self,
        speech_encoder_dim=1024,
        llm_dim=2048,
        downsampling_rate=5,
        path_to_adapter_weights=None,
    ):
        super().__init__()
        self.k = downsampling_rate
        self.speech_encoder_dim = speech_encoder_dim
        self.llm_dim = llm_dim

        self.conv1 = nn.Conv1d(
            speech_encoder_dim, 1024, kernel_size=3, stride=2, padding=1
        )
        self.conv2 = nn.Conv1d(1024, 1024, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(1024, llm_dim, kernel_size=3, stride=2, padding=1)

        if path_to_adapter_weights is not None:
            self.load_state_dict(torch.load(path_to_adapter_weights)["model"])

    def forward(self, x):
        x = x.transpose(1, 2)  # [bs, speech_enc_dim * k, L]
        x = self.conv1(x)  # [bs, 1024, L]
        x = self.conv2(x)  # [bs, 1024, L]
        x = self.conv3(x)  # [bs, llm_dim, L]
        x = x.transpose(1, 2)  # [bs, L, llm_dim]
        return x

    def project(self, x):
        return self.forward(x)


@register_adapter("resnet-projector")
class ResNetProjector(Adapter):
    def __init__(
        self,
        speech_encoder_dim=1024,
        llm_dim=2048,
        downsampling_rate=5,
        path_to_adapter_weights=None,
    ):
        super().__init__()
        self.k = downsampling_rate
        self.speech_encoder_dim = speech_encoder_dim
        self.llm_dim = llm_dim

        # 3 convs at different resolutions
        self.conv1 = nn.Conv1d(
            speech_encoder_dim, 64, kernel_size=3, stride=2, padding=1
        )
        self.conv2 = nn.Conv1d(
            speech_encoder_dim, 64, kernel_size=5, stride=2, padding=2
        )
        self.conv3 = nn.Conv1d(
            speech_encoder_dim, 64, kernel_size=7, stride=2, padding=3
        )

        # final conv to project to llm_dim
        self.conv_final = nn.Conv1d(64 * 3, llm_dim, kernel_size=3, stride=1, padding=1)

        if path_to_adapter_weights is not None:
            self.load_state_dict(torch.load(path_to_adapter_weights)["model"])

    def forward(self, x):
        x = x.transpose(1, 2)  # [bs, speech_enc_dim * k, L]
        conv1_out = self.conv1(x)  # [bs, 64, L]
        conv2_out = self.conv2(x)  # [bs, 64, L]
        conv3_out = self.conv3(x)  # [bs, 64, L]
        all_conv_out = torch.cat(
            [conv1_out, conv2_out, conv3_out], dim=1
        )  # [bs, 192, L]
        x = self.conv_final(all_conv_out)  # [bs, llm_dim, L]
        x = x.transpose(1, 2)  # [bs, L, llm_dim]
        return x

    def project(self, x):
        return self.forward(x)


@register_adapter("resnet-projector-1024")
class ResNetProjector1024(Adapter):
    def __init__(
        self,
        speech_encoder_dim=1024,
        llm_dim=2048,
        downsampling_rate=5,
        path_to_adapter_weights=None,
    ):
        super().__init__()
        self.k = downsampling_rate
        self.speech_encoder_dim = speech_encoder_dim
        self.llm_dim = llm_dim

        # 3 convs at different resolutions
        self.conv1 = nn.Conv1d(
            speech_encoder_dim, 1024, kernel_size=3, stride=2, padding=1
        )
        self.conv2 = nn.Conv1d(
            speech_encoder_dim, 1024, kernel_size=5, stride=2, padding=2
        )
        self.conv3 = nn.Conv1d(
            speech_encoder_dim, 1024, kernel_size=7, stride=2, padding=3
        )

        # final conv to project to llm_dim
        self.conv_final = nn.Conv1d(
            1024 * 3, llm_dim, kernel_size=3, stride=1, padding=1
        )

        if path_to_adapter_weights is not None:
            self.load_state_dict(torch.load(path_to_adapter_weights)["model"])

    def forward(self, x):
        x = x.transpose(1, 2)  # [bs, speech_enc_dim * k, L]
        conv1_out = self.conv1(x)  # [bs, 1024, L]
        conv2_out = self.conv2(x)  # [bs, 1024, L]
        conv3_out = self.conv3(x)  # [bs, 1024, L]
        all_conv_out = torch.cat(
            [conv1_out, conv2_out, conv3_out], dim=1
        )  # [bs, 3072, L]
        x = self.conv_final(all_conv_out)  # [bs, llm_dim, L]
        x = x.transpose(1, 2)  # [bs, L, llm_dim]
        return x

    def project(self, x):
        return self.forward(x)


@register_adapter("resnet-projector-large")
class ResNetProjectorLarge(Adapter):
    def __init__(
        self,
        speech_encoder_dim=1024,
        llm_dim=2048,
        downsampling_rate=5,
        path_to_adapter_weights=None,
    ):
        super().__init__()
        self.k = downsampling_rate
        self.speech_encoder_dim = speech_encoder_dim
        self.llm_dim = llm_dim

        # 3 convs at different resolutions
        self.conv1 = nn.Conv1d(
            speech_encoder_dim, 256, kernel_size=3, stride=2, padding=1
        )
        self.conv2 = nn.Conv1d(
            speech_encoder_dim, 256, kernel_size=5, stride=2, padding=2
        )
        self.conv3 = nn.Conv1d(
            speech_encoder_dim, 256, kernel_size=7, stride=2, padding=3
        )
        self.conv4 = nn.Conv1d(
            speech_encoder_dim, 256, kernel_size=9, stride=2, padding=4
        )

        # final conv to project to llm_dim
        self.conv_final = nn.Conv1d(
            256 * 4, llm_dim, kernel_size=3, stride=1, padding=1
        )

        if path_to_adapter_weights is not None:
            self.load_state_dict(torch.load(path_to_adapter_weights)["model"])

    def forward(self, x):
        x = x.transpose(1, 2)  # [bs, speech_enc_dim * k, L]
        conv1_out = self.conv1(x)  # [bs, 256, L]
        conv2_out = self.conv2(x)  # [bs, 256, L]
        conv3_out = self.conv3(x)  # [bs, 256, L]
        conv4_out = self.conv4(x)  # [bs, 256, L]
        all_conv_out = torch.cat(
            [conv1_out, conv2_out, conv3_out, conv4_out], dim=1
        )  # [bs, 1024, L]
        x = self.conv_final(all_conv_out)  # [bs, llm_dim, L]
        x = x.transpose(1, 2)  # [bs, L, llm_dim]
        return x

    def project(self, x):
        return self.forward(x)


@register_adapter("multi-res-conv-small-projector")
class MultiResConvSmallProjector(Adapter):
    def __init__(
        self,
        speech_encoder_dim=1024,
        llm_dim=2048,
        downsampling_rate=5,
        path_to_adapter_weights=None,
    ):
        super().__init__()
        self.k = downsampling_rate
        self.speech_encoder_dim = speech_encoder_dim
        self.llm_dim = llm_dim

        # 3 convs at different resolutions
        self.conv1 = nn.Conv1d(
            speech_encoder_dim, 32, kernel_size=3, stride=2, padding=1
        )
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.relu1 = nn.SiLU()
        self.conv2 = nn.Conv1d(
            speech_encoder_dim, 32, kernel_size=5, stride=2, padding=2
        )
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.relu2 = nn.SiLU()
        self.conv3 = nn.Conv1d(
            speech_encoder_dim, 32, kernel_size=7, stride=2, padding=3
        )
        self.batch_norm3 = nn.BatchNorm1d(32)
        self.relu3 = nn.SiLU()

        # final conv to project to llm_dim
        self.conv_final = nn.Conv1d(32 * 3, llm_dim, kernel_size=1, stride=1, padding=1)
        self.batch_norm_final = nn.BatchNorm1d(llm_dim)
        self.relu_final = nn.SiLU()

        if path_to_adapter_weights is not None:
            self.load_state_dict(torch.load(path_to_adapter_weights)["model"])

    def forward(self, x):
        x = x.transpose(1, 2)  # [bs, speech_enc_dim * k, L]
        conv1_out = self.conv1(x)  # [bs, 32, L]
        conv1_out = self.batch_norm1(conv1_out)  # [bs, 32, L]
        conv1_out = self.relu1(conv1_out)  # [bs, 32, L]

        conv2_out = self.conv2(x)  # [bs, 32, L]
        conv2_out = self.batch_norm2(conv2_out)  # [bs, 32, L]
        conv2_out = self.relu2(conv2_out)  # [bs, 32, L]

        conv3_out = self.conv3(x)  # [bs, 32, L]
        conv3_out = self.batch_norm3(conv3_out)  # [bs, 32, L]
        conv3_out = self.relu3(conv3_out)  # [bs, 32, L]

        all_conv_out = torch.cat(
            [conv1_out, conv2_out, conv3_out], dim=1
        )  # [bs, 192, L]
        x = self.conv_final(all_conv_out)  # [bs, llm_dim, L]
        x = self.batch_norm_final(x)  # [bs, llm_dim, L]
        x = self.relu_final(x)  # [bs, llm_dim, L]

        x = x.transpose(1, 2)  # [bs, L, llm_dim]
        return x

    def project(self, x):
        return self.forward(x)


@register_adapter("multi-res-conv-large-projector")
class MultiResConvLargeProjector(Adapter):
    def __init__(
        self,
        speech_encoder_dim=1024,
        llm_dim=2048,
        downsampling_rate=5,
        path_to_adapter_weights=None,
    ):
        super().__init__()
        self.k = downsampling_rate
        self.speech_encoder_dim = speech_encoder_dim
        self.llm_dim = llm_dim

        # 3 convs at different resolutions
        self.conv1 = nn.Conv1d(
            speech_encoder_dim, 128, kernel_size=3, stride=2, padding=1
        )
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.relu1 = nn.SiLU()
        self.conv2 = nn.Conv1d(
            speech_encoder_dim, 128, kernel_size=5, stride=2, padding=2
        )
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.relu2 = nn.SiLU()
        self.conv3 = nn.Conv1d(
            speech_encoder_dim, 128, kernel_size=7, stride=2, padding=3
        )
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.relu3 = nn.SiLU()
        self.conv4 = nn.Conv1d(
            speech_encoder_dim, 128, kernel_size=9, stride=2, padding=4
        )
        self.batch_norm4 = nn.BatchNorm1d(128)
        self.relu4 = nn.SiLU()

        # final conv to project to llm_dim
        self.conv_final = nn.Conv1d(
            128 * 4, llm_dim, kernel_size=1, stride=1, padding=0
        )
        self.batch_norm_final = nn.BatchNorm1d(llm_dim)
        self.relu_final = nn.SiLU()

        if path_to_adapter_weights is not None:
            self.load_state_dict(torch.load(path_to_adapter_weights)["model"])

    def forward(self, x):
        x = x.transpose(1, 2)  # [bs, speech_enc_dim * k, L]
        conv1_out = self.conv1(x)  # [bs, 32, L]
        conv1_out = self.batch_norm1(conv1_out)  # [bs, 32, L]
        conv1_out = self.relu1(conv1_out)  # [bs, 32, L]

        conv2_out = self.conv2(x)  # [bs, 32, L]
        conv2_out = self.batch_norm2(conv2_out)  # [bs, 32, L]
        conv2_out = self.relu2(conv2_out)  # [bs, 32, L]

        conv3_out = self.conv3(x)  # [bs, 32, L]
        conv3_out = self.batch_norm3(conv3_out)  # [bs, 32, L]
        conv3_out = self.relu3(conv3_out)  # [bs, 32, L]

        conv4_out = self.conv4(x)  # [bs, 32, L]
        conv4_out = self.batch_norm4(conv4_out)  # [bs, 32, L]
        conv4_out = self.relu4(conv4_out)  # [bs, 32, L]

        all_conv_out = torch.cat(
            [conv1_out, conv2_out, conv3_out, conv4_out], dim=1
        )  # [bs, 192, L]
        x = self.conv_final(all_conv_out)  # [bs, llm_dim, L]
        x = self.batch_norm_final(x)  # [bs, llm_dim, L]
        x = self.relu_final(x)  # [bs, llm_dim, L]

        x = x.transpose(1, 2)  # [bs, L, llm_dim]
        return x

    def project(self, x):
        return self.forward(x)


@register_adapter("multi-res-conv-xlarge-projector")
class MultiResConvXLargeProjector(Adapter):
    def __init__(
        self,
        speech_encoder_dim=1024,
        llm_dim=2048,
        downsampling_rate=5,
        path_to_adapter_weights=None,
    ):
        super().__init__()
        self.k = downsampling_rate
        self.speech_encoder_dim = speech_encoder_dim
        self.llm_dim = llm_dim

        # 3 convs at different resolutions
        self.conv1 = nn.Conv1d(
            speech_encoder_dim, 256, kernel_size=3, stride=2, padding=1
        )
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.relu1 = nn.SiLU()
        self.conv2 = nn.Conv1d(
            speech_encoder_dim, 256, kernel_size=5, stride=2, padding=2
        )
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.relu2 = nn.SiLU()
        self.conv3 = nn.Conv1d(
            speech_encoder_dim, 256, kernel_size=7, stride=2, padding=3
        )
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.relu3 = nn.SiLU()
        self.conv4 = nn.Conv1d(
            speech_encoder_dim, 256, kernel_size=9, stride=2, padding=4
        )
        self.batch_norm4 = nn.BatchNorm1d(256)
        self.relu4 = nn.SiLU()

        self.conv5 = nn.Conv1d(
            speech_encoder_dim, 256, kernel_size=11, stride=2, padding=5
        )
        self.batch_norm5 = nn.BatchNorm1d(256)
        self.relu5 = nn.SiLU()

        # final conv to project to llm_dim
        self.conv_final = nn.Conv1d(
            256 * 5, llm_dim, kernel_size=1, stride=1, padding=0
        )
        self.batch_norm_final = nn.BatchNorm1d(llm_dim)
        self.relu_final = nn.SiLU()

        if path_to_adapter_weights is not None:
            self.load_state_dict(torch.load(path_to_adapter_weights)["model"])

    def forward(self, x):
        x = x.transpose(1, 2)  # [bs, speech_enc_dim * k, L]
        conv1_out = self.conv1(x)  # [bs, 256, L]
        conv1_out = self.batch_norm1(conv1_out)  # [bs, 256, L]
        conv1_out = self.relu1(conv1_out)  # [bs, 256, L]

        conv2_out = self.conv2(x)  # [bs, 256, L]
        conv2_out = self.batch_norm2(conv2_out)  # [bs, 256, L]
        conv2_out = self.relu2(conv2_out)  # [bs, 256, L]

        conv3_out = self.conv3(x)  # [bs, 256, L]
        conv3_out = self.batch_norm3(conv3_out)  # [bs, 256, L]
        conv3_out = self.relu3(conv3_out)  # [bs, 256, L]

        conv4_out = self.conv4(x)  # [bs, 256, L]
        conv4_out = self.batch_norm4(conv4_out)  # [bs, 256, L]
        conv4_out = self.relu4(conv4_out)  # [bs, 256, L]

        conv5_out = self.conv5(x)  # [bs, 256, L]
        conv5_out = self.batch_norm5(conv5_out)  # [bs, 256, L]
        conv5_out = self.relu5(conv5_out)  # [bs, 256, L]

        all_conv_out = torch.cat(
            [conv1_out, conv2_out, conv3_out, conv4_out, conv5_out], dim=1
        )  # [bs, 1280, L]
        x = self.conv_final(all_conv_out)  # [bs, llm_dim, L]
        x = self.batch_norm_final(x)  # [bs, llm_dim, L]
        x = self.relu_final(x)  # [bs, llm_dim, L]

        x = x.transpose(1, 2)  # [bs, L, llm_dim]
        return x

    def project(self, x):
        return self.forward(x)


@register_adapter("multi-res-conv-xxlarge-projector")
class MultiResConvXXLargeProjector(Adapter):
    def __init__(
        self,
        speech_encoder_dim=1024,
        llm_dim=2048,
        downsampling_rate=5,
        path_to_adapter_weights=None,
    ):
        super().__init__()
        self.k = downsampling_rate
        self.speech_encoder_dim = speech_encoder_dim
        self.llm_dim = llm_dim

        # 3 convs at different resolutions
        self.conv1 = nn.Conv1d(
            speech_encoder_dim, 512, kernel_size=3, stride=2, padding=1
        )
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.relu1 = nn.SiLU()
        self.conv2 = nn.Conv1d(
            speech_encoder_dim, 512, kernel_size=5, stride=2, padding=2
        )
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.relu2 = nn.SiLU()
        self.conv3 = nn.Conv1d(
            speech_encoder_dim, 512, kernel_size=7, stride=2, padding=3
        )
        self.batch_norm3 = nn.BatchNorm1d(512)
        self.relu3 = nn.SiLU()
        self.conv4 = nn.Conv1d(
            speech_encoder_dim, 512, kernel_size=9, stride=2, padding=4
        )
        self.batch_norm4 = nn.BatchNorm1d(512)
        self.relu4 = nn.SiLU()

        self.conv5 = nn.Conv1d(
            speech_encoder_dim, 512, kernel_size=11, stride=2, padding=5
        )
        self.batch_norm5 = nn.BatchNorm1d(512)
        self.relu5 = nn.SiLU()

        self.conv6 = nn.Conv1d(
            speech_encoder_dim, 512, kernel_size=13, stride=2, padding=6
        )
        self.batch_norm6 = nn.BatchNorm1d(512)
        self.relu6 = nn.SiLU()

        # final conv to project to llm_dim
        self.conv_final = nn.Conv1d(
            512 * 6, llm_dim, kernel_size=1, stride=1, padding=0
        )
        self.batch_norm_final = nn.BatchNorm1d(llm_dim)
        self.relu_final = nn.SiLU()

        if path_to_adapter_weights is not None:
            self.load_state_dict(torch.load(path_to_adapter_weights)["model"])

    def forward(self, x):
        x = x.transpose(1, 2)  # [bs, speech_enc_dim * k, L]
        conv1_out = self.conv1(x)  # [bs, 512, L]
        conv1_out = self.batch_norm1(conv1_out)  # [bs, 512, L]
        conv1_out = self.relu1(conv1_out)  # [bs, 512, L]

        conv2_out = self.conv2(x)  # [bs, 512, L]
        conv2_out = self.batch_norm2(conv2_out)  # [bs, 512, L]
        conv2_out = self.relu2(conv2_out)  # [bs, 512, L]

        conv3_out = self.conv3(x)  # [bs, 512, L]
        conv3_out = self.batch_norm3(conv3_out)  # [bs, 512, L]
        conv3_out = self.relu3(conv3_out)  # [bs, 512, L]

        conv4_out = self.conv4(x)  # [bs, 512, L]
        conv4_out = self.batch_norm4(conv4_out)  # [bs, 512, L]
        conv4_out = self.relu4(conv4_out)  # [bs, 512, L]

        conv5_out = self.conv5(x)  # [bs, 512, L]
        conv5_out = self.batch_norm5(conv5_out)  # [bs, 512, L]
        conv5_out = self.relu5(conv5_out)  # [bs, 512, L]

        conv6_out = self.conv6(x)  # [bs, 512, L]
        conv6_out = self.batch_norm6(conv6_out)  # [bs, 512, L]
        conv6_out = self.relu6(conv6_out)  # [bs, 512, L]

        all_conv_out = torch.cat(
            [conv1_out, conv2_out, conv3_out, conv4_out, conv5_out, conv6_out], dim=1
        )  # [bs, 3072, L]
        x = self.conv_final(all_conv_out)  # [bs, llm_dim, L]
        x = self.batch_norm_final(x)  # [bs, llm_dim, L]
        x = self.relu_final(x)  # [bs, llm_dim, L]

        x = x.transpose(1, 2)  # [bs, L, llm_dim]
        return x

    def project(self, x):
        return self.forward(x)


@register_adapter("multi-res-dc1d-projector")
class MultiResDC1DProjector(Adapter):
    def __init__(
        self,
        speech_encoder_dim=1024,
        llm_dim=2048,
        downsampling_rate=5,
        path_to_adapter_weights=None,
    ):
        from src.dc1d import PackedDeformConv1d

        super().__init__()
        self.k = downsampling_rate
        self.speech_encoder_dim = speech_encoder_dim
        self.llm_dim = llm_dim
        self.conv1 = PackedDeformConv1d(
            in_channels=speech_encoder_dim,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.relu1 = nn.SiLU()
        self.conv2 = PackedDeformConv1d(
            in_channels=speech_encoder_dim,
            out_channels=256,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.relu2 = nn.SiLU()
        self.conv3 = PackedDeformConv1d(
            in_channels=speech_encoder_dim,
            out_channels=256,
            kernel_size=7,
            stride=1,
            padding=3,
        )
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.relu3 = nn.SiLU()
        self.conv4 = PackedDeformConv1d(
            in_channels=speech_encoder_dim,
            out_channels=256,
            kernel_size=9,
            stride=1,
            padding=4,
        )
        self.batch_norm4 = nn.BatchNorm1d(256)
        self.relu4 = nn.SiLU()
        self.conv_final = PackedDeformConv1d(
            in_channels=256 * 4,
            out_channels=llm_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # [bs, speech_enc_dim * k, L]
        x_1 = self.conv1(x)  # [bs, llm_dim, L]
        x_1 = self.batch_norm1(x_1)  # [bs, llm_dim, L]
        x_1 = self.relu1(x_1)  # [bs, llm_dim, L]
        x_2 = self.conv2(x)  # [bs, llm_dim, L]
        x_2 = self.batch_norm2(x_2)  # [bs, llm_dim, L]
        x_2 = self.relu2(x_2)  # [bs, llm_dim, L]
        x_3 = self.conv3(x)  # [bs, llm_dim, L]
        x_3 = self.batch_norm3(x_3)  # [bs, llm_dim, L]
        x_3 = self.relu3(x_3)  # [bs, llm_dim, L]
        x_4 = self.conv4(x)  # [bs, llm_dim, L]
        x_4 = self.batch_norm4(x_4)  # [bs, llm_dim, L]
        x_4 = self.relu4(x_4)  # [bs, llm_dim, L]
        x_cat = torch.cat([x_1, x_2, x_3, x_4], dim=1)  # [bs, 512 * 4, L]
        x = self.conv_final(x_cat)  # [bs, llm_dim, L]
        x = x.transpose(1, 2)  # [bs, L, llm_dim]
        return x

    def project(self, x):
        return self.forward(x)


@register_adapter("input-conditioned-resnet-projector")
class InputConditionedResNetProjector(Adapter):
    def __init__(
        self,
        speech_encoder_dim=1024,
        llm_dim=2048,
        downsampling_rate=5,
        path_to_adapter_weights=None,
        num_blocks=33,
        num_mels=128,
    ):
        super().__init__()
        self.k = downsampling_rate
        self.speech_encoder_dim = speech_encoder_dim
        self.llm_dim = llm_dim

        # 3 convs at different resolutions
        self.conv1 = nn.Conv1d(
            speech_encoder_dim, 64, kernel_size=3, stride=2, padding=1
        )
        self.conv2 = nn.Conv1d(
            speech_encoder_dim, 64, kernel_size=5, stride=2, padding=2
        )
        self.conv3 = nn.Conv1d(
            speech_encoder_dim, 64, kernel_size=7, stride=2, padding=3
        )

        # final conv to project to llm_dim
        self.conv_final = nn.Conv1d(64 * 3, llm_dim, kernel_size=3, stride=1, padding=1)

        # HOTFIX: ad hoc for auxiliary module
        self._add_auxilliary_input_conditioning_layers(num_blocks=33, num_mels=128)

        if path_to_adapter_weights is not None:
            self.load_state_dict(torch.load(path_to_adapter_weights)["model"])

    def forward(self, x, aux_input):
        # auxilliary input conditioning on aux_input [bs, num_mels, L_audio]
        aux_conv_out = self.aux_input(aux_input)  # [bs, 64, L_audio]
        aux_pooled_out = self.aux_global_pooling(aux_conv_out)  # [bs, 64, 1]
        aux_pooled_out = aux_pooled_out.squeeze(2)  # [bs, 64]
        aux_linear_out = self.aux_linear(aux_pooled_out)  # [bs, 33]
        aux_block_weights = self.aux_softmax(aux_linear_out)  # [bs, 33]

        aux_block_weights = aux_block_weights.unsqueeze(-1).unsqueeze(
            -1
        )  # [bs, 33, 1, 1]
        x = x * aux_block_weights  # [bs, 33, L, D]

        x = x.sum(dim=1)  # [bs, L, D]

        x = x.transpose(1, 2)  # [bs, D, L]
        conv1_out = self.conv1(x)  # [bs, 64, L]
        conv2_out = self.conv2(x)  # [bs, 64, L]
        conv3_out = self.conv3(x)  # [bs, 64, L]
        all_conv_out = torch.cat(
            [conv1_out, conv2_out, conv3_out], dim=1
        )  # [bs, 192, L]
        x = self.conv_final(all_conv_out)  # [bs, llm_dim, L]
        x = x.transpose(1, 2)  # [bs, L, llm_dim]
        return x

    def project(self, x, aux_input):
        return self.forward(x, aux_input)

    def _add_auxilliary_input_conditioning_layers(self, num_blocks, num_mels):
        """this is a hack to add auxilliary input conditioning layers to the adapter

        this block will ingest the mel spec of shape [bs, mels, L] and output a tensor of shape [bs, num_blocks]
        """
        self.aux_input = nn.Conv1d(num_mels, 64, kernel_size=3, stride=2, padding=1)
        self.aux_global_pooling = nn.AdaptiveAvgPool1d(1)
        self.aux_linear = nn.Linear(64, num_blocks)
        self.aux_softmax = nn.Softmax(dim=1)


@register_adapter("input-conditioned-mlp-large-projector-ln-per-5-blocks")
class InputConditionedMLPLargeProjectorLNPer5Blocks(Adapter):
    def __init__(
        self,
        downsampling_rate=5,
        speech_encoder_dim=1024,
        llm_dim=2048,
        path_to_adapter_weights=None,
        num_blocks=33,
        num_mels=128,
    ):
        """Projection Layer for the encoder output to the LLM input dimension.

        Args:
            downsampling_rate (int): Downsampling rate of the encoder output as discussed in the paper
            speech_ (int): Dimension of the encoder output
            llm_dim (int): Dimension of the LLM's input dimension. NOTE: this is not the vocab size,
                 the embedding layer in the llm maps vocab_size -> llm_dim. so this is not a very large number.
        """
        super().__init__()
        self.k = downsampling_rate
        self.speech_encoder_dim = speech_encoder_dim
        self.llm_dim = llm_dim
        self.num_blocks = 5  # NOTE: this is a hack to create a module dict with keys as block_0, block_1, ...
        self.num_mels = num_mels
        self.block2layers = nn.ModuleDict(
            {
                f"block_{i}": nn.Sequential(
                    nn.Linear(self.speech_encoder_dim * self.k, 1024),
                    nn.LayerNorm(1024),
                    nn.SiLU(),
                    nn.Linear(1024, self.llm_dim),
                    nn.LayerNorm(self.llm_dim),
                    nn.SiLU(),
                )
                for i in range(self.num_blocks)
            }
        )

        self._add_auxilliary_input_conditioning_layers()

        if path_to_adapter_weights is not None:
            self.load_state_dict(torch.load(path_to_adapter_weights)["model"])

    def _add_auxilliary_input_conditioning_layers(self):
        """this is a hack to add auxilliary input conditioning layers to the adapter

        this block will ingest the mel spec of shape [bs, mels, L_audio] and output a tensor of shape [bs, num_blocks]
        """
        self.aux_input = nn.Conv1d(
            self.num_mels, 64, kernel_size=3, stride=2, padding=1
        )
        self.aux_global_pooling = nn.AdaptiveAvgPool1d(1)
        self.aux_silu = nn.SiLU()
        self.aux_linear = nn.Linear(64, self.num_blocks)
        self.aux_softmax = nn.Softmax(dim=1)

    def forward(self, x, aux_input):
        # NOTE: we will only use the last 5 blocks of the speech encoder
        x = x[:, -self.num_blocks :, :, :]  # [bs, num_blocks, L, D]

        # auxilliary input conditioning on aux_input [bs, num_mels, L_audio]
        aux_conv_out = self.aux_input(aux_input)  # [bs, 64, L_audio]
        aux_pooled_out = self.aux_global_pooling(aux_conv_out)  # [bs, 64, 1]
        aux_pooled_out = self.aux_silu(aux_pooled_out)  # [bs, 64, 1]
        aux_pooled_out = aux_pooled_out.squeeze(2)  # [bs, 64]
        aux_linear_out = self.aux_linear(aux_pooled_out)  # [bs, num_blocks]
        aux_block_weights = self.aux_softmax(aux_linear_out)  # [bs, num_blocks]

        list_of_per_block_outputs = []
        for block_idx in range(self.num_blocks):
            layers = self.block2layers[f"block_{block_idx}"]
            aux_weight_this_block = (
                aux_block_weights[:, block_idx].unsqueeze(1).unsqueeze(1)
            )  # [bs, 1, 1]
            x_i = x[:, block_idx, :, :]  # [bs, L, D]
            x_i = self.downsample_speech_features(x_i, self.k)  # [bs, L', D]
            x_i = x_i.contiguous()
            x_i = layers(x_i)  # [bs, L', D]
            x_i = x_i * aux_weight_this_block  # [bs, L', D]
            list_of_per_block_outputs.append(x_i)  # [bs, L', D]

        output_tensor = torch.stack(
            list_of_per_block_outputs, dim=1
        )  # [bs, num_blocks, L', D]
        output_tensor = torch.sum(output_tensor, dim=1)  # [bs, L', D]
        return output_tensor

    def project(self, x, aux_input):
        return self.forward(x, aux_input)


@register_adapter("titanet-projector")
class TitanetProjector(Adapter):
    """
    TitaNet projector with configurable input/output dimensions and length dimension.
    """

    def __init__(
        self,
        downsampling_rate=5,
        speech_encoder_dim=256,
        llm_dim=256,
        path_to_adapter_weights=None,
    ):
        super().__init__()
        from src.modules import TitanetMegaBlock

        # Set an intermediate dimension for hidden layers
        hidden_dim = 256

        # Block 1: speech_encoder_dim -> hidden_dim
        self.block1 = TitanetMegaBlock(
            input_size=speech_encoder_dim,
            output_size=hidden_dim,
            kernel_size=3,
            n_sub_blocks=1,
            se_reduction=16,
            dropout=0.1,
        )
        # Block 2: hidden_dim -> hidden_dim
        self.block2 = TitanetMegaBlock(
            input_size=hidden_dim,
            output_size=hidden_dim,
            kernel_size=7,
            n_sub_blocks=1,
            se_reduction=16,
            dropout=0.1,
        )
        # Block 3: hidden_dim -> llm_dim
        self.block3 = TitanetMegaBlock(
            input_size=hidden_dim,
            output_size=llm_dim,
            kernel_size=11,
            n_sub_blocks=1,
            se_reduction=16,
            dropout=0.1,
        )

        if path_to_adapter_weights is not None:
            self.load_state_dict(torch.load(path_to_adapter_weights)["model"])

    def forward(self, x):
        # x: [bs, L, speech_encoder_dim]
        x = x.transpose(1, 2)  # [bs, speech_encoder_dim, L]
        x = self.block1(x)  # [bs, hidden_dim, L]
        x = self.block2(x)  # [bs, hidden_dim, L]
        x = self.block3(x)  # [bs, llm_dim, L]
        x = x.transpose(1, 2)  # [bs, L, llm_dim]
        return x

    def project(self, x):
        return self.forward(x)

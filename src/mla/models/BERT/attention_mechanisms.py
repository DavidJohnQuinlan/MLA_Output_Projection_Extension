import logging
from collections.abc import Callable
from typing import Unpack

import torch
from torch import nn
from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.modeling_utils import AttentionInterface
from transformers.utils import TransformersKwargs

logger = logging.getLogger(__name__)

ALL_ATTENTION_FUNCTIONS = AttentionInterface()


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class BertBaseAttention(nn.Module):
    """
    """
    def __init__(self, config, is_causal=False, layer_idx=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.config = config

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.is_decoder = config.is_decoder
        self.is_causal = is_causal
        self.layer_idx = layer_idx


class BertSelfAttention(BertBaseAttention):
    """
    Computes Multi-Head Self-Attention (MHA) for BERT-based architectures.

    This layer transforms input hidden states into Query (Q), Key (K), and Value (V)
    vectors, computes attention scores using scaled dot-product attention,
    and optionally incorporates relative positional information.

    The attention mechanism follows the formula:
    Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V

    Args:
        config (PretrainedConfig): The model configuration containing hyper-parameters
            such as hidden_size, num_attention_heads, and dropout rates.
        position_embedding_type (str, optional): The type of position embeddings to use.
            Options include "absolute", "relative_key", or "relative_key_query".
            Defaults to "absolute" via config.
        layer_idx (int, optional): The index of the layer in the encoder/decoder stack.
            Required for proper KV cache indexing. Defaults to None.
    """
    def __init__(self, config, is_causal=False, layer_idx=None):
        super().__init__(config, is_causal=is_causal, layer_idx=layer_idx)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        """
        Calculates the attention context layer and attention probabilities.

        Args:
            hidden_states (torch.Tensor): Input sequence of shape (batch_size, sequence_length, hidden_size).
            attention_mask (torch.FloatTensor, optional): Mask to avoid performing attention on padding tokens.
                Shape (batch_size, 1, 1, seq_length) or (batch_size, 1, seq_length, seq_length). Defaults to None.
            past_key_values (Cache, optional): Cached Key and Value states for accelerating sequential decoding.
                Defaults to None.

        Returns:
            tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - context_layer: The output of the attention mechanism, shape
                  (batch_size, sequence_length, hidden_size).
                - attention_probs: The attention weights (if output_attentions=True),
                  shape (batch_size, num_heads, sequence_length, sequence_length).
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)

        # get all proj
        query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)

        if past_key_values is not None:
            # decoder-only bert can have a simple dynamic cache for example
            current_past_key_values = past_key_values
            if isinstance(past_key_values, EncoderDecoderCache):
                current_past_key_values = past_key_values.self_attention_cache

            # save all key/value_layer to cache to be re-used for fast auto-regressive generation
            key_layer, value_layer = current_past_key_values.update(key_layer, value_layer, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout.p,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return attn_output, attn_weights


class MultiHeadedLatentAttention(BertBaseAttention):
    """
    Computes Multi-Head Latent Attention (MLA) for BERT-based architectures.

    More specifically, the keys and values are projected into a much smaller latent space before
    being re-projected back up to the hidden state space. The idea being that when using decoder
    models that it will be much cheaper to cache the kv shared latent vector given that it is much
    smaller in size compared to standard MHA key and value attention vectors.

    The attention mechanism follows the formula:
    Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V

    Args:
        config (PretrainedConfig): The model configuration containing hyper-parameters
            such as hidden_size, num_attention_heads, and dropout rates.
        position_embedding_type (str, optional): The type of position embeddings to use.
            Options include "absolute", "relative_key", or "relative_key_query".
            Defaults to "absolute" via config.
        layer_idx (int, optional): The index of the layer in the encoder/decoder stack.
            Required for proper KV cache indexing. Defaults to None.
    """

    def __init__(self, config, is_causal=False, layer_idx=None):
        super().__init__(config, is_causal=is_causal, layer_idx=layer_idx)
        
        # Define the layers associated with low rank joint compression of keys and values
        self.kv_down_projection = nn.Linear(config.hidden_size, config.kv_compression_dim)
        self.k_up_projection = nn.Linear(config.kv_compression_dim, config.hidden_size)
        self.v_up_projection = nn.Linear(config.kv_compression_dim, config.hidden_size)

        # Define the layers associated with low rank compression of queries
        self.q_down_projection = nn.Linear(config.hidden_size, config.q_compression_dim)
        self.q_up_projection = nn.Linear(config.q_compression_dim, config.hidden_size)

        # Define the norm layer
        self.kv_layer_norm = nn.LayerNorm(config.kv_compression_dim)
        self.q_layer_norm = nn.LayerNorm(config.q_compression_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        """
        Project the keys/values and queries down to their respective latent spaces before projecting them back up to the
        hidden state space. Once the keys/values and queries are returned to the hidden state space apply attention as
        normal.

        Additional code has been added to ensure that the compressed latent kv vector is cached.

        Args:
            hidden_states (torch.Tensor): Input sequence of shape (batch_size, sequence_length, hidden_size).
            attention_mask (torch.FloatTensor, optional): Mask to avoid performing attention on padding tokens.
                Shape (batch_size, 1, 1, seq_length) or (batch_size, 1, seq_length, seq_length). Defaults to None.
            past_key_values (Cache, optional): Cached Key and Value states for accelerating sequential decoding.
                Defaults to None.

        Returns:
            tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - context_layer: The output of the attention mechanism, shape
                  (batch_size, sequence_length, hidden_size).
                - attention_probs: The attention weights (if output_attentions=True),
                  shape (batch_size, num_heads, sequence_length, sequence_length).
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)

        # Down project to the KV latent space -> (batch_size, seq_len, q/kv_compression_dim)
        kv_compressed_latent_vector = self.kv_down_projection(hidden_states)
        q_compressed_latent_vector = self.q_down_projection(hidden_states)

        # For stability pass the KV latent vector through a layer norm -> (batch_size, seq_len, q/kv_compression_dim)
        kv_compressed_latent_vector_norm = self.kv_layer_norm(kv_compressed_latent_vector)
        q_compressed_latent_vector_norm = self.q_layer_norm(q_compressed_latent_vector)

        # Up project back to the hidden space -> (batch_size, seq_len, hidden_size)
        # -> (batch_size, seq_len, n_heads, head_dim) -> (batch_size, n_heads, seq_len, head_dim)
        key_layer = self.k_up_projection(kv_compressed_latent_vector_norm).view(*hidden_shape).transpose(1, 2)
        value_layer = self.v_up_projection(kv_compressed_latent_vector_norm).view(*hidden_shape).transpose(1, 2)
        query_layer = self.q_up_projection(q_compressed_latent_vector_norm).view(*hidden_shape).transpose(1, 2)

        if past_key_values is not None:
            # decoder-only bert can have a simple dynamic cache for example
            current_past_key_values = past_key_values
            if isinstance(past_key_values, EncoderDecoderCache):
                current_past_key_values = past_key_values.self_attention_cache

            # save all key/value_layer to cache to be re-used for fast auto-regressive generation
            key_layer, value_layer = current_past_key_values.update(key_layer, value_layer, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout.p,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return attn_output, attn_weights


class BertSelfOutput(nn.Module):
    """
    Applies the final linear projection, dropout, and residual connection
    to the output of the Self-Attention layer.

    This module performs the "Add & Norm" step described in the original
    Transformer architecture: LayerNorm(input_tensor + Dropout(Linear(hidden_states))).

    Args:
        config (PretrainedConfig): The model configuration containing
            hidden_size, layer_norm_eps, and hidden_dropout_prob.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): The output from the attention layer (batch_size, seq_len, hidden_size).
            input_tensor (torch.Tensor): The original input to the attention layer (the "shortcut" connection).

        Returns:
            torch.Tensor: The normalized and combined hidden states.
        """

        # Mix the attention heads
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Residual connection + Layer Normalization
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MultiHeadedLatentAttentionBertSelfOutput(BertSelfOutput):
    """
    Implement the compress-decompress trick on the output projections.

    Args:
        config (PretrainedConfig): The model configuration containing
            hidden_size, layer_norm_eps, and hidden_dropout_prob.
    """
    def __init__(self, config):
        super().__init__(config)

        # Define the output down/up projection and norm layer
        self.output_down_projection = nn.Linear(config.hidden_size, config.output_compression_dim)
        self.output_up_projection = nn.Linear(config.output_compression_dim, config.hidden_size)
        self.output_norm_layer = nn.LayerNorm(config.output_compression_dim)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): The output from the attention layer (batch_size, seq_len, hidden_size).
            input_tensor (torch.Tensor): The original input to the attention layer (the "shortcut" connection).

        Returns:
            torch.Tensor: The normalized and combined hidden states.
        """

        # Project down, apply norm layer, project up
        output_latent_vector = self.output_down_projection(hidden_states)
        output_hidden_states = self.output_norm_layer(output_latent_vector)
        output_hidden_states = self.output_up_projection(output_hidden_states)

        # Mix the attention heads + Residual connection + Layer Normalization
        output_hidden_states = self.dense(output_hidden_states)
        output_hidden_states = self.dropout(output_hidden_states)
        output_hidden_states = self.LayerNorm(output_hidden_states + input_tensor)
        return output_hidden_states


class BertCrossAttention(BertSelfAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.FloatTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        past_key_values: EncoderDecoderCache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        # determine input shapes
        input_shape = hidden_states.shape[:-1]

        hidden_shape = (*input_shape, -1, self.attention_head_size)

        # get query proj
        query_layer = self.query(hidden_states).view(hidden_shape).transpose(1, 2)

        is_updated = past_key_values.is_updated.get(self.layer_idx) if past_key_values is not None else False
        if past_key_values is not None and is_updated:
            # reuse k,v, cross_attentions
            key_layer = past_key_values.cross_attention_cache.layers[self.layer_idx].keys
            value_layer = past_key_values.cross_attention_cache.layers[self.layer_idx].values
        else:
            kv_shape = (*encoder_hidden_states.shape[:-1], -1, self.attention_head_size)
            key_layer = self.key(encoder_hidden_states).view(kv_shape).transpose(1, 2)
            value_layer = self.value(encoder_hidden_states).view(kv_shape).transpose(1, 2)

            if past_key_values is not None:
                # save all states to the cache
                key_layer, value_layer = past_key_values.cross_attention_cache.update(
                    key_layer, value_layer, self.layer_idx
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                past_key_values.is_updated[self.layer_idx] = True

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout.p,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return attn_output, attn_weights

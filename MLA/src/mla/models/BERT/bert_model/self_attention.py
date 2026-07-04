import logging
import math

import torch
from torch import nn
from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.utils.deprecation import deprecate_kwarg

logger = logging.getLogger(__name__)


class BertSelfAttention(nn.Module):
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

    def __init__(self, config, position_embedding_type=None, layer_idx=None):
        super().__init__()

        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.position_embedding_type = position_embedding_type or getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder
        self.layer_idx = layer_idx

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool | None = False,
        cache_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        """
        Calculates the attention context layer and attention probabilities.

        Args:
            hidden_states (torch.Tensor): Input sequence of shape (batch_size, sequence_length, hidden_size).
            attention_mask (torch.FloatTensor, optional): Mask to avoid performing attention on padding tokens.
                Shape (batch_size, 1, 1, seq_length) or (batch_size, 1, seq_length, seq_length). Defaults to None.
            head_mask (torch.FloatTensor, optional): Mask to nullify specific attention heads. Shape (num_heads,)
                or (num_layers, num_heads). Defaults to None.
            encoder_hidden_states (torch.FloatTensor, optional): Sequence of contextual states from an encoder, used
                for cross-attention. Defaults to None.
            past_key_values (Cache, optional): Cached Key and Value states for accelerating sequential decoding.
                Defaults to None.
            output_attentions (bool, optional): Whether to return the attention probabilities matrix. Defaults to False.
            cache_position (torch.Tensor, optional): Indices indicating the position of the current tokens in the global
                sequence for caching purposes. Defaults to None.

        Returns:
            tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - context_layer: The output of the attention mechanism, shape
                  (batch_size, sequence_length, hidden_size).
                - attention_probs: The attention weights (if output_attentions=True),
                  shape (batch_size, num_heads, sequence_length, sequence_length).
        """
        batch_size, seq_length, _ = hidden_states.shape

        # Project and reshape Q from (batch_size, seq_len, n_heads, head_size) to (batch_size, n_heads, seq_len, head_size)
        query_layer = self.query(hidden_states)
        query_layer = query_layer.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        is_updated = False
        is_cross_attention = encoder_hidden_states is not None

        # Extract either the cross or self attention K/V cache
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):

                # Has the encoders data already been processed
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    curr_past_key_value = past_key_values.cross_attention_cache
                else:
                    curr_past_key_value = past_key_values.self_attention_cache
            else:
                curr_past_key_value = past_key_values

        current_states = encoder_hidden_states if is_cross_attention else hidden_states

        # If using cross attention, the encoders data has been processed and there is a KV cache
        if is_cross_attention and past_key_values is not None and is_updated:

            # Extract the K/Vs directly
            key_layer = curr_past_key_value.layers[self.layer_idx].keys
            value_layer = curr_past_key_value.layers[self.layer_idx].values

        else:

            # Project and reshape K,V from (batch_size, seq_len, n_heads, head_size) to (batch_size, n_heads, seq_len, head_size)
            key_layer = self.key(current_states)
            value_layer = self.value(current_states)
            key_layer = key_layer.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
            value_layer = value_layer.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

            # If there exists a KV cache
            if past_key_values is not None:
                cache_position = cache_position if not is_cross_attention else None

                # Save self attention KVs to cache to be re-used for fast auto-regressive generation
                key_layer, value_layer = curr_past_key_value.update(
                    key_layer, value_layer, self.layer_idx, {"cache_position": cache_position}
                )
                if is_cross_attention and isinstance(past_key_values, EncoderDecoderCache):
                    past_key_values.is_updated[self.layer_idx] = True

        # Scaled Dot-Product Attention: Q @ K^T -> (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # Adjust attention scores for relative distances
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if past_key_values is not None:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            # Shift the index to positive values only
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)

            # Add the relative position (and query) scores to the attention scores
            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        # Scale the attention scores
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Mask out irrelevant positions from the attention scores
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Apply softmax to the attention weights to convert to probabilities (each query asks which key I should pay attention to?)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to (from Transformer paper)
        attention_probs = self.dropout(attention_probs)

        # Mask out any heads we want to remove
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # Combine with the values -> (batch_size, num_heads, seq_len, head_size)
        context_layer = torch.matmul(attention_probs, value_layer)

        # Reshape the tensors for other layers (batch_size, seq_len, num_heads, head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)

        # Concatenate the the attention heads together -> (batch_size, seq_len, hidden_size)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, attention_probs


class BertSdpaSelfAttention(BertSelfAttention):
    """
    BERT self-attention implementation that utilizes PyTorch's Scaled Dot-Product Attention (SDPA).

    This class is an optimized subclass of `BertSelfAttention`. It leverages fused kernels
    (like FlashAttention and Memory-Efficient Attention) through `torch.nn.functional.scaled_dot_product_attention`.

    Note:
        This implementation automatically falls back to the manual `BertSelfAttention` if:
        1. `position_embedding_type` is not "absolute".
        2. `output_attentions` is set to True.
        3. A `head_mask` is provided.
    """
    def __init__(self, config, position_embedding_type=None, layer_idx=None):
        super().__init__(config, position_embedding_type=position_embedding_type, layer_idx=layer_idx)
        self.dropout_prob = config.attention_probs_dropout_prob

    # Adapted from BertSelfAttention
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool | None = False,
        cache_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        """
        Calculates the attention context layer using optimized SDPA kernels.

        Args:
            hidden_states (torch.Tensor): Input sequence of shape (batch_size, sequence_length, hidden_size).
            attention_mask (torch.FloatTensor, optional): Mask to avoid performing attention on padding tokens.
                Shape (batch_size, 1, 1, seq_length) or (batch_size, 1, seq_length, seq_length). Defaults to None.
            head_mask (torch.FloatTensor, optional): Mask to nullify specific attention heads. Shape (num_heads,)
                or (num_layers, num_heads). Defaults to None.
            encoder_hidden_states (torch.FloatTensor, optional): Sequence of contextual states from an encoder, used
                for cross-attention. Defaults to None.
            past_key_values (Cache, optional): Cached Key and Value states for accelerating sequential decoding.
                Defaults to None.
            output_attentions (bool, optional): Whether to return the attention probabilities matrix. Defaults to False.
            cache_position (torch.Tensor, optional): Indices indicating the position of the current tokens in the global
                sequence for caching purposes. Defaults to None.

        Returns:
            tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - context_layer: The output of the attention mechanism, shape
                  (batch_size, sequence_length, hidden_size).
                - None
        """
        # If not using absolute embeddings default to BertSelfAttention
        if self.position_embedding_type != "absolute" or output_attentions or head_mask is not None:
            # TODO: Improve this warning with e.g. `model.config._attn_implementation = "manual"` once implemented.
            logger.warning_once(
                "BertSdpaSelfAttention is used but `torch.nn.functional.scaled_dot_product_attention` does not support "
                "non-absolute `position_embedding_type` or `output_attentions=True` or `head_mask`. Falling back to "
                "the manual attention implementation, but specifying the manual implementation will be required from "
                "Transformers version v5.0.0 onwards. This warning can be removed using the argument "
                '`attn_implementation="eager"` when loading the model.'
            )

            # Default to BertSelfAttention
            return super().forward(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                past_key_values,
                output_attentions,
                cache_position,
            )

        batch_size, target_len, _ = hidden_states.size()

        # Project and reshape the Q vector
        query_layer = (
            self.query(hidden_states).view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        )

        # Define the current hidden states
        is_updated = False
        is_cross_attention = encoder_hidden_states is not None
        current_states = encoder_hidden_states if is_cross_attention else hidden_states

        # Extract the KV cache
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):

                # Has the encoders data already been processed
                is_updated = past_key_values.is_updated.get(self.layer_idx)

                # Extract either the cross attention cache or the self attention cache
                if is_cross_attention:
                    curr_past_key_value = past_key_values.cross_attention_cache
                else:
                    curr_past_key_value = past_key_values.self_attention_cache
            else:
                curr_past_key_value = past_key_values

        # Is this needed -> Redundant code
        current_states = encoder_hidden_states if is_cross_attention else hidden_states

        if is_cross_attention and past_key_values is not None and is_updated:
            # Directly extract KV vectors
            key_layer = curr_past_key_value.layers[self.layer_idx].keys
            value_layer = curr_past_key_value.layers[self.layer_idx].values

        else:

            # Project and reshape K,V from (batch_size, seq_len, n_heads, head_size) to (batch_size, n_heads, seq_len, head_size)
            key_layer = (
                self.key(current_states)
                .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
                .transpose(1, 2)
            )
            value_layer = (
                self.value(current_states)
                .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
                .transpose(1, 2)
            )

            # If there exists a KV cache
            if past_key_values is not None:

                # Save all key/value to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None

                # Update the keys and values with the current values
                key_layer, value_layer = curr_past_key_value.update(
                    key_layer, value_layer, self.layer_idx, {"cache_position": cache_position}
                )
                # Set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention and isinstance(past_key_values, EncoderDecoderCache):
                    past_key_values.is_updated[self.layer_idx] = True

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional
        # assignment in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic
        # shapes from compiling. The target_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create
        # a causal mask in case target_len == 1.
        is_causal = self.is_decoder and not is_cross_attention and attention_mask is None and target_len > 1

        # Use the SDPA scaled dot product attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=is_causal,
        )

        # Reshape to concatenate the the attention heads together -> (batch_size, seq_len, hidden_size)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, target_len, self.all_head_size)

        return attn_output, None


class MultiHeadedLatentAttention(nn.Module):
    """
    Class that implements Multi-headed Latent Attention was defined in DeepSeeks V2 paper.
    More specifically, the keys and values are projected into a much smaller latent space before
    being re-projected back up to the hidden state space. The idea being that when using decoder
    models that it will be much cheaper to cache the kv shared latent vector given that it is much
    smaller in size compared to standard MHA key and value attention vectors.

    There is mention of how to use it in inference which I am not sure about. Will have to return to that!
    """

    def __init__(self, config, position_embedding_type=None, layer_idx=None):
        super().__init__()

        # Define the dimensions of the down and up projections
        self.kv_compression_dim = config.kv_compression_dim
        self.q_compression_dim = config.q_compression_dim

        # Define the number and size of the attention heads
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_dim = int(config.hidden_size / config.num_attention_heads)

        # Define the layers associated with low rank joint compression of keys and values
        self.kv_down_projection = nn.Linear(self.hidden_size, self.kv_compression_dim)
        self.k_up_projection = nn.Linear(self.kv_compression_dim, self.hidden_size)
        self.v_up_projection = nn.Linear(self.kv_compression_dim, self.hidden_size)

        # Define the layers associated with low rank compression of queries
        self.q_down_projection = nn.Linear(self.hidden_size, self.q_compression_dim)
        self.q_up_projection = nn.Linear(self.q_compression_dim, self.hidden_size)

        # Define the attention dropout layer
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.dropout = nn.Dropout(self.attention_probs_dropout_prob)

        # Define the norm layer
        self.kv_layer_norm = nn.LayerNorm(self.kv_compression_dim)
        self.q_layer_norm = nn.LayerNorm(self.q_compression_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool | None = False,
        cache_position: torch.Tensor | None = None,
        use_cache=False,
        do_train=True,
    ) -> tuple[torch.Tensor]:
        """
        Project the keys/values and queries down to their respective latent spaces before projecting them back up to the hidden state
        space. Once the keys/values and queries are returned to the hidden state space apply attention as normal.

        Additional code has been added to ensure that the compressed latent kv vector is cached.
        """

        # Define the batch size and sequence length -> (batch_size, seq_len, hidden_size)
        batch_size, seq_length, _ = hidden_states.shape

        # Down project to the KV latent space -> (batch_size, seq_len, q/kv_compression_dim)
        kv_compressed_latent_vector = self.kv_down_projection(hidden_states)
        q_compressed_latent_vector = self.q_down_projection(hidden_states)

        # For stability pass the KV latent vector through a layer norm -> (batch_size, seq_len, q/kv_compression_dim)
        kv_compressed_latent_vector_norm = self.kv_layer_norm(kv_compressed_latent_vector)
        q_compressed_latent_vector_norm = self.q_layer_norm(q_compressed_latent_vector)

        # Up project back to the hidden space -> (batch_size, seq_len, hidden_size)
        key = self.k_up_projection(kv_compressed_latent_vector_norm)
        value = self.v_up_projection(kv_compressed_latent_vector_norm)
        query = self.q_up_projection(q_compressed_latent_vector_norm)

        # Prepare Q, K and V for attention -> (batch_size, seq_len, n_heads, head_dim) -> (batch_size, n_heads, seq_len, head_dim)
        query = query.view(batch_size, -1, self.num_attention_heads, self.attention_head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_attention_heads, self.attention_head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_attention_heads, self.attention_head_dim).transpose(1, 2)

        # Dot product between Q and K -> (batch_size, n_heads, seq_len, seq_len)
        attention_scores = query @ key.transpose(-1, -2)

        # attention_scores/sqrt(head_size) -> (batch_size, n_heads, seq_len, seq_len)
        attention_scores = attention_scores / math.sqrt(self.attention_head_dim)

        # Apply the attention mask
        if attention_mask is not None:
            # BERT mask is usually (batch_size, 1, 1, seq_length) or (batch_size, seq_length)
            # It needs to be added to the scores before softmax
            attention_scores = attention_scores + attention_mask

        # Apply softmax to the attention weights to convert to probabilities
        # We apply the softmax over the set of keys associated with each query
        # In particular, each query is asking which key should I pay attention to?
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Multiply the attention weights by the values
        # This produces a single vector for each query token, a combination of all values weighted by attention probabilities
        # So the values which correspond to the most likely keys are selected -> (batch_size, n_heads, seq_len, head_dim)
        context_layer = attention_probs @ value

        # Reorder the dims and create a new tenor (with values stored sequentually in memory) -> (batch_size, seq_len, n_heads, head_dim)
        # Reshape to concatenate the the attention heads together -> (batch_size, seq_len, hidden_size)
        context_layer = context_layer.transpose(1, 2)

        # Convert the attention outputs back to heads grouped by tokens
        # (batch_size, query_len, hidden_size)
        context_layer = context_layer.reshape(batch_size, seq_length, self.hidden_size)

        return context_layer, attention_probs


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


class MultiHeadedLatentAttentionBertSelfOutput(nn.Module):
    """
    Implement the compress-decompress trick on the output projections.
    """
    def __init__(self, config):
        super().__init__()

        # Define the type of output down projection we are implementing
        self.output_compression_dim = config.output_compression_dim

        # Define the number of heads, head dim
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_dim = int(self.hidden_size / self.num_attention_heads)

        # Define the output down projection
        self.output_down_projection = nn.Linear(config.hidden_size, self.output_compression_dim)
        self.output_up_projection = nn.Linear(self.output_compression_dim, config.hidden_size)

        # Define the output layer norm
        self.output_norm_layer = nn.LayerNorm(self.output_compression_dim)

        # Define the norm and dropout layers
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Project the output projections down to the output projection hidden space before projecting it back up to hidden state space.
        """

        # Project the hidden states to smaller latent space
        output_latent_vector = self.output_down_projection(hidden_states)

        # Add the initial input in the form of a residual connection prior to performing normalisation
        output_hidden_states = self.output_norm_layer(output_latent_vector)

        # Project the output latent vector back to the hidden state space
        output_hidden_states = self.output_up_projection(output_hidden_states)

        # Apply dropout
        output_hidden_states = self.dropout(output_hidden_states)

        # Add the initial input in the form of a residual connection prior to performing normalisation
        output_hidden_states = self.LayerNorm(output_hidden_states + input_tensor)

        return output_hidden_states


class BertAttention(nn.Module):
    """
    The BERT attention layer, encompassing both the self-attention mechanism
    and the subsequent output projection with a residual connection.

    This module manages the flow of data through the attention heads and
    handles head pruning, which allows for the removal of less important
    attention heads to optimize model size and speed.
    """
    def __init__(self, config, position_embedding_type=None, layer_idx=None):
        super().__init__()

        self.attention_mechanism = {
            "MHA": BertSdpaSelfAttention,
            "MLA": MultiHeadedLatentAttention,
            "MLAE": MultiHeadedLatentAttention,
        }
        self.attention_mechanism_output = {
            "MHA": BertSelfOutput,
            "MLA": BertSelfOutput,
            "MLAE": MultiHeadedLatentAttentionBertSelfOutput,
        }

        self.self = self.attention_mechanism[config.attention_mechanism](config, position_embedding_type=position_embedding_type, layer_idx=layer_idx)
        self.output = self.attention_mechanism_output[config.attention_mechanism](config)
        self.pruned_heads = set()

    # def prune_heads(self, heads):
    #     """
    #     Prunes specified attention heads from the model.

    #     This method permanently removes the weights associated with the given
    #     head indices from the Query, Key, Value, and Output linear layers.

    #     Args:
    #         heads (list[int]): List of attention head indices to be pruned.
    #     """
    #     if len(heads) == 0:
    #         return

    #     # Identify the heads and their associated index values to be pruned
    #     heads, index = find_pruneable_heads_and_indices(
    #         heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
    #     )

    #     # Physically shrink the linear layers by removing the identified indices
    #     self.self.query = prune_linear_layer(self.self.query, index)
    #     self.self.key = prune_linear_layer(self.self.key, index)
    #     self.self.value = prune_linear_layer(self.self.value, index)
    #     self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

    #     # Update the metadata so the model knows it has fewer heads now
    #     self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
    #     self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
    #     self.pruned_heads = self.pruned_heads.union(heads)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool | None = False,
        cache_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        """
        Processes the hidden states through the entire attention block.

        Args:
            hidden_states (torch.Tensor): Input hidden states of shape
                (batch_size, sequence_length, hidden_size).
            attention_mask (torch.FloatTensor, optional): Mask to avoid attention
                on padding tokens.
            head_mask (torch.FloatTensor, optional): Mask to nullify specific heads.
            encoder_hidden_states (torch.FloatTensor, optional): States for
                cross-attention if applicable.
            past_key_values (Cache, optional): KV cache for fast decoding.
            output_attentions (bool, optional): Whether to return attention probabilities.
            cache_position (torch.Tensor, optional): Positional indices for caching.

        Returns:
            tuple[torch.Tensor, ...]: A tuple where the first element is the
                attention output. If `output_attentions` is True, the second
                element contains the attention probabilities.
        """

        # Apply the attention mechanism to return the attention scores
        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )

        # Apply the Dense projection, Dropout, Residual connection, and LayerNorm
        attention_output = self.output(self_outputs[0], hidden_states)

        # Concatenate any extra outputs (like attention weights or cache)
        outputs = (attention_output,) + self_outputs[1:]

        return outputs

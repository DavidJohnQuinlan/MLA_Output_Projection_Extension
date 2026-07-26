from collections.abc import Callable
from typing import Unpack

import torch
from torch import nn
from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.utils import TransformersKwargs

from mla.models.BERT.attention_mechanisms import ALL_ATTENTION_FUNCTIONS, BertSelfAttention, BertSelfOutput, MultiHeadedLatentAttention, MultiHeadedLatentAttentionBertSelfOutput, eager_attention_forward

_ATTENTION_MECHANISM = {
    "MHA": BertSelfAttention,
    "MHAE": BertSelfAttention,
    "MLA": MultiHeadedLatentAttention,
    "MLAE": MultiHeadedLatentAttention,
}

_ATTENTION_MECHANISM_OUTPUT = {
    "MHA": BertSelfOutput,
    "MHAE": MultiHeadedLatentAttentionBertSelfOutput,
    "MLA": BertSelfOutput,
    "MLAE": MultiHeadedLatentAttentionBertSelfOutput,
}

class BertAttention(nn.Module):
    """
    The BERT attention layer, encompassing both the self-attention mechanism
    and the subsequent output projection with a residual connection.

    This module manages the flow of data through the attention heads and
    handles head pruning, which allows for the removal of less important
    attention heads to optimize model size and speed.
    """
    def __init__(self, config, is_causal=False, layer_idx=None, is_cross_attention=False):
        super().__init__()

        self.is_cross_attention = is_cross_attention
        attention_class = BertCrossAttention if is_cross_attention else _ATTENTION_MECHANISM[config.attention_mechanism]
        self.self = attention_class(config, is_causal=is_causal, layer_idx=layer_idx)
        self.output = _ATTENTION_MECHANISM_OUTPUT[config.attention_mechanism](config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
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

        attention_mask = attention_mask if not self.is_cross_attention else encoder_attention_mask
        attention_output, attn_weights = self.self(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        attention_output = self.output(attention_output, hidden_states)
        return attention_output, attn_weights

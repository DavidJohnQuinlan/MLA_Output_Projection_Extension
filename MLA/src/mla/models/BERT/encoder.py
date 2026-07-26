import logging
from typing import Unpack

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.utils import TransformersKwargs

from mla.models.BERT.dataclasses import BaseModelOutputWithPastAndCrossAttentions
from mla.models.BERT.layer import BertLayer

logger = logging.getLogger(__name__)


class BertEncoder(nn.Module):
    """
    The core encoder stack for BERT, responsible for sequentially processing hidden states
    through a series of identical transformer layers (`BertLayer`).

    This module handles the horizontal orchestration of the transformer block list, manages
    KV caching across generation passes for decoder variants, collects hidden states or
    attention maps for diagnostic outputs, and safely handles gradient checkpointing configuration.

    Attributes:
        config (PretrainedConfig): The configuration object containing model architecture parameters.
        layer (nn.ModuleList): A list containing the individual `BertLayer` blocks.
        gradient_checkpointing (bool): Flag stating whether to use memory-saving checkpointing during the backward pass.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPastAndCrossAttentions:
        """
        Passes input hidden states sequentially through all layers of the encoder stack.

        Args:
            hidden_states (torch.Tensor): Input embeddings/hidden features of shape (batch_size, sequence_length, hidden_size).
            attention_mask (torch.FloatTensor, optional): Mask preventing attention over padding tokens.
                Shape (batch_size, 1, 1, to_seq_length).
            head_mask (torch.FloatTensor, optional): Mask to nullify specific attention heads.
            encoder_hidden_states (torch.FloatTensor, optional): Cross-attention sequence from an external encoder (used if
                configured as a decoder).
            encoder_attention_mask (torch.FloatTensor, optional): Mask for cross-attention sequences.
            past_key_values (Cache, optional): Pre-computed key/value states used to accelerate sequential inference generation loops.
            use_cache (bool, optional): Whether to retain past K/V tensors for text generation.
                Automatically overridden to `False` if training with gradient checkpointing.

        Returns:
            Union[tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
                A subclass container or structured tuple containing the last layer output,
                dynamic context cache metadata, and optional full-stack hidden tensors/attention profiles.
        """
        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

import logging

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache

from mla.models.BERT.bert_model.bert_dataclasses import BaseModelOutputWithPastAndCrossAttentions
from mla.models.BERT.bert_model.bert_layer import BertLayer

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
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
        cache_position: torch.Tensor | None = None,
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
            output_attentions (bool, optional): Whether to return attention probability maps.
                Defaults to `False`. Note: Elements will be `None` if an SDPA backend is active.
            output_hidden_states (bool, optional): Whether to return hidden states from all layers. Defaults to `False`.
            return_dict (bool, optional): Whether to package results in a
                `BaseModelOutputWithPastAndCrossAttentions` structure or a raw tuple. Defaults to `True`.
            cache_position (torch.Tensor, optional): Index tensors tracking the relative structural
                position of current tokens within the cache object.

        Returns:
            Union[tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
                A subclass container or structured tuple containing the last layer output,
                dynamic context cache metadata, and optional full-stack hidden tensors/attention profiles.
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # If training do not use a cache (if used with gradient checkpointing then the KV cache would grow during the forward pass re-runs)
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        # If performing inference (use_cache=True) and no cache already exists
        if use_cache and self.config.is_decoder and past_key_values is None:

            # Create a cache for self and cross attention K/Vs
            past_key_values = EncoderDecoderCache(DynamicCache(config=self.config), DynamicCache(config=self.config))

        # If performing inference (use_cache=True) and past_key_values is a tuple
        if use_cache and self.config.is_decoder and isinstance(past_key_values, tuple):
            logger.warning_once(
                "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.58.0. "
                "You should pass an instance of `EncoderDecoderCache` instead, e.g. "
                "`past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`."
            )

            # Convert tuple to cache instance
            past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:

                # Add hidden states to the tuple
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Identify the relevant layers head mask
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # Apply the full transformer layer
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )

            # Extract the layers output hidden states
            hidden_states = layer_outputs[0]

            if output_attentions:
                # Add the layers attentions (self and/or cross) raw probs to the all self attentions tuple (if using SDPA -> none)
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # Add the final hidden state to the tuple (from the last layer)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Format the output
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

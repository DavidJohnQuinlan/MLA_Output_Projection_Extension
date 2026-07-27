
from typing import Unpack

import torch
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.masking_utils import create_bidirectional_mask, create_causal_mask
from transformers.utils import TransformersKwargs

from mla.models.BERT.dataclasses import BaseModelOutputWithPoolingAndCrossAttentions
from mla.models.BERT.embeddings import BertEmbeddings
from mla.models.BERT.encoder import BertEncoder
from mla.models.BERT.pooler import BertPooler
from mla.models.BERT.pretrained_model import BertPreTrainedModel


class BertModel(BertPreTrainedModel):
    """
    The core, top-level orchestrator module for the BERT architecture.

    This class inherits from `BertPreTrainedModel` and orchestrates the complete
    forward execution graph: transforming raw token IDs into multi-layered dense embeddings,
    handling modern fast-attention (SDPA) mask preprocessing, piping spatial matrices through
    the encoder stack (`BertEncoder`), and optionally projecting the sequence via a structural
    pooling mechanism (`BertPooler`).

    Attributes:
        config (PretrainedConfig): The configuration object containing architectural hyper-parameters.
        embeddings (BertEmbeddings): Module managing token, position, and segment embedding lookups.
        encoder (BertEncoder): The main transformer layer loop stack.
        pooler (BertPooler, optional): Segment pooling module to resolve sentence-level representations.
        attn_implementation (str): Named string flag tracking the mathematical backend engine (e.g., `"sdpa"`, `"eager"`).
        position_embedding_type (str): Explicitly logs the embedding type (e.g., `"absolute"`).
    """
    _no_split_modules = ["BertEmbeddings", "BertLayer"]

    def __init__(self, config, add_pooling_layer=True):
        """
        Initializes the top-level structural layout of the BertModel.

        Args:
            config (PretrainedConfig): The foundational architecture configuration profile.
            add_pooling_layer (bool, optional): Determines whether to attach a `BertPooler`
                on top of the sequence outputs. Defaults to `True`.
        """
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieves the underlying primary token vocabulary word embedding weights.

        Returns:
            nn.Embedding: The primitive vocabulary lookup table layer module.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        Overwrites or hot-swaps the core token vocabulary embeddings with a custom configuration.

        Args:
            value (nn.Embedding): The replacement embedding layer to bind to the model framework.
        """
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPoolingAndCrossAttentions:
        """
        Executes the main forward pass logic across embeddings, encoder layers, and pooling heads.

        Args:
            input_ids (torch.Tensor, optional): Tensor indices of input sequence tokens in the vocabulary.
                Shape: `(batch_size, sequence_length)`.
            attention_mask (torch.Tensor, optional): Mask avoiding computations over padding tokens.
                Shape: `(batch_size, sequence_length)`.
            token_type_ids (torch.Tensor, optional): Segment token indices identifying structural pairs.
                Shape: `(batch_size, sequence_length)`.
            position_ids (torch.Tensor, optional): Explicit spatial index coordinates mapping tokens to positions.
                Shape: `(batch_size, sequence_length)`.
            inputs_embeds (torch.Tensor, optional): Directly provided embedded features instead of input token IDs.
                Shape: `(batch_size, sequence_length, hidden_size)`.
            encoder_hidden_states (torch.Tensor, optional): Continuous sequence vectors from a separate encoder
                serving cross-attention nodes. Shape: `(batch_size, encoder_sequence_length, hidden_size)`.
            encoder_attention_mask (torch.Tensor, optional): Specialized cross-attention pad mask boundaries.
            past_key_values (Cache, optional): Multi-pass evaluation state context history tracker.
            use_cache (bool, optional): Controls whether KV states are cached. Hard-set to `False` if not a decoder.

        Raises:
            ValueError: If both `input_ids` and `inputs_embeds` are simultaneously declared, or if neither is set.

        Returns:
            Union[tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
                A custom dataclass summary container or a standard python tuple presenting the final structural
                sequence states, pooled sentence projections, historical caches, and optional multi-layer analytics arrays.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = (
                EncoderDecoderCache(DynamicCache(config=self.config), DynamicCache(config=self.config))
                if encoder_hidden_states is not None or self.config.is_encoder_decoder
                else DynamicCache(config=self.config)
            )

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        attention_mask, encoder_attention_mask = self._create_attention_masks(
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            embedding_output=embedding_output,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_ids=position_ids,
            **kwargs,
        )
        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
        )

    def _create_attention_masks(
        self,
        attention_mask,
        encoder_attention_mask,
        embedding_output,
        encoder_hidden_states,
        past_key_values,
    ):
        if self.config.is_decoder:
            attention_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=embedding_output,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
        else:
            attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=embedding_output,
                attention_mask=attention_mask,
            )

        if encoder_attention_mask is not None:
            encoder_attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=embedding_output,
                attention_mask=encoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )

        return attention_mask, encoder_attention_mask

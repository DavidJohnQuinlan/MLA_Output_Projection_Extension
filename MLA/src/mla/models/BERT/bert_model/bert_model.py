
import torch
from transformers.cache_utils import Cache
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask_for_sdpa

from mla.models.BERT.bert_model.bert_dataclasses import BaseModelOutputWithPoolingAndCrossAttentions
from mla.models.BERT.bert_model.bert_embeddings import BertEmbeddings
from mla.models.BERT.bert_model.bert_encoder import BertEncoder
from mla.models.BERT.bert_model.bert_pooler import BertPooler
from mla.models.BERT.bert_model.bert_pretrained_model import BertPreTrainedModel


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

        # Define the main layers of the Bert model -> embedding, encoder layer (attention) * 12, pooler
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Define the attention and position embedding methods
        self.attn_implementation = config._attn_implementation
        self.position_embedding_type = config.position_embedding_type

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

    def _prune_heads(self, heads_to_prune):
        """
        Prunes specific multi-head attention units across specified encoder blocks.

        Args:
            heads_to_prune (Dict[int, List[int]]): A mapping dictionary where keys correspond
                to targeted 0-indexed layer numbers and values represent list indices of individual heads to excise.
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs
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
            head_mask (torch.Tensor, optional): Custom structural mask to zero out chosen attention heads.
            inputs_embeds (torch.Tensor, optional): Directly provided embedded features instead of input token IDs.
                Shape: `(batch_size, sequence_length, hidden_size)`.
            encoder_hidden_states (torch.Tensor, optional): Continuous sequence vectors from a separate encoder
                serving cross-attention nodes. Shape: `(batch_size, encoder_sequence_length, hidden_size)`.
            encoder_attention_mask (torch.Tensor, optional): Specialized cross-attention pad mask boundaries.
            past_key_values (Cache, optional): Multi-pass evaluation state context history tracker.
            use_cache (bool, optional): Controls whether KV states are cached. Hard-set to `False` if not a decoder.
            output_attentions (bool, optional): Returns full attention map layers. Overridden by SDPA configurations.
            output_hidden_states (bool, optional): Aggregates hidden output tensors from all layer blocks.
            return_dict (bool, optional): Packages output data into a structured object class or a tuple.
            cache_position (torch.Tensor, optional): Tracks active step offsets inside sequential generation loops.

        Raises:
            ValueError: If both `input_ids` and `inputs_embeds` are simultaneously declared, or if neither is set.

        Returns:
            Union[tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
                A custom dataclass summary container or a standard python tuple presenting the final structural
                sequence states, pooled sentence projections, historical caches, and optional multi-layer analytics arrays.
        """
        # Prepare training/inference configuration (what data to store/output, KV cache)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        # Define the input shape, batch size and sequence length
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # Calculate the KV cache length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = (
                past_key_values[0][0].shape[-2]
                if not isinstance(past_key_values, Cache)
                else past_key_values.get_seq_length()
            )

        # Define the token type ids
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # Generate the embeddings for a set of input/position/token type ids
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        # Define a dummy attention mask if none was set (adding additional space for the K/V cache length)
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)

        # Define wheather to use SDPA attention masks
        use_sdpa_attention_masks = (
            self.attn_implementation == "sdpa"
            and self.position_embedding_type == "absolute"
            and head_mask is None
            and not output_attentions
        )

        # Prepare the attention mask
        if use_sdpa_attention_masks and attention_mask.dim() == 2:

            # Expand the attention mask for SDPA
            # [batch_size, sequence_length] -> [batch_size, 1, sequence_length, sequence_length]
            if self.config.is_decoder:

                # Prepare the masking to ensure that any padding or future tokens are ignored
                # Creates a mask of 0s for those to be included and -inf for those to be ignored
                extended_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    input_shape,
                    embedding_output,
                    past_key_values_length,
                )
            else:

                # In the case that we are using the encoder transformer architecture then we only need to mask the padding tokens
                # with -inf masking for the padding tokens
                extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
        else:
            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

            # Generate the encoder attention mask (if none exists)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if use_sdpa_attention_masks and encoder_attention_mask.dim() == 2:

                # Expand the attention mask for SDPA
                # [batch_size, sequence_length] -> [batch_size, 1, sequence_length, sequence_length]
                encoder_extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare attention head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch_size x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # Pass the embeddings through the encoder layers
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        # Extract the last hidden state
        sequence_output = encoder_outputs[0]

        # Apply CLS pooling (if true)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

import inspect
from collections.abc import Callable

import torch
from transformers.cache_utils import Cache
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.utils.deprecation import deprecate_kwarg

from mla.models.BERT.bert_model.bert_ffn import BertIntermediate, BertOutput
from mla.models.BERT.bert_model.self_attention import BertAttention


class BertLayer(GradientCheckpointingLayer):
    """
    A single block of a BERT/Transformer layer.

    Depending on the configuration, this layer can act either as an encoder layer
    (self-attention followed by a feed-forward network) or a decoder layer
    (self-attention, cross-attention looking at an encoder's state, followed by a
    feed-forward network). It also supports feed-forward chunking along the sequence
    dimension to optimize peak memory footprint.
    """
    def __init__(self, config, layer_idx=None):
        """
        Initializes the BERT layer components.

        Args:
            config (BertConfig): The model configuration object containing architectural parameters.
            layer_idx (int, optional): The absolute index of this layer within the overall transformer block. Defaults to None.

        Raises:
            ValueError: If `config.add_cross_attention` is True but `config.is_decoder` is False.
        """
        super().__init__()

        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config, layer_idx=layer_idx)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention

        # Only decoders can use cross attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")

            # Define the cross attention layer as another bertattention instance
            self.crossattention = BertAttention(config, position_embedding_type="absolute", layer_idx=layer_idx)

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool | None = False,
        cache_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        """
        Executes the forward pass of the Transformer layer block.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape `(batch_size, sequence_length, hidden_size)`.
            attention_mask (torch.FloatTensor, optional): Mask to avoid performing attention on padding token indices.
                Shape `(batch_size, 1, 1, to_seq_length)`. Defaults to None.
            head_mask (torch.FloatTensor, optional): Mask to nullify selected heads of the self-attention modules.
                Defaults to None.
            encoder_hidden_states (torch.FloatTensor, optional): Sequence of hidden-states at the output of the last layer
                of the encoder. Used for cross-attention if the layer is a decoder. Defaults to None.
                Shape `(batch_size, encoder_sequence_length, hidden_size)`.
            encoder_attention_mask (torch.FloatTensor, optional): Mask to avoid performing cross-attention on padding tokens
                of the encoder input. Defaults to None.
            past_key_values (Cache, optional): Cached hidden-states (keys and values in the attention blocks) to speed up
                sequential decoding. Defaults to None.
            output_attentions (bool, optional): Whether or not to return the attention weights of all attention layers.
                Defaults to False.
            cache_position (torch.Tensor, optional): Indices indicating the positions of the incoming tokens in the sequence cache.
                Defaults to None.

        Returns:
            tuple[torch.Tensor]: A tuple containing:
                - **layer_output** (torch.Tensor): The final processed hidden states of shape (batch_size, sequence_length, hidden_size).
                - **self_attn_weights** (torch.Tensor, optional): Self-attention weights, returned if `output_attentions=True`.
                - **cross_attn_weights** (torch.Tensor, optional): Cross-attention weights, returned if `output_attentions=True` and
                    the model acts as a decoder.
        Raises:
            ValueError:
                If `encoder_hidden_states` are passed but the layer was not configured with cross-attention capabilities.
        """

        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )

        # Extract the attention layer outputs (attention + hidden states residual connection)
        attention_output = self_attention_outputs[0]

        # Extract the raw self attention outputs
        outputs = self_attention_outputs[1:]

        # If using encoder/decoder model
        if self.is_decoder and encoder_hidden_states is not None:

            # Ensure that crossattention layer exists
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask=encoder_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )

            # Extract the cross attention output (attention + hidden states residual connection)
            attention_output = cross_attention_outputs[0]

            # Combine raw self attention outputs with raw cross attention outputs
            outputs = outputs + cross_attention_outputs[1:]

        # Take the attention output (hidden states + attention heads) and apply FFN to each chunk
        # In this case we are appling the feed forward chunk function along the sequence length dimension
        # This can be done because the FFN only applies to each token independently
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        # Add chunked processed outputs and raw self attention outputs to a tuple
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        """
        Processes a chunk of attention outputs through the intermediate (FFN)
        and output projection layers.

        Args:
            attention_output (torch.Tensor): A slice or full tensor of attention outputs representing the hidden states.

        Returns:
            torch.Tensor: The residual-connected and normalized final hidden states for this chunk.
        """
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


def apply_chunking_to_forward(
    forward_fn: Callable[..., torch.Tensor],
    chunk_size: int,
    chunk_dim: int,
    *input_tensors,
) -> torch.Tensor:
    """
    This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the dimension
    `chunk_dim`. It then applies a layer `forward_fn` to each chunk independently to save memory.

    If the `forward_fn` is independent across the `chunk_dim` this function will yield the same result as directly
    applying `forward_fn` to `input_tensors`.

    Args:
        forward_fn (`Callable[..., torch.Tensor]`):
            The forward function of the model.
        chunk_size (`int`):
            The chunk size of a chunked tensor: `num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_dim (`int`):
            The dimension over which the `input_tensors` should be chunked.
        input_tensors (`tuple[torch.Tensor]`):
            A tuple of input tensors of `forward_fn` which will be chunked. Input tensor can be many tensors i.e. hidden_states and
            attention_mask.

    Returns:
        `torch.Tensor`: A tensor with the same shape as the `forward_fn` would have given if applied`.

    Examples:

    ```python
    # rename the usual forward() fn to forward_chunk()
    def forward_chunk(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states

    # implement a chunked forward function
    def forward(self, hidden_states):
        return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
    ```"""
    assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"

    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
    # Count the number of arguments that exist in the forward fn
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)

    # Verify that the number of input tensors matches what forward_fn expects
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(
            f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input tensors are given"
        )
    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for input_tensor in input_tensors:

            if input_tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError(
                    f"All input tenors have to be of the same shape: {tensor_shape}, found shape {input_tensor.shape[chunk_dim]}"
                )

        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk size {chunk_size}"
            )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # Slice each input tensor (e.g., hidden_states, attention_mask) into chunks
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)

        # Zip chunks together so corresponding pieces (e.g., hidden_states[0], mask[0])
        # are processed sequentially through the forward pass
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks, strict=True))

        return torch.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors)

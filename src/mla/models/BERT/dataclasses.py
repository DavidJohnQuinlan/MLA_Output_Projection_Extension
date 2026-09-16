from dataclasses import dataclass

import torch
from transformers.cache_utils import Cache
from transformers.utils import ModelOutput


@dataclass
class BaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (batch_size, sequence_length, hidden_size):
            Sequence of hidden-states at the output of the last layer of the model.

        pooler_output (batch_size, hidden_size):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.

        hidden_states (returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer,
            + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

        attentions (returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

        cross_attentions (returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when
            `config.output_attentions=True`): Tuple of `torch.FloatTensor` (one for each layer) of shape (batch_size, num_heads,
            sequence_length, sequence_length). Attentions weights of the decoder's cross-attention layer, after the attention
            softmax, used to compute the weighted average in the cross-attention heads.

        past_key_values (returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). Contains pre-computed hidden-states
            (key and values in the self-attention blocks and optionally if `config.is_encoder_decoder=True` in the
            cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    """
    last_hidden_state: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    past_key_values: Cache | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    cross_attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class BaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (batch_size, sequence_length, hidden_size):
            Sequence of hidden-states at the output of the last layer of the model.
            If `past_key_values` is used only the last hidden-state of the sequences of shape (batch_size, 1, hidden_size) is output.

        past_key_values (returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). Contains pre-computed hidden-states
            (key and values in the self-attention blocks and optionally if `config.is_encoder_decoder=True` in the cross-attention blocks)
            that can be used (see `past_key_values` input) to speed up sequential decoding.

        hidden_states (returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the
            output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of
            each layer plus the optional initial embedding outputs.

        attentions (returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

        cross_attentions (returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when
            `config.output_attentions=True`): Tuple of `torch.FloatTensor` (one for each layer) of shape
            (batch_size, num_heads, sequence_length, sequence_length). Attentions weights of the decoder's cross-attention layer, after
            the attention softmax, used to compute the weighted average in the cross-attention heads.
    """
    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    cross_attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class MaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when
            `config.output_hidden_states=True`): Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has
            an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when
            `config.output_attentions=True`): Tuple of `torch.FloatTensor` (one for each layer) of shape
            `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute
            the weighted average in the self-attention heads.
    """
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class BertForPreTrainingOutput(ModelOutput):
    r"""
    loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
        Total loss as the sum of the masked language modeling loss and the next sequence prediction
        (classification) loss.
    prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
        Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
        before SoftMax).
    """

    loss: torch.FloatTensor | None = None
    prediction_logits: torch.FloatTensor | None = None
    seq_relationship_logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput

from mla.models.BERT.bert_model.bert_dataclasses import MaskedLMOutput
from mla.models.BERT.bert_model.bert_model import BertModel
from mla.models.BERT.bert_model.bert_pretrained_model import BertPreTrainedModel


class BertModelForMLM(BertPreTrainedModel):
    """
    BERT model with a language modeling head on top for Masked Language Modeling (MLM).

    The base `BertModel` acts strictly as a feature extractor, outputting continuous multi-dimensional
    vectors. This class attaches a linear prediction head (`lm_head`) to map those feature vectors
    back into the vocabulary space, calculating the prediction probability for every token position.
    It automatically handles cross-entropy loss extraction if target labels are supplied.

    Attributes:
        bert (BertModel): The structural backbone encoder model.
        lm_head (nn.Linear): The classification projection layer mapping from hidden size dimensions
            to the vocabulary length.
        bias (nn.Parameter): A standalone trainable tensor bound to the language modeling head's bias.
    """
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.lm_head.bias = self.bias
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        Executes the forward pass for masked language model training or prediction.

        Args:
            input_ids (torch.Tensor, optional): Indices of input sequence tokens in the vocabulary.
                Shape: `(batch_size, sequence_length)`.
            attention_mask (torch.Tensor, optional): Mask preventing attention over padding tokens.
                Shape: `(batch_size, sequence_length)`.
            labels (torch.Tensor, optional): Ground-truth target token indices for calculating
                the masked language modeling loss. Position values corresponding to unmasked tokens
                should be flagged with `-100` to be ignored. Shape: `(batch_size, sequence_length)`.

        Returns:
            MaskedLMOutput: A structured dataclass object containing the optional training loss value,
                unnormalized prediction logits across the entire vocabulary, and downstream tracking states.
        """
        # For a given set of input_ids/attention_masks return the bert last hidden state
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        # Convert the output to logits (prob for each word in the vocab)
        # Shape transition: [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, vocab_size]
        logits = self.lm_head(outputs.last_hidden_state)

        loss = None
        if labels is not None:

            # Calculate the cross entropy loss between the true labels and predicted logits
            # PyTorch's cross_entropy requires a 2D matrix of predictions (N, classes) and a 1D vector of targets (N).
            # We flatten both matrices using .view(-1) to create a single long stream of individual tokens.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BERTModelForClassification(nn.Module):
    """
    A sequence classification wrapper sitting on top of a base BERT encoder.

    Extracts the contextual representation of the pooler target ([CLS] token)
    from the final hidden state and projects it through a regularized dropout
    layer and linear classification head.
    """
    def __init__(self, bert_model: Callable, config):
        """
        Initializes the classification wrapper with an encoder and head layers.

        Args:
            bert_model (nn.Module): The underlying pre-trained or raw base BERT
                encoder model.
            config (Callable): Configuration dataclass containing
                structural properties (`hidden_size`, `dropout_rate`, `num_labels`).
        """
        super().__init__()
        self.bert = bert_model
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels

        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, config.num_labels)

        self.loss_fn = nn.CrossEntropyLoss()

        # Initialize weights for the new head
        self._init_weights(self.classifier)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Applies standard BERT normal distribution initialization to a module.

        Args:
            module (nn.Module): The structural sub-component layer (typically a
                nn.Linear layer) requiring weight calibration.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs
    ):
        """
        Executes a sequence classification forward pass over the batch.

        Args:
            input_ids (torch.Tensor): Tensor of shape (batch_size, seq_len)
                containing vocabulary token indices.
            attention_mask (torch.Tensor): Tensor of shape (batch_size, seq_len)
                containing attention masking bits (1 for data, 0 for padding).
            labels (torch.Tensor, optional): Ground truth training target indices
                for loss calculations. Defaults to None.
            **kwargs: Additional keyword arguments passed down directly to the
                underlying base BERT encoder.

        Returns:
            BERTClassifierOutput: A lightweight structural container holding:
                - loss (torch.Tensor or None): Calculated Cross-Entropy scalar if
                  labels are provided, otherwise None.
                - logits (torch.Tensor): Raw, unnormalized prediction scores of
                  shape (batch_size, num_labels).
        """

        # Forward pass through the Transformer layers
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        # Extract the [CLS] token representation -> [batch_size, seq_len, 768]
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Apply classification head
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)

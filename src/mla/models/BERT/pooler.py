import torch
from torch import nn


class BertPooler(nn.Module):
    """
    Pooler layer used to extract a sentence-level classification by returning a
    fixed-size representation of the entire input sequence.

    This module implements the standard BERT pooling strategy: it extracts the hidden
    state corresponding to the first token (the `[CLS]` token) (across a batch), passes
    it through a linear projection, and applies a Tanh activation function. This pooled
    output is typically used as the input for downstream classification heads.

    Attributes:
        dense (nn.Linear): The linear projection layer mapping from hidden size to hidden size.
        activation (nn.Tanh): The non-linear activation function applied to the projected token.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Pools the sequence of hidden states by isolating and projecting the first token.

        Args:
            hidden_states (torch.Tensor): The final hidden states from the encoder stack.
                Shape: `(batch_size, sequence_length, hidden_size)`.

        Returns:
            torch.Tensor: The sentence-level pooled representation.
                Shape: `(batch_size, hidden_size)`.
        """
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

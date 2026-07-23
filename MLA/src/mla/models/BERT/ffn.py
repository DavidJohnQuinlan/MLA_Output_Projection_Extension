import torch
from torch import nn
from transformers.activations import GELUActivation


class BertIntermediate(nn.Module):
    """
    The intermediate "expansion" layer of the Transformer's Feed-Forward Network (FFN).

    This module projects the hidden states into a higher-dimensional space
    (typically 4x the hidden size) and applies a non-linear activation function.
    This allows the model to learn complex feature representations for each token
    independently.

    Args:
        config (PretrainedConfig): The model configuration containing
            hidden_size, intermediate_size, and hidden_act.
    """
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = GELUActivation()
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Processes the hidden states through the expansion layer.

        Args:
            hidden_states (torch.Tensor): The output from the Attention layer
                of shape (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor: The expanded and activated hidden states of shape
                (batch_size, seq_len, intermediate_size).
        """

        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """
    The final component of the Transformer's Feed-Forward Network (FFN).

    This module performs the "contraction" phase: projecting the high-dimensional
    intermediate states back down to the model's hidden size. It then applies
    dropout, adds a residual connection from the attention output, and
    finalizes the layer with normalization.

    Args:
        config (PretrainedConfig): The model configuration containing
            intermediate_size, hidden_size, and layer_norm_eps.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Finalizes the FFN computation and applies a residual connection.

        Args:
            hidden_states (torch.Tensor): The expanded output from BertIntermediate
                of shape (batch_size, seq_len, intermediate_size).
            input_tensor (torch.Tensor): The output from the Attention layer
                (residual shortcut) of shape (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor: The final output of the Transformer layer of shape
                (batch_size, seq_len, hidden_size).
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

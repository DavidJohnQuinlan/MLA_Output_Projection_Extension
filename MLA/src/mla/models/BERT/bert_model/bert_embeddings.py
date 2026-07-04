
import torch
from torch import nn


class BertEmbeddings(nn.Module):
    """
    Constructs BERT input embeddings by summing word, position, and token type embeddings.

    This layer transforms discrete input IDs into continuous vectors and applies
    normalization and dropout to stabilize the initial representations.

    Args:
        config: Model configuration object containing hyperparameters like vocab_size,
                hidden_size, and max_position_embeddings.
    """

    def __init__(self, config):
        super().__init__()

        # Trainable Lookup Tables
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        # Fixed Buffers: Pre-allocate templates for IDs to avoid redundant ops on every forward pass.
        # These are persistent=False because they are generated from arange and don't need to be saved in weights.
        # Automatically they are added to the correct device are included in the state dict and are excluded from the optimizer
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)
        self.register_buffer("token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len) indices of input sequence tokens.
            token_type_ids: (batch_size, seq_len) segment indices (0 for sentence A, 1 for B).
            position_ids: (batch_size, seq_len) indices of positions in the sequence.
            inputs_embeds: (batch_size, seq_len, hidden_size) optional pre-computed word vectors.
            past_key_values_length: Number of previous tokens (used in KV-caching for inference).

        Returns:
            embeddings: (batch_size, seq_len, hidden_size) normalized summed representations.
        """

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]

        # Generate position IDs if not provided, accounting for KV-cache offset
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Generate token type IDs (default to 0s) if not provided
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # Retrieve and sum all embedding components
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # Post-processing: Normalize to prevent any one embedding from dominating the signal
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

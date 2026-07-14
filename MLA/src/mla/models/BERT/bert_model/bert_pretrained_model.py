from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.modeling_bert import BertLMPredictionHead  # , load_tf_weights_in_bert

from mla.models.BERT.bert_model.bert_config import BertConfig


class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a python interface
    for downloading and loading pretrained models.

    Attributes:
        config (BertConfig): Model configuration class with all hyperparameters.
        base_model_prefix (str): Prefix used for the base model attribute ("bert").
    """
    config: BertConfig
    # load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True
    _supports_sdpa = True

    def _init_weights(self, module):
        """
        Initializes the weights of the provided module using BERT-specific defaults.

        BERT uses a truncated normal distribution for initialization, which differs
        from PyTorch's default Kaiming initialization. This ensures training stability
        across the deep Transformer architecture.

        Args:
            module (nn.Module): The layer or sub-module to initialize (e.g., Linear,
                                Embedding, or LayerNorm).
        """
        if isinstance(module, nn.Linear):
            # BERT paper specifies a normal distribution (mean=0.0, std=0.02)
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # Ensure the [PAD] token vector remains zeroed so it doesn't contribute signal
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.LayerNorm):
            # LayerNorm is initialized to 'identity' (bias 0, weight 1)
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        elif isinstance(module, BertLMPredictionHead):
            # Specifically zero out bias for the prediction head
            module.bias.data.zero_()

import torch
from transformers import initialization as init
from transformers.modeling_utils import PreTrainedModel

from mla.models.BERT.attention_mechanisms import BertBaseAttention, BertCrossAttention
from mla.models.BERT.config import BertConfig
from mla.models.BERT.embeddings import BertEmbeddings
from mla.models.BERT.layer import BertLayer


class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a python interface
    for downloading and loading pretrained models.

    Attributes:
        config (BertConfig): Model configuration class with all hyperparameters.
        base_model_prefix (str): Prefix used for the base model attribute ("bert").
    """
    config_class = BertConfig
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": BertLayer,
        "attentions": BertBaseAttention,
        "cross_attentions": BertCrossAttention,
    }

    @torch.no_grad()
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
        from mla.models.BERT.heads import BertLMPredictionHead
        super()._init_weights(module)
        if isinstance(module, BertLMPredictionHead):
            init.zeros_(module.bias)
        elif isinstance(module, BertEmbeddings):
            init.copy_(module.position_ids, torch.arange(module.position_ids.shape[-1]).expand((1, -1)))
            init.zeros_(module.token_type_ids)

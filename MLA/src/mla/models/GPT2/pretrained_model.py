
import math

import torch
from transformers import initialization as init
from transformers.pytorch_utils import Conv1D
from transformers.utils.output_capturing import OutputRecorder

from mla.models.GPT2.attention import GPT2Attention
from mla.models.GPT2.block import GPT2Block
from mla.models.GPT2.config import GPT2Config
from mla.models.GPT2.pretrained_model import PreTrainedModel


class GPT2PreTrainedModel(PreTrainedModel):
    config: GPT2Config
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPT2Block"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True
    _can_compile_fullgraph = True
    _canr_record_outputs = {
        "hidden_states": GPT2Block,
        "attentions": OutputRecorder(GPT2Attention, layer_name=".attn", index=1),
        "cross_attentions": OutputRecorder(GPT2Attention, layer_name=".crossattention", index=1),
    }

    # No longer used as we directly use our masks instead
    _keys_to_ignore_on_load_unexpected = ["attn.bias", "crossattention.bias"]

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights."""
        super()._init_weights(module)
        if isinstance(module, Conv1D):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        if isinstance(module, PreTrainedModel):
            for name, p in module.named_parameters():
                if name == "c_proj.weight":
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    init.normal_(p, mean=0.0, std=self.config.initializer_range / math.sqrt(2 * self.config.n_layer))

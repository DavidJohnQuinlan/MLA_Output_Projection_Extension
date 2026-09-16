import torch
from torch.utils.data import DataLoader

from mla.model_training.model_training import BaseModelTraining


class AttentionHeadHook:
    """
    Registers forward hooks on each encoder layer's self-attention module to capture
    per-head activations for downstream analysis (e.g. CKA similarity).

    Handles both compiled (`torch.compile`) and uncompiled models by unwrapping
    `_orig_mod` if present.

    Attributes:
        model (nn.Module): The unwrapped base model.
        handles (list): Active hook handles, used for cleanup via `remove()`.
        activations (dict): Maps layer index to captured head activations of
            shape `(n_heads, batch_size * seq_len, head_dim)`.
        n_heads (int): Number of attention heads.
        hidden_size (int): Model hidden dimension.
        head_dim (int): Per-head feature dimension.
    """
    def __init__(self, model):
        """
        Args:
            model (nn.Module): The model to hook into. Compiled models are unwrapped automatically.
        """
        self.model = model
        self.handles = []
        self.activations = {}

        self.n_heads = self.model.config.num_attention_heads
        self.hidden_size = self.model.config.hidden_size
        self.head_dim = self.hidden_size // self.n_heads

    def bert_hook_fn(self, layer_idx: int):
        """
        Returns a forward hook that captures and reshapes attention head activations for a given layer.

        Args:
            layer_idx (int): Index of the encoder layer being hooked.
        """
        def hook(module, inputs, outputs):
            context_layer = outputs[0]
            batch_size, seq_len, _ = context_layer.shape

            heads = (
                context_layer
                .reshape(batch_size, seq_len, self.n_heads, self.head_dim)
                .permute(2, 0, 1, 3)
                .reshape(self.n_heads, -1, self.head_dim)
                .detach()
                .cpu()
                .float()
            )
            self.activations[layer_idx] = heads

        return hook

    def gpt2_hook_fn(self, layer_idx: int):
        """
        Returns a forward hook that captures and reshapes attention head activations for a given layer.

        Args:
            layer_idx (int): Index of the encoder layer being hooked.
        """
        def hook(module, inputs):
            attn_weights = inputs[0]
            batch_size, seq_len, _ = attn_weights.shape

            heads = (
                attn_weights
                .reshape(batch_size, seq_len, self.n_heads, self.head_dim)
                .permute(2, 0, 1, 3)
                .reshape(self.n_heads, -1, self.head_dim)
                .detach()
                .cpu()
                .float()
            )
            self.activations[layer_idx] = heads

        return hook

    def register(self) -> None:
        """Register forward hooks on all encoder self-attention layers."""
        if hasattr(self.model, "bert"):
            for i, layer in enumerate(self.model.bert.encoder.layer):
                handle = layer.attention.self.register_forward_hook(
                    self.bert_hook_fn(i)
                )
                self.handles.append(handle)
        elif hasattr(self.model, "transformer"):
            for i, h in enumerate(self.model.transformer.h):
                if hasattr(h.attn, "c_proj"):
                    handle = h.attn.c_proj.register_forward_pre_hook(
                        self.gpt2_hook_fn(i)
                    )
                elif hasattr(h.attn, "output_down_proj"):
                    handle = h.attn.output_down_proj.register_forward_pre_hook(
                        self.gpt2_hook_fn(i)
                    )
                self.handles.append(handle)

    def remove(self) -> None:
        """Remove all registered hooks and clear stored handles."""
        for handle in self.handles:
            handle.remove()

        self.handles.clear()


def collect_attention_head_activations(pretrainer: BaseModelTraining, validation_dataloader: DataLoader) -> list[torch.Tensor]:
    """
    Collects per-layer attention head activations by running a forward pass with registered hooks.

    Registers forward hooks on all encoder self-attention layers, runs evaluation
    over the validation set to populate activations, then removes the hooks.

    Args:
        pretrainer (BaseModelTraining): Trainer instance used to run evaluation.
        validation_dataloader (DataLoader): Validation dataloader used to drive the forward pass.

    Returns:
        list[torch.Tensor]: Per-layer head activations, where each element has
            shape `(n_heads, batch_size * seq_len, head_dim)`.
    """

    # Register the hooks
    torch._dynamo.reset()
    collector = AttentionHeadHook(pretrainer.unwrapped_model)
    collector.register()

    # Output the loss/metric values
    _, _ = pretrainer.eval_model(validation_dataloader=validation_dataloader, training_eval=False)

    # Extract the attention heads
    activations_list = list(collector.activations.values())

    # Remove the hooks
    collector.remove()

    return activations_list

from collections.abc import Callable

import torch
from torch import nn
from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.pytorch_utils import Conv1D
from transformers.utils.generic import maybe_autocast


def eager_attention_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)

    return attn_output, attn_weights


class GPT2Attention(nn.Module):
    """
    Standard GPT-2 multi-head self-attention (MHA), and the base class for every
    attention variant in this module.

    The forward pass is factored into two overridable seams so that variants only
    need to change what actually differs between them:

      * the query/key/value projection  -> ``_build_qkv_proj`` / ``_project_qkv``
      * the attention-output projection  -> ``_build_output_proj`` / ``_project_output``

    Everything else, KV-cache handling, the eager vs. configured-backend dispatch,
    and the optional upcast/reorder path, lives here and is shared by all variants.
    Subclasses override the hooks above and should not need to reimplement
    ``__init__`` or ``forward``.
    """
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn
        self.is_cross_attention = is_cross_attention
        self.layer_idx = layer_idx

        # Precompute unified scaling factor (accounts for both head_dim and layer-wise scaling)
        self.scaling = 1.0
        if self.scale_attn_weights:
            self.scaling = self.head_dim**-0.5
        if self.scale_attn_by_inverse_layer_idx:
            self.scaling /= float(self.layer_idx + 1)

        # Build the QKV and output projections
        self._build_qkv_proj()
        self._build_output_proj()

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = not is_cross_attention

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with maybe_autocast(query.device.type, enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=self.scaling)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2)

        return attn_output, attn_weights

    def _build_qkv_proj(self) -> None:
        """
        Construct the query/key/value input projection for standard MHA.

        Self-attention uses a single fused ``c_attn`` producing Q, K and V;
        cross-attention takes K/V from ``c_attn`` (applied to the encoder states)
        and Q from a separate ``q_attn``. Called once from ``__init__``.
        """
        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)

    def _build_output_proj(self) -> None:
        """
        Construct the attention-output projection for standard MHA: a single dense
        ``c_proj`` mapping the concatenated heads back to ``embed_dim``. Called once
        from ``__init__``.
        """
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

    def _project_qkv(
            self,
            hidden_states: tuple[torch.FloatTensor] | None,
            attention_mask: torch.FloatTensor | None = None,
            past_key_values: EncoderDecoderCache | Cache | None = None,
            curr_past_key_values: Cache | None = None,
            encoder_hidden_states: torch.Tensor | None = None,
            encoder_attention_mask: torch.FloatTensor | None = None,
            is_updated: bool = False,
            is_cross_attention: bool = False
        ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor | None]:
        """
        Project the inputs into per-head query, key and value tensors for standard MHA.

        Handles both the self-attention path (fused ``c_attn`` on ``hidden_states``)
        and the cross-attention path (Q from ``q_attn``; K/V from the encoder states
        or reused from the cache). Returns tensors shaped
        ``(batch, num_heads, seq_len, head_dim)`` plus the ``attention_mask`` that
        ``forward`` should use downstream -- swapped to ``encoder_attention_mask`` on
        the cross-attention path, otherwise passed through unchanged.

        Returns:
            tuple: ``(query_states, key_states, value_states, attention_mask)``.
        """
        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            query_states = self.q_attn(hidden_states)
            attention_mask = encoder_attention_mask

            # Try to get key/value states from cache if possible
            # Pre-computed encoder step
            if past_key_values is not None and is_updated:
                key_states = curr_past_key_values.layers[self.layer_idx].keys
                value_states = curr_past_key_values.layers[self.layer_idx].values
            else:
                key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
                shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
                key_states = key_states.view(shape_kv).transpose(1, 2)
                value_states = value_states.view(shape_kv).transpose(1, 2)
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
            shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        return query_states, key_states, value_states, attention_mask

    def _project_output(self, attn_output: torch.FloatTensor) -> torch.FloatTensor:
        """
        Merge the per-head attention output and project it back to ``embed_dim``.

        Flattens the heads, applies the dense ``c_proj``, then residual dropout.
        Returns a tensor shaped ``(batch, seq_len, embed_dim)``.
        """
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output

    def forward(
        self,
        hidden_states: tuple[torch.FloatTensor] | None,
        past_key_values: Cache | None = None,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor], ...]:
        """
        Run attention for one block.

        Resolves the KV cache, delegates the Q/K/V projection to ``_project_qkv``,
        computes attention (eager-with-upcast or the configured backend), and
        delegates the output projection to ``_project_output``. Those two delegated
        steps are the only behaviour that varies across attention variants.

        Returns:
            tuple: ``(attn_output, attn_weights)``.
        """
        is_updated = False
        is_cross_attention = encoder_hidden_states is not None
        curr_past_key_values = None
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    # after the first generated id, we can subsequently re-use all key/value_layer from cache
                    curr_past_key_values = past_key_values.cross_attention_cache
                else:
                    curr_past_key_values = past_key_values.self_attention_cache
            else:
                curr_past_key_values = past_key_values

        query_states, key_states, value_states, attention_mask = self._project_qkv(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            curr_past_key_values=curr_past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            is_updated=is_updated,
            is_cross_attention=is_cross_attention
        )

        if (past_key_values is not None and not is_cross_attention) or (
            past_key_values is not None and is_cross_attention and not is_updated
        ):
            key_states, value_states = curr_past_key_values.update(key_states, value_states, self.layer_idx)
            # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
            if is_cross_attention:
                past_key_values.is_updated[self.layer_idx] = True

        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        if using_eager and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                scaling=self.scaling,
                **kwargs,
            )

        attn_output = self._project_output(attn_output)

        return attn_output, attn_weights


class GPT2AttentionExtension(GPT2Attention):
    """
    MHA with a low-rank ("latent") compression on the attention output (MHAE).

    Identical to ``GPT2Attention`` on the Q/K/V path; only the output projection is
    replaced. Instead of a single ``c_proj``, the merged heads are projected down to
    ``output_compression_dim``, layer-normalised, and projected back up to
    ``embed_dim``.
    """
    def _build_output_proj(self) -> None:
        """
        Construct the compressed output projection: down-projection to
        ``output_compression_dim``, a ``LayerNorm`` in the compressed space, and an
        up-projection back to ``embed_dim``. Replaces the base ``c_proj``.
        """
        self.output_compression_dim = self.config.output_compression_dim
        self.output_down_proj = Conv1D(self.output_compression_dim, self.embed_dim)
        self.output_up_proj = Conv1D(self.embed_dim, self.output_compression_dim)
        self.output_norm_layer = nn.LayerNorm(self.output_compression_dim)

    def _project_output(self, attn_output: torch.FloatTensor) -> torch.FloatTensor:
        """
        Merge the per-head attention output, then compress and reconstruct it.

        Flattens the heads, down-projects to ``output_compression_dim``, applies the
        LayerNorm, up-projects back to ``embed_dim``, then residual dropout. Returns
        a tensor shaped ``(batch, seq_len, embed_dim)``.
        """
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_latent_output = self.output_down_proj(attn_output)
        attn_latent_norm_output = self.output_norm_layer(attn_latent_output)
        attn_output = self.output_up_proj(attn_latent_norm_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output


class GPT2MultiHeadedLatentAttention(GPT2Attention):
    """
    Multi-Head Latent Attention (MLA).

    Replaces the fused ``c_attn``/``q_attn`` projections with low-rank ("latent")
    query and key/value paths: inputs are projected down to ``q_compression_dim`` /
    ``kv_compression_dim``, layer-normalised, then projected back up before being
    split into heads. The output projection is unchanged from standard MHA.
    """
    def _build_qkv_proj(self) -> None:
        """
        Construct the latent Q and K/V projections.

        Query: down-project to ``q_compression_dim`` -> LayerNorm -> up-project to
        ``embed_dim``. Key/value: down-project to ``kv_compression_dim`` -> LayerNorm
        -> up-project to ``2 * embed_dim`` (later split into K and V). Replaces the
        base fused ``c_attn`` -- no ``c_attn`` is built.
        """
        self.kv_compression_dim = self.config.kv_compression_dim
        self.q_compression_dim = self.config.q_compression_dim
        self.kv_down_proj = Conv1D(self.kv_compression_dim, self.embed_dim)
        self.kv_up_proj = Conv1D(2 * self.embed_dim, self.kv_compression_dim)
        self.q_down_proj = Conv1D(self.q_compression_dim, self.embed_dim)
        self.q_up_proj = Conv1D(self.embed_dim, self.q_compression_dim)

        self.kv_norm_layer = nn.LayerNorm(self.kv_compression_dim)
        self.q_norm_layer = nn.LayerNorm(self.q_compression_dim)

    def _project_qkv(
            self,
            hidden_states: tuple[torch.FloatTensor] | None,
            attention_mask: torch.FloatTensor | None = None,
            past_key_values: EncoderDecoderCache | Cache | None = None,
            curr_past_key_values: Cache | None = None,
            encoder_hidden_states: torch.Tensor | None = None,
            encoder_attention_mask: torch.FloatTensor | None = None,
            is_updated: bool = False,
            is_cross_attention: bool = False
        ) -> list[torch.FloatTensor]:
        """
        Project the inputs into per-head query, key and value tensors via the latent
        (low-rank) Q and K/V paths.

        Mirrors ``GPT2Attention._project_qkv`` but routes through the down/norm/up
        projections instead of ``c_attn``/``q_attn``. Returns tensors shaped
        ``(batch, num_heads, seq_len, head_dim)`` plus the ``attention_mask`` for
        downstream use.
        """
        if is_cross_attention:
            if not hasattr(self, "q_down_proj"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_down_proj` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            q_latent_states = self.q_down_proj(hidden_states)
            q_latent_norm_states = self.q_norm_layer(q_latent_states)
            query_states = self.q_up_proj(q_latent_norm_states)
            attention_mask = encoder_attention_mask

            # Try to get key/value states from cache if possible
            # Pre-computed encoder step
            if past_key_values is not None and is_updated:
                key_states = curr_past_key_values.layers[self.layer_idx].keys
                value_states = curr_past_key_values.layers[self.layer_idx].values
            else:
                kv_latent_states = self.kv_down_proj(encoder_hidden_states)
                kv_latent_norm_states = self.kv_norm_layer(kv_latent_states)
                key_states, value_states = self.kv_up_proj(kv_latent_norm_states).split(self.split_size, dim=2)
                shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
                key_states = key_states.view(shape_kv).transpose(1, 2)
                value_states = value_states.view(shape_kv).transpose(1, 2)
        else:
            kv_latent_states = self.kv_down_proj(hidden_states)
            kv_latent_norm_states = self.kv_norm_layer(kv_latent_states)
            key_states, value_states = self.kv_up_proj(kv_latent_norm_states).split(self.split_size, dim=2)

            q_latent_states = self.q_down_proj(hidden_states)
            q_latent_norm_states = self.q_norm_layer(q_latent_states)
            query_states = self.q_up_proj(q_latent_norm_states)

            shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        return query_states, key_states, value_states, attention_mask


class GPT2MultiHeadedLatentAttentionExtension(GPT2AttentionExtension, GPT2MultiHeadedLatentAttention):
    """
    MLA combined with the compressed output projection (MLAE) -- latent Q/K/V *and*
    latent output.

    This class deliberately has no body; it is composed entirely by method
    resolution order (MRO) from its two parents:

        MRO: MLAE -> GPT2AttentionExtension -> GPT2MultiHeadedLatentAttention
                  -> GPT2Attention -> nn.Module

      * Q/K/V hooks (``_build_qkv_proj`` / ``_project_qkv``) resolve to
        ``GPT2MultiHeadedLatentAttention``: ``GPT2AttentionExtension`` does not
        define them, so the MRO falls through to the latent path.
      * Output hooks (``_build_output_proj`` / ``_project_output``) resolve to
        ``GPT2AttentionExtension``, which is earlier in the MRO than MLA, so its
        compressed-output versions win.
      * ``__init__`` and ``forward`` resolve to ``GPT2Attention`` and receive the
        real constructor arguments unchanged.

    The base-class order ``(GPT2AttentionExtension, GPT2MultiHeadedLatentAttention)``
    is load-bearing: it selects the compressed output from the first parent and the
    latent Q/K/V from the second. This composes cleanly only because each parent
    overrides a *disjoint* pair of hooks; if both overrode the same hook, the
    left-most parent would silently win. Reordering the parents would swap which
    output projection is used.
    """
    pass

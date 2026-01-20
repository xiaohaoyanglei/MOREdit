"""Attention recorder to capture cross-attention probabilities."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, List, Optional

import torch
from diffusers.models.transformers.transformer_flux import FluxAttention, _get_qkv_projections, apply_rotary_emb


class PointerRecorder(torch.nn.Module):
    """Replace Flux attention processor to record conditional attention maps."""

    def __init__(self, owner: "PointerTrainer", layer_key: str):
        super().__init__()
        self.owner = owner
        self.layer_key = layer_key

    def __call__(
        self,
        attn: FluxAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        enable_ctx = torch.enable_grad() if not torch.is_grad_enabled() else nullcontext()
        with enable_ctx:
            query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
                attn, hidden_states, encoder_hidden_states
            )

            query = query.unflatten(-1, (attn.heads, -1))
            key = key.unflatten(-1, (attn.heads, -1))
            value = value.unflatten(-1, (attn.heads, -1))

            query = attn.norm_q(query)
            key = attn.norm_k(key)

            encoder_length = 0
            if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
                encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
                encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
                encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

                encoder_query = attn.norm_added_q(encoder_query)
                encoder_key = attn.norm_added_k(encoder_key)

                query = torch.cat([encoder_query, query], dim=1)
                key = torch.cat([encoder_key, key], dim=1)
                value = torch.cat([encoder_value, value], dim=1)
                encoder_length = encoder_query.shape[1]

            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

            query = query.permute(0, 2, 1, 3)
            key = key.permute(0, 2, 1, 3)
            value = value.permute(0, 2, 1, 3)

            scale = 1.0 / torch.sqrt(torch.tensor(attn.head_dim, dtype=query.dtype, device=query.device))
            scores = torch.matmul(query, key.transpose(-1, -2)) * scale
            if attention_mask is not None:
                scores = scores + attention_mask.to(dtype=scores.dtype)

            probs = torch.softmax(scores, dim=-1)
            context = torch.matmul(probs, value)

            context = context.permute(0, 2, 1, 3).reshape(hidden_states.shape[0], -1, attn.heads * attn.head_dim)
            context = context.to(hidden_states.dtype)

            if encoder_length > 0:
                pointer_slice = probs[:, :, encoder_length:, :encoder_length]
                self.owner.record_pointer(self.layer_key, pointer_slice)
                if getattr(self.owner, "_pointer_debug_active", False):
                    self.owner.report_pointer_slice(self.layer_key, tuple(pointer_slice.shape), encoder_length)

            if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
                encoder_context, hidden_context = context.split(
                    [encoder_length, context.shape[1] - encoder_length], dim=1
                )
                hidden_context = attn.to_out[0](hidden_context)
                hidden_context = attn.to_out[1](hidden_context)
                encoder_context = attn.to_add_out(encoder_context)
                return hidden_context, encoder_context

            return context


class PointerCache:
    """Store attention probabilities for later aggregation."""

    def __init__(self):
        self._store: Dict[str, List[torch.Tensor]] = {}

    def reset(self):
        self._store = {}

    def append(self, key: str, tensor: torch.Tensor):
        self._store.setdefault(key, []).append(tensor)

    def items(self):
        return self._store.items()



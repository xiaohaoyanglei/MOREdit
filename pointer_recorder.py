"""Attention recorder to capture Qwen joint-attention probabilities."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, List, Optional

import torch
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen


class PointerRecorder(torch.nn.Module):
    """Replace Qwen double-stream attention processor to record conditional attention maps."""

    def __init__(self, owner: "PointerTrainer", layer_key: str):
        super().__init__()
        self.owner = owner
        self.layer_key = layer_key

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        enable_ctx = torch.enable_grad() if not torch.is_grad_enabled() else nullcontext()
        with enable_ctx:
            if encoder_hidden_states is None:
                raise ValueError("PointerRecorder requires encoder_hidden_states for joint attention.")

            seq_txt = encoder_hidden_states.shape[1]

            img_query = attn.to_q(hidden_states)
            img_key = attn.to_k(hidden_states)
            img_value = attn.to_v(hidden_states)

            txt_query = attn.add_q_proj(encoder_hidden_states)
            txt_key = attn.add_k_proj(encoder_hidden_states)
            txt_value = attn.add_v_proj(encoder_hidden_states)

            img_query = img_query.unflatten(-1, (attn.heads, -1))
            img_key = img_key.unflatten(-1, (attn.heads, -1))
            img_value = img_value.unflatten(-1, (attn.heads, -1))
            txt_query = txt_query.unflatten(-1, (attn.heads, -1))
            txt_key = txt_key.unflatten(-1, (attn.heads, -1))
            txt_value = txt_value.unflatten(-1, (attn.heads, -1))

            if attn.norm_q is not None:
                img_query = attn.norm_q(img_query)
            if attn.norm_k is not None:
                img_key = attn.norm_k(img_key)
            if attn.norm_added_q is not None:
                txt_query = attn.norm_added_q(txt_query)
            if attn.norm_added_k is not None:
                txt_key = attn.norm_added_k(txt_key)

            if image_rotary_emb is not None:
                img_freqs, txt_freqs = image_rotary_emb
                img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
                img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
                txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
                txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

            # Order follows Qwen processor: [text, image]
            joint_query = torch.cat([txt_query, img_query], dim=1).permute(0, 2, 1, 3)
            joint_key = torch.cat([txt_key, img_key], dim=1).permute(0, 2, 1, 3)
            joint_value = torch.cat([txt_value, img_value], dim=1).permute(0, 2, 1, 3)

            head_dim = joint_query.shape[-1]
            scale = 1.0 / torch.sqrt(torch.tensor(head_dim, dtype=joint_query.dtype, device=joint_query.device))
            scores = torch.matmul(joint_query, joint_key.transpose(-1, -2)) * scale
            if attention_mask is not None:
                scores = scores + attention_mask.to(dtype=scores.dtype)

            probs = torch.softmax(scores, dim=-1)
            context = torch.matmul(probs, joint_value)
            context = context.permute(0, 2, 1, 3).reshape(hidden_states.shape[0], -1, attn.heads * head_dim)
            context = context.to(hidden_states.dtype)

            pointer_slice = probs[:, :, seq_txt:, :seq_txt]
            self.owner.record_pointer(self.layer_key, pointer_slice)
            if getattr(self.owner, "_pointer_debug_active", False):
                self.owner.report_pointer_slice(self.layer_key, tuple(pointer_slice.shape), seq_txt)

            txt_context, img_context = context.split([seq_txt, context.shape[1] - seq_txt], dim=1)
            img_context = attn.to_out[0](img_context.contiguous())
            if len(attn.to_out) > 1:
                img_context = attn.to_out[1](img_context)
            txt_context = attn.to_add_out(txt_context.contiguous())
            return img_context, txt_context


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

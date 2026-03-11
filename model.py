"""Model loading and inference helpers for Qwen pointer training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer


@dataclass
class PromptEmbeds:
    text_embeds: torch.Tensor
    pooled_embeds: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None


class QwenPromptEncoder:
    """Shared prompt encode/tokenize helper for Qwen image models."""

    def __init__(
        self,
        tokenizer: Qwen2Tokenizer,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        max_length: int = 1024,
    ) -> None:
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.max_length = int(max_length)

    @property
    def model_max_length(self) -> int:
        return self.max_length

    def to(self, *args, **kwargs):
        self.text_encoder.to(*args, **kwargs)
        return self

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(ids)

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> List[torch.Tensor]:
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return list(torch.split(selected, valid_lengths.tolist(), dim=0))

    def _tokenize_raw_prompts(
        self,
        prompts: List[str],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        return encoded.input_ids.to(device), encoded.attention_mask.to(device)

    def encode_pointer_tokens(self, prompts: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._tokenize_raw_prompts(prompts, device=device)

    def encode_prompts(
        self,
        prompts: List[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> PromptEmbeds:
        # `device` is the target device for returned embeddings.
        # The text encoder itself may be offloaded to CPU.
        encoder_device = self.text_encoder.get_input_embeddings().weight.device
        input_ids, attention_mask = self._tokenize_raw_prompts(prompts, device=encoder_device)

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        split_hidden = self._extract_masked_hidden(hidden_states, attention_mask)

        attn_mask_list = [torch.ones(h.size(0), dtype=torch.long, device=encoder_device) for h in split_hidden]
        max_seq = max(h.size(0) for h in split_hidden)

        prompt_embeds = torch.stack(
            [torch.cat([h, h.new_zeros(max_seq - h.size(0), h.size(1))], dim=0) for h in split_hidden], dim=0
        ).to(device=device, dtype=dtype)
        encoder_attention_mask = torch.stack(
            [torch.cat([m, m.new_zeros(max_seq - m.size(0))], dim=0) for m in attn_mask_list], dim=0
        ).to(device=device)
        return PromptEmbeds(text_embeds=prompt_embeds, attention_mask=encoder_attention_mask)


def load_qwen_components(
    model_root: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    offload_text_encoders: bool = False,
) -> Tuple[
    QwenImageTransformer2DModel,
    List[QwenPromptEncoder],
    List[QwenPromptEncoder],
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
]:
    """Load Qwen transformer, prompt encoder, VAE, and scheduler."""

    transformer = QwenImageTransformer2DModel.from_pretrained(model_root, subfolder="transformer", torch_dtype=dtype)
    transformer.to(device, dtype=dtype)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_root, subfolder="tokenizer")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_root, subfolder="text_encoder")
    if offload_text_encoders:
        text_encoder.to("cpu", dtype=torch.float32)
    else:
        text_encoder.to(device, dtype=dtype)
    text_encoder.eval()

    prompt_encoder = QwenPromptEncoder(tokenizer=tokenizer, text_encoder=text_encoder)

    vae = AutoencoderKLQwenImage.from_pretrained(model_root, subfolder="vae", torch_dtype=dtype)
    vae.to(device, dtype=dtype)
    vae.eval()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_root, subfolder="scheduler")

    # Keep both return slots as wrapper list for backward compatibility.
    tokenizers = [prompt_encoder]
    text_encoders = [prompt_encoder]

    return transformer, text_encoders, tokenizers, vae, scheduler


def encode_prompts(
    prompts: List[str],
    tokenizers: List[QwenPromptEncoder],
    text_encoders: List[QwenPromptEncoder],
    device: torch.device,
    dtype: torch.dtype,
) -> PromptEmbeds:
    """Encode prompts using Qwen text encoder."""

    encoder = text_encoders[0] if text_encoders else tokenizers[0]
    return encoder.encode_prompts(prompts=prompts, device=device, dtype=dtype)


def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.dim() != 5:
        raise ValueError(f"Expected 5D latents [B, 1, C, H, W], got shape {tuple(latents.shape)}")
    bsz, _, channels, height, width = latents.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"Packed latents require even H/W, got {(height, width)}")
    latents = latents.view(bsz, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(bsz, (height // 2) * (width // 2), channels * 4)
    return latents


def encode_images(vae: AutoencoderKLQwenImage, pixel_values: torch.Tensor) -> torch.Tensor:
    """Encode images into Qwen packed latent space."""

    if pixel_values.ndim == 3:
        pixel_values = pixel_values.unsqueeze(0)
    if pixel_values.ndim != 4:
        raise ValueError(f"pixel_values must be [B,C,H,W], got {tuple(pixel_values.shape)}")

    pixel_values = pixel_values.to(device=vae.device, dtype=vae.dtype)
    with torch.no_grad():
        image_latents = vae.encode(pixel_values.unsqueeze(2)).latent_dist.sample()

    latents_mean = (
        torch.tensor(vae.config.latents_mean, device=image_latents.device, dtype=image_latents.dtype)
        .view(1, vae.config.z_dim, 1, 1, 1)
    )
    latents_std = (
        1.0
        / torch.tensor(vae.config.latents_std, device=image_latents.device, dtype=image_latents.dtype)
        .view(1, vae.config.z_dim, 1, 1, 1)
    )
    image_latents = (image_latents - latents_mean) * latents_std
    image_latents = image_latents.transpose(1, 2)  # [B,1,C,H,W]
    return _pack_latents(image_latents)


def predict_noise(
    transformer: QwenImageTransformer2DModel,
    scheduler: FlowMatchEulerDiscreteScheduler,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: PromptEmbeds,
) -> torch.Tensor:
    """Run Qwen transformer to predict noise with gradient enabled."""

    del scheduler  # Kept for backward-compatible signature.

    if latents.dim() != 3:
        raise ValueError(f"Qwen latents must be [B, Seq, C], got {tuple(latents.shape)}")

    bsz, seq_len, _ = latents.shape
    side = int(seq_len**0.5)
    if side * side != seq_len:
        raise ValueError(f"Unexpected latent sequence length {seq_len}; expected square number.")
    packed_h = side * 2
    packed_w = side * 2

    timestep = timesteps.float().to(latents.device)
    while timestep.ndim < 1:
        timestep = timestep.unsqueeze(0)
    if timestep.shape[0] != bsz:
        timestep = timestep.expand(bsz)

    guidance = None
    if getattr(transformer.config, "guidance_embeds", False):
        guidance = torch.ones((bsz,), device=latents.device, dtype=torch.float32)

    img_shapes = [[(1, packed_h // 2, packed_w // 2)]] * bsz
    outputs = transformer(
        hidden_states=latents,
        timestep=timestep / 1000.0,
        guidance=guidance,
        encoder_hidden_states=prompt_embeds.text_embeds,
        encoder_hidden_states_mask=prompt_embeds.attention_mask,
        img_shapes=img_shapes,
        return_dict=False,
    )[0]
    return outputs.to(dtype=latents.dtype)

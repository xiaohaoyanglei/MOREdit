"""Model loading and inference helpers for Flux Kontext pointer training."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
from diffusers import FluxTransformer2DModel, AutoencoderKL
from einops import rearrange, repeat
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from .scheduler import CustomFlowMatchEulerDiscreteScheduler


@dataclass
class PromptEmbeds:
    text_embeds: torch.Tensor
    pooled_embeds: torch.Tensor


def load_flux_kontext_components(
    model_root: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    offload_text_encoders: bool = False,
) -> Tuple[
    FluxTransformer2DModel,
    List[CLIPTextModel],
    List[CLIPTokenizer],
    AutoencoderKL,
    CustomFlowMatchEulerDiscreteScheduler,
]:
    """Load Flux Kontext transformer, text encoders, tokenizers, VAE, and scheduler."""

    transformer_path = os.path.join(model_root, "transformer")
    transformer = FluxTransformer2DModel.from_pretrained(transformer_path, torch_dtype=dtype)
    transformer.to(device, dtype=dtype)

    tokenizer_clip = CLIPTokenizer.from_pretrained(model_root, subfolder="tokenizer")
    text_encoder_clip = CLIPTextModel.from_pretrained(model_root, subfolder="text_encoder")
    tokenizer_t5 = T5TokenizerFast.from_pretrained(model_root, subfolder="tokenizer_2")
    text_encoder_t5 = T5EncoderModel.from_pretrained(model_root, subfolder="text_encoder_2")

    if offload_text_encoders:
        text_encoder_clip.to("cpu", dtype=torch.float32)
        text_encoder_t5.to("cpu", dtype=torch.float32)
    else:
        text_encoder_clip.to(device, dtype=dtype)
        text_encoder_t5.to(device, dtype=dtype)
    text_encoder_clip.eval()
    text_encoder_t5.eval()

    vae = AutoencoderKL.from_pretrained(model_root, subfolder="vae", torch_dtype=dtype)
    vae.to(device, dtype=dtype)
    vae.eval()

    scheduler = CustomFlowMatchEulerDiscreteScheduler.from_pretrained(model_root, subfolder="scheduler")

    tokenizers = [tokenizer_clip, tokenizer_t5]
    text_encoders = [text_encoder_clip, text_encoder_t5]

    return transformer, text_encoders, tokenizers, vae, scheduler


def encode_prompts(
    prompts: List[str],
    tokenizers: List[CLIPTokenizer],
    text_encoders: List[CLIPTextModel],
    device: torch.device,
    dtype: torch.dtype,
) -> PromptEmbeds:
    """Encode prompts using CLIP and T5 encoders."""

    clip_tok, t5_tok = tokenizers
    clip_enc, t5_enc = text_encoders

    clip_device = next(clip_enc.parameters()).device
    t5_device = next(t5_enc.parameters()).device

    clip_inputs = clip_tok(
        prompts,
        padding="max_length",
        max_length=clip_tok.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    clip_input_ids = clip_inputs.input_ids.to(clip_device)

    with torch.no_grad():
        clip_outputs = clip_enc(clip_input_ids, output_hidden_states=False)
    pooled_prompt_embeds = clip_outputs.pooler_output.to(dtype=dtype, device=device)

    t5_inputs = t5_tok(
        prompts,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    t5_input_ids = t5_inputs.input_ids.to(t5_device)

    with torch.no_grad():
        text_embeds = t5_enc(t5_input_ids, output_hidden_states=False)[0]
    text_embeds = text_embeds.to(dtype=dtype, device=device)

    return PromptEmbeds(text_embeds=text_embeds, pooled_embeds=pooled_prompt_embeds)


def encode_images(vae: AutoencoderKL, pixel_values: torch.Tensor) -> torch.Tensor:
    """Encode images into latent space using the VAE."""

    if pixel_values.ndim == 3:
        pixel_values = pixel_values.unsqueeze(0)
    pixel_values = pixel_values.to(vae.device)
    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    return latents


def predict_noise(
    transformer: FluxTransformer2DModel,
    scheduler: CustomFlowMatchEulerDiscreteScheduler,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: PromptEmbeds,
) -> torch.Tensor:
    """Run the Flux transformer to predict noise with gradient enabled."""

    device = latents.device
    dtype = latents.dtype

    if latents.dim() != 4:
        raise ValueError("Latents must have shape [B, C, H, W]")

    bs, _, h, w = latents.shape
    timestep = timesteps.float().to(device)
    while timestep.ndim < 1:
        timestep = timestep.unsqueeze(0)
    guidance = torch.zeros_like(timestep, device=device, dtype=timestep.dtype)

    latent_model_input = scheduler.scale_model_input(latents, timestep)
    latent_model_input = latent_model_input.to(device=device, dtype=dtype)
    ph = pw = 2
    latent_model_input_packed = rearrange(
        latent_model_input,
        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
        ph=ph,
        pw=pw,
    )

    img_ids = torch.zeros(h // ph, w // pw, 3, device=device, dtype=dtype)
    img_ids[..., 1] = torch.arange(h // ph, device=device, dtype=dtype)[:, None]
    img_ids[..., 2] = torch.arange(w // pw, device=device, dtype=dtype)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    txt_ids = torch.zeros(bs, prompt_embeds.text_embeds.shape[1], 3, device=device, dtype=dtype)

    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        img_ids = img_ids[0]

    outputs = transformer(
        hidden_states=latent_model_input_packed,
        timestep=timestep / 1000.0,
        encoder_hidden_states=prompt_embeds.text_embeds,
        pooled_projections=prompt_embeds.pooled_embeds,
        txt_ids=txt_ids,
        img_ids=img_ids,
        guidance=guidance,
        return_dict=False,
    )[0]

    noise_pred = rearrange(
        outputs,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=h // ph,
        w=w // pw,
        ph=ph,
        pw=pw,
        c=transformer.config.out_channels if transformer.config.out_channels is not None else latents.shape[1],
    )
    noise_pred = noise_pred.to(dtype=dtype)
    return noise_pred



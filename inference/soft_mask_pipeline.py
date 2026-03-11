"""Pointer LoRA soft-mask inference pipeline for Qwen models."""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

from ..lora import LoRAInjectionConfig, inject_lora_into_attention, is_qwen_joint_attention
from ..model import (
    encode_images,
    encode_prompts,
    load_qwen_components,
    predict_noise,
)
from ..pointer_recorder import PointerCache, PointerRecorder
from ..trainer import _to_dtype
from .mask_refiner import MaskRefinerConfig, load_mask_refiner


def _load_yaml(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_lora_state(lora_modules: List[torch.nn.Module], weight_path: str) -> None:
    payload = torch.load(weight_path, map_location="cpu")
    state_dict = payload.get("state_dict", payload)
    name_to_module: Dict[str, torch.nn.Module] = {}
    for idx, module in enumerate(lora_modules):
        name = getattr(module, "lora_name", f"module_{idx}")
        name_to_module[name] = module

    for name, data in state_dict.items():
        module = name_to_module.get(name)
        if module is None:
            continue
        module.lora_down.weight.data.copy_(data["lora_down"].to(module.lora_down.weight.device))
        module.lora_up.weight.data.copy_(data["lora_up"].to(module.lora_up.weight.device))
        if module.bias is not None and data.get("bias") is not None:
            module.bias.data.copy_(data["bias"].to(module.bias.device))


@dataclass
class PointerSoftMaskConfig:
    model_path: str
    base_resolution: int
    dtype: str
    offload_text_encoders: bool
    pointer_token_mode: str
    pointer_token_filters: Tuple[str, ...]
    pointer_heatmap_combine: str
    pointer_temperature: float
    pointer_epsilon: float
    pointer_superres_factor: int
    pointer_superres_sharpness: float
    pointer_noise_timestep: float
    pointer_attn_prefixes: Tuple[str, ...]
    refiner_in_channels: int
    refiner_base_channels: int
    refiner_sigma_ratio: float
    refiner_dilate_iters: int
    refiner_weight_path: Optional[str]


class PointerSoftMaskPipeline:
    """Compute soft masks via pointer LoRA and run Qwen-based editing."""

    def __init__(
        self,
        config_path: str,
        lora_weights: str,
        device: Optional[str] = None,
        refiner_weights: Optional[str] = None,
        qwen_controlnet_path: Optional[str] = None,
    ) -> None:
        cfg = _load_yaml(config_path)
        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})
        training_cfg = cfg.get("training", {})
        pointer_cfg = cfg.get("pointer", {})
        refiner_cfg = cfg.get("refiner", {})
        lora_cfg = cfg.get("lora", {})

        pointer_heatmap_cfg = pointer_cfg.get("heatmap", {})
        pointer_superres_cfg = pointer_cfg.get("superres", {})

        self.config = PointerSoftMaskConfig(
            model_path=model_cfg["path"],
            base_resolution=int(data_cfg.get("resolution", 640)),
            dtype=str(training_cfg.get("dtype", "float32")),
            offload_text_encoders=bool(model_cfg.get("offload_text_encoders", False)),
            pointer_token_mode=str(pointer_cfg.get("token_mode", "phrase")).lower(),
            pointer_token_filters=tuple(pointer_cfg.get("token_filters", [])),
            pointer_heatmap_combine=pointer_heatmap_cfg.get("combine", "mean"),
            pointer_temperature=float(pointer_heatmap_cfg.get("temperature", 1.0)),
            pointer_epsilon=float(pointer_heatmap_cfg.get("epsilon", 1.0e-6)),
            pointer_superres_factor=int(pointer_superres_cfg.get("factor", 1)),
            pointer_superres_sharpness=float(pointer_superres_cfg.get("sharpness", 0.0)),
            pointer_noise_timestep=float(pointer_cfg.get("noise_timestep", 500.0)),
            pointer_attn_prefixes=tuple(pointer_cfg.get("attn_blocks", lora_cfg.get("attn_blocks", []))),
            refiner_in_channels=int(refiner_cfg.get("in_channels", 4)),
            refiner_base_channels=int(refiner_cfg.get("base_channels", 32)),
            refiner_sigma_ratio=float(refiner_cfg.get("gaussian_sigma", 0.05)),
            refiner_dilate_iters=int(refiner_cfg.get("dilate_iters", 0)),
            refiner_weight_path=refiner_cfg.get("weights"),
        )

        lora_config = LoRAInjectionConfig(
            rank=int(lora_cfg.get("rank", 16)),
            alpha=float(lora_cfg.get("alpha", 32.0)),
            attn_block_prefixes=tuple(lora_cfg.get("attn_blocks", ["transformer_blocks.14", "transformer_blocks.15", "transformer_blocks.16"])),
            target_projections=tuple(lora_cfg.get("projections", LoRAInjectionConfig.target_projections)),
        )

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = _to_dtype(self.config.dtype)

        (
            self.transformer,
            self.text_encoders,
            self.tokenizers,
            self.vae,
            self.scheduler,
        ) = load_qwen_components(
            self.config.model_path,
            device=self.device,
            dtype=self.dtype,
            offload_text_encoders=self.config.offload_text_encoders,
        )

        self.lora_modules = inject_lora_into_attention(
            self.transformer,
            lora_config,
        )
        if not os.path.exists(lora_weights):
            raise FileNotFoundError(f"LoRA weight file not found: {lora_weights}")
        _load_lora_state(self.lora_modules, lora_weights)

        self.pointer_cache = PointerCache()
        self._pointer_modules: Dict[str, torch.nn.Module] = {}
        self._original_processors: Dict[str, torch.nn.Module] = {}
        self._recorders_active = False

        self._guidance_resize_meta: Optional[Dict[str, float]] = None
        self._last_peak_info: Optional[Dict[str, float]] = None
        self._last_pointer_map: Optional[torch.Tensor] = None
        self._refiner_input_tensor: Optional[torch.Tensor] = None
        self._last_coarse_hint: Optional[torch.Tensor] = None
        self._last_refined_mask: Optional[torch.Tensor] = None
        # version1: 仅使用热图 argmax（按 value），提供可选的视觉高斯提示

        self._superres_factor = max(1, self.config.pointer_superres_factor)
        self._superres_sharpness = max(0.0, self.config.pointer_superres_sharpness)
        self._laplacian_kernel = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        # Flux/Kontext path is intentionally disabled. We keep fields for backward compatibility.
        self.pipeline = None
        self.inpaint_pipeline = None

        self.refiner: Optional[torch.nn.Module] = None
        refiner_weight_path = refiner_weights or self.config.refiner_weight_path
        if refiner_weight_path:
            refiner_cfg = MaskRefinerConfig(
                in_channels=self.config.refiner_in_channels,
                base_channels=self.config.refiner_base_channels,
                weight_path=refiner_weight_path,
            )
            self.refiner = load_mask_refiner(refiner_cfg, self.device)
            print(f"[mask-refiner] loaded weights from {refiner_weight_path}")
        else:
            print("[mask-refiner] 未提供 refiner 权重，将仅输出粗略高斯提示。")
        self._refiner_sigma_ratio = max(1e-4, float(self.config.refiner_sigma_ratio))
        self._refiner_dilate_iters = max(0, int(self.config.refiner_dilate_iters))
        # ClickSEG cache (lazy load)
        self._clickseg_cache: Dict[str, Any] = {}
        self._qwen_backends: Dict[str, Any] = {}
        self._qwen_controlnet_path = qwen_controlnet_path
        self._last_qwen_mask_used: Optional[Image.Image] = None
        self._last_qwen_image_resized: Optional[Image.Image] = None
        self._last_qwen_full_edit: Optional[Image.Image] = None

    # ------------------------------------------------------------------ Pointer heatmap helpers
    def _attach_pointer_recorders(self) -> None:
        if self._recorders_active:
            return
        prefixes = self.config.pointer_attn_prefixes or ()
        self._pointer_modules = {}
        self._original_processors = {}
        for name, module in self.transformer.named_modules():
            if not is_qwen_joint_attention(module):
                continue
            if prefixes and not any(name.startswith(prefix) or prefix in name for prefix in prefixes):
                continue
            recorder = PointerRecorder(owner=self, layer_key=name)
            self._original_processors[name] = module.processor
            module.set_processor(recorder)
            self._pointer_modules[name] = module
        if not self._pointer_modules:
            raise RuntimeError("No attention modules matched pointer_attn_prefixes; cannot record pointer heatmaps.")
        self._recorders_active = True

    def _restore_pointer_processors(self) -> None:
        if not self._recorders_active:
            return
        for name, module in self._pointer_modules.items():
            original = self._original_processors.get(name)
            if original is not None:
                module.set_processor(original)
        self._pointer_modules = {}
        self._original_processors = {}
        self._recorders_active = False

    def record_pointer(self, key: str, tensor: torch.Tensor) -> None:
        self.pointer_cache.append(key, tensor)

    def _build_token_masks(self, prompts: List[str]) -> torch.Tensor:
        tokenizer = self.tokenizers[0]
        if hasattr(tokenizer, "encode_pointer_tokens"):
            input_ids, attention_mask = tokenizer.encode_pointer_tokens(prompts, device=self.device)
        else:
            encoded = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded.input_ids.to(self.device)
            attention_mask = encoded.attention_mask.to(self.device)
        masks = torch.zeros_like(input_ids, dtype=torch.bool)

        token_filters = {tok.lower().strip(): True for tok in self.config.pointer_token_filters}
        ordinal_keywords = {
            "first", "second", "third", "fourth", "fifth", "sixth",
            "seventh", "eighth", "ninth", "tenth", "left", "right",
            "middle", "center", "centre", "front", "back",
        }

        def normalize_token(token: str) -> str:
            return token.replace("▁", "").replace("Ġ", "").replace("ġ", "").strip().lower()

        for batch_idx, ids in enumerate(input_ids):
            tokens = tokenizer.convert_ids_to_tokens(ids.tolist())
            for token_idx, token in enumerate(tokens):
                if attention_mask[batch_idx, token_idx] == 0:
                    continue
                normalized = normalize_token(token)
                use_token = False
                if self.config.pointer_token_mode == "phrase":
                    use_token = True
                elif self.config.pointer_token_mode == "ordinal":
                    use_token = (
                        normalized in ordinal_keywords
                        or normalized.isdigit()
                        or any(char.isdigit() for char in normalized)
                    )
                if not use_token and normalized in token_filters:
                    use_token = True
                if use_token:
                    masks[batch_idx, token_idx] = True

        for batch_idx in range(masks.size(0)):
            if not masks[batch_idx].any():
                prompt_preview = prompts[batch_idx] if batch_idx < len(prompts) else ""
                print(
                    f"[soft-mask] Warning: no pointer tokens matched for prompt={prompt_preview!r}; "
                    "falling back to full attention mask."
                )
                masks[batch_idx] = attention_mask[batch_idx].bool()

        return masks.to(self.device)

    def _compose_pointer_heatmap(
        self,
        token_masks: torch.Tensor,
        return_layer_sizes: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[int]]]:
        layer_heatmaps: List[torch.Tensor] = []
        layer_sizes: List[int] = []
        token_masks = token_masks.to(self.device)

        for _, tensors in self.pointer_cache.items():
            if not tensors:
                continue
            stacked = torch.stack(tensors, dim=0).mean(dim=0)
            mask = token_masks.unsqueeze(1).unsqueeze(2).to(stacked.dtype)
            if mask.shape[-1] != stacked.shape[-1]:
                mask = mask[..., : stacked.shape[-1]]
            masked = stacked * mask
            denom = mask.sum(dim=-1).clamp_min(1.0)
            heat = masked.sum(dim=-1) / denom
            heat = heat.mean(dim=1)
            spatial_tokens = heat.shape[-1]
            side = int(spatial_tokens ** 0.5)
            if side * side != spatial_tokens:
                raise RuntimeError(f"Unexpected spatial token count {spatial_tokens}")
            heat = heat / max(self.config.pointer_temperature, 1e-6)
            heat = heat - heat.amin(dim=-1, keepdim=True)
            heat = heat / (heat.amax(dim=-1, keepdim=True) + self.config.pointer_epsilon)
            heat = heat.view(heat.shape[0], 1, side, side)
            layer_heatmaps.append(heat)
            if return_layer_sizes:
                layer_sizes.append(spatial_tokens)

        if not layer_heatmaps:
            raise RuntimeError("Pointer cache is empty; attention processors may not be attached correctly.")

        stack = torch.stack(layer_heatmaps, dim=0)
        if self.config.pointer_heatmap_combine == "max":
            combined = stack.max(dim=0).values
        else:
            combined = stack.mean(dim=0)
        combined = combined.clamp(0.0, 1.0)
        pointer_map = self._apply_super_resolution(combined)
        pointer_map = F.interpolate(
            pointer_map,
            size=(self.config.base_resolution, self.config.base_resolution),
            mode="bilinear",
            align_corners=False,
        )
        pointer_map = self._mask_letterbox_padding(pointer_map)
        return pointer_map.clamp(0.0, 1.0), (layer_sizes if return_layer_sizes else None)

    def _apply_super_resolution(self, pointer_map: torch.Tensor) -> torch.Tensor:
        if self._superres_factor <= 1:
            return pointer_map
        upsampled = F.interpolate(
            pointer_map,
            scale_factor=self._superres_factor,
            mode="bicubic",
            align_corners=False,
        )
        if self._superres_sharpness > 0.0:
            kernel = self._laplacian_kernel.to(device=upsampled.device, dtype=upsampled.dtype)
            laplacian = F.conv2d(upsampled, kernel, padding=1)
            upsampled = torch.clamp(upsampled + self._superres_sharpness * laplacian, 0.0, 1.0)
        return upsampled

    def _mask_letterbox_padding(self, pointer_map: torch.Tensor) -> torch.Tensor:
        """
        Suppress padded (letterbox) area in square pointer maps.
        This avoids peaks collapsing to padding borders (e.g. y=0 for wide images).
        """
        meta = getattr(self, "_guidance_resize_meta", None)
        if not meta or pointer_map.ndim != 4:
            return pointer_map

        h = int(pointer_map.shape[-2])
        w = int(pointer_map.shape[-1])
        pad_top = int(meta.get("pad_top", 0))
        pad_left = int(meta.get("pad_left", 0))
        resized_h = int(meta.get("resized_height", h))
        resized_w = int(meta.get("resized_width", w))

        y0 = max(0, min(h, pad_top))
        x0 = max(0, min(w, pad_left))
        y1 = max(y0, min(h, y0 + resized_h))
        x1 = max(x0, min(w, x0 + resized_w))

        valid = torch.zeros((1, 1, h, w), dtype=pointer_map.dtype, device=pointer_map.device)
        if y1 > y0 and x1 > x0:
            valid[:, :, y0:y1, x0:x1] = 1.0

        masked = pointer_map * valid
        # Renormalize per sample after masking.
        denom = masked.amax(dim=(-2, -1), keepdim=True).clamp_min(self.config.pointer_epsilon)
        masked = (masked / denom).clamp(0.0, 1.0)
        return masked

    # ------------------------------------------------------------------ Public API
    def compute_pointer_map(self, image_path: str, pointer_prompt: str) -> torch.Tensor:
        self._attach_pointer_recorders()
        tensor, resize_meta, refiner_tensor = self._preprocess_image(image_path)
        self._guidance_resize_meta = resize_meta
        self._refiner_input_tensor = refiner_tensor
        pixel_values = tensor.unsqueeze(0).to(self.device, dtype=self.dtype)

        latents = encode_images(self.vae, pixel_values).to(self.device, dtype=self.dtype)
        # Use a non-zero noise timestep for pointer extraction. t=0 often collapses to near-flat maps.
        timesteps = torch.full(
            (latents.size(0),),
            float(self.config.pointer_noise_timestep),
            dtype=torch.float32,
            device=self.device,
        )
        latents_for_pointer = latents
        if hasattr(self.scheduler, "scale_noise"):
            noise = torch.randn_like(latents)
            latents_for_pointer = self.scheduler.scale_noise(
                latents,
                timesteps,
                noise=noise,
            ).to(device=self.device, dtype=self.dtype)

        embeds = encode_prompts(
            [pointer_prompt],
            self.tokenizers,
            self.text_encoders,
            device=self.device,
            dtype=self.dtype,
        )
        self.pointer_cache.reset()
        with torch.no_grad():
            predict_noise(
                self.transformer,
                self.scheduler,
                latents_for_pointer,
                timesteps,
                embeds,
            )
        token_masks = self._build_token_masks([pointer_prompt])
        pointer_map, _ = self._compose_pointer_heatmap(token_masks)
        map_tensor = pointer_map.squeeze().detach().to(torch.float32)
        if map_tensor.ndim == 3:
            map_tensor = map_tensor[0]
        self._last_peak_info = self._extract_peak_info(map_tensor)
        self._last_pointer_map = pointer_map.detach().clone()
        self.pointer_cache.reset()
        self._restore_pointer_processors()
        return pointer_map.detach()

    def build_refined_mask(self) -> torch.Tensor:
        if self._last_peak_info is None:
            raise RuntimeError("尚未计算 pointer heatmap，无法生成 refined mask。")
        coarse_hint = self._build_gaussian_hint(self._last_peak_info)
        # If refiner weights are not available, fall back to a soft region cue derived from the pointer heatmap.
        # This is generally more informative than using a pure gaussian blob.
        if self.refiner is None:
            if self._last_pointer_map is None:
                refined = coarse_hint.detach().clone()
            else:
                heat = self._last_pointer_map.squeeze().detach().to(torch.float32)
                if heat.ndim == 3:
                    heat = heat[0]
                heat = heat.clamp(0.0, 1.0)
                soft = (heat * coarse_hint).clamp(0.0, 1.0)
                soft = soft / soft.max().clamp(min=1e-6)
                refined = soft
        else:
            refined = self._run_mask_refiner(coarse_hint)
        self._last_refined_mask = refined.clone()
        return refined

    def build_gaussian_mask(self) -> torch.Tensor:
        """
        Build a pure gaussian hint mask centered at the pointer peak.

        This is a legacy/debug option and does NOT run any refiner.
        """
        if self._last_peak_info is None:
            raise RuntimeError("尚未计算 pointer heatmap，无法生成 gaussian mask。")
        mask = self._build_gaussian_hint(self._last_peak_info)
        self._last_refined_mask = mask.clone()
        self._last_coarse_hint = mask.clone()
        return mask

    def build_peak_region_mask(
        self,
        threshold: float = 0.5,
        threshold_mode: str = "rel",
        connectivity: int = 8,
        max_iters: int = 2048,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Build a soft mask by growing a connected region from the peak on the pointer heatmap.

        Steps:
        - Find the peak location (already cached in `_last_peak_info`).
        - Define candidate pixels by thresholding the heatmap.
        - Grow the connected component starting from the peak within the candidate pixels.
        - Return `heatmap * component_mask` as a soft mask (optionally normalized to max=1).

        Args:
            threshold: In "rel" mode, it's a ratio of peak value (0..1). In "abs" mode, it's an absolute heat threshold (0..1).
            threshold_mode: "rel" or "abs".
            connectivity: 4 or 8-neighborhood connectivity.
            max_iters: Safety upper bound for region growing iterations.
            normalize: If True, normalize the output soft mask to have max=1 (when non-empty).
        """
        if self._last_peak_info is None:
            raise RuntimeError("尚未计算 pointer heatmap，无法生成 peak-region mask。")
        if self._last_pointer_map is None:
            raise RuntimeError("pointer heatmap 缓存为空，无法生成 peak-region mask。")

        heat = self._last_pointer_map.squeeze().detach().to(torch.float32)
        if heat.ndim == 3:
            heat = heat[0]
        heat = heat.clamp(0.0, 1.0)

        peak_x, peak_y = self._last_peak_info["map_index"]
        peak_x = int(peak_x)
        peak_y = int(peak_y)
        if not (0 <= peak_y < heat.shape[0] and 0 <= peak_x < heat.shape[1]):
            raise RuntimeError("peak 坐标超出 heatmap 范围。")

        peak_val = float(heat[peak_y, peak_x].item())
        mode = str(threshold_mode).lower().strip()
        if mode not in ("rel", "abs"):
            raise ValueError("threshold_mode must be 'rel' or 'abs'.")
        if mode == "rel":
            thr = float(peak_val * float(threshold))
        else:
            thr = float(threshold)
        thr = float(max(0.0, min(1.0, thr)))

        candidates = (heat >= thr)

        # Seed at peak; if peak itself is below threshold (can happen when thr=abs and too high), force include it.
        if not bool(candidates[peak_y, peak_x].item()):
            candidates = candidates.clone()
            candidates[peak_y, peak_x] = True

        # Region grow via iterative dilation + intersection (GPU-friendly).
        comp = torch.zeros_like(heat, dtype=torch.float32)
        comp[peak_y, peak_x] = 1.0
        comp4 = comp.unsqueeze(0).unsqueeze(0)
        cand4 = candidates.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        if int(connectivity) == 4:
            kernel = torch.tensor(
                [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
                dtype=torch.float32,
                device=comp4.device,
            ).view(1, 1, 3, 3)
            def dilate(x: torch.Tensor) -> torch.Tensor:
                return (F.conv2d(x, kernel, padding=1) > 0.0).to(x.dtype)
        else:
            def dilate(x: torch.Tensor) -> torch.Tensor:
                return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

        last = comp4
        for _ in range(int(max_iters)):
            grown = dilate(last)
            grown = (grown * cand4).clamp(0.0, 1.0)
            if torch.equal((grown > 0.5), (last > 0.5)):
                last = grown
                break
            last = grown

        component = (last[0, 0] > 0.5)
        self._last_coarse_hint = component.to(torch.float32).clone()

        soft = (heat * component.to(dtype=heat.dtype)).clamp(0.0, 1.0)
        if normalize:
            m = soft.max().clamp(min=1e-6)
            soft = (soft / m).clamp(0.0, 1.0)
        self._last_refined_mask = soft.clone()
        return soft

    def pointer_map_to_grayscale(self, pointer_map: torch.Tensor, size: Optional[Tuple[int, int]] = None) -> Image.Image:
        arr = pointer_map.squeeze().detach().cpu().to(torch.float32).numpy()
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
        if size is not None:
            img = img.resize(size, Image.BILINEAR)
        return img

    def pointer_map_overlay(self, pointer_map: torch.Tensor, base_image: Image.Image, alpha: float = 0.6) -> Image.Image:
        # pointer_map lives in letterboxed square space (base_resolution x base_resolution).
        # For correct visualization on the original image, recover aspect ratio first.
        recovered = self._recover_square_map_to_original(pointer_map.squeeze().detach())
        mask = self.pointer_map_to_grayscale(recovered, base_image.size)
        heat = ImageOps.colorize(mask, black="#00000000", white="#ff0000")
        heat = heat.convert("RGBA")
        base_rgba = base_image.convert("RGBA")
        blended = Image.blend(base_rgba, heat, alpha=alpha)
        return blended.convert("RGB")

    def _build_gaussian_hint(self, peak_info: Dict[str, Any]) -> torch.Tensor:
        target = self.config.base_resolution
        cx = peak_info["map_norm"][0] * target
        cy = peak_info["map_norm"][1] * target
        sigma = max(2.0, self._refiner_sigma_ratio * target)
        xs = np.arange(target, dtype=np.float32)
        ys = np.arange(target, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        gaussian = np.exp(-dist_sq / (2.0 * sigma ** 2))
        tensor = torch.from_numpy(gaussian).to(torch.float32)
        if self._refiner_dilate_iters > 0:
            dil = tensor.unsqueeze(0).unsqueeze(0)
            for _ in range(self._refiner_dilate_iters):
                dil = F.max_pool2d(dil, kernel_size=3, stride=1, padding=1)
            tensor = dil.squeeze(0).squeeze(0)
        tensor = tensor / tensor.max().clamp(min=1e-6)
        return tensor.clamp(0.0, 1.0)

    def _run_mask_refiner(self, coarse_hint: torch.Tensor) -> torch.Tensor:
        self._last_coarse_hint = coarse_hint.detach().clone()
        if self.refiner is None or self._refiner_input_tensor is None:
            return coarse_hint.detach().clone()
        if coarse_hint.ndim != 2:
            raise ValueError("coarse_hint 需要是 2D 张量")
        image_tensor = self._refiner_input_tensor
        if image_tensor.ndim != 3:
            raise ValueError("refiner 输入图像缓存异常")
        if image_tensor.shape[1:] != coarse_hint.shape:
            raise ValueError("refiner 输入尺寸与 coarse hint 不一致")
        channels = [image_tensor]
        channels.append(coarse_hint.unsqueeze(0))
        stacked = torch.cat(channels, dim=0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.refiner(stacked)
            probs = torch.sigmoid(logits)[0, 0].detach().cpu().to(torch.float32)
        self._last_refined_mask = probs.clone()
        return probs

    def _recover_mask_to_original(self, mask_tensor: torch.Tensor) -> torch.Tensor:
        mask = mask_tensor.unsqueeze(0).unsqueeze(0)
        meta = getattr(self, "_guidance_resize_meta", None)
        if meta:
            orig_height = int(meta["orig_height"])
            orig_width = int(meta["orig_width"])
            # If already at original size, skip cropping/resizing to avoid zeroing masks from external refiner.
            if mask.shape[-2] == orig_height and mask.shape[-1] == orig_width:
                return mask.squeeze(0).squeeze(0).clamp(0.0, 1.0)
            pad_top = int(meta["pad_top"])
            pad_left = int(meta["pad_left"])
            resized_height = int(meta["resized_height"])
            resized_width = int(meta["resized_width"])
            mask = mask[:, :, pad_top : pad_top + resized_height, pad_left : pad_left + resized_width]
            mask = F.interpolate(mask, size=(orig_height, orig_width), mode="bilinear", align_corners=False)
        return mask.squeeze(0).squeeze(0).clamp(0.0, 1.0)

    def _recover_square_map_to_original(self, map_tensor: torch.Tensor) -> torch.Tensor:
        """
        Recover a square (letterboxed) heatmap/mask back to the original image aspect ratio.

        Preprocess does: resize to fit within target_size square + pad (letterbox).
        Pointer heatmaps live in that square space. For visualization we should:
        1) crop away padding, 2) resize back to (orig_height, orig_width).
        """
        m = map_tensor
        if m.ndim == 3:
            # allow (1, H, W) or (C, H, W); pick first channel
            m = m[0]
        if m.ndim != 2:
            raise ValueError("map_tensor must be 2D (H,W) or 3D (C,H,W).")
        m = m.unsqueeze(0).unsqueeze(0).to(torch.float32)
        meta = getattr(self, "_guidance_resize_meta", None)
        if meta:
            pad_top = int(meta["pad_top"])
            pad_left = int(meta["pad_left"])
            resized_height = int(meta["resized_height"])
            resized_width = int(meta["resized_width"])
            m = m[:, :, pad_top : pad_top + resized_height, pad_left : pad_left + resized_width]
            orig_height = int(meta["orig_height"])
            orig_width = int(meta["orig_width"])
            m = F.interpolate(m, size=(orig_height, orig_width), mode="bilinear", align_corners=False)
        return m.squeeze(0).squeeze(0).clamp(0.0, 1.0)

    def _resize_mask_for_edit(self, mask_tensor: torch.Tensor, width: int, height: int) -> torch.Tensor:
        mask_orig = self._recover_mask_to_original(mask_tensor)
        mask = F.interpolate(
            mask_orig.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return mask.squeeze(0).squeeze(0).clamp(0.0, 1.0)

    def _mask_tensor_to_image(self, mask_tensor: torch.Tensor, size: Optional[Tuple[int, int]] = None) -> Image.Image:
        arr = mask_tensor.detach().cpu().numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
        if size is not None and img.size != size:
            img = img.resize(size, Image.BILINEAR)
        return img

    @staticmethod
    def _bbox_from_mask(mask_2d: torch.Tensor, threshold: float = 0.5) -> Optional[Tuple[int, int, int, int]]:
        """
        Compute tight bbox (x0,y0,x1,y1) from a 2D mask tensor.
        - mask_2d: [H,W] in [0,1]
        - returns None if mask is empty
        """
        if mask_2d.ndim != 2:
            raise ValueError("mask_2d must be 2D (H,W).")
        m = (mask_2d.detach().to(torch.float32) >= float(threshold)).cpu()
        ys, xs = torch.where(m)
        if ys.numel() == 0:
            return None
        y0 = int(ys.min().item())
        y1 = int(ys.max().item())
        x0 = int(xs.min().item())
        x1 = int(xs.max().item())
        return x0, y0, x1, y1

    def get_last_peak_hint(self) -> Optional[str]:
        info = getattr(self, "_last_peak_info", None)
        if not info:
            return None
        x_norm, y_norm = info["norm"]
        return f"Target person is around {x_norm * 100:.1f}% width and {y_norm * 100:.1f}% height of the image."

    def get_last_peak_info(self) -> Optional[Dict[str, Any]]:
        if self._last_peak_info is None:
            return None
        return dict(self._last_peak_info)

    # ------------------------------------------------------------------ ClickSEG refiner
    def _load_clickseg_model(self, checkpoint: str):
        """
        Lazy-load ClickSEG checkpoint (isegm HRNet) and return (model, device).
        """
        if checkpoint in self._clickseg_cache:
            return self._clickseg_cache[checkpoint]
        # Import ClickSEG dependencies
        try:
            _third_party = str(Path(__file__).resolve().parent.parent / "third_party")
            if _third_party not in sys.path:
                sys.path.insert(0, _third_party)
            from isegm.utils.serialization import load_model  # type: ignore
        except Exception as e:
            raise ImportError(f"Cannot import ClickSEG; please ensure third_party/isegm is available: {e}")

        ckpt = torch.load(checkpoint, map_location="cpu")
        model = load_model(ckpt["config"])
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._clickseg_cache[checkpoint] = model
        return model

    def build_clickseg_mask(
        self,
        image_path: str,
        clickseg_checkpoint: str,
        clickseg_threshold: float = 0.4,
        clickseg_filter_component: bool = True,
        clickseg_use_prev_mask: bool = False,
        clickseg_infer_size: int = 384,
        clickseg_prev_mask_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Use ClickSEG (isegm HRNet) with a single positive click at peak to produce a binary mask.
        - clickseg_use_prev_mask=False (default) means prev_mask = zeros (pure click).
        - If filter_component=True, keep only the connected component containing the click.
        Returns: torch.Tensor [H,W] float in {0,1}.
        """
        peak = self.get_last_peak_info()
        if not peak or "pixel" not in peak:
            raise RuntimeError("ClickSEG refiner requires peak info; call compute_pointer_map() first.")

        px, py = peak["pixel"]
        px_i, py_i = int(px), int(py)

        img = Image.open(image_path).convert("RGB")
        W, H = img.size

        # prev_mask: either zeros or resized refined mask if provided
        if clickseg_use_prev_mask and self._last_refined_mask is not None:
            prev = self._resize_mask_for_edit(self._last_refined_mask, W, H).detach().cpu().numpy().astype(np.float32)
            if clickseg_prev_mask_scale != 1.0:
                prev = np.clip(prev * float(clickseg_prev_mask_scale), 0.0, 1.0)
        else:
            prev = np.zeros((H, W), dtype=np.float32)

        # Load model and predictor
        model = self._load_clickseg_model(clickseg_checkpoint)
        # Import predictor utilities
        from isegm.inference.predictors import get_predictor  # type: ignore
        from isegm.inference.clicker import Clicker, Click  # type: ignore

        predictor = get_predictor(
            model,
            brs_mode="Baseline",
            device=self.device,
            infer_size=clickseg_infer_size,
            with_flip=False,
            zoom_in_params=None,
        )
        predictor.set_input_image(img)
        predictor.set_prev_mask(prev)

        clicker = Clicker()
        clicker.add_click(Click(is_positive=True, coords=(py_i, px_i)))
        pred = predictor.get_prediction(clicker)  # numpy [H,W] 0..1

        mask = (pred > float(clickseg_threshold)).astype(np.uint8)
        if clickseg_filter_component:
            import cv2

            num, labels = cv2.connectedComponents(mask)
            label_at_click = labels[py_i, px_i] if 0 <= py_i < H and 0 <= px_i < W else 0
            if label_at_click != 0:
                mask = (labels == label_at_click).astype(np.uint8)
            else:
                mask = np.zeros_like(mask, dtype=np.uint8)

        # debug stats
        nonzero = int(mask.sum())
        print(
            f"[clickseg] mask stats: shape={mask.shape}, sum={nonzero}, "
            f"threshold={clickseg_threshold}, filter_component={clickseg_filter_component}, "
            f"click=({px_i},{py_i}), image={image_path}, prev_used={clickseg_use_prev_mask}, prev_scale={clickseg_prev_mask_scale}"
        )
        mask_t = torch.from_numpy(mask.astype(np.float32)).to(self.device)
        return mask_t

    def _extract_peak_info(self, map_tensor: torch.Tensor) -> Dict[str, Any]:
        if map_tensor.ndim != 2:
            raise ValueError("map_tensor must be 2D after squeeze.")
        height, width = map_tensor.shape
        flat_index = torch.argmax(map_tensor)
        peak_y = (flat_index // width).item()
        peak_x = (flat_index % width).item()
        map_x_norm = (peak_x + 0.5) / width
        map_y_norm = (peak_y + 0.5) / height

        info: Dict[str, Any] = {
            "map_norm": (float(map_x_norm), float(map_y_norm)),
            "map_index": (int(peak_x), int(peak_y)),
            "map_size": (int(width), int(height)),
        }

        meta = getattr(self, "_guidance_resize_meta", None)
        if meta:
            target_size = meta["target_size"]
            pad_left = meta["pad_left"]
            pad_top = meta["pad_top"]
            resized_width = meta["resized_width"]
            resized_height = meta["resized_height"]
            scale_w = meta["scale_width"]
            scale_h = meta["scale_height"]
            orig_width = meta["orig_width"]
            orig_height = meta["orig_height"]

            peak_x_px = map_x_norm * target_size
            peak_y_px = map_y_norm * target_size
            inner_x = float(torch.clamp(torch.tensor(peak_x_px - pad_left), 0.0, resized_width))
            inner_y = float(torch.clamp(torch.tensor(peak_y_px - pad_top), 0.0, resized_height))
            orig_x = inner_x / max(scale_w, 1e-6)
            orig_y = inner_y / max(scale_h, 1e-6)
            orig_x_norm = float(orig_x / max(orig_width, 1e-6))
            orig_y_norm = float(orig_y / max(orig_height, 1e-6))
            info["norm"] = (
                float(torch.clamp(torch.tensor(orig_x_norm), 0.0, 1.0).item()),
                float(torch.clamp(torch.tensor(orig_y_norm), 0.0, 1.0).item()),
            )
            info["pixel"] = (float(orig_x), float(orig_y))
            info["image_size"] = (float(orig_width), float(orig_height))
        else:
            info["norm"] = (float(map_x_norm), float(map_y_norm))
            info["pixel"] = (
                float(map_x_norm * width),
                float(map_y_norm * height),
            )
            info["image_size"] = (float(width), float(height))

        return info

    def _preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        image = Image.open(image_path).convert("RGB")
        tensor_linear = TF.to_tensor(image)
        tensor = tensor_linear * 2.0 - 1.0
        orig_height = tensor.shape[1]
        orig_width = tensor.shape[2]
        target_size = self.config.base_resolution
        scale = target_size / max(orig_height, orig_width)
        resized_height = max(1, int(round(orig_height * scale)))
        resized_width = max(1, int(round(orig_width * scale)))

        tensor = tensor.unsqueeze(0)
        tensor_linear = tensor_linear.unsqueeze(0)
        resized = F.interpolate(tensor, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
        resized_linear = F.interpolate(
            tensor_linear, size=(resized_height, resized_width), mode="bilinear", align_corners=False
        )

        pad_top = (target_size - resized_height) // 2
        pad_bottom = target_size - resized_height - pad_top
        pad_left = (target_size - resized_width) // 2
        pad_right = target_size - resized_width - pad_left
        padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom))
        padded_linear = F.pad(resized_linear, (pad_left, pad_right, pad_top, pad_bottom))

        meta = {
            "orig_height": float(orig_height),
            "orig_width": float(orig_width),
            "target_size": float(target_size),
            "resized_height": float(resized_height),
            "resized_width": float(resized_width),
            "pad_top": float(pad_top),
            "pad_left": float(pad_left),
            "scale_height": float(resized_height / max(orig_height, 1)),
            "scale_width": float(resized_width / max(orig_width, 1)),
        }
        return padded.squeeze(0), meta, padded_linear.squeeze(0)

    def offload_kontext(self, destroy: bool = True) -> None:
        """Move Kontext-related modules to CPU (or drop them) to free GPU memory before Qwen."""
        try:
            self.transformer.to("cpu")
        except Exception:
            pass
        try:
            self.vae.to("cpu")
        except Exception:
            pass
        try:
            for enc in self.text_encoders:
                enc.to("cpu")
        except Exception:
            pass
        try:
            self.pipeline.to("cpu")
        except Exception:
            pass
        try:
            if self.inpaint_pipeline is not None:
                self.inpaint_pipeline.to("cpu")
        except Exception:
            pass
        if destroy:
            for attr in ("pipeline", "inpaint_pipeline", "transformer", "vae", "scheduler", "text_encoders", "tokenizers"):
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass
            try:
                self._pointer_modules = {}
                self._original_processors = {}
            except Exception:
                pass
            try:
                self.pointer_cache.reset()
            except Exception:
                pass
            try:
                import gc

                gc.collect()
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_qwen_backend(self, mode: str):
        mode = str(mode).strip().lower()
        if mode not in self._qwen_backends:
            from .qwen_inpaint_backend import QwenInpaintBackend

            self._qwen_backends[mode] = QwenInpaintBackend(
                device=self.device,
                dtype=self.dtype,
                mode=mode,
                controlnet_path=self._qwen_controlnet_path,
            )
        return self._qwen_backends[mode]

    # ------------------------------------------------------------------ Editing helpers (soft latent blending)
    def _encode_vae_image_argmax(self, image: Image.Image) -> torch.Tensor:
        """Encode image into VAE latents using argmax/mode (matches diffusers FluxKontextPipeline behavior)."""
        tensor = TF.to_tensor(image) * 2.0 - 1.0  # [-1,1]
        pixel_values = tensor.unsqueeze(0).to(self.device, dtype=self.dtype)
        with torch.no_grad():
            enc = self.vae.encode(pixel_values)
            latents = enc.latent_dist.mode()
        # Match diffusers: (latents - shift) * scaling
        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return latents

    @staticmethod
    def _pack_latents_flux(latents: torch.Tensor) -> torch.Tensor:
        """Pack [B,C,H,W] latents into Flux packed format [B,(H/2)*(W/2),C*4]."""
        b, c, h, w = latents.shape
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError("Flux latent packing requires even H and W.")
        latents = latents.view(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(b, (h // 2) * (w // 2), c * 4)
        return latents

    @staticmethod
    def _make_masked_condition_image(
        base_img: Image.Image,
        mask_l: Image.Image,
        blur_radius: float = 8.0,
        fill: str = "noise",
        noise_strength: float = 1.0,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Create a masked conditioning image for inpainting-style editing.

        This is the classic diffusion inpainting trick: provide a *masked image* (hole) so the model cannot copy
        original content inside the editable region.

        - `mask_l`: white = editable region, black = keep region
        - `fill`: "noise" or "gray"
        - `noise_strength`: when fill="noise", scales noise amplitude (0..1 recommended)
        - `blur_radius`: feather mask edge to allow slight spill
        """
        base = base_img.convert("RGB")
        m = mask_l.convert("L")
        if blur_radius and blur_radius > 0:
            from PIL import ImageFilter

            m = m.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
        m_arr = np.array(m, dtype=np.float32) / 255.0
        m_arr = np.clip(m_arr, 0.0, 1.0)

        base_arr = np.array(base, dtype=np.float32)
        if str(fill).lower() == "gray":
            fill_arr = np.full_like(base_arr, 127.0)
        else:
            rng = np.random.default_rng(int(seed) if seed is not None else None)
            noise = rng.standard_normal(size=base_arr.shape).astype(np.float32)
            ns = float(noise_strength)
            ns = max(0.0, min(2.0, ns))
            fill_arr = np.clip(127.0 + noise * 50.0 * ns, 0.0, 255.0)

        # composite: out = base*(1-m) + fill*m
        m3 = m_arr[..., None]
        out = base_arr * (1.0 - m3) + fill_arr * m3
        out = np.clip(out, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(out, mode="RGB")

    def _mask_to_flux_token_weights(
        self,
        mask_for_edit: torch.Tensor,
        latent_h: int,
        latent_w: int,
    ) -> torch.Tensor:
        """
        Convert a [H_img,W_img] soft mask into Flux packed-token weights [num_patches].
        Each packed token corresponds to a 2x2 patch in latent space.
        """
        m = mask_for_edit.detach().to(device=self.device, dtype=torch.float32)
        if m.ndim != 2:
            raise ValueError("mask_for_edit must be 2D.")
        m = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(latent_h, latent_w), mode="bilinear", align_corners=False)[0, 0]
        # average pool 2x2 to match packed tokens grid
        m2 = F.avg_pool2d(m.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=2)[0, 0]
        return m2.reshape(-1).clamp(0.0, 1.0)

    def generate_control_image(
        self,
        pointer_map: torch.Tensor,
        width: int,
        height: int,
    ) -> Image.Image:
        gray = self.pointer_map_to_grayscale(pointer_map, size=(width, height))
        return Image.merge("RGB", (gray, gray, gray))

    def edit_with_soft_mask(
        self,
        refined_mask: torch.Tensor,
        source_image_path: str,
        edit_prompt: str,
        width: int,
        height: int,
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
        true_cfg_scale: float = 1.0,
        seed: Optional[int] = 42,
        negative_prompt: Optional[str] = None,
        latent_inpaint_strength: float = 1.0,
        mask_feather: float = 6.0,
        mask_dilate: int = 0,
        latent_blend_alpha: float = 0.6,
        conditioning_masked_image: bool = True,
        conditioning_fill: str = "noise",
        conditioning_noise_strength: float = 1.0,
        use_inpaint_pipeline: bool = False,
        padding_mask_crop: Optional[int] = None,
        inpaint_auto_resize: bool = False,
        append_bbox_hint: bool = True,
        bbox_threshold: float = 0.5,
        edit_backend: str = "kontext",
        qwen_strength: Optional[float] = None,
        qwen_mode: str = "inpaint",
        controlnet_conditioning_scale: Optional[float] = None,
    ) -> Image.Image:
        base_img = Image.open(source_image_path).convert("RGB")
        if base_img.size != (width, height):
            base_img = base_img.resize((width, height), Image.BILINEAR)

        mask_for_edit = self._resize_mask_for_edit(refined_mask, width, height)

        # Kontext inpaint uses `mask_image` as a *where-to-edit* hard gate, not a *who-to-edit* selector.
        # If the prompt points to the "wrong" instance, the model often refuses to change anything inside the mask.
        #
        # We try to make behavior closer to classic inpainting by explicitly telling the model where the editable
        # subject is, using bbox coordinates derived from the mask itself.
        # (Prompts are kept in English by design.)
        if bool(append_bbox_hint):
            bbox = self._bbox_from_mask(mask_for_edit, threshold=bbox_threshold)
            if bbox is not None:
                x0, y0, x1, y1 = bbox
                x0n = (x0 + 0.5) / max(1, int(width))
                x1n = (x1 + 0.5) / max(1, int(width))
                y0n = (y0 + 0.5) / max(1, int(height))
                y1n = (y1 + 0.5) / max(1, int(height))
                edit_prompt = (
                    str(edit_prompt).rstrip()
                    + "\n"
                    + "Edit only the masked region.\n"
                    + "The person to edit is inside the bounding box: "
                    + f"x=[{x0},{x1}] y=[{y0},{y1}] pixels; "
                    + f"x=[{x0n*100:.1f}%,{x1n*100:.1f}%] y=[{y0n*100:.1f}%,{y1n*100:.1f}%]."
                )

        backend = str(edit_backend).strip().lower()
        if backend == "qwen":
            mode = str(qwen_mode).strip().lower()
            qwen_backend = self._get_qwen_backend(mode)
            mask_img = self._mask_tensor_to_image(mask_for_edit, size=(width, height)).convert("L")
            if int(mask_dilate) > 0:
                from PIL import ImageFilter

                k = max(3, 2 * int(mask_dilate) + 1)
                mask_img = mask_img.filter(ImageFilter.MaxFilter(size=k))
            if float(mask_feather) > 0:
                from PIL import ImageFilter

                mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=float(mask_feather)))
            if mode == "edit":
                full_edit = qwen_backend.run_edit(
                    image_pil=base_img,
                    prompt=edit_prompt,
                    seed=seed,
                    steps=num_inference_steps,
                    guidance=guidance_scale,
                    out_size=(width, height),
                    negative_prompt=negative_prompt,
                    true_cfg_scale=true_cfg_scale,
                )
                self._last_qwen_full_edit = full_edit.copy()
                composite = Image.composite(full_edit, base_img, mask_img)
                self._last_qwen_mask_used = mask_img.copy()
                self._last_qwen_image_resized = qwen_backend.last_image_resized
                return composite
            edited = qwen_backend.run_inpaint(
                image_pil=base_img,
                mask_pil=mask_img,
                prompt=edit_prompt,
                seed=seed,
                steps=num_inference_steps,
                guidance=guidance_scale,
                out_size=(width, height),
                negative_prompt=negative_prompt,
                true_cfg_scale=true_cfg_scale,
                strength=qwen_strength,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
            )
            self._last_qwen_mask_used = qwen_backend.last_mask_used
            self._last_qwen_image_resized = qwen_backend.last_image_resized
            return edited
        raise ValueError(
            "Kontext/Flux backend has been removed. Please use `edit_backend='qwen'`."
        )


    def edit_image_inpaint(
        self,
        image: Image.Image,
        mask_image_l: Image.Image,
        edit_prompt: str,
        width: int,
        height: int,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        true_cfg_scale: float = 1.0,
        seed: Optional[int] = 42,
        negative_prompt: Optional[str] = None,
        strength: float = 1.0,
        mask_feather: float = 2.0,
        padding_mask_crop: Optional[int] = None,
    ) -> Image.Image:
        """
        Explicit mask-gated editing using FluxKontextInpaintPipeline on a provided (image, mask).
        Intended for crop+inpaint workflows where the crop isolates the target subject.
        """
        raise RuntimeError("Kontext/Flux inpaint path has been removed. Use Qwen backend instead.")

    def edit_image(
        self,
        image: Image.Image,
        edit_prompt: str,
        width: int,
        height: int,
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
        true_cfg_scale: float = 1.0,
        seed: Optional[int] = 42,
        negative_prompt: Optional[str] = None,
    ) -> Image.Image:
        """
        Plain Kontext edit without any mask logic. Useful when we crop the image to isolate the target.
        """
        raise RuntimeError("Kontext/Flux edit path has been removed. Use Qwen backend instead.")

    def save_pointer_artifacts(
        self,
        pointer_map: torch.Tensor,
        refined_mask: torch.Tensor,
        source_image_path: str,
        output_dir: str | Path,
        prefix: str,
    ) -> Dict[str, str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_image = Image.open(source_image_path).convert("RGB")
        heat_overlay = self.pointer_map_overlay(pointer_map, base_image)

        peak_info = self.get_last_peak_info() or {}
        peak_img = heat_overlay.copy()
        if "pixel" in peak_info and "image_size" in peak_info:
            from PIL import ImageDraw

            draw = ImageDraw.Draw(peak_img)
            px, py = peak_info["pixel"]
            r = max(3, int(min(peak_img.size) * 0.01))
            draw.ellipse([px - r, py - r, px + r, py + r], outline="cyan", width=3)

        refined_orig = self._recover_mask_to_original(refined_mask)
        refined_img = self._mask_tensor_to_image(refined_orig)

        heat_path = output_dir / f"{prefix}_heatmap_peak.png"
        refined_path = output_dir / f"{prefix}_mask_refined.png"
        peak_meta_path = output_dir / f"{prefix}_peak.json"

        peak_img.save(heat_path)
        refined_img.save(refined_path)

        if peak_info:
            with open(peak_meta_path, "w", encoding="utf-8") as fh:
                json.dump(peak_info, fh, ensure_ascii=False, indent=2)

        return {
            "heatmap_peak": str(heat_path),
            "refined_mask": str(refined_path),
            "peak_meta": str(peak_meta_path),
        }

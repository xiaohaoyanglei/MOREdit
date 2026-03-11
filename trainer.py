"""Pointer LoRA trainer built on top of Qwen image components."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from PIL import Image, ImageFile

from .dataset import PointerTripletDataset
from .losses import PointerLossConfig, compute_pointer_losses
from .lora import LoRAInjectionConfig, inject_lora_into_attention, LoRALinear, is_qwen_joint_attention
from .model import PromptEmbeds, encode_images, encode_prompts, load_qwen_components, predict_noise
from .pointer_recorder import PointerCache, PointerRecorder


@dataclass
class PointerTrainerConfig:
    data_path: str
    data_root: str
    model_path: str
    output_dir: str
    resolution: int = 640
    batch_size: int = 1
    num_steps: int = 10000
    lr: float = 1e-4
    weight_decay: float = 0.0
    seed: int = 42
    log_every: int = 10
    heatmap_every: int = 1000
    heatmap_max_samples: int = 2
    checkpoint_every: int = 1000
    pointer_superres_factor: int = 1
    pointer_superres_sharpness: float = 0.0
    dtype: str = "float32"
    offload_text_encoders: bool = False
    test_image_path: Optional[str] = None
    test_mask_path: Optional[str] = None
    test_prompt: Optional[str] = None
    pointer_debug_every: int = 0
    pointer_token_mode: str = "phrase"
    pointer_token_filters: Tuple[str, ...] = field(default_factory=tuple)
    pointer_heatmap_combine: str = "mean"
    pointer_temperature: float = 1.0
    pointer_epsilon: float = 1.0e-6


def _to_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name.lower() in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_name.lower() in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


class PointerTrainer:
    def __init__(
        self,
        config: PointerTrainerConfig,
        pointer_loss: PointerLossConfig,
        lora_config: LoRAInjectionConfig,
    ) -> None:
        self.config = config
        self.pointer_loss = pointer_loss
        self.lora_config = lora_config

        torch.manual_seed(config.seed)

        if self.config.pointer_heatmap_combine not in {"mean", "max"}:
            raise ValueError("pointer_heatmap_combine must be 'mean' or 'max'.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = _to_dtype(config.dtype)

        dataset = PointerTripletDataset(
            triples_path=config.data_path,
            root_dir=config.data_root,
            resolution=config.resolution,
            pad_to_square=True,
        )
        self.dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, drop_last=True)
        self.data_iter = iter(self.dataloader)

        (
            self.transformer,
            self.text_encoders,
            self.tokenizers,
            self.vae,
            self.scheduler,
        ) = load_qwen_components(
            config.model_path,
            device=self.device,
            dtype=self.dtype,
            offload_text_encoders=config.offload_text_encoders,
        )

        self.lora_modules: List[LoRALinear] = inject_lora_into_attention(
            self.transformer,
            self.lora_config,
        )
        lora_params = []
        for module in self.lora_modules:
            for param in module.lora_parameters:
                lora_params.append(param)

        self.optimizer = torch.optim.AdamW(lora_params, lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.999))

        self.pointer_cache = PointerCache()
        self._attach_pointer_recorders()

        os.makedirs(config.output_dir, exist_ok=True)
        self._init_run_dirs()

        self._pointer_debug_enabled = self.config.pointer_debug_every > 0
        self._pointer_debug_active = False
        self._pointer_slice_stats: Dict[str, Tuple[Tuple[int, ...], int]] = {}
        self._current_prompts: List[str] = []
        self._superres_factor = max(1, int(self.config.pointer_superres_factor))
        self._superres_sharpness = max(0.0, float(self.config.pointer_superres_sharpness))
        self._laplacian_kernel = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        self.test_sample = None
        if self.config.test_image_path:
            self.test_sample = self._prepare_test_sample(
                self.config.test_image_path,
                self.config.test_mask_path,
                self.config.test_prompt,
            )

    def _attach_pointer_recorders(self):
        self._original_processors: Dict[str, torch.nn.Module] = {}
        for name, module in self.transformer.named_modules():
            if not is_qwen_joint_attention(module):
                continue
            if not any(name.startswith(prefix) for prefix in self.lora_config.attn_block_prefixes):
                continue
            recorder = PointerRecorder(owner=self, layer_key=name)
            self._original_processors[name] = module.processor
            module.set_processor(recorder)

    def record_pointer(self, key: str, tensor: torch.Tensor):
        self.pointer_cache.append(key, tensor)

    def _init_run_dirs(self) -> None:
        base_output = os.path.abspath(self.config.output_dir)
        os.makedirs(base_output, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        candidate = os.path.join(base_output, timestamp)
        suffix = 1
        while os.path.exists(candidate):
            suffix += 1
            candidate = os.path.join(base_output, f"{timestamp}_{suffix:02d}")

        self.run_id = os.path.basename(candidate)
        self.run_dir = candidate
        self.heatmap_dir = os.path.join(self.run_dir, "heatmaps")
        self.sample_dir = os.path.join(self.run_dir, "samples")
        self.weights_dir = os.path.join(self.run_dir, "weights")

        for path in (self.run_dir, self.heatmap_dir, self.sample_dir, self.weights_dir):
            os.makedirs(path, exist_ok=True)

        metadata = {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "started_at": datetime.now().isoformat(),
            "trainer_config": asdict(self.config),
            "lora_config": asdict(self.lora_config),
            "pointer_loss": asdict(self.pointer_loss),
            "device": str(self.device),
            "dtype": str(self.dtype),
        }
        meta_path = os.path.join(self.run_dir, "run_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

    def _next_batch(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        return batch

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
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "sixth",
            "seventh",
            "eighth",
            "ninth",
            "tenth",
            "left",
            "right",
            "middle",
            "center",
            "centre",
            "front",
            "back",
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
                    f"[pointer-trainer] Warning: no pointer tokens matched for prompt={prompt_preview!r}; "
                    "falling back to full attention mask."
                )
                masks[batch_idx] = attention_mask[batch_idx].bool()

        return masks.to(self.device)

    def _compose_pointer_heatmap(
        self,
        token_masks: torch.Tensor,
        return_layer_sizes: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[str, int]]]]:
        layer_heatmaps: List[torch.Tensor] = []
        layer_sizes: List[Tuple[str, int]] = []
        token_masks = token_masks.to(self.device)

        for layer_key, tensors in self.pointer_cache.items():
            stacked = torch.stack(tensors, dim=0).mean(dim=0)  # (B, heads, tokens_q, tokens_k)
            mask = token_masks.unsqueeze(1).unsqueeze(2).to(stacked.dtype)  # (B, 1, 1, tokens_k_max)
            if mask.shape[-1] != stacked.shape[-1]:
                mask = mask[..., : stacked.shape[-1]]
            masked = stacked * mask
            denom = mask.sum(dim=-1).clamp_min(1.0)
            heat = masked.sum(dim=-1) / denom  # (B, heads, tokens_q)
            heat = heat.mean(dim=1)  # (B, tokens_q)

            spatial_tokens = heat.shape[-1]
            side = int(math.sqrt(spatial_tokens))
            if side * side != spatial_tokens:
                raise RuntimeError(f"Unexpected spatial token count {spatial_tokens} in {layer_key}")

            heat = heat / max(self.config.pointer_temperature, 1e-6)
            heat = heat - heat.amin(dim=-1, keepdim=True)
            heat = heat / (heat.amax(dim=-1, keepdim=True) + self.config.pointer_epsilon)
            heat = heat.view(heat.shape[0], 1, side, side)
            layer_heatmaps.append(heat)
            if return_layer_sizes:
                layer_sizes.append((layer_key, side))

        if not layer_heatmaps:
            raise RuntimeError("Pointer cache is empty; attention recorders may not be attached correctly.")

        stack = torch.stack(layer_heatmaps, dim=0)
        if self.config.pointer_heatmap_combine == "max":
            combined = stack.max(dim=0).values
        else:
            combined = stack.mean(dim=0)

        combined = combined.clamp(0.0, 1.0)

        pointer_map = self._apply_super_resolution(combined)

        pointer_map = F.interpolate(
            pointer_map,
            size=(self.config.resolution, self.config.resolution),
            mode="bilinear",
            align_corners=False,
        )
        pointer_map = pointer_map.clamp(0.0, 1.0)
        return pointer_map, (layer_sizes if return_layer_sizes else None)

    def _prepare_test_sample(
        self,
        image_path: str,
        mask_path: Optional[str],
        prompt: Optional[str],
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not os.path.exists(image_path):
            print(f"[pointer-trainer] Test image not found at {image_path}, skipping test heatmaps.")
            return None

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(image_path).convert("RGB")
        image_tensor = TF.to_tensor(image) * 2.0 - 1.0
        image_tensor = PointerTripletDataset._resize_tensor(
            image_tensor,
            self.config.resolution,
            pad_to_square=True,
            mode="bilinear",
        )

        mask_tensor: Optional[torch.Tensor] = None
        if mask_path:
            if os.path.exists(mask_path):
                mask_img = Image.open(mask_path).convert("L")
                mask_tensor = TF.to_tensor(mask_img)
                mask_tensor = (mask_tensor > 0.5).float()
                mask_tensor = PointerTripletDataset._resize_tensor(
                    mask_tensor,
                    self.config.resolution,
                    pad_to_square=True,
                    mode="nearest",
                )
            else:
                print(f"[pointer-trainer] Test mask not found at {mask_path}, using blank mask.")
        if mask_tensor is None:
            mask_tensor = torch.zeros(1, self.config.resolution, self.config.resolution)

        if not prompt:
            print("[pointer-trainer] No test prompt provided; using empty string.")
            prompt = ""

        return {
            "pixel_values": image_tensor.unsqueeze(0),
            "target_mask": mask_tensor.unsqueeze(0),
            "prompt": prompt,
            "image_path": image_path,
            "mask_path": mask_path,
        }

    def _run_pointer_forward(
        self,
        pixel_values: torch.Tensor,
        prompts: List[str],
    ) -> torch.Tensor:
        pixel_values = pixel_values.to(self.device, dtype=self.dtype)
        latents = encode_images(self.vae, pixel_values).to(self.device, dtype=self.dtype)
        timesteps = torch.zeros((latents.size(0),), dtype=torch.long, device=self.device)

        embeds: PromptEmbeds = encode_prompts(
            prompts,
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
                latents,
                timesteps,
                embeds,
            )
        token_masks = self._build_token_masks(prompts)
        pointer_map, _ = self._compose_pointer_heatmap(token_masks, return_layer_sizes=False)
        return pointer_map

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
            laplacian_kernel = self._laplacian_kernel.to(device=upsampled.device, dtype=upsampled.dtype)
            laplacian = F.conv2d(upsampled, laplacian_kernel, padding=1)
            upsampled = torch.clamp(upsampled + self._superres_sharpness * laplacian, 0.0, 1.0)

        return upsampled

    def _write_heatmaps(
        self,
        pointer_map: torch.Tensor,
        images: torch.Tensor,
        target_mask: Optional[torch.Tensor],
        step: int,
        tag: str,
        prompts: Optional[List[str]] = None,
    ):
        from torchvision.utils import save_image

        target_dir = os.path.join(self.heatmap_dir, tag)
        os.makedirs(target_dir, exist_ok=True)

        pointer_map = pointer_map.detach().cpu()
        images = images.detach().cpu()
        if target_mask is not None:
            target_mask = target_mask.detach().cpu()

        if pointer_map.dim() == 3:
            pointer_map = pointer_map.unsqueeze(1)
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if target_mask is not None and target_mask.dim() == 3:
            target_mask = target_mask.unsqueeze(0)

        total_samples = pointer_map.size(0)
        if self.config.heatmap_max_samples is None or self.config.heatmap_max_samples <= 0:
            num_samples = total_samples
        else:
            num_samples = min(self.config.heatmap_max_samples, total_samples)

        for idx in range(num_samples):
            orig = (images[idx] + 1.0) / 2.0
            heat = pointer_map[idx]
            mask = target_mask[idx] if target_mask is not None else torch.zeros_like(heat)

            heat_norm = heat - heat.min()
            heat_norm = heat_norm / (heat_norm.max() + 1e-6)
            heat_rgb = heat_norm.repeat(3, 1, 1)

            if mask.dim() == 2:
                mask_rgb = mask.unsqueeze(0)
            else:
                mask_rgb = mask
            if mask_rgb.size(0) == 1:
                mask_rgb = mask_rgb.repeat(3, 1, 1)

            grid = torch.stack([orig.clamp(0, 1), mask_rgb.clamp(0, 1), heat_rgb.clamp(0, 1)], dim=0)
            save_path = os.path.join(
                target_dir,
                f"{tag}_step_{step:06d}_sample_{idx}.png",
            )
            save_image(grid, save_path, nrow=3)

        self._persist_samples(
            pointer_map=pointer_map,
            images=images,
            target_mask=target_mask,
            prompts=prompts,
            step=step,
            tag=tag,
            num_samples=num_samples,
        )

    def _persist_samples(
        self,
        pointer_map: torch.Tensor,
        images: torch.Tensor,
        target_mask: Optional[torch.Tensor],
        prompts: Optional[List[str]],
        step: int,
        tag: str,
        num_samples: int,
    ) -> None:
        sample_dir = os.path.join(self.sample_dir, tag)
        os.makedirs(sample_dir, exist_ok=True)

        record = {
            "step": step,
            "tag": tag,
            "saved_at": datetime.now().isoformat(),
            "prompts": list(prompts[:num_samples]) if prompts else [],
            "pointer_map": pointer_map[:num_samples].to(dtype=torch.float32).cpu(),
            "pixel_values": images[:num_samples].to(dtype=torch.float32).cpu(),
            "target_mask": target_mask[:num_samples].to(dtype=torch.float32).cpu()
            if target_mask is not None
            else None,
        }
        save_path = os.path.join(sample_dir, f"{tag}_step_{step:06d}.pt")
        torch.save(record, save_path)

    def _save_heatmap(self, pointer_map: torch.Tensor, images: torch.Tensor, target_mask: torch.Tensor, step: int):
        if self.config.heatmap_every <= 0:
            return
        if step % self.config.heatmap_every != 0:
            return

        self._write_heatmaps(pointer_map, images, target_mask, step, tag="train", prompts=self._current_prompts)

        if self.test_sample is not None:
            test_pointer_map = self._run_pointer_forward(
                self.test_sample["pixel_values"],
                [self.test_sample["prompt"]],
            )
            self._write_heatmaps(
                test_pointer_map.to(self.device),
                self.test_sample["pixel_values"].to(self.device),
                self.test_sample["target_mask"].to(self.device),
                step,
                tag="test",
                prompts=[self.test_sample["prompt"]],
            )

        self.pointer_cache.reset()

    def _save_lora_weights(self, step: int) -> str:
        os.makedirs(self.weights_dir, exist_ok=True)
        state_dict = {}
        for idx, module in enumerate(self.lora_modules):
            name = getattr(module, "lora_name", f"module_{idx}")
            state_dict[name] = {
                "lora_down": module.lora_down.weight.detach().cpu(),
                "lora_up": module.lora_up.weight.detach().cpu(),
                "bias": module.bias.detach().cpu() if module.bias is not None else None,
                "rank": module.rank,
                "scale": module.scale,
            }

        payload = {
            "step": step,
            "saved_at": datetime.now().isoformat(),
            "run_id": getattr(self, "run_id", ""),
            "state_dict": state_dict,
            "trainer_config": asdict(self.config),
            "lora_config": asdict(self.lora_config),
        }
        save_path = os.path.join(self.weights_dir, f"lora_step_{step:06d}.pt")
        torch.save(payload, save_path)
        return save_path

    def report_pointer_slice(self, layer_key: str, tensor_shape: Tuple[int, ...], encoder_length: int):
        if self._pointer_debug_active:
            self._pointer_slice_stats[layer_key] = (tensor_shape, encoder_length)

    def train(self):
        self.transformer.train()
        for param in self.transformer.parameters():
            param.requires_grad = False
        for module in self.lora_modules:
            for param in module.lora_parameters:
                param.requires_grad = True

        for step in range(1, self.config.num_steps + 1):
            batch = self._next_batch()
            pixel_values = batch["pixel_values"].to(self.device, dtype=self.dtype)
            target_mask = batch["target_mask"].to(self.device, dtype=self.dtype)
            prompts = list(batch["prompt_loc"])
            self._current_prompts = prompts

            if pixel_values.dim() == 3:
                pixel_values = pixel_values.unsqueeze(0)
            if target_mask.dim() == 3:
                target_mask = target_mask.unsqueeze(0)

            latents = encode_images(self.vae, pixel_values).to(self.device, dtype=self.dtype)
            noise = torch.randn_like(latents, dtype=self.dtype, device=self.device)
            schedule_timesteps = self.scheduler.timesteps.to(device=self.device)
            timestep_indices = torch.randint(
                0,
                schedule_timesteps.shape[0],
                (latents.size(0),),
                device=self.device,
                dtype=torch.long,
            )
            timesteps = schedule_timesteps[timestep_indices]
            if hasattr(self.scheduler, "add_noise"):
                noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
            elif hasattr(self.scheduler, "scale_noise"):
                noisy_latents = self.scheduler.scale_noise(latents, timesteps, noise=noise)
            else:
                raise RuntimeError("Scheduler does not support adding/scaling noise for training.")
            noisy_latents = noisy_latents.to(self.device, dtype=self.dtype)

            embeds: PromptEmbeds = encode_prompts(
                prompts,
                self.tokenizers,
                self.text_encoders,
                device=self.device,
                dtype=self.dtype,
            )

            self.pointer_cache.reset()
            self._pointer_debug_active = self._pointer_debug_enabled and (
                self.config.pointer_debug_every > 0 and step % self.config.pointer_debug_every == 0
            )
            if not self._pointer_debug_active:
                self._pointer_slice_stats.clear()
            with torch.enable_grad():
                predict_noise(
                    self.transformer,
                    self.scheduler,
                    noisy_latents,
                    timesteps,
                    embeds,
                )

            token_masks = self._build_token_masks(prompts)
            pointer_map, layer_sizes = self._compose_pointer_heatmap(
                token_masks=token_masks,
                return_layer_sizes=self._pointer_debug_active,
            )
            if self._pointer_debug_active and layer_sizes is not None:
                layer_report = ", ".join(f"{name.split('.')[-1]}:{side}" for name, side in layer_sizes)
                print(f"[step {step}] pointer grids -> {layer_report}")
                if self._pointer_slice_stats:
                    slice_report = ", ".join(
                        f"{name.split('.')[-1]}:{shape}->{enc_len}"
                        for name, (shape, enc_len) in self._pointer_slice_stats.items()
                    )
                    print(f"[step {step}] pointer slices -> {slice_report}")
                self._pointer_slice_stats = {}
            pointer_map = pointer_map.to(self.device, dtype=target_mask.dtype)
            if pointer_map.shape[0] != target_mask.shape[0]:
                raise RuntimeError("Pointer map batch size mismatch with target mask.")

            loss_dict = compute_pointer_losses(pointer_map, target_mask, self.pointer_loss)
            total_loss = sum(loss_dict.values())

            self.optimizer.zero_grad()
            total_loss.backward()
            if self._pointer_debug_active:
                grad_norms = []
                for module in self.lora_modules:
                    for param in module.lora_parameters:
                        if param.grad is not None:
                            grad_norms.append(param.grad.norm().detach())
                if grad_norms:
                    grad_norms_tensor = torch.stack(grad_norms)
                    print(
                        f"[step {step}] LoRA grad norm mean={grad_norms_tensor.mean().item():.4f} "
                        f"max={grad_norms_tensor.max().item():.4f}"
                    )
            torch.nn.utils.clip_grad_norm_(list(self.optimizer.param_groups[0]["params"]), max_norm=1.0)
            self.optimizer.step()

            if step % self.config.log_every == 0:
                loss_values = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
                print(f"[step {step}] loss={total_loss.detach().cpu().item():.6f} {loss_values}")

            self._save_heatmap(pointer_map.detach(), pixel_values.detach(), target_mask.detach(), step)

            if self.config.checkpoint_every > 0 and step % self.config.checkpoint_every == 0:
                self._save_lora_weights(step)

        if self.config.checkpoint_every <= 0 or self.config.num_steps % self.config.checkpoint_every != 0:
            self._save_lora_weights(self.config.num_steps)

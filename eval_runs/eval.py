"""Evaluation script for pointer-style LoRA models using pointing accuracy."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageFile
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from tqdm import tqdm

from pointer_lora.dataset import PointerTripletDataset
from pointer_lora.lora import LoRAInjectionConfig, inject_lora_into_attention
from pointer_lora.model import PromptEmbeds, encode_images, encode_prompts, load_flux_kontext_components, predict_noise
from pointer_lora.pointer_recorder import PointerCache, PointerRecorder
from pointer_lora.scheduler import CustomFlowMatchEulerDiscreteScheduler

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _to_dtype(dtype_name: str) -> torch.dtype:
    name = (dtype_name or "float32").lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


@dataclass
class EvalModelConfig:
    model_path: str
    resolution: int
    dtype: str
    offload_text_encoders: bool
    pointer_token_mode: str
    pointer_token_filters: Tuple[str, ...]
    pointer_heatmap_combine: str
    pointer_temperature: float
    pointer_epsilon: float
    pointer_superres_factor: int
    pointer_superres_sharpness: float


@dataclass
class EvalRunConfig:
    annotations: str
    data_root: str
    per_bucket: int
    seed: int
    output_dir: str
    compute_iou: bool
    heatmap_threshold: float
    save_details: bool
    max_images: Optional[int]
    save_failures: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pointer LoRA via pointing game accuracy.")
    parser.add_argument("--config", required=True, help="Training YAML，用于复用模型/指针参数。")
    parser.add_argument("--lora-weights", required=True, help="LoRA 权重路径，例如 output/.../weights/lora_step_xxx.pt。")
    parser.add_argument("--annotations", default=None, help="评测三元组 annotations.json 路径（默认沿用 config.data.annotations）。")
    parser.add_argument("--data-root", default=None, help="评测数据根目录（包含 images/masks），默认沿用 config.data.root。")
    parser.add_argument("--per-bucket", type=int, default=300, help="每个‘人数桶’抽样的图像数量，0 表示使用全部。")
    parser.add_argument("--max-images", type=int, default=None, help="整体抽样的图像上限，用于快速 smoke test。")
    parser.add_argument("--seed", type=int, default=42, help="随机采样种子。")
    parser.add_argument("--output-dir", default="pointer_lora/output/eval_runs", help="评测结果输出根目录。")
    parser.add_argument("--compute-iou", action="store_true", help="除了 pointing accuracy 以外，同时统计 IoU / Dice。")
    parser.add_argument("--heatmap-threshold", type=float, default=0.35, help="当 compute-iou=True 时用于二值化热力图的阈值。")
    parser.add_argument("--save-details", action="store_true", help="将每条样本的命中结果保存成 CSV，便于排查。")
    parser.add_argument(
        "--save-failures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="命中失败时保存热力图 PNG 与对应 mask（默认开启，可用 --no-save-failures 关闭）。",
    )
    return parser.parse_args()


def load_yaml_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_model_config(yaml_cfg: Dict, args: argparse.Namespace) -> EvalModelConfig:
    pointer_section = yaml_cfg.get("pointer", {})
    heatmap_section = pointer_section.get("heatmap", {})
    superres_section = pointer_section.get("superres", {})
    token_filters = pointer_section.get("token_filters", [])
    if not isinstance(token_filters, (list, tuple)):
        raise ValueError("pointer.token_filters must be a list.")

    return EvalModelConfig(
        model_path=yaml_cfg["model"]["path"],
        resolution=int(yaml_cfg.get("data", {}).get("resolution", 640)),
        dtype=yaml_cfg.get("training", {}).get("dtype", "float32"),
        offload_text_encoders=bool(yaml_cfg.get("model", {}).get("offload_text_encoders", False)),
        pointer_token_mode=pointer_section.get("token_mode", "phrase").lower(),
        pointer_token_filters=tuple(token_filters),
        pointer_heatmap_combine=heatmap_section.get("combine", "mean"),
        pointer_temperature=float(heatmap_section.get("temperature", 1.0)),
        pointer_epsilon=float(heatmap_section.get("epsilon", 1.0e-6)),
        pointer_superres_factor=int(superres_section.get("factor", 1)),
        pointer_superres_sharpness=float(superres_section.get("sharpness", 0.0)),
    )


def resolve_lora_config(yaml_cfg: Dict) -> LoRAInjectionConfig:
    section = yaml_cfg.get("lora", {})
    return LoRAInjectionConfig(
        rank=int(section.get("rank", 16)),
        alpha=float(section.get("alpha", 32.0)),
        attn_block_prefixes=tuple(section.get("attn_blocks", LoRAInjectionConfig.attn_block_prefixes)),
        target_projections=tuple(section.get("projections", LoRAInjectionConfig.target_projections)),
    )


def resolve_run_config(yaml_cfg: Dict, args: argparse.Namespace) -> EvalRunConfig:
    data_section = yaml_cfg.get("data", {})
    annotations = args.annotations or data_section.get("annotations")
    data_root = args.data_root or data_section.get("root")

    if annotations is None or data_root is None:
        raise ValueError("annotations 和 data_root 需要在 CLI 或 YAML 中指定。")

    return EvalRunConfig(
        annotations=annotations,
        data_root=data_root,
        per_bucket=max(0, int(args.per_bucket)),
        seed=int(args.seed),
        output_dir=args.output_dir,
        compute_iou=bool(args.compute_iou),
        heatmap_threshold=float(args.heatmap_threshold),
        save_details=bool(args.save_details),
        max_images=int(args.max_images) if args.max_images is not None else None,
        save_failures=bool(args.save_failures),
    )


class PointerHeatmapPredictor:
    """Thin inference wrapper to reproduce pointer heatmaps for evaluation."""

    def __init__(
        self,
        model_cfg: EvalModelConfig,
        lora_cfg: LoRAInjectionConfig,
        lora_weights: str,
    ):
        self.cfg = model_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = _to_dtype(model_cfg.dtype)

        (
            self.transformer,
            self.text_encoders,
            self.tokenizers,
            self.vae,
            self.scheduler,
        ) = load_flux_kontext_components(
            model_cfg.model_path,
            device=self.device,
            dtype=self.dtype,
            offload_text_encoders=model_cfg.offload_text_encoders,
        )
        if isinstance(self.scheduler, CustomFlowMatchEulerDiscreteScheduler):
            self.scheduler.set_train_timesteps(self.scheduler.config.num_train_timesteps, device=self.device)

        self.lora_modules = inject_lora_into_attention(self.transformer, lora_cfg)
        if lora_weights:
            self._load_lora_weights(lora_weights)

        self.pointer_cache = PointerCache()
        self._attach_pointer_recorders(lora_cfg.attn_block_prefixes)

        self._superres_factor = max(1, int(model_cfg.pointer_superres_factor))
        self._superres_sharpness = max(0.0, float(model_cfg.pointer_superres_sharpness))
        self._laplacian_kernel = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

    def _attach_pointer_recorders(self, attn_prefixes: Iterable[str]):
        self._original_processors: Dict[str, torch.nn.Module] = {}
        for name, module in self.transformer.named_modules():
            if not getattr(module, "processor", None):
                continue
            if not any(name.startswith(prefix) for prefix in attn_prefixes):
                continue
            recorder = PointerRecorder(owner=self, layer_key=name)
            self._original_processors[name] = module.processor
            module.set_processor(recorder)

    def record_pointer(self, key: str, tensor: torch.Tensor):
        self.pointer_cache.append(key, tensor)

    def _load_lora_weights(self, weight_path: str):
        payload = torch.load(weight_path, map_location="cpu")
        state_dict = payload.get("state_dict", payload)

        name_to_module = {getattr(module, "lora_name", f"module_{idx}"): module for idx, module in enumerate(self.lora_modules)}
        for name, params in state_dict.items():
            module = name_to_module.get(name)
            if module is None:
                continue
            module.lora_down.weight.data.copy_(params["lora_down"].to(module.lora_down.weight.device, dtype=module.lora_down.weight.dtype))
            module.lora_up.weight.data.copy_(params["lora_up"].to(module.lora_up.weight.device, dtype=module.lora_up.weight.dtype))
            if module.bias is not None and params.get("bias") is not None:
                module.bias.data.copy_(params["bias"].to(module.bias.device, dtype=module.bias.dtype))

    def _build_token_masks(self, prompts: List[str]) -> torch.Tensor:
        tokenizer = self.tokenizers[1]
        encoded = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids
        attention_mask = encoded.attention_mask
        masks = torch.zeros_like(input_ids, dtype=torch.bool)

        token_filters = {tok.lower().strip(): True for tok in self.cfg.pointer_token_filters}
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

        def normalize(token: str) -> str:
            return token.lower().replace("▁", "").strip()

        for batch_idx, ids in enumerate(input_ids):
            tokens = tokenizer.convert_ids_to_tokens(ids.tolist())
            for token_idx, token in enumerate(tokens):
                if attention_mask[batch_idx, token_idx] == 0:
                    continue
                normalized = normalize(token)
                use_token = False
                if self.cfg.pointer_token_mode == "phrase":
                    use_token = True
                elif self.cfg.pointer_token_mode == "ordinal":
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
                masks[batch_idx] = attention_mask[batch_idx].bool()

        return masks.to(self.device)

    def _compose_pointer_heatmap(self, token_masks: torch.Tensor) -> torch.Tensor:
        layer_heatmaps: List[torch.Tensor] = []
        token_masks = token_masks.to(self.device)

        for _, tensors in self.pointer_cache.items():
            stacked = torch.stack(tensors, dim=0).mean(dim=0)
            mask = token_masks.unsqueeze(1).unsqueeze(2).to(stacked.dtype)
            if mask.shape[-1] != stacked.shape[-1]:
                mask = mask[..., : stacked.shape[-1]]
            masked = stacked * mask
            denom = mask.sum(dim=-1).clamp_min(1.0)
            heat = masked.sum(dim=-1) / denom
            heat = heat.mean(dim=1)

            spatial_tokens = heat.shape[-1]
            side = int(math.sqrt(spatial_tokens))
            if side * side != spatial_tokens:
                raise RuntimeError(f"Unexpected spatial token count {spatial_tokens}")

            heat = heat / max(self.cfg.pointer_temperature, 1e-6)
            heat = heat - heat.amin(dim=-1, keepdim=True)
            heat = heat / (heat.amax(dim=-1, keepdim=True) + self.cfg.pointer_epsilon)
            heat = heat.view(heat.shape[0], 1, side, side)
            layer_heatmaps.append(heat)

        if not layer_heatmaps:
            raise RuntimeError("Pointer cache is empty; attention processors may not be attached.")

        stack = torch.stack(layer_heatmaps, dim=0)
        if self.cfg.pointer_heatmap_combine == "max":
            combined = stack.max(dim=0).values
        else:
            combined = stack.mean(dim=0)

        combined = combined.clamp(0.0, 1.0)
        pointer_map = self._apply_super_resolution(combined)
        pointer_map = F.interpolate(
            pointer_map,
            size=(self.cfg.resolution, self.cfg.resolution),
            mode="bilinear",
            align_corners=False,
        )
        return pointer_map.clamp(0.0, 1.0)

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

    def predict_pointer_map(self, pixel_values: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        pixel_values = pixel_values.to(self.device, dtype=self.dtype)
        latents = encode_images(self.vae, pixel_values).to(self.device, dtype=self.dtype)
        noise = torch.zeros_like(latents, dtype=self.dtype, device=self.device)
        timesteps = torch.zeros((latents.size(0),), dtype=torch.long, device=self.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps).to(self.device, dtype=self.dtype)

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
                noisy_latents,
                timesteps,
                embeds,
            )
        token_masks = self._build_token_masks(prompts)
        pointer_map = self._compose_pointer_heatmap(token_masks)
        self.pointer_cache.reset()
        return pointer_map


def normalize_rel_path(path: str) -> str:
    path = path.replace("\\", "/").strip()
    if path.startswith("./"):
        path = path[2:]
    return path


def bucket_label(count: int) -> Optional[str]:
    if count < 2:
        return None
    if count == 2:
        return "2"
    if count == 3:
        return "3"
    if count == 4:
        return "4"
    return "5+"


def load_annotations_grouped(run_cfg: EvalRunConfig) -> Tuple[Dict[str, List[Dict]], Dict[str, str]]:
    with open(run_cfg.annotations, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    grouped: Dict[str, List[Dict]] = {}
    bucket_map: Dict[str, str] = {}
    for rec in data:
        img_rel = normalize_rel_path(rec.get("image_path", ""))
        mask_rel = normalize_rel_path(rec.get("mask_path", ""))
        if not img_rel or not mask_rel:
            continue
        rec = dict(rec)
        rec["image_path"] = img_rel
        rec["mask_path"] = mask_rel
        grouped.setdefault(img_rel, []).append(rec)
    for img_rel, records in list(grouped.items()):
        bucket = bucket_label(len(records))
        if bucket is None:
            grouped.pop(img_rel)
            continue
        bucket_map[img_rel] = bucket
    if not grouped:
        raise ValueError("No valid multi-person samples found for evaluation.")
    return grouped, bucket_map


def sample_images_by_bucket(
    grouped: Dict[str, List[Dict]],
    bucket_map: Dict[str, str],
    run_cfg: EvalRunConfig,
) -> Dict[str, List[str]]:
    rng = random.Random(run_cfg.seed)
    bucket_to_images: Dict[str, List[str]] = {"2": [], "3": [], "4": [], "5+": []}
    for img_rel, bucket in bucket_map.items():
        if bucket in bucket_to_images:
            bucket_to_images[bucket].append(img_rel)

    sampled: Dict[str, List[str]] = {}
    for bucket, images in bucket_to_images.items():
        if not images:
            continue
        target = images
        if run_cfg.per_bucket > 0 and len(images) > run_cfg.per_bucket:
            target = rng.sample(images, run_cfg.per_bucket)
        sampled[bucket] = sorted(target)

    if run_cfg.max_images is not None:
        # flatten, then cut to max_images, keeping bucket balance roughly proportional
        flat = [(bucket, img_rel) for bucket, imgs in sampled.items() for img_rel in imgs]
        if len(flat) > run_cfg.max_images:
            flat = rng.sample(flat, run_cfg.max_images)
        trimmed: Dict[str, List[str]] = {"2": [], "3": [], "4": [], "5+": []}
        for bucket, img_rel in flat:
            trimmed.setdefault(bucket, []).append(img_rel)
        sampled = {bucket: imgs for bucket, imgs in trimmed.items() if imgs}

    return sampled


def prepare_image_tensor(image_path: Path, resolution: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    tensor = TF.to_tensor(image) * 2.0 - 1.0
    tensor = PointerTripletDataset._resize_tensor(tensor, resolution, pad_to_square=True, mode="bilinear")
    return tensor


def prepare_mask_tensor(mask_path: Path, resolution: int) -> torch.Tensor:
    mask = Image.open(mask_path).convert("L")
    tensor = TF.to_tensor(mask)
    tensor = (tensor > 0.5).float()
    tensor = PointerTripletDataset._resize_tensor(tensor, resolution, pad_to_square=True, mode="nearest")
    return tensor


def save_failure_visual(image_tensor: torch.Tensor, mask_tensor: torch.Tensor, heat: np.ndarray, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    image = ((image_tensor.detach().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)
    mask = mask_tensor.detach().cpu()
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    heat_tensor = torch.from_numpy(heat).unsqueeze(0)
    heat_tensor = heat_tensor - heat_tensor.min()
    heat_tensor = heat_tensor / (heat_tensor.max() + 1e-6)
    heat_rgb = heat_tensor.repeat(3, 1, 1)
    mask_rgb = mask.repeat(3, 1, 1)
    grid = torch.stack([image, mask_rgb, heat_rgb], dim=0)
    save_image(grid, str(dest), nrow=3)


class MetricBucket:
    def __init__(self):
        self.count = 0
        self.hit = 0
        self.iou_sum = 0.0
        self.dice_sum = 0.0

    def update(self, hit: bool, iou: Optional[float] = None, dice: Optional[float] = None):
        self.count += 1
        if hit:
            self.hit += 1
        if iou is not None:
            self.iou_sum += float(iou)
        if dice is not None:
            self.dice_sum += float(dice)

    def to_dict(self, include_iou: bool) -> Dict[str, float]:
        result = {
            "samples": self.count,
            "pointing_acc": self.hit / self.count if self.count > 0 else 0.0,
        }
        if include_iou and self.count > 0:
            result["mean_iou"] = self.iou_sum / self.count
            result["mean_dice"] = self.dice_sum / self.count
        return result


def evaluate(
    run_cfg: EvalRunConfig,
    model_cfg: EvalModelConfig,
    lora_cfg: LoRAInjectionConfig,
    lora_weights: str,
    run_dir: Path,
):
    grouped, bucket_map = load_annotations_grouped(run_cfg)
    sampled = sample_images_by_bucket(grouped, bucket_map, run_cfg)
    if not sampled:
        raise RuntimeError("No images were selected for evaluation. Check sampling条件。")

    predictor = PointerHeatmapPredictor(model_cfg, lora_cfg, lora_weights)
    resolution = model_cfg.resolution

    image_cache: Dict[str, torch.Tensor] = {}
    detail_rows: List[str] = []
    buckets = {bucket: MetricBucket() for bucket in ["2", "3", "4", "5+"]}
    overall = MetricBucket()
    failure_root = run_dir / "failures"

    total_records = sum(len(grouped[img_rel]) for imgs in sampled.values() for img_rel in imgs)
    progress = tqdm(total=total_records, desc="Evaluating", unit="sample")

    for bucket, images in sampled.items():
        for img_rel in images:
            image_path = Path(run_cfg.data_root) / img_rel
            if not image_path.exists():
                progress.update(len(grouped[img_rel]))
                continue

            if img_rel not in image_cache:
                image_cache[img_rel] = prepare_image_tensor(image_path, resolution)
            image_tensor = image_cache[img_rel]

            per_image_records = grouped[img_rel]
            per_image_results = []
            fail_any = False

            for rec in per_image_records:
        mask_rel = rec["mask_path"]
        prompt = rec.get("prompt_loc", "")
        mask_path = Path(run_cfg.data_root) / mask_rel
                if not mask_path.exists():
                    progress.update(1)
            continue

        mask_tensor = prepare_mask_tensor(mask_path, resolution)

        pointer_map = predictor.predict_pointer_map(image_tensor.unsqueeze(0), [prompt])
        heat = pointer_map[0, 0].detach().float().cpu().numpy()
        mask_np = mask_tensor.squeeze(0).detach().cpu().numpy()

        peak_index = np.argmax(heat)
        h, w = heat.shape
        peak_y, peak_x = divmod(int(peak_index), w)
        hit = bool(mask_np[peak_y, peak_x] > 0.5)
                fail_any = fail_any or (not hit)

        iou = dice = None
        if run_cfg.compute_iou:
            pred_mask = (heat >= run_cfg.heatmap_threshold).astype(np.float32)
            intersection = float((pred_mask * mask_np).sum())
            pred_sum = float(pred_mask.sum())
            gt_sum = float(mask_np.sum())
            union = pred_sum + gt_sum - intersection
            if union <= 0:
                iou = 1.0 if gt_sum == 0 else 0.0
                dice = 1.0 if gt_sum == 0 else 0.0
            else:
                iou = intersection / union
                dice = (2.0 * intersection) / (pred_sum + gt_sum + 1e-6)

        buckets[bucket].update(hit, iou, dice)
        overall.update(hit, iou, dice)

        if run_cfg.save_details:
            detail_rows.append(
                "{},{},{},{},{},{:.4f},{:.4f}".format(
                    bucket,
                            img_rel,
                    mask_rel,
                    prompt.replace(",", " "),
                    int(hit),
                    peak_x,
                    peak_y,
                )
            )

                per_image_results.append(
                    {
                        "heat": heat,
                        "mask_tensor": mask_tensor.cpu(),
                        "mask_path": mask_path,
                        "mask_rel": mask_rel,
                        "prompt": prompt,
                    }
                )
                progress.update(1)

            if run_cfg.save_failures and fail_any and per_image_results:
                image_subdir = failure_root / bucket / Path(img_rel).stem
                image_subdir.mkdir(parents=True, exist_ok=True)
                for idx, item in enumerate(per_image_results, start=1):
                    mask_rel = item["mask_rel"]
                    heat = item["heat"]
                    mask_tensor = item["mask_tensor"]
                    mask_path = item["mask_path"]
                    base_name = Path(mask_rel).stem
                    heat_name = image_subdir / f"{idx:02d}_{base_name}_heat.png"
                    mask_dest = image_subdir / f"{idx:02d}_{Path(mask_path).name}"
                    save_failure_visual(image_tensor, mask_tensor, heat, heat_name)
                    if mask_path.exists():
                        shutil.copy2(mask_path, mask_dest)

    metrics = {
        "overall": overall.to_dict(run_cfg.compute_iou),
        "by_count": {
            bucket: stats.to_dict(run_cfg.compute_iou)
            for bucket, stats in buckets.items()
            if stats.count > 0
        },
    }
    return metrics, detail_rows


def create_run_dir(base_output: str) -> Path:
    base = Path(base_output)
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base / timestamp
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = base / f"{timestamp}_{suffix:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main():
    args = parse_args()
    yaml_cfg = load_yaml_config(args.config)
    model_cfg = resolve_model_config(yaml_cfg, args)
    lora_cfg = resolve_lora_config(yaml_cfg)
    run_cfg = resolve_run_config(yaml_cfg, args)

    run_dir = create_run_dir(run_cfg.output_dir)
    metrics, details = evaluate(run_cfg, model_cfg, lora_cfg, args.lora_weights, run_dir)

    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "metrics": metrics,
                "eval_config": asdict(run_cfg),
                "model_config": asdict(model_cfg),
                "generated_at": datetime.now().isoformat(),
            },
            fh,
            indent=2,
        )

    if details:
        header = "bucket,image_path,mask_path,prompt_loc,hit,peak_x,peak_y\n"
        details_path = run_dir / "details.csv"
        details_path.write_text(header + "\n".join(details), encoding="utf-8")

    print(f"[pointer-eval] Results saved to {run_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()


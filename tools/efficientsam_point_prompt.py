#!/usr/bin/env python3
"""
Minimal point-prompt demo for EfficientSAM.

Reads a single peak point from pointer_peak.json (key: "pixel": [x, y]),
uses it as a positive click prompt, and exports:
  - mask_k.png / overlay_k.png (top-K candidate masks from EfficientSAM)
  - best_mask.png / best_overlay.png (best IoU candidate)

Run (recommended):
  conda run -n edit python pointer_lora/tools/efficientsam_point_prompt.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Paths:
    efficientsam_repo: str
    checkpoint: str
    image: str
    pointer_json: str
    outdir: str
    topk: int


def _parse_args() -> Paths:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--efficientsam_repo",
        default="/root/autodl-tmp/EfficientSAM_code",
        help="Path to EfficientSAM git repo (contains efficient_sam/).",
    )
    ap.add_argument(
        "--checkpoint",
        default="/root/autodl-tmp/EfficientSAM/weights/efficient_sam_vitt.pt",
        help="Path to EfficientSAM-Ti checkpoint (.pt).",
    )
    ap.add_argument(
        "--image",
        default="/root/autodl-tmp/test_images/test4.png",
        help="Input image path (should match pointer_peak.json image_size).",
    )
    ap.add_argument(
        "--pointer_json",
        default="/workspace/MOREdit/output/softmask_demo/20251223-172332/pointer_peak.json",
        help="pointer_peak.json path (expects key: pixel=[x,y]).",
    )
    ap.add_argument(
        "--outdir",
        default="/workspace/MOREdit/output/softmask_demo/20251223-172332/efficientsam_point_prompt",
        help="Output directory for mask.png / overlay.png.",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=3,
        help="How many candidate masks to export (EfficientSAM outputs 3 by default).",
    )
    args = ap.parse_args()
    return Paths(
        efficientsam_repo=args.efficientsam_repo,
        checkpoint=args.checkpoint,
        image=args.image,
        pointer_json=args.pointer_json,
        outdir=args.outdir,
        topk=int(args.topk),
    )


def _load_point(pointer_json_path: str) -> Tuple[float, float]:
    with open(pointer_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if "pixel" not in obj or not isinstance(obj["pixel"], (list, tuple)) or len(obj["pixel"]) != 2:
        raise ValueError(f"Expected pointer json to have pixel=[x,y], got keys: {list(obj.keys())}")
    x, y = float(obj["pixel"][0]), float(obj["pixel"][1])
    return x, y


def _ensure_importable(efficientsam_repo: str) -> None:
    if not os.path.isdir(efficientsam_repo):
        raise FileNotFoundError(f"EfficientSAM repo not found: {efficientsam_repo}")
    sys.path.insert(0, efficientsam_repo)


def _build_model(checkpoint_path: str):
    import torch  # noqa: WPS433 (runtime import)

    # Imported after we add `--efficientsam_repo` to sys.path at runtime.
    # Some linters may not resolve this import statically.
    from efficient_sam.efficient_sam import build_efficient_sam  # type: ignore  # noqa: WPS433

    model = build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint=checkpoint_path,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), device


def _overlay_mask_and_point(image_rgb: Image.Image, mask_hw: np.ndarray, xy: Tuple[float, float]) -> Image.Image:
    img = np.asarray(image_rgb).astype(np.uint8)
    mask = (mask_hw > 0).astype(np.uint8)

    color = np.array([30, 144, 255], dtype=np.uint8)  # dodgerblue
    alpha = 0.45
    out = img.copy()
    out[mask == 1] = (out[mask == 1] * (1.0 - alpha) + color * alpha).astype(np.uint8)

    out_img = Image.fromarray(out)
    draw = ImageDraw.Draw(out_img)
    x, y = xy
    r = max(3, int(round(min(out_img.size) * 0.006)))
    draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0), width=3)
    draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 0, 0))
    return out_img


def _save_mask_pair(
    outdir: str,
    name: str,
    image: Image.Image,
    mask_bool: np.ndarray,
    xy: Tuple[float, float],
    extra_text: str | None = None,
) -> Tuple[str, str]:
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    mask_img = Image.fromarray(mask_u8)
    mask_path = os.path.join(outdir, f"{name}.png")
    mask_img.save(mask_path)

    overlay = _overlay_mask_and_point(image, mask_hw=mask_bool, xy=xy)
    if extra_text:
        draw = ImageDraw.Draw(overlay)
        draw.rectangle([0, 0, 520, 40], fill=(0, 0, 0))
        draw.text((10, 10), extra_text, fill=(255, 255, 255))
    overlay_path = os.path.join(outdir, f"{name.replace('mask', 'overlay')}.png")
    overlay.save(overlay_path)
    return mask_path, overlay_path


def main() -> None:
    paths = _parse_args()
    os.makedirs(paths.outdir, exist_ok=True)

    _ensure_importable(paths.efficientsam_repo)
    model, device = _build_model(paths.checkpoint)

    import torch  # noqa: WPS433
    from torchvision import transforms  # noqa: WPS433

    x, y = _load_point(paths.pointer_json)

    image = Image.open(paths.image).convert("RGB")
    # Make array writable to avoid PyTorch warning.
    image_np = np.asarray(image).copy()
    image_tensor = transforms.ToTensor()(image_np)

    input_points = torch.tensor([[[[x, y]]]], dtype=torch.float32, device=device)
    input_labels = torch.tensor([[[1]]], dtype=torch.int64, device=device)

    with torch.no_grad():
        predicted_logits, predicted_iou = model(
            image_tensor[None, ...].to(device),
            input_points,
            input_labels,
        )

    # Sort candidate masks by predicted IoU.
    sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
    predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
    predicted_logits = torch.take_along_dim(predicted_logits, sorted_ids[..., None, None], dim=2)

    # Export top-K candidates so you can see whether it picked "clothes" vs "full person".
    k = max(1, min(int(paths.topk), predicted_logits.shape[2]))
    saved: List[Tuple[str, str]] = []
    for i in range(k):
        logit = predicted_logits[0, 0, i, :, :]
        iou = float(predicted_iou[0, 0, i].detach().cpu().item())
        mask_bool = (logit >= 0).detach().cpu().numpy()
        mask_path, overlay_path = _save_mask_pair(
            paths.outdir,
            name=f"mask_{i}",
            image=image,
            mask_bool=mask_bool,
            xy=(x, y),
            extra_text=f"candidate {i} | pred_iou={iou:.4f}",
        )
        saved.append((mask_path, overlay_path))

    # Also write aliases for the best one.
    best_mask_path = os.path.join(paths.outdir, "best_mask.png")
    best_overlay_path = os.path.join(paths.outdir, "best_overlay.png")
    if saved:
        os.replace(saved[0][0], best_mask_path)
        os.replace(saved[0][1], best_overlay_path)
        # re-save the original names for candidate 0 (since we moved them)
        _save_mask_pair(
            paths.outdir,
            name="mask_0",
            image=image,
            mask_bool=(predicted_logits[0, 0, 0, :, :] >= 0).detach().cpu().numpy(),
            xy=(x, y),
            extra_text=f"candidate 0 | pred_iou={float(predicted_iou[0, 0, 0].detach().cpu().item()):.4f}",
        )

    print("saved candidates:")
    for mp, op in saved:
        print(" -", mp)
        print(" -", op)
    print("saved best:")
    print(" -", best_mask_path)
    print(" -", best_overlay_path)


if __name__ == "__main__":
    main()



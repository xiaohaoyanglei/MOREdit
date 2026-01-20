#!/usr/bin/env python3
"""
Train a lightweight "click -> person instance mask" head on top of a frozen EfficientSAM encoder.

Data format (reuse refiner_train.py style):
  annotations.json: list (or {"records": list}) where each record has:
    - image_path: str (relative to --root or absolute)
    - mask_path:  str (relative to --root or absolute), single-person instance mask (0/255)

Training:
  - sample one positive click inside the GT mask
  - map click to 1024x1024 space (EfficientSAM preprocess resizes to 1024)
  - build a gaussian click map at 64x64 (encoder embedding resolution)
  - decoder head predicts 1024x1024 logits

Run (recommended env has torch):
  conda run -n edit python -m pointer_lora.person_head.train_click_person_head \
    --annotations /root/autodl-tmp/mhpv2_triples_en_val/annotations.json \
    --root /root/autodl-tmp/mhpv2_triples_en_val \
    --save /root/autodl-tmp/pointer_lora/output/person_head/click_person_head.pt \
    --efficientsam-repo /root/autodl-tmp/EfficientSAM_code \
    --efficientsam-ckpt /root/autodl-tmp/EfficientSAM/weights/efficient_sam_vitt.pt \
    --epochs 2 --batch-size 8 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from PIL import ImageFile
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from .model import ClickPersonHead, EfficientSamFrozenEncoder, make_gaussian_map


TARGET_SIZE = 1024  # EfficientSAM encoder expects 1024
EMBED_SIZE = 64  # 1024/16


def _load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "records" in data:
        data = data["records"]
    if not isinstance(data, list):
        raise ValueError("annotations 文件应为 list 或包含 records 字段")
    return data


def _resolve_path(root: str, p: str) -> str:
    p = p.replace("\\", "/").strip()
    if not p:
        return ""
    if os.path.isabs(p):
        return p
    return os.path.join(root, p)


def _sample_point_from_mask(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return mask.shape[1] / 2.0, mask.shape[0] / 2.0
    idx = np.random.randint(0, len(xs))
    return float(xs[idx]), float(ys[idx])


@dataclass
class Item:
    image_path: str
    mask_path: str


class PersonClickDataset(Dataset):
    def __init__(
        self,
        annotations: str,
        root: str,
        image_dir: str | None = None,
        mask_dir: str | None = None,
        max_samples: int | None = None,
    ) -> None:
        recs = _load_records(annotations)
        items: List[Item] = []
        for rec in recs:
            img_rel = str(rec.get("image_path", "")).replace("\\", "/").strip()
            msk_rel = str(rec.get("mask_path", "")).replace("\\", "/").strip()
            if image_dir and img_rel and not os.path.isabs(img_rel):
                img_rel = f"{image_dir.rstrip('/')}/{img_rel}"
            if mask_dir and msk_rel and not os.path.isabs(msk_rel):
                msk_rel = f"{mask_dir.rstrip('/')}/{msk_rel}"
            img = _resolve_path(root, img_rel)
            msk = _resolve_path(root, msk_rel)
            if not (img and msk and os.path.exists(img) and os.path.exists(msk)):
                continue
            items.append(Item(image_path=img, mask_path=msk))
        if max_samples:
            items = items[: int(max_samples)]
        if not items:
            raise ValueError("数据集中没有有效样本")
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Some MPH2 converted samples may contain truncated/corrupted images.
        # Robustly retry a few times before giving up.
        for _ in range(10):
            item = self.items[idx]
            try:
                image = Image.open(item.image_path).convert("RGB")
                mask = Image.open(item.mask_path).convert("L")
                img_t = TF.to_tensor(image)  # [0,1], [3,H,W]
                mask_t = TF.to_tensor(mask)  # [0,1], [1,H,W]
            except OSError:
                idx = np.random.randint(0, len(self.items))
                continue

            mask_bin = (mask_t > 0.5).float()
            _, h, w = mask_bin.shape
            cx, cy = _sample_point_from_mask(mask_bin.squeeze(0).numpy())
            # map click to TARGET_SIZE space (EfficientSAM preprocess stretches to 1024x1024)
            x1024 = cx * (TARGET_SIZE / float(w))
            y1024 = cy * (TARGET_SIZE / float(h))

            # Resize inputs to fixed 1024x1024 so dataloader can stack batches.
            img_1024 = F.interpolate(
                img_t.unsqueeze(0),
                size=(TARGET_SIZE, TARGET_SIZE),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            mask_1024 = F.interpolate(mask_bin.unsqueeze(0), size=(TARGET_SIZE, TARGET_SIZE), mode="nearest").squeeze(0)

            return {
                "image": img_1024,
                "mask": mask_1024,
                "click_xy_1024": (float(x1024), float(y1024)),
            }

        raise RuntimeError("Failed to load a valid (image, mask) pair after retries (dataset may be severely corrupted).")


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * inter + smooth) / (union + smooth)
    return bce + dice.mean()


def _build_efficientsam_encoder(efficientsam_repo: str, ckpt: str, device: torch.device) -> EfficientSamFrozenEncoder:
    sys.path.insert(0, efficientsam_repo)
    from efficient_sam.efficient_sam import build_efficient_sam  # type: ignore

    sam = build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint=ckpt,
    ).to(device)
    sam.eval()
    for p in sam.parameters():
        p.requires_grad_(False)
    return EfficientSamFrozenEncoder(model=sam, device=device)


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    xs = torch.tensor([b["click_xy_1024"][0] for b in batch], dtype=torch.float32)
    ys = torch.tensor([b["click_xy_1024"][1] for b in batch], dtype=torch.float32)
    return {"image": images, "mask": masks, "x1024": xs, "y1024": ys}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--efficientsam-repo", default="/root/autodl-tmp/EfficientSAM_code")
    ap.add_argument("--efficientsam-ckpt", default="/root/autodl-tmp/EfficientSAM/weights/efficient_sam_vitt.pt")
    ap.add_argument("--image-dir", default=None, help="Optional subdir for images under --root (e.g. images)")
    ap.add_argument("--mask-dir", default=None, help="Optional subdir for masks under --root (e.g. masks)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sigma64", type=float, default=2.0, help="Click gaussian sigma in 64x64 space.")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--preview-every", type=int, default=1000, help="Save preview every N steps (0=disable).")
    ap.add_argument(
        "--preview-outdir",
        default=None,
        help="Where to save previews. Default: <save_dir>/preview",
    )
    return ap.parse_args()


def _to_u8(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().to(torch.float32).numpy()
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def save_preview(
    step: int,
    outdir: Path,
    img: torch.Tensor,  # [B,3,1024,1024]
    mask_gt_1024: torch.Tensor,  # [B,1,1024,1024]
    prob_1024: torch.Tensor,  # [B,1,1024,1024]
    click64: torch.Tensor,  # [B,1,64,64]
    max_items: int = 4,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    b = min(int(max_items), int(img.shape[0]))

    img1024 = img[:b]
    click1024 = F.interpolate(click64[:b], size=(TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False)

    tiles: List[Image.Image] = []
    for i in range(b):
        rgb = Image.fromarray(_to_u8(img1024[i].permute(1, 2, 0)))
        g = Image.fromarray(_to_u8(click1024[i, 0])).convert("RGB")
        pred = Image.fromarray(_to_u8(prob_1024[i, 0])).convert("RGB")
        gt = Image.fromarray(_to_u8(mask_gt_1024[i, 0])).convert("RGB")

        # overlay
        rgb_np = np.asarray(rgb).astype(np.uint8)
        pred_np = (np.asarray(pred.convert("L")) > 127)
        overlay_np = rgb_np.copy()
        overlay_np[pred_np] = (overlay_np[pred_np] * 0.55 + np.array([30, 144, 255]) * 0.45).astype(np.uint8)
        overlay = Image.fromarray(overlay_np)

        row = Image.new("RGB", (rgb.width * 5, rgb.height))
        row.paste(rgb, (0, 0))
        row.paste(g, (rgb.width, 0))
        row.paste(pred, (rgb.width * 2, 0))
        row.paste(gt, (rgb.width * 3, 0))
        row.paste(overlay, (rgb.width * 4, 0))
        tiles.append(row)

    grid = Image.new("RGB", (tiles[0].width, tiles[0].height * len(tiles)))
    for r, row in enumerate(tiles):
        grid.paste(row, (0, r * row.height))

    grid.save(outdir / f"step_{step:07d}.png")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # Best-effort handling for partially downloaded images; 0-byte files still need skipping (handled in dataset).
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Auto-detect common converted dataset layout: <root>/images and <root>/masks
    image_dir = args.image_dir
    mask_dir = args.mask_dir
    if image_dir is None and os.path.isdir(os.path.join(args.root, "images")):
        image_dir = "images"
    if mask_dir is None and os.path.isdir(os.path.join(args.root, "masks")):
        mask_dir = "masks"

    ds = PersonClickDataset(
        args.annotations,
        args.root,
        image_dir=image_dir,
        mask_dir=mask_dir,
        max_samples=args.max_samples,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate,
    )

    encoder = _build_efficientsam_encoder(args.efficientsam_repo, args.efficientsam_ckpt, device=device)
    head = ClickPersonHead(in_embed_ch=256, base_ch=128).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=float(args.lr), weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    preview_dir = Path(args.preview_outdir) if args.preview_outdir else (save_path.parent / "preview")

    global_step = 0
    head.train()
    for epoch in range(int(args.epochs)):
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            global_step += 1
            img = batch["image"].to(device)
            mask_1024 = batch["mask"].to(device)

            # EfficientSAM encoder embedding: [B,256,64,64]
            with torch.no_grad():
                embed = encoder.encode_image(img)

            # Build click gaussian at 64x64 (map x1024,y1024 -> x64,y64)
            x64 = batch["x1024"].to(device) * (EMBED_SIZE / float(TARGET_SIZE))
            y64 = batch["y1024"].to(device) * (EMBED_SIZE / float(TARGET_SIZE))
            click64 = make_gaussian_map(
                h=EMBED_SIZE,
                w=EMBED_SIZE,
                x=x64,
                y=y64,
                sigma=float(args.sigma64),
                device=device,
                dtype=embed.dtype,
            )

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
                logits = head(embed, click64)
                loss = bce_dice_loss(logits, mask_1024)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix(loss=float(loss.detach().cpu().item()))

            if int(args.preview_every) > 0 and (global_step % int(args.preview_every) == 0):
                with torch.no_grad():
                    prob = torch.sigmoid(logits).detach()
                save_preview(
                    step=global_step,
                    outdir=preview_dir,
                    img=img.detach(),
                    mask_gt_1024=mask_1024.detach(),
                    prob_1024=prob,
                    click64=click64.detach(),
                )

        # Save per-epoch
        torch.save(
            {
                "head": head.state_dict(),
                "meta": {
                    "epoch": epoch,
                    "target_size": TARGET_SIZE,
                    "embed_size": EMBED_SIZE,
                    "sigma64": float(args.sigma64),
                },
            },
            str(save_path),
        )

    print("saved:", str(save_path))


if __name__ == "__main__":
    main()



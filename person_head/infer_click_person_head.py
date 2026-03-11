#!/usr/bin/env python3
"""
Quick inference for ClickPersonHead.

Example:
  conda run -n edit python -m MOREdit.person_head.infer_click_person_head \
    --weights /workspace/MOREdit/output/person_head/click_person_head.pt \
    --image /root/autodl-tmp/test_images/test4.png \
    --x 232 --y 391 \
    --efficientsam-repo /root/autodl-tmp/EfficientSAM_code \
    --efficientsam-ckpt /root/autodl-tmp/EfficientSAM/weights/efficient_sam_vitt.pt \
    --outdir /workspace/MOREdit/output/person_head/demo
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms

from .model import ClickPersonHead, EfficientSamFrozenEncoder, make_gaussian_map

TARGET_SIZE = 1024
EMBED_SIZE = 64


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--x", type=float, required=True)
    ap.add_argument("--y", type=float, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--efficientsam-repo", default="/root/autodl-tmp/EfficientSAM_code")
    ap.add_argument("--efficientsam-ckpt", default="/root/autodl-tmp/EfficientSAM/weights/efficient_sam_vitt.pt")
    ap.add_argument("--sigma64", type=float, default=2.0)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def _build_encoder(repo: str, ckpt: str, device: torch.device) -> EfficientSamFrozenEncoder:
    sys.path.insert(0, repo)
    from efficient_sam.efficient_sam import build_efficient_sam  # type: ignore

    sam = build_efficient_sam(encoder_patch_embed_dim=192, encoder_num_heads=3, checkpoint=ckpt).to(device).eval()
    for p in sam.parameters():
        p.requires_grad_(False)
    return EfficientSamFrozenEncoder(model=sam, device=device)


def overlay(image_rgb: Image.Image, mask_bool: np.ndarray, x: float, y: float) -> Image.Image:
    img = np.asarray(image_rgb).astype(np.uint8)
    out = img.copy()
    color = np.array([30, 144, 255], dtype=np.uint8)
    alpha = 0.45
    out[mask_bool] = (out[mask_bool] * (1.0 - alpha) + color * alpha).astype(np.uint8)
    out_img = Image.fromarray(out)
    draw = ImageDraw.Draw(out_img)
    r = max(3, int(round(min(out_img.size) * 0.006)))
    draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0), width=3)
    return out_img


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    ckpt = torch.load(args.weights, map_location="cpu")
    head = ClickPersonHead(in_embed_ch=256, base_ch=128).to(device).eval()
    head.load_state_dict(ckpt["head"], strict=True)

    encoder = _build_encoder(args.efficientsam_repo, args.efficientsam_ckpt, device=device)

    img_pil = Image.open(args.image).convert("RGB")
    w, h = img_pil.size
    img_t = transforms.ToTensor()(np.asarray(img_pil).copy()).unsqueeze(0).to(device)

    # map click to 1024-space then 64-space
    x1024 = args.x * (TARGET_SIZE / float(w))
    y1024 = args.y * (TARGET_SIZE / float(h))
    x64 = torch.tensor([x1024 * (EMBED_SIZE / float(TARGET_SIZE))], device=device)
    y64 = torch.tensor([y1024 * (EMBED_SIZE / float(TARGET_SIZE))], device=device)

    with torch.no_grad():
        embed = encoder.encode_image(img_t)
        click64 = make_gaussian_map(EMBED_SIZE, EMBED_SIZE, x64, y64, float(args.sigma64), device, embed.dtype)
        logits = head(embed, click64)
        prob = torch.sigmoid(logits)

    # recover to original size (inverse of 1024 stretch)
    prob_orig = F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
    mask = (prob_orig >= 0.5)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(os.path.join(args.outdir, "mask.png"))
    overlay(img_pil, mask, args.x, args.y).save(os.path.join(args.outdir, "overlay.png"))
    print("saved:", os.path.join(args.outdir, "mask.png"))
    print("saved:", os.path.join(args.outdir, "overlay.png"))


if __name__ == "__main__":
    main()


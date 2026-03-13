#!/usr/bin/env python3
"""Replay paste-back with a larger final keep mask."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay paste-back with mask dilation.")
    parser.add_argument("--base-image", required=True, help="Original source image.")
    parser.add_argument("--aligned-edit", required=True, help="Aligned edited image, e.g. pointer_orb_aligned.png.")
    parser.add_argument("--mask", required=True, help="Refined mask image, e.g. pointer_mask_refined.png.")
    parser.add_argument("--output", required=True, help="Output final composited image.")
    parser.add_argument("--mask-dilate", type=float, default=6.0, help="Final paste-back mask dilation radius.")
    parser.add_argument("--feather", type=float, default=3.0, help="Gaussian blur radius for final alpha.")
    parser.add_argument("--save-mask", default=None, help="Optional path to save the dilated paste-back mask.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    aligned = Image.open(args.aligned_edit).convert("RGB")
    width, height = aligned.size

    base = Image.open(args.base_image).convert("RGB")
    if base.size != (width, height):
        base = base.resize((width, height), Image.BILINEAR)

    mask = Image.open(args.mask).convert("L")
    if mask.size != (width, height):
        mask = mask.resize((width, height), Image.NEAREST)
    mask = mask.point(lambda p: 255 if p >= 128 else 0)

    dilate_radius = max(0.0, float(args.mask_dilate))
    if dilate_radius > 0:
        mask_arr = np.array(mask, dtype=np.uint8)
        inv = np.where(mask_arr > 0, 0, 255).astype(np.uint8)
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
        keep_arr = np.where((mask_arr > 0) | (dist <= dilate_radius), 255, 0).astype(np.uint8)
        keep_mask = Image.fromarray(keep_arr, mode="L")
    else:
        keep_mask = mask

    if args.save_mask:
        Path(args.save_mask).parent.mkdir(parents=True, exist_ok=True)
        keep_mask.save(args.save_mask)

    feather_mask = keep_mask
    if float(args.feather) > 0:
        feather_mask = feather_mask.filter(ImageFilter.GaussianBlur(radius=float(args.feather)))

    orig_arr = np.array(base, dtype=np.uint8)
    aligned_arr = np.array(aligned, dtype=np.uint8)
    keep_arr = np.array(keep_mask, dtype=np.uint8)
    feather_arr = np.array(feather_mask, dtype=np.float32)[..., None] / 255.0

    # Use the dilated keep mask for both content retention and final blend.
    recovered_arr = np.where((keep_arr > 0)[..., None], aligned_arr, orig_arr).astype(np.uint8)
    composite_arr = (recovered_arr * feather_arr + orig_arr * (1.0 - feather_arr)).astype(np.uint8)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(composite_arr, mode="RGB").save(output_path)
    print(f"[pasteback] saved final image to {output_path}")
    if args.save_mask:
        print(f"[pasteback] saved dilated mask to {args.save_mask}")


if __name__ == "__main__":
    main()

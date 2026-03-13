#!/usr/bin/env python3
"""Merge aligned edited isolated image over original isolated image, then paste back."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Use isolated image as background and aligned edit as foreground.")
    parser.add_argument("--base-image", required=True, help="Original source image.")
    parser.add_argument("--isolated-image", required=True, help="Original black-background isolated image.")
    parser.add_argument("--aligned-edit", required=True, help="Aligned edited image on black background.")
    parser.add_argument("--output", required=True, help="Output final composited image.")
    parser.add_argument("--black-threshold", type=int, default=8, help="Threshold for extracting support masks.")
    parser.add_argument("--edit-choke", type=float, default=0.0, help="Shrink aligned edit support inward by this many pixels.")
    parser.add_argument("--save-merged", default=None, help="Optional path to save merged isolated-domain image.")
    parser.add_argument("--save-edit-mask", default=None, help="Optional path to save foreground edit support mask.")
    parser.add_argument("--save-iso-mask", default=None, help="Optional path to save isolated support mask.")
    return parser


def load_rgb(path: str, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def main() -> None:
    args = build_parser().parse_args()

    aligned = load_rgb(args.aligned_edit)
    h, w = aligned.shape[:2]
    size = (w, h)
    isolated = load_rgb(args.isolated_image, size=size)
    base = load_rgb(args.base_image, size=size)

    thr = int(args.black_threshold)
    edit_support = aligned.max(axis=2) > thr
    iso_support = isolated.max(axis=2) > thr
    choke = max(0.0, float(args.edit_choke))
    if choke > 0:
        support_u8 = (edit_support.astype(np.uint8) * 255)
        dist = cv2.distanceTransform(support_u8, cv2.DIST_L2, 5)
        edit_support = dist > choke

    merged = np.where(edit_support[..., None], aligned, isolated).astype(np.uint8)
    final = np.where(iso_support[..., None], merged, base).astype(np.uint8)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(final, mode="RGB").save(output_path)

    if args.save_merged:
        Path(args.save_merged).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(merged, mode="RGB").save(args.save_merged)
    if args.save_edit_mask:
        Path(args.save_edit_mask).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((edit_support.astype(np.uint8) * 255), mode="L").save(args.save_edit_mask)
    if args.save_iso_mask:
        Path(args.save_iso_mask).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((iso_support.astype(np.uint8) * 255), mode="L").save(args.save_iso_mask)

    print(f"[merge-isolated-edit] saved final image to {output_path}")
    if args.save_merged:
        print(f"[merge-isolated-edit] saved merged isolated-domain image to {args.save_merged}")
    if args.save_edit_mask:
        print(f"[merge-isolated-edit] saved edit support mask to {args.save_edit_mask}")
    if args.save_iso_mask:
        print(f"[merge-isolated-edit] saved isolated support mask to {args.save_iso_mask}")


if __name__ == "__main__":
    main()

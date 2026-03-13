#!/usr/bin/env python3
"""Fix dark edge fringe on an aligned isolate-edit patch, then paste back."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inpaint dark edge fringe on an aligned patch, then paste back.")
    parser.add_argument("--base-image", required=True, help="Original source image.")
    parser.add_argument("--aligned-edit", required=True, help="Aligned edited patch on black background.")
    parser.add_argument("--output", required=True, help="Output composited image.")
    parser.add_argument("--black-threshold", type=int, default=8, help="Pixels with max RGB <= threshold are outside support.")
    parser.add_argument("--ring-width", type=int, default=3, help="Only inspect this many pixels inward from the support edge.")
    parser.add_argument("--dark-threshold", type=int, default=28, help="Dark pixels inside the ring are treated as fringe.")
    parser.add_argument("--inpaint-radius", type=float, default=2.0, help="OpenCV inpaint radius.")
    parser.add_argument("--save-mask", default=None, help="Optional path to save detected fringe mask.")
    parser.add_argument("--save-fixed", default=None, help="Optional path to save the repaired aligned patch.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    aligned = Image.open(args.aligned_edit).convert("RGB")
    width, height = aligned.size

    base = Image.open(args.base_image).convert("RGB")
    if base.size != (width, height):
        base = base.resize((width, height), Image.BILINEAR)

    aligned_arr = np.array(aligned, dtype=np.uint8)
    base_arr = np.array(base, dtype=np.uint8)

    support = (aligned_arr.max(axis=2) > int(args.black_threshold)).astype(np.uint8) * 255

    ring_width = max(1, int(args.ring_width))
    kernel = np.ones((2 * ring_width + 1, 2 * ring_width + 1), dtype=np.uint8)
    eroded = cv2.erode(support, kernel, iterations=1)
    ring = cv2.subtract(support, eroded)

    dark = (aligned_arr.max(axis=2) <= int(args.dark_threshold)).astype(np.uint8) * 255
    fringe_mask = cv2.bitwise_and(ring, dark)

    # Expand a touch so the inpaint region fully covers the dark halo.
    fringe_mask = cv2.dilate(fringe_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)

    fixed_bgr = cv2.inpaint(
        cv2.cvtColor(aligned_arr, cv2.COLOR_RGB2BGR),
        fringe_mask,
        inpaintRadius=float(args.inpaint_radius),
        flags=cv2.INPAINT_TELEA,
    )
    fixed_arr = cv2.cvtColor(fixed_bgr, cv2.COLOR_BGR2RGB)

    alpha = (support.astype(np.float32) / 255.0)[..., None]
    composite_arr = (fixed_arr * alpha + base_arr * (1.0 - alpha)).astype(np.uint8)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(composite_arr, mode="RGB").save(output_path)

    if args.save_mask:
        mask_path = Path(args.save_mask)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(fringe_mask, mode="L").save(mask_path)

    if args.save_fixed:
        fixed_path = Path(args.save_fixed)
        fixed_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(fixed_arr, mode="RGB").save(fixed_path)

    print(f"[replay-inpaint] saved final image to {output_path}")
    if args.save_mask:
        print(f"[replay-inpaint] saved fringe mask to {args.save_mask}")
    if args.save_fixed:
        print(f"[replay-inpaint] saved repaired patch to {args.save_fixed}")


if __name__ == "__main__":
    main()

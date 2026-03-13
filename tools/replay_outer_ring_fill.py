#!/usr/bin/env python3
"""Fill the thin black outer ring around a hard support mask, then composite."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fill a thin black outer ring around a support mask.")
    parser.add_argument("--base-image", required=True, help="Original source image.")
    parser.add_argument("--aligned-edit", required=True, help="Aligned edited patch on black background.")
    parser.add_argument("--support-mask", required=True, help="Hard non-black support mask.")
    parser.add_argument("--output", required=True, help="Output composited image.")
    parser.add_argument("--expand", type=int, default=2, help="Outer ring width in pixels.")
    parser.add_argument("--inpaint-radius", type=float, default=2.0, help="OpenCV inpaint radius.")
    parser.add_argument("--black-threshold", type=int, default=8, help="Threshold for recomputing support after repair.")
    parser.add_argument("--save-ring", default=None, help="Optional path to save the repaired outer-ring mask.")
    parser.add_argument("--save-fixed", default=None, help="Optional path to save the repaired patch.")
    parser.add_argument("--save-mask", default=None, help="Optional path to save the recomputed hard support mask.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    aligned = Image.open(args.aligned_edit).convert("RGB")
    width, height = aligned.size
    aligned_arr = np.array(aligned, dtype=np.uint8)

    base = Image.open(args.base_image).convert("RGB")
    if base.size != (width, height):
        base = base.resize((width, height), Image.BILINEAR)
    base_arr = np.array(base, dtype=np.uint8)

    support_mask = Image.open(args.support_mask).convert("L")
    if support_mask.size != (width, height):
        support_mask = support_mask.resize((width, height), Image.NEAREST)
    support = (np.array(support_mask, dtype=np.uint8) > 127).astype(np.uint8) * 255

    expand = max(1, int(args.expand))
    kernel = np.ones((2 * expand + 1, 2 * expand + 1), dtype=np.uint8)
    dilated = cv2.dilate(support, kernel, iterations=1)
    outer_ring = cv2.subtract(dilated, support)

    # Only repair truly dark pixels in that immediate outer ring.
    dark = (aligned_arr.max(axis=2) <= int(args.black_threshold)).astype(np.uint8) * 255
    repair_mask = cv2.bitwise_and(outer_ring, dark)

    fixed_bgr = cv2.inpaint(
        cv2.cvtColor(aligned_arr, cv2.COLOR_RGB2BGR),
        repair_mask,
        inpaintRadius=float(args.inpaint_radius),
        flags=cv2.INPAINT_TELEA,
    )
    fixed_arr = cv2.cvtColor(fixed_bgr, cv2.COLOR_BGR2RGB)

    repaired_support = (fixed_arr.max(axis=2) > int(args.black_threshold)).astype(np.uint8) * 255
    alpha = (repaired_support.astype(np.float32) / 255.0)[..., None]
    composite_arr = (fixed_arr * alpha + base_arr * (1.0 - alpha)).astype(np.uint8)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(composite_arr, mode="RGB").save(output_path)

    if args.save_ring:
        ring_path = Path(args.save_ring)
        ring_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(repair_mask, mode="L").save(ring_path)

    if args.save_fixed:
        fixed_path = Path(args.save_fixed)
        fixed_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(fixed_arr, mode="RGB").save(fixed_path)

    if args.save_mask:
        mask_path = Path(args.save_mask)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(repaired_support, mode="L").save(mask_path)

    print(f"[outer-ring-fill] saved final image to {output_path}")
    if args.save_ring:
        print(f"[outer-ring-fill] saved repair ring to {args.save_ring}")
    if args.save_fixed:
        print(f"[outer-ring-fill] saved repaired patch to {args.save_fixed}")
    if args.save_mask:
        print(f"[outer-ring-fill] saved repaired support mask to {args.save_mask}")


if __name__ == "__main__":
    main()

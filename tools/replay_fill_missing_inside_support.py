#!/usr/bin/env python3
"""Fill missing pixels where edited support shrank inside the original isolated support."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use isolated support as final boundary and inpaint the missing inner ring on aligned edit."
    )
    parser.add_argument("--base-image", required=True, help="Original source image.")
    parser.add_argument("--isolated-image", required=True, help="Original black-background isolated image.")
    parser.add_argument("--aligned-edit", required=True, help="ORB-aligned edited image on black background.")
    parser.add_argument("--output", required=True, help="Output composited image.")
    parser.add_argument("--black-threshold", type=int, default=8, help="Threshold for support extraction.")
    parser.add_argument("--inpaint-radius", type=float, default=2.0, help="OpenCV inpaint radius.")
    parser.add_argument("--save-gap-mask", default=None, help="Optional path to save iso_support - edited_support mask.")
    parser.add_argument("--save-iso-mask", default=None, help="Optional path to save isolated support mask.")
    parser.add_argument("--save-edit-mask", default=None, help="Optional path to save aligned edited support mask.")
    parser.add_argument("--save-fixed", default=None, help="Optional path to save repaired aligned patch.")
    return parser


def _load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def main() -> None:
    args = build_parser().parse_args()

    aligned_arr = _load_rgb(args.aligned_edit)
    height, width = aligned_arr.shape[:2]

    isolated_arr = _load_rgb(args.isolated_image)
    if isolated_arr.shape[:2] != (height, width):
        isolated_arr = np.array(
            Image.fromarray(isolated_arr, mode="RGB").resize((width, height), Image.BILINEAR),
            dtype=np.uint8,
        )

    base_arr = _load_rgb(args.base_image)
    if base_arr.shape[:2] != (height, width):
        base_arr = np.array(
            Image.fromarray(base_arr, mode="RGB").resize((width, height), Image.BILINEAR),
            dtype=np.uint8,
        )

    thr = int(args.black_threshold)
    iso_support = (isolated_arr.max(axis=2) > thr).astype(np.uint8) * 255
    edit_support = (aligned_arr.max(axis=2) > thr).astype(np.uint8) * 255

    gap_mask = cv2.subtract(iso_support, edit_support)
    # Slightly expand so inpaint fully covers the shrink gap at the boundary.
    gap_mask = cv2.dilate(gap_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    gap_mask = cv2.bitwise_and(gap_mask, iso_support)

    fixed_bgr = cv2.inpaint(
        cv2.cvtColor(aligned_arr, cv2.COLOR_RGB2BGR),
        gap_mask,
        inpaintRadius=float(args.inpaint_radius),
        flags=cv2.INPAINT_TELEA,
    )
    fixed_arr = cv2.cvtColor(fixed_bgr, cv2.COLOR_BGR2RGB)

    alpha = (iso_support.astype(np.float32) / 255.0)[..., None]
    composite_arr = (fixed_arr * alpha + base_arr * (1.0 - alpha)).astype(np.uint8)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(composite_arr, mode="RGB").save(output_path)

    if args.save_gap_mask:
        Path(args.save_gap_mask).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(gap_mask, mode="L").save(args.save_gap_mask)
    if args.save_iso_mask:
        Path(args.save_iso_mask).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(iso_support, mode="L").save(args.save_iso_mask)
    if args.save_edit_mask:
        Path(args.save_edit_mask).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(edit_support, mode="L").save(args.save_edit_mask)
    if args.save_fixed:
        Path(args.save_fixed).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(fixed_arr, mode="RGB").save(args.save_fixed)

    print(f"[fill-missing-support] saved final image to {output_path}")
    if args.save_gap_mask:
        print(f"[fill-missing-support] saved gap mask to {args.save_gap_mask}")
    if args.save_fixed:
        print(f"[fill-missing-support] saved repaired patch to {args.save_fixed}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Use support overlap to locate the shrink fringe, then inpaint that band."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Overlap-guided fringe fill for aligned edited patch.")
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--isolated-image", required=True)
    parser.add_argument("--aligned-edit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--black-threshold", type=int, default=8)
    parser.add_argument("--gap-expand", type=int, default=2, help="Expand iso-edit gap inward/outward by this many pixels.")
    parser.add_argument("--inpaint-radius", type=float, default=2.0)
    parser.add_argument("--save-gap", default=None)
    parser.add_argument("--save-target", default=None)
    parser.add_argument("--save-fixed", default=None)
    return parser


def load_rgb(path: str, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def save_gray(arr: np.ndarray, path: str | None) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(path)


def save_rgb(arr: np.ndarray, path: str | None) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)


def main() -> None:
    args = build_parser().parse_args()

    aligned = load_rgb(args.aligned_edit)
    h, w = aligned.shape[:2]
    size = (w, h)
    isolated = load_rgb(args.isolated_image, size=size)
    base = load_rgb(args.base_image, size=size)

    thr = int(args.black_threshold)
    iso_support = (isolated.max(axis=2) > thr).astype(np.uint8) * 255
    edit_support = (aligned.max(axis=2) > thr).astype(np.uint8) * 255

    gap = cv2.subtract(iso_support, edit_support)
    expand = max(1, int(args.gap_expand))
    kernel = np.ones((2 * expand + 1, 2 * expand + 1), dtype=np.uint8)

    # The actual black line typically sits just inside the edited support next to the shrink gap.
    target = cv2.dilate(gap, kernel, iterations=1)
    target = cv2.bitwise_and(target, iso_support)

    fixed_bgr = cv2.inpaint(
        cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR),
        target,
        inpaintRadius=float(args.inpaint_radius),
        flags=cv2.INPAINT_TELEA,
    )
    fixed = cv2.cvtColor(fixed_bgr, cv2.COLOR_BGR2RGB)

    alpha = (iso_support.astype(np.float32) / 255.0)[..., None]
    composite = (fixed * alpha + base * (1.0 - alpha)).astype(np.uint8)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(composite, mode="RGB").save(out)

    save_gray(gap, args.save_gap)
    save_gray(target, args.save_target)
    save_rgb(fixed, args.save_fixed)

    print(f"[overlap-guided-fill] saved final image to {out}")
    if args.save_gap:
        print(f"[overlap-guided-fill] saved gap mask to {args.save_gap}")
    if args.save_target:
        print(f"[overlap-guided-fill] saved repair target to {args.save_target}")
    if args.save_fixed:
        print(f"[overlap-guided-fill] saved repaired patch to {args.save_fixed}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Replay isolate-edit paste-back without refined-mask restriction."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ORB-align edited isolated image and paste it back without mask gating.")
    parser.add_argument("--base-image", required=True, help="Original source image.")
    parser.add_argument("--isolated-image", required=True, help="Black-background isolated input image.")
    parser.add_argument("--edited-isolated", required=True, help="Edited isolated output image.")
    parser.add_argument("--output", required=True, help="Output final composited image.")
    parser.add_argument("--save-aligned", default=None, help="Optional path to save the ORB-aligned image.")
    parser.add_argument("--save-mask", default=None, help="Optional path to save the non-black paste mask.")
    parser.add_argument("--black-threshold", type=int, default=8, help="Pixels with max RGB <= threshold are treated as background.")
    parser.add_argument("--feather", type=float, default=3.0, help="Gaussian blur radius for the non-black paste mask.")
    parser.add_argument("--mask-choke", type=int, default=0, help="Erode the non-black paste mask inward by this many pixels.")
    parser.add_argument(
        "--border-mode",
        choices=["replicate", "constant"],
        default="replicate",
        help="OpenCV warpAffine border mode for ORB alignment.",
    )
    return parser


def orb_align(
    edited: np.ndarray,
    reference: np.ndarray,
    max_features: int = 5000,
    border_mode: str = "replicate",
) -> np.ndarray:
    gray_edit = cv2.cvtColor(edited, cv2.COLOR_RGB2GRAY)
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(gray_edit, None)
    kp2, des2 = orb.detectAndCompute(gray_ref, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return edited

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 4:
        return edited

    matches = sorted(matches, key=lambda m: m.distance)
    matches = matches[: max(10, len(matches) // 3)]

    pts_edit = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts_ref = np.float32([kp2[m.trainIdx].pt for m in matches])

    matrix, _ = cv2.estimateAffinePartial2D(
        pts_edit, pts_ref, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if matrix is None:
        return edited

    h, w = reference.shape[:2]
    cv_border_mode = cv2.BORDER_REPLICATE if border_mode == "replicate" else cv2.BORDER_CONSTANT
    return cv2.warpAffine(
        edited,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv_border_mode,
        borderValue=(0, 0, 0),
    )


def main() -> None:
    args = build_parser().parse_args()

    edited = Image.open(args.edited_isolated).convert("RGB")
    width, height = edited.size

    isolated = Image.open(args.isolated_image).convert("RGB")
    if isolated.size != (width, height):
        isolated = isolated.resize((width, height), Image.BILINEAR)

    base = Image.open(args.base_image).convert("RGB")
    if base.size != (width, height):
        base = base.resize((width, height), Image.BILINEAR)

    edited_arr = np.array(edited, dtype=np.uint8)
    isolated_arr = np.array(isolated, dtype=np.uint8)
    base_arr = np.array(base, dtype=np.uint8)

    aligned_arr = orb_align(edited_arr, isolated_arr, border_mode=args.border_mode)
    aligned_img = Image.fromarray(aligned_arr, mode="RGB")

    if args.save_aligned:
        save_aligned = Path(args.save_aligned)
        save_aligned.parent.mkdir(parents=True, exist_ok=True)
        aligned_img.save(save_aligned)

    non_black = (aligned_arr.max(axis=2) > int(args.black_threshold)).astype(np.uint8) * 255
    if int(args.mask_choke) > 0:
        kernel = max(3, 2 * int(args.mask_choke) + 1)
        non_black = cv2.erode(non_black, np.ones((kernel, kernel), dtype=np.uint8), iterations=1)
    mask_img = Image.fromarray(non_black, mode="L")
    if float(args.feather) > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=float(args.feather)))

    if args.save_mask:
        save_mask = Path(args.save_mask)
        save_mask.parent.mkdir(parents=True, exist_ok=True)
        mask_img.save(save_mask)

    alpha = np.array(mask_img, dtype=np.float32)[..., None] / 255.0
    composite_arr = (aligned_arr * alpha + base_arr * (1.0 - alpha)).astype(np.uint8)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(composite_arr, mode="RGB").save(output_path)
    print(f"[replay-full-orb] saved final image to {output_path}")
    if args.save_aligned:
        print(f"[replay-full-orb] saved aligned image to {args.save_aligned}")
    if args.save_mask:
        print(f"[replay-full-orb] saved non-black mask to {args.save_mask}")


if __name__ == "__main__":
    main()

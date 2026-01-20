"""CLI script to generate pointer soft masks and optional Kontext edits."""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageFilter

from pointer_lora.inference import PointerSoftMaskPipeline


def _load_yaml(path: str):
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pointer LoRA soft-mask inference")
    parser.add_argument("--config", default="pointer_lora/pointer_lora_config.yaml", help="路径：训练用 YAML（复用 pointer 配置）")
    parser.add_argument(
        "--lora-weights",
        default="/root/autodl-tmp/pointer_lora/output/20251201-141737/weights/lora_step_007000.pt",
        help="指针 LoRA 权重路径",
    )
    parser.add_argument("--refiner-weights", default=None, help="mask refiner 权重路径")
    parser.add_argument("--image", default=None, help="原始图像路径（可在 job YAML 中指定）")
    parser.add_argument(
        "--prompt",
        default=None,
        help="单句 prompt 作为默认基准：未显式提供 --pointer-prompt/--edit-prompt 时会复用它（可为空，用于解耦）",
    )
    parser.add_argument(
        "--pointer-prompt",
        default=None,
        help="定位 prompt（若不填则回退为 --prompt；若 --prompt 也为空且提供了 --edit-prompt，则会复用 edit_prompt 做定位）",
    )
    parser.add_argument("--edit-prompt", default=None, help="编辑 prompt（若不填则回退为 --prompt；仅描述要改什么，建议不包含位置信息）")
    parser.add_argument("--output-dir", default="pointer_lora/output/softmask_infer", help="输出目录")
    parser.add_argument("--width", type=int, default=None, help="编辑输出宽度（默认跟原图保持一致，需为 16 的倍数）")
    parser.add_argument("--height", type=int, default=None, help="编辑输出高度（默认跟原图保持一致，需为 16 的倍数）")
    parser.add_argument("--steps", type=int, default=20, help="Kontext 推理步数")
    parser.add_argument("--guidance-scale", type=float, default=3.5, help="CFG guidance scale")
    parser.add_argument("--true-cfg-scale", type=float, default=1.0, help="Kontext true CFG scale（>1 需要 negative_prompt 才生效）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", default=None, help="可选：cuda / cuda:1 / cpu，默认自动检测")
    parser.add_argument("--negative-prompt", default=None, help="负向 prompt（用于 true_cfg_scale>1 的 true-CFG）")
    parser.add_argument(
        "--edit-backend",
        choices=["kontext", "qwen"],
        default="kontext",
        help="编辑后端：kontext=FluxKontextInpaintPipeline，qwen=Qwen inpaint/edit pipeline",
    )
    parser.add_argument("--qwen-strength", type=float, default=None, help="Qwen inpaint strength（0~1，越小越保留原图）")
    parser.add_argument("--qwen-mode", choices=["inpaint", "edit"], default="inpaint", help="Qwen backend 模式：inpaint 或 edit")
    parser.add_argument(
        "--qwen-controlnet-path",
        type=str,
        default=None,
        help="Qwen ControlNet Inpainting 模型路径（默认从 QWEN_CONTROLNET_PATH 环境变量读取，或 /root/autodl-tmp/Qwen-Image-ControlNet-Inpainting）",
    )
    parser.add_argument("--latent-inpaint-strength", type=float, default=1.0, help="latents 初始化时在 mask 内注入噪声的强度（0=不注入；1=全注入）")
    parser.add_argument("--mask-feather", type=float, default=6.0, help="mask feather/blur 半径（像素），允许轻微外溢")
    parser.add_argument("--mask-dilate", type=int, default=0, help="mask 膨胀像素（在编辑前扩大 mask 范围）")
    parser.add_argument(
        "--latent-blend-alpha",
        type=float,
        default=0.6,
        help="每步 soft pull-back 强度 α（0=关闭；1=mask 外完全拉回原图；推荐 0.4~0.8）",
    )
    parser.add_argument("--conditioning-masked-image", action="store_true", help="将 mask 区域在条件图里挖空/填噪声（inpainting范式，推荐开启）")
    parser.add_argument("--conditioning-fill", choices=["noise", "gray"], default="noise", help="条件图挖空区域的填充方式")
    parser.add_argument("--conditioning-noise-strength", type=float, default=1.0, help="条件图填噪声强度（0~1 推荐）")
    parser.add_argument(
        "--use-inpaint-pipeline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="优先使用 diffusers 的 FluxKontextInpaintPipeline（显式 mask_image 通道；推荐开启）",
    )
    parser.add_argument(
        "--inpaint-auto-resize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否允许 FluxKontextInpaintPipeline 自动选择偏好的 Kontext 分辨率（默认关闭，避免 mask 坐标系变化导致看似“不吃mask”）。",
    )
    parser.add_argument(
        "--append-bbox-hint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="将 mask 的 bbox 坐标（像素+百分比）追加到编辑 prompt（英文），用于减少“想改错人→mask内摆烂不改”。",
    )
    parser.add_argument(
        "--bbox-threshold",
        type=float,
        default=0.5,
        help="计算 bbox 时的 mask 二值化阈值（0~1）。",
    )
    parser.add_argument(
        "--padding-mask-crop",
        type=int,
        default=None,
        help="Kontext Inpaint 的 padding_mask_crop：自动围绕 mask 裁剪再还原（mask 很小时推荐 16~96）",
    )
    parser.add_argument("--crop-edit", action="store_true", help="使用 mask bbox 裁剪后再编辑，避免模型改错人（推荐）")
    parser.add_argument("--crop-pad", type=int, default=48, help="裁剪 bbox 外扩像素（兼容旧参数；未指定 crop_pad_x/y 时生效）")
    parser.add_argument("--crop-pad-x", type=int, default=None, help="裁剪 bbox 左右外扩像素（优先生效）")
    parser.add_argument("--crop-pad-y", type=int, default=None, help="裁剪 bbox 上下外扩像素（优先生效）")
    parser.add_argument("--crop-full-height", action="store_true", help="单排/横向排布场景：裁剪框上下直接拉满到整张图高度")
    parser.add_argument("--crop-feather", type=float, default=10.0, help="贴回原图时 mask feather 半径（像素）")
    parser.add_argument(
        "--crop-paste-mode",
        choices=["mask", "full"],
        default="mask",
        help="crop_edit 的贴回方式：mask=用mask合成(更保守)；full=直接整块贴回(不压制任何改动，可能有接缝)",
    )
    parser.add_argument("--crop-dilate", type=int, default=0, help="mask 贴回前膨胀像素（放宽mask，覆盖边缘残留/半截问题）")
    parser.add_argument("--crop-alpha-gamma", type=float, default=0.85, help="mask alpha 的 gamma（<1 更放宽，>1 更严格）")
    parser.add_argument("--clickseg-checkpoint", type=str, default=None, help="ClickSEG/isegm HRNet checkpoint 路径（.pth），开启后可用 clickseg mask 生成")
    parser.add_argument("--clickseg-threshold", type=float, default=0.40, help="ClickSEG 输出阈值，默认 0.40")
    parser.add_argument("--clickseg-filter-component", action="store_true", default=True, help="仅保留包含点击点的连通域，去除其它块")
    parser.add_argument("--clickseg-no-filter-component", dest="clickseg_filter_component", action="store_false")
    parser.add_argument("--clickseg-use-prev-mask", action="store_true", default=False, help="使用 pointer mask 作为 prev_mask 输入 ClickSEG（默认不用，纯点击）")
    parser.add_argument("--clickseg-infer-size", type=int, default=384, help="ClickSEG 推理输入尺寸")
    parser.add_argument("--clickseg-prev-mask-scale", type=float, default=0.9, help="ClickSEG prev_mask 缩放系数（配合 clickseg-use-prev-mask）")
    parser.add_argument("--clickseg-save-noprev", action="store_true", default=False, help="额外保存不使用 prev_mask 的 ClickSEG mask")
    parser.add_argument(
        "--mask-mode",
        choices=["peak_region", "refiner", "gaussian", "clickseg"],
        default="peak_region",
        help="soft mask 生成策略：peak_region=峰值连通域阈值扩张(不走refiner), refiner=高斯hint+可选refiner, gaussian=仅高斯hint",
    )
    parser.add_argument("--peak-region-threshold", type=float, default=0.5, help="峰值连通域阈值：rel=peak*ratio / abs=absolute")
    parser.add_argument("--peak-region-threshold-mode", choices=["rel", "abs"], default="rel", help="峰值连通域阈值模式")
    parser.add_argument("--peak-region-connectivity", type=int, choices=[4, 8], default=8, help="连通域邻域：4 或 8")
    parser.add_argument("--peak-region-max-iters", type=int, default=2048, help="连通域扩张最大迭代次数（安全上限）")
    parser.add_argument("--no-peak-region-normalize", action="store_true", help="不对 peak_region 输出做 max=1 归一化")
    parser.add_argument("--job-config", default=None, help="推理 YAML，提供上述所有字段，命令行仅需这一项")
    return parser.parse_args()


def _create_run_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base_dir / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def main() -> None:
    args = parse_args()

    if args.job_config:
        job_cfg = _load_yaml(args.job_config)
        model_cfg = job_cfg.get("model", {})
        infer_cfg = job_cfg.get("inference", job_cfg)

        args.config = model_cfg.get("config", args.config)
        args.lora_weights = model_cfg.get("lora_weights", args.lora_weights)
        args.refiner_weights = model_cfg.get("refiner_weights", args.refiner_weights)
        for key in [
            "image",
            "prompt",
            "pointer_prompt",
            "edit_prompt",
            "negative_prompt",
            "output_dir",
            "width",
            "height",
            "steps",
            "guidance_scale",
            "true_cfg_scale",
            "seed",
            "device",
            "edit_backend",
            "qwen_strength",
            "qwen_mode",
            "qwen_controlnet_path",
            "latent_blend_alpha",
            "latent_inpaint_strength",
            "mask_feather",
            "mask_dilate",
            "conditioning_masked_image",
            "conditioning_fill",
            "conditioning_noise_strength",
            "use_inpaint_pipeline",
            "padding_mask_crop",
            "crop_edit",
            "crop_pad",
            "crop_pad_x",
            "crop_pad_y",
            "crop_full_height",
            "crop_feather",
            "crop_paste_mode",
            "crop_dilate",
            "crop_alpha_gamma",
            "clickseg_checkpoint",
            "clickseg_threshold",
            "clickseg_filter_component",
            "clickseg_use_prev_mask",
            "clickseg_infer_size",
            "clickseg_prev_mask_scale",
            "clickseg_save_noprev",
            "mask_mode",
            "peak_region_threshold",
            "peak_region_threshold_mode",
            "peak_region_connectivity",
            "peak_region_max_iters",
            "no_peak_region_normalize",
        ]:
            if key in infer_cfg:
                setattr(args, key, infer_cfg[key])

    if not args.image:
        raise ValueError("必须提供 --image（可在 job YAML 的 inference 段落中指定）")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _create_run_dir(output_dir)
    print(f"[soft-mask] run directory: {run_dir}")

    pipeline = PointerSoftMaskPipeline(
        config_path=args.config,
        lora_weights=args.lora_weights,
        device=args.device,
        refiner_weights=args.refiner_weights,
        qwen_controlnet_path=getattr(args, "qwen_controlnet_path", None),
    )

    # Prompt resolution strategy:
    # - If user provides a single `--prompt`, it is used for both pointer + edit by default.
    # - If user provides `--pointer-prompt` / `--edit-prompt`, they override independently.
    # - If `--prompt` is omitted but `--edit-prompt` is provided, fall back to using edit_prompt for pointer too.
    base_prompt = args.prompt
    pointer_prompt = args.pointer_prompt or base_prompt or args.edit_prompt
    edit_prompt = args.edit_prompt or base_prompt

    if not pointer_prompt:
        raise ValueError("必须提供定位用的 prompt：--pointer-prompt（推荐）或 --prompt（或仅提供 --edit-prompt 也可回退复用）")

    print(f"[soft-mask] pointer_prompt: {pointer_prompt!r}")
    print(f"[soft-mask] edit_prompt: {edit_prompt!r}")

    edit_backend = str(getattr(args, "edit_backend", "kontext")).lower().strip()

    pointer_map = pipeline.compute_pointer_map(args.image, pointer_prompt)
    if args.mask_mode == "peak_region":
        refined_mask = pipeline.build_peak_region_mask(
            threshold=float(args.peak_region_threshold),
            threshold_mode=str(args.peak_region_threshold_mode),
            connectivity=int(args.peak_region_connectivity),
            max_iters=int(args.peak_region_max_iters),
            normalize=not bool(args.no_peak_region_normalize),
        )
    elif args.mask_mode == "gaussian":
        refined_mask = pipeline.build_gaussian_mask()
    elif args.mask_mode == "clickseg":
        if not args.clickseg_checkpoint:
            raise ValueError("mask_mode=clickseg 需要提供 --clickseg-checkpoint")
        # If we want prev_mask for ClickSEG, build a pointer mask first and stash it for reuse.
        if bool(getattr(args, "clickseg_use_prev_mask", False)):
            pipeline._last_refined_mask = pipeline.build_peak_region_mask(
                threshold=float(args.peak_region_threshold),
                threshold_mode=str(args.peak_region_threshold_mode),
                connectivity=int(args.peak_region_connectivity),
                max_iters=int(args.peak_region_max_iters),
                normalize=not bool(args.no_peak_region_normalize),
            ) * float(getattr(args, "clickseg_prev_mask_scale", 0.9))
        refined_mask = pipeline.build_clickseg_mask(
            image_path=args.image,
            clickseg_checkpoint=args.clickseg_checkpoint,
            clickseg_threshold=float(getattr(args, "clickseg_threshold", 0.40)),
            clickseg_filter_component=bool(getattr(args, "clickseg_filter_component", True)),
            clickseg_use_prev_mask=bool(getattr(args, "clickseg_use_prev_mask", False)),
            clickseg_infer_size=int(getattr(args, "clickseg_infer_size", 384)),
            clickseg_prev_mask_scale=float(getattr(args, "clickseg_prev_mask_scale", 1.0)),
        )
        if bool(getattr(args, "clickseg_save_noprev", False)):
            no_prev = pipeline.build_clickseg_mask(
                image_path=args.image,
                clickseg_checkpoint=args.clickseg_checkpoint,
                clickseg_threshold=float(getattr(args, "clickseg_threshold", 0.40)),
                clickseg_filter_component=bool(getattr(args, "clickseg_filter_component", True)),
                clickseg_use_prev_mask=False,
                clickseg_infer_size=int(getattr(args, "clickseg_infer_size", 384)),
                clickseg_prev_mask_scale=1.0,
            )
            noprev_img = pipeline._mask_tensor_to_image(no_prev)
            noprev_path = run_dir / "pointer_mask_clickseg_noprev.png"
            noprev_img.save(noprev_path)
            print(f"[soft-mask] Saved ClickSEG mask (no prev) to {noprev_path}")
    else:
        refined_mask = pipeline.build_refined_mask()
    artifacts = pipeline.save_pointer_artifacts(pointer_map, refined_mask, args.image, run_dir, prefix="pointer")
    print("[soft-mask] Saved artifacts:", artifacts)
    peak_info = pipeline.get_last_peak_info()
    if peak_info:
        print(
            "[soft-mask] peak:",
            f"map_norm=({peak_info['map_norm'][0]:.4f},{peak_info['map_norm'][1]:.4f}), "
            f"orig_norm=({peak_info['norm'][0]:.4f},{peak_info['norm'][1]:.4f})",
        )

    if edit_prompt:
        if edit_backend == "qwen":
            print("[soft-mask] offloading Kontext modules before Qwen to save GPU memory...")
            pipeline.offload_kontext()
        print(f"[soft-mask] using seed: {args.seed if args.seed is not None else 'random'}")
        if args.width is None or args.height is None:
            with Image.open(args.image) as img:
                img_width, img_height = img.size
            width = max(16, ((img_width + 15) // 16) * 16)
            height = max(16, ((img_height + 15) // 16) * 16)
        else:
            width = max(16, args.width // 16 * 16)
            height = max(16, args.height // 16 * 16)
        if bool(getattr(args, "crop_edit", False)):
            # 1) Work in the same edit resolution space
            base = Image.open(args.image).convert("RGB")
            if base.size != (width, height):
                base = base.resize((width, height), Image.BILINEAR)
            # 2) Build mask at edit resolution
            mask_for_edit = pipeline._resize_mask_for_edit(refined_mask, width, height)
            mask_img = pipeline._mask_tensor_to_image(mask_for_edit, size=(width, height)).convert("L")
            mask_dilate = int(getattr(args, "mask_dilate", 0))
            if mask_dilate > 0:
                # PIL MaxFilter size must be odd.
                k = max(3, 2 * mask_dilate + 1)
                mask_img = mask_img.filter(ImageFilter.MaxFilter(size=k))
            m = (Image.eval(mask_img, lambda p: 255 if p >= 128 else 0))
            import numpy as np
            arr = np.array(m, dtype=np.uint8)
            ys, xs = np.where(arr > 0)
            if ys.size == 0:
                raise RuntimeError("mask is empty; cannot crop-edit.")
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            pad_default = int(getattr(args, "crop_pad", 48))
            pad_x = getattr(args, "crop_pad_x", None)
            pad_y = getattr(args, "crop_pad_y", None)
            pad_x = int(pad_x) if pad_x is not None else pad_default
            pad_y = int(pad_y) if pad_y is not None else pad_default
            x0 = max(0, x0 - pad_x); y0 = max(0, y0 - pad_y)
            x1 = min(width - 1, x1 + pad_x); y1 = min(height - 1, y1 + pad_y)
            if bool(getattr(args, "crop_full_height", False)):
                y0 = 0
                y1 = height - 1

            crop_box = (x0, y0, x1 + 1, y1 + 1)
            crop_img = base.crop(crop_box)
            crop_mask = mask_img.crop(crop_box)

            # Round crop size to multiple of 16 for Kontext.
            cw, ch = crop_img.size
            cw2 = max(16, (cw // 16) * 16)
            ch2 = max(16, (ch // 16) * 16)
            if (cw2, ch2) != (cw, ch):
                crop_img = crop_img.resize((cw2, ch2), Image.BILINEAR)
                crop_mask = crop_mask.resize((cw2, ch2), Image.BILINEAR)

            # If inpaint pipeline is enabled, do crop+inpaint:
            # - Inpaint enforces the mask inside the crop
            # - We can paste the whole crop back (no alpha composite), minimizing seams and avoiding residue.
            if edit_backend == "qwen":
                qwen_mode = str(getattr(args, "qwen_mode", "inpaint")).lower().strip()
                qwen_backend = pipeline._get_qwen_backend(qwen_mode)
                if qwen_mode == "edit":
                    full_edit = qwen_backend.run_edit(
                        image_pil=crop_img,
                        prompt=edit_prompt,
                        seed=args.seed,
                        steps=args.steps,
                        guidance=args.guidance_scale,
                        out_size=(cw2, ch2),
                        negative_prompt=args.negative_prompt,
                        true_cfg_scale=args.true_cfg_scale,
                    )
                    pipeline._last_qwen_full_edit = full_edit.copy()
                    pasted_crop = Image.composite(full_edit.convert("RGB"), crop_img.convert("RGB"), crop_mask.convert("L"))
                    pipeline._last_qwen_mask_used = crop_mask.copy()
                    pipeline._last_qwen_image_resized = qwen_backend.last_image_resized
                    pasted = base.copy()
                    pasted.paste(pasted_crop, (x0, y0))
                else:
                    crop_edit = qwen_backend.run_inpaint(
                        image_pil=crop_img,
                        mask_pil=crop_mask,
                        prompt=edit_prompt,
                        seed=args.seed,
                        steps=args.steps,
                        guidance=args.guidance_scale,
                        out_size=(cw2, ch2),
                        negative_prompt=args.negative_prompt,
                        true_cfg_scale=args.true_cfg_scale,
                        strength=getattr(args, "qwen_strength", None),
                    )
                    pipeline._last_qwen_mask_used = qwen_backend.last_mask_used
                    pipeline._last_qwen_image_resized = qwen_backend.last_image_resized
                    pasted = base.copy()
                    pasted.paste(crop_edit.convert("RGB"), (x0, y0))
            else:
                use_inpaint = bool(getattr(args, "use_inpaint_pipeline", True))
                if use_inpaint and getattr(pipeline, "inpaint_pipeline", None) is not None:
                    crop_edit = pipeline.edit_image_inpaint(
                        image=crop_img,
                        mask_image_l=crop_mask,
                        edit_prompt=edit_prompt,
                        negative_prompt=args.negative_prompt,
                        width=cw2,
                        height=ch2,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance_scale,
                        true_cfg_scale=args.true_cfg_scale,
                        seed=args.seed,
                        strength=float(getattr(args, "latent_inpaint_strength", 1.0)),
                        mask_feather=float(getattr(args, "crop_feather", 2.0)),
                        padding_mask_crop=None,
                    )
                    pasted = base.copy()
                    pasted.paste(crop_edit.convert("RGB"), (x0, y0))
                else:
                    # Fallback: plain Kontext edit on crop, then composite back with soft mask.
                    crop_edit = pipeline.edit_image(
                        image=crop_img,
                        edit_prompt=edit_prompt,
                        negative_prompt=args.negative_prompt,
                        width=cw2,
                        height=ch2,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance_scale,
                        true_cfg_scale=args.true_cfg_scale,
                        seed=args.seed,
                    )

                    paste_mode = str(getattr(args, "crop_paste_mode", "mask"))
                    pasted = base.copy()
                    if paste_mode == "full":
                        # No mask suppression: paste the whole edited crop back.
                        pasted.paste(crop_edit.convert("RGB"), (x0, y0))
                    else:
                        # Masked composite (conservative). Optionally relax via dilate+gamma.
                        a = crop_mask.convert("L")
                        dil = int(getattr(args, "crop_dilate", 0))
                        if dil > 0:
                            # PIL MaxFilter size must be odd.
                            k = max(3, 2 * dil + 1)
                            a = a.filter(ImageFilter.MaxFilter(size=k))
                        feather = float(getattr(args, "crop_feather", 10.0))
                        if feather > 0:
                            a = a.filter(ImageFilter.GaussianBlur(radius=feather))
                        gamma = float(getattr(args, "crop_alpha_gamma", 1.0))
                        if gamma > 0 and abs(gamma - 1.0) > 1e-3:
                            import numpy as np

                            aa = np.array(a, dtype=np.float32) / 255.0
                            aa = np.clip(aa, 0.0, 1.0) ** gamma
                            a = Image.fromarray((aa * 255.0).astype(np.uint8), mode="L")
                        if crop_edit.size != a.size:
                            crop_edit = crop_edit.resize(a.size, Image.BILINEAR)
                        pasted_crop = Image.composite(crop_edit.convert("RGB"), crop_img.convert("RGB"), a)
                        pasted.paste(pasted_crop, (x0, y0))

            crop_path = run_dir / "pointer_edit_crop.png"
            crop_edit.save(crop_path)
            output_path = run_dir / "pointer_edit.png"
            pasted.save(output_path)
            print(f"[soft-mask] Saved crop edit to {crop_path}")
            print(f"[soft-mask] Saved edited image to {output_path}")
        else:
            result = pipeline.edit_with_soft_mask(
                refined_mask=refined_mask,
                source_image_path=args.image,
                edit_prompt=edit_prompt,
                negative_prompt=args.negative_prompt,
                width=width,
                height=height,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                true_cfg_scale=args.true_cfg_scale,
                seed=args.seed,
                latent_inpaint_strength=float(getattr(args, "latent_inpaint_strength", 1.0)),
                mask_feather=float(getattr(args, "mask_feather", 6.0)),
                latent_blend_alpha=float(getattr(args, "latent_blend_alpha", 0.6)),
                conditioning_masked_image=bool(getattr(args, "conditioning_masked_image", False)),
                conditioning_fill=str(getattr(args, "conditioning_fill", "noise")),
                conditioning_noise_strength=float(getattr(args, "conditioning_noise_strength", 1.0)),
                use_inpaint_pipeline=bool(getattr(args, "use_inpaint_pipeline", True)),
                padding_mask_crop=getattr(args, "padding_mask_crop", None),
                inpaint_auto_resize=bool(getattr(args, "inpaint_auto_resize", False)),
                append_bbox_hint=bool(getattr(args, "append_bbox_hint", True)),
                bbox_threshold=float(getattr(args, "bbox_threshold", 0.5)),
                edit_backend=edit_backend,
                mask_dilate=int(getattr(args, "mask_dilate", 0)),
                qwen_strength=getattr(args, "qwen_strength", None),
                qwen_mode=str(getattr(args, "qwen_mode", "inpaint")),
            )
            output_path = run_dir / "pointer_edit.png"
            result.save(output_path)
            print(f"[soft-mask] Saved edited image to {output_path}")
            inpaint_mask = getattr(pipeline, "_last_inpaint_mask_image", None)
            if inpaint_mask is not None:
                mpath = run_dir / "pointer_inpaint_mask_used.png"
                inpaint_mask.save(mpath)
                print(f"[soft-mask] Saved inpaint mask used to {mpath}")
            cond = getattr(pipeline, "_last_conditioning_image", None)
            if cond is not None:
                cond_path = run_dir / "pointer_conditioning_masked.png"
                cond.save(cond_path)
                print(f"[soft-mask] Saved conditioning image to {cond_path}")
        if edit_backend == "qwen":
            qwen_mask = getattr(pipeline, "_last_qwen_mask_used", None)
            if qwen_mask is not None:
                qwen_mask_path = run_dir / "qwen_mask_used.png"
                qwen_mask.save(qwen_mask_path)
                print(f"[soft-mask] Saved qwen mask to {qwen_mask_path}")
            qwen_img = getattr(pipeline, "_last_qwen_image_resized", None)
            if qwen_img is not None:
                qwen_img_path = run_dir / "qwen_image_resized.png"
                qwen_img.save(qwen_img_path)
                print(f"[soft-mask] Saved qwen resized image to {qwen_img_path}")
            qwen_full = getattr(pipeline, "_last_qwen_full_edit", None)
            if qwen_full is not None:
                qwen_full_path = run_dir / "qwen_edit_full.png"
                qwen_full.save(qwen_full_path)
                print(f"[soft-mask] Saved qwen full edit to {qwen_full_path}")
    else:
        print("[soft-mask] Edit prompt未提供，仅导出软掩码。")


if __name__ == "__main__":
    main()

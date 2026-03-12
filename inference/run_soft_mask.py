"""CLI script to generate pointer soft masks and optional Kontext edits."""

from __future__ import annotations

import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from PIL import Image

from . import PointerSoftMaskPipeline


def _load_yaml(path: str):
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_external_mask(path: str | Path) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    arr = np.array(mask, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).clamp(0.0, 1.0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pointer LoRA soft-mask inference")
    parser.add_argument("--config", default="/workspace/MOREdit/pointer_lora_config.yaml", help="路径：训练用 YAML（复用 pointer 配置）")
    parser.add_argument(
        "--lora-weights",
        default="/workspace/MOREdit/output/20260307-092048/weights/lora_step_007000.pt",
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
    parser.add_argument("--output-dir", default="/workspace/MOREdit/output/softmask_infer", help="输出目录")
    parser.add_argument("--width", type=int, default=None, help="编辑输出宽度（默认跟原图保持一致，需为 16 的倍数）")
    parser.add_argument("--height", type=int, default=None, help="编辑输出高度（默认跟原图保持一致，需为 16 的倍数）")
    parser.add_argument("--steps", type=int, default=20, help="Qwen 推理步数")
    parser.add_argument("--guidance-scale", type=float, default=3.5, help="CFG guidance scale")
    parser.add_argument("--true-cfg-scale", type=float, default=1.0, help="Qwen true CFG scale（>1 需要 negative_prompt 才生效）")
    parser.add_argument("--seed", default="42", help="随机种子；可填整数，或填 random 使用非固定随机种子")
    parser.add_argument("--device", default=None, help="可选：cuda / cuda:1 / cpu，默认自动检测")
    parser.add_argument("--negative-prompt", default=None, help="负向 prompt（用于 true_cfg_scale>1 的 true-CFG）")
    parser.add_argument(
        "--edit-backend",
        choices=["qwen"],
        default="qwen",
        help="编辑后端：仅支持 qwen",
    )
    parser.add_argument("--qwen-strength", type=float, default=None, help="Qwen inpaint strength（0~1，越小越保留原图）")
    parser.add_argument("--qwen-mode", choices=["inpaint", "edit"], default="inpaint", help="Qwen backend 模式：inpaint 或 edit")
    parser.add_argument(
        "--qwen-controlnet-path",
        type=str,
        default=None,
        help="Qwen ControlNet Inpainting 模型路径（默认从 QWEN_CONTROLNET_PATH 环境变量读取，或 /workspace/models/Qwen-Image-ControlNet-Inpainting）",
    )
    parser.add_argument(
        "--qwen-model-path",
        type=str,
        default=None,
        help="Qwen 编辑模型路径；仅影响编辑后端，不影响前面的 pointer LoRA 主模型。",
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
        help='是否允许 FluxKontextInpaintPipeline 自动选择偏好的 Kontext 分辨率（默认关闭，避免 mask 坐标系变化导致看似"不吃mask"）。',
    )
    parser.add_argument(
        "--append-bbox-hint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help='将 mask 的 bbox 坐标（像素+百分比）追加到编辑 prompt（英文），用于减少"想改错人->mask内摆烂不改"。',
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
    parser.add_argument(
        "--isolate-edit",
        action="store_true",
        help="整图黑底隔离→Qwen Edit→ORB对齐→mask外区域填回原图→原始mask feathered composite",
    )
    parser.add_argument("--isolate-dilate", type=int, default=12, help="黑底隔离时的 mask 膨胀半径（像素），默认 12")
    parser.add_argument("--no-orb-align", action="store_true", help="关闭 ORB 对齐（直接贴回）")
    parser.add_argument("--paste-feather", type=float, default=3.0, help="贴回原图时原始 mask 的 feather 半径（像素）")
    parser.add_argument("--paste-choke", type=int, default=0, help="贴回前先将最终 mask 内缩的像素数，用于压掉边缘亮线/脏边")
    parser.add_argument("--clickseg-checkpoint", type=str, default=None, help="ClickSEG/isegm HRNet checkpoint 路径（.pth），开启后可用 clickseg mask 生成")
    parser.add_argument("--clickseg-threshold", type=float, default=0.40, help="ClickSEG 输出阈值，默认 0.40")
    parser.add_argument("--clickseg-filter-component", action="store_true", default=True, help="仅保留包含点击点的连通域，去除其它块")
    parser.add_argument("--clickseg-no-filter-component", dest="clickseg_filter_component", action="store_false")
    parser.add_argument("--clickseg-use-prev-mask", action="store_true", default=False, help="使用 pointer mask 作为 prev_mask 输入 ClickSEG（默认不用，纯点击）")
    parser.add_argument("--clickseg-infer-size", type=int, default=384, help="ClickSEG 推理输入尺寸")
    parser.add_argument("--clickseg-prev-mask-scale", type=float, default=0.9, help="ClickSEG prev_mask 缩放系数（配合 clickseg-use-prev-mask）")
    parser.add_argument("--clickseg-save-noprev", action="store_true", default=False, help="额外保存不使用 prev_mask 的 ClickSEG mask")
    parser.add_argument(
        "--refined-mask-path",
        type=str,
        default=None,
        help="直接使用现成 refined mask PNG/L 图，跳过 pointer/mask 生成，只跑后半段编辑与贴回。",
    )
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
    parser.add_argument("--decompose-lora", default=None, help="Prompt 解耦 LoRA 权重路径（仅在提供 --prompt 且未显式提供 --pointer-prompt/--edit-prompt 时生效）")
    parser.add_argument("--decompose-base-model", default="/workspace/models/Qwen2.5-0.5B-Instruct", help="Prompt 解耦用的 base 模型路径")
    return parser


def parse_args() -> argparse.Namespace:
    return _build_parser().parse_args()


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
    parser = _build_parser()
    args = parser.parse_args()

    if args.job_config:
        job_cfg = _load_yaml(args.job_config)
        model_cfg = job_cfg.get("model", {})
        infer_cfg = job_cfg.get("inference", job_cfg)

        if args.config == parser.get_default("config"):
            args.config = model_cfg.get("config", args.config)
        if args.lora_weights == parser.get_default("lora_weights"):
            args.lora_weights = model_cfg.get("lora_weights", args.lora_weights)
        if args.refiner_weights == parser.get_default("refiner_weights"):
            args.refiner_weights = model_cfg.get("refiner_weights", args.refiner_weights)
        for key in [
            "image",
            "prompt",
            "pointer_prompt",
            "edit_prompt",
            "decompose_lora",
            "decompose_base_model",
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
            "qwen_model_path",
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
            "isolate_edit",
            "isolate_dilate",
            "no_orb_align",
            "paste_feather",
            "paste_choke",
            "clickseg_checkpoint",
            "clickseg_threshold",
            "clickseg_filter_component",
            "clickseg_use_prev_mask",
            "clickseg_infer_size",
            "clickseg_prev_mask_scale",
            "clickseg_save_noprev",
            "refined_mask_path",
            "mask_mode",
            "peak_region_threshold",
            "peak_region_threshold_mode",
            "peak_region_connectivity",
            "peak_region_max_iters",
            "no_peak_region_normalize",
            "controlnet_conditioning_scale",
        ]:
            if key in infer_cfg and getattr(args, key) == parser.get_default(key):
                setattr(args, key, infer_cfg[key])
        if (
            "crop_isolate_edit" in infer_cfg
            and "isolate_edit" not in infer_cfg
            and args.isolate_edit == parser.get_default("isolate_edit")
        ):
            args.isolate_edit = infer_cfg["crop_isolate_edit"]
        if (
            "crop_feather" in infer_cfg
            and "paste_feather" not in infer_cfg
            and args.paste_feather == parser.get_default("paste_feather")
        ):
            args.paste_feather = infer_cfg["crop_feather"]

    seed_value = getattr(args, "seed", None)
    if isinstance(seed_value, str):
        seed_text = seed_value.strip().lower()
        if seed_text in ("", "none", "null", "random"):
            args.seed = None
        else:
            args.seed = int(seed_value)
    elif seed_value is not None:
        args.seed = int(seed_value)

    if not args.image:
        raise ValueError("必须提供 --image（可在 job YAML 的 inference 段落中指定）")

    using_external_mask = bool(getattr(args, "refined_mask_path", None))

    # Prompt resolution strategy:
    # - If user provides a single `--prompt` (no explicit pointer/edit prompts) and --decompose-lora is set,
    #   automatically decompose the prompt into pointer + edit via the LoRA model.
    #   Decomposition runs BEFORE the main pipeline loads to avoid GPU memory contention.
    # - If user provides `--pointer-prompt` / `--edit-prompt`, they override independently (no decompose).
    # - If `--prompt` is omitted but `--edit-prompt` is provided, fall back to using edit_prompt for pointer too.
    base_prompt = args.prompt
    decompose_lora = getattr(args, "decompose_lora", None)
    if base_prompt and not args.pointer_prompt and not args.edit_prompt and decompose_lora and not using_external_mask:
        from MOREdit.inference.prompt_decompose import PromptDecomposer
        decomposer = PromptDecomposer(
            base_model=getattr(args, "decompose_base_model", "/workspace/models/Qwen2.5-0.5B-Instruct"),
            lora_weights=decompose_lora,
            device=str(getattr(args, "device", "cpu") or "cpu"),
        )
        pointer_prompt, edit_prompt = decomposer.decompose(base_prompt)
        decomposer.offload()  # free GPU before loading main pipeline
    else:
        pointer_prompt = args.pointer_prompt or base_prompt or (None if using_external_mask else args.edit_prompt)
        edit_prompt = args.edit_prompt or base_prompt

    if not pointer_prompt and not using_external_mask:
        raise ValueError("必须提供定位用的 prompt：--pointer-prompt（推荐）或 --prompt（或仅提供 --edit-prompt 也可回退复用）")

    if using_external_mask and not pointer_prompt:
        print(f"[soft-mask] pointer_prompt: <skipped; using refined_mask_path={args.refined_mask_path}>")
    else:
        print(f"[soft-mask] pointer_prompt: {pointer_prompt!r}")
    print(f"[soft-mask] edit_prompt: {edit_prompt!r}")

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
        qwen_model_path=getattr(args, "qwen_model_path", None),
        skip_pointer_model_init=using_external_mask,
    )

    edit_backend = str(getattr(args, "edit_backend", "kontext")).lower().strip()

    if using_external_mask:
        mask_path = Path(args.refined_mask_path)
        if not mask_path.exists():
            raise FileNotFoundError(f"refined mask not found: {mask_path}")
        refined_mask = _load_external_mask(mask_path)
        pipeline._last_refined_mask = refined_mask.clone()
        refined_img = pipeline._mask_tensor_to_image(refined_mask)
        refined_path = run_dir / "pointer_mask_refined.png"
        refined_img.save(refined_path)
        print(f"[soft-mask] using external refined mask: {mask_path}")
        print(f"[soft-mask] Saved refined mask copy to {refined_path}")
    else:
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
            if bool(getattr(args, "isolate_edit", False)):
                width = max(16, (img_width // 16) * 16)
                height = max(16, (img_height // 16) * 16)
            else:
                width = max(16, ((img_width + 15) // 16) * 16)
                height = max(16, ((img_height + 15) // 16) * 16)
        else:
            width = max(16, args.width // 16 * 16)
            height = max(16, args.height // 16 * 16)
        if bool(getattr(args, "isolate_edit", False)):
            result = pipeline.isolate_edit_pasteback(
                source_image_path=args.image,
                refined_mask=refined_mask,
                edit_prompt=edit_prompt,
                width=width,
                height=height,
                isolate_dilate=int(getattr(args, "isolate_dilate", 16)),
                paste_feather=float(getattr(args, "paste_feather", 10.0)),
                paste_choke=int(getattr(args, "paste_choke", 0)),
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                true_cfg_scale=args.true_cfg_scale,
                seed=args.seed,
                negative_prompt=args.negative_prompt,
                use_orb_align=not bool(getattr(args, "no_orb_align", False)),
                qwen_mode=str(getattr(args, "qwen_mode", "edit")),
                qwen_strength=getattr(args, "qwen_strength", None),
                controlnet_conditioning_scale=getattr(args, "controlnet_conditioning_scale", None),
            )
            output_path = run_dir / "pointer_edit.png"
            result.save(output_path)
            print(f"[soft-mask] Saved isolate-edit result to {output_path}")
            isolated = getattr(pipeline, "_last_isolated_crop", None)
            if isolated is not None:
                isolated.save(run_dir / "pointer_isolated_image.png")
                print(f"[soft-mask] Saved isolated image to {run_dir / 'pointer_isolated_image.png'}")
            edited_iso = getattr(pipeline, "_last_edited_isolated", None)
            if edited_iso is not None:
                edited_iso.save(run_dir / "pointer_edited_isolated.png")
                print(f"[soft-mask] Saved edited isolated image to {run_dir / 'pointer_edited_isolated.png'}")
            orb_aligned = getattr(pipeline, "_last_orb_aligned", None)
            if orb_aligned is not None:
                orb_aligned.save(run_dir / "pointer_orb_aligned.png")
                print(f"[soft-mask] Saved ORB-aligned image to {run_dir / 'pointer_orb_aligned.png'}")
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
                controlnet_conditioning_scale=getattr(args, "controlnet_conditioning_scale", None),
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

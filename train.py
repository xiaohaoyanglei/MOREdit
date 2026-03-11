"""Entry point for pointer-style Qwen LoRA training."""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional

import yaml

from .losses import PointerLossConfig
from .lora import LoRAInjectionConfig
from .trainer import PointerTrainer, PointerTrainerConfig


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _parse_bool(value) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pointer-style LoRA on Qwen image models.")
    parser.add_argument("--config", type=str, help="Path to YAML configuration.")

    parser.add_argument("--model-path", default=None, help="Path to the local Qwen image model directory.")
    parser.add_argument("--data-path", default=None, help="Path to annotations JSON/JSONL file.")
    parser.add_argument("--data-root", default=None, help="Root directory containing images and masks.")
    parser.add_argument("--output-dir", default=None, help="Directory to save checkpoints and heatmaps.")

    parser.add_argument("--steps", type=int, default=None, help="Total training steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate for LoRA parameters.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay.")
    parser.add_argument("--dtype", type=str, default=None, help="Computation dtype (float32|float16|bf16).")
    parser.add_argument("--log-every", type=int, default=None, help="Logging frequency.")
    parser.add_argument("--heatmap-every", type=int, default=None, help="Heatmap saving interval.")
    parser.add_argument("--heatmap-max-samples", type=int, default=None, help="Heatmap samples per interval.")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="LoRA checkpoint interval.")
    parser.add_argument("--pointer-superres-factor", type=int, default=None, help="Pointer super-resolution scale.")
    parser.add_argument(
        "--pointer-superres-sharpness",
        type=float,
        default=None,
        help="Laplacian sharpening applied after super-resolution.",
    )
    parser.add_argument("--pointer-token-mode", type=str, default=None, help="Pointer token selection mode.")
    parser.add_argument(
        "--pointer-token-filters",
        nargs="*",
        default=None,
        help="Additional token filters to include during pointer aggregation.",
    )
    parser.add_argument(
        "--pointer-heatmap-combine",
        type=str,
        default=None,
        help="Pointer heatmap combination strategy (mean|max).",
    )
    parser.add_argument(
        "--pointer-temperature",
        type=float,
        default=None,
        help="Pointer heatmap temperature before normalization.",
    )
    parser.add_argument(
        "--pointer-epsilon",
        type=float,
        default=None,
        help="Pointer heatmap epsilon to avoid division by zero.",
    )
    parser.add_argument(
        "--pointer-debug-every",
        type=int,
        default=None,
        help="Step interval for reporting pointer heatmap grid sizes.",
    )

    parser.add_argument("--rank", type=int, default=None, help="LoRA rank.")
    parser.add_argument("--alpha", type=float, default=None, help="LoRA alpha scaling.")

    parser.add_argument("--bce-weight", type=float, default=None, help="Pointer loss BCE weight.")
    parser.add_argument("--dice-weight", type=float, default=None, help="Pointer loss Dice weight.")
    parser.add_argument("--dice-smooth", type=float, default=None, help="Pointer loss Dice smoothing term.")
    parser.add_argument("--loss-background-weight", type=float, default=None, help="Background suppression weight.")
    parser.add_argument("--loss-contrast-weight", type=float, default=None, help="Contrast margin weight.")
    parser.add_argument("--loss-contrast-margin", type=float, default=None, help="Contrast margin threshold.")

    parser.add_argument(
        "--offload-text-encoders",
        type=str,
        default=None,
        help="Whether to keep Qwen text encoder on CPU (true/false).",
    )

    return parser.parse_args()


def _value_from_config(
    config: Dict[str, Any],
    section: str,
    key: str,
    fallback,
    default=None,
):
    if fallback is not None:
        return fallback
    if section in config and key in config[section]:
        return config[section][key]
    return default


def _resolve_configs(args: argparse.Namespace) -> Dict[str, Any]:
    yaml_config: Dict[str, Any] = {}
    if args.config:
        yaml_config = _load_yaml(args.config)

    offload_flag = _parse_bool(
        _value_from_config(
            yaml_config,
            "model",
            "offload_text_encoders",
            args.offload_text_encoders,
            False,
        )
    )
    if offload_flag is None:
        offload_flag = False

    test_section = yaml_config.get("test_image", {})

    pointer_section = yaml_config.get("pointer", {})
    heatmap_section = pointer_section.get("heatmap", {})
    superres_section = pointer_section.get("superres", {})

    token_filters_cfg = pointer_section.get("token_filters", [])
    if not isinstance(token_filters_cfg, (list, tuple)):
        raise ValueError("pointer.token_filters must be a list if specified.")

    pointer_token_filters = tuple(args.pointer_token_filters) if args.pointer_token_filters is not None else tuple(token_filters_cfg)
    pointer_token_mode = (args.pointer_token_mode or pointer_section.get("token_mode", "phrase")).lower()
    pointer_heatmap_combine = (
        args.pointer_heatmap_combine or heatmap_section.get("combine", "mean")
    )
    pointer_temperature = (
        args.pointer_temperature if args.pointer_temperature is not None else float(heatmap_section.get("temperature", 1.0))
    )
    pointer_epsilon = (
        args.pointer_epsilon if args.pointer_epsilon is not None else float(heatmap_section.get("epsilon", 1.0e-6))
    )
    pointer_superres_factor = (
        args.pointer_superres_factor if args.pointer_superres_factor is not None else int(superres_section.get("factor", 1))
    )
    pointer_superres_sharpness = (
        args.pointer_superres_sharpness
        if args.pointer_superres_sharpness is not None
        else float(superres_section.get("sharpness", 0.0))
    )

    trainer_cfg = PointerTrainerConfig(
        data_path=_value_from_config(yaml_config, "data", "annotations", args.data_path),
        data_root=_value_from_config(yaml_config, "data", "root", args.data_root),
        model_path=_value_from_config(yaml_config, "model", "path", args.model_path),
        output_dir=_value_from_config(yaml_config, "training", "output_dir", args.output_dir),
        resolution=_value_from_config(yaml_config, "data", "resolution", getattr(args, "resolution", None), 640),
        batch_size=_value_from_config(yaml_config, "training", "batch_size", args.batch_size, 1),
        num_steps=_value_from_config(yaml_config, "training", "steps", args.steps, 10000),
        lr=_value_from_config(yaml_config, "training", "lr", args.lr, 1e-4),
        weight_decay=_value_from_config(yaml_config, "training", "weight_decay", args.weight_decay, 0.0),
        seed=_value_from_config(yaml_config, "training", "seed", getattr(args, "seed", None), 42),
        dtype=_value_from_config(yaml_config, "training", "dtype", args.dtype, "float32"),
        log_every=_value_from_config(yaml_config, "training", "log_every", args.log_every, 10),
        heatmap_every=_value_from_config(yaml_config, "training", "heatmap_every", args.heatmap_every, 1000),
        heatmap_max_samples=_value_from_config(
            yaml_config, "training", "heatmap_max_samples", args.heatmap_max_samples, 2
        ),
        checkpoint_every=_value_from_config(
            yaml_config, "training", "checkpoint_every", args.checkpoint_every, 1000
        ),
        pointer_superres_factor=max(1, pointer_superres_factor),
        pointer_superres_sharpness=max(0.0, pointer_superres_sharpness),
        offload_text_encoders=offload_flag,
        test_image_path=test_section.get("path"),
        test_mask_path=test_section.get("mask"),
        test_prompt=test_section.get("prompt"),
        pointer_debug_every=_value_from_config(
            yaml_config,
            "training",
            "pointer_debug_every",
            args.pointer_debug_every,
            0,
        ),
        pointer_token_mode=pointer_token_mode,
        pointer_token_filters=pointer_token_filters,
        pointer_heatmap_combine=str(pointer_heatmap_combine).lower(),
        pointer_temperature=pointer_temperature,
        pointer_epsilon=pointer_epsilon,
    )

    lora_cfg = LoRAInjectionConfig(
        rank=_value_from_config(yaml_config, "lora", "rank", args.rank, 16),
        alpha=_value_from_config(yaml_config, "lora", "alpha", args.alpha, 32.0),
        attn_block_prefixes=tuple(
            _value_from_config(
                yaml_config,
                "lora",
                "attn_blocks",
                getattr(args, "attn_blocks", None),
                (
                    "transformer_blocks.14",
                    "transformer_blocks.15",
                    "transformer_blocks.16",
                ),
            )
        ),
        target_projections=tuple(
            _value_from_config(
                yaml_config,
                "lora",
                "projections",
                getattr(args, "projections", None),
                (
                    "to_q",
                    "to_k",
                    "to_v",
                    "add_q_proj",
                    "add_k_proj",
                    "add_v_proj",
                    "to_add_out",
                ),
            )
        ),
    )

    pointer_loss_cfg = PointerLossConfig(
        bce_weight=_value_from_config(yaml_config, "loss", "bce_weight", args.bce_weight, 1.0),
        dice_weight=_value_from_config(yaml_config, "loss", "dice_weight", args.dice_weight, 0.5),
        smooth=_value_from_config(yaml_config, "loss", "smooth", args.dice_smooth, 1.0e-6),
        background_weight=_value_from_config(
            yaml_config,
            "loss",
            "background_weight",
            args.loss_background_weight,
            0.0,
        ),
        contrast_weight=_value_from_config(
            yaml_config,
            "loss",
            "contrast_weight",
            args.loss_contrast_weight,
            0.0,
        ),
        contrast_margin=_value_from_config(
            yaml_config,
            "loss",
            "contrast_margin",
            args.loss_contrast_margin,
            0.2,
        ),
    )

    required = {
        "model-path": trainer_cfg.model_path,
        "data-path": trainer_cfg.data_path,
        "data-root": trainer_cfg.data_root,
        "output-dir": trainer_cfg.output_dir,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"Missing required configuration values: {', '.join(missing)}")

    return {
        "trainer_cfg": trainer_cfg,
        "lora_cfg": lora_cfg,
        "pointer_loss_cfg": pointer_loss_cfg,
    }


def main():
    args = parse_args()
    resolved = _resolve_configs(args)

    os.makedirs(resolved["trainer_cfg"].output_dir, exist_ok=True)

    trainer = PointerTrainer(
        resolved["trainer_cfg"],
        resolved["pointer_loss_cfg"],
        resolved["lora_cfg"],
    )
    trainer.train()


if __name__ == "__main__":
    main()

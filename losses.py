"""Loss helpers for pointer-style supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class PointerLossConfig:
    bce_weight: float = 1.0
    dice_weight: float = 0.0
    smooth: float = 1.0e-6
    background_weight: float = 0.0
    contrast_weight: float = 0.0
    contrast_margin: float = 0.2

    def has_bce(self) -> bool:
        return self.bce_weight > 0.0

    def has_dice(self) -> bool:
        return self.dice_weight > 0.0

    def has_background(self) -> bool:
        return self.background_weight > 0.0

    def has_contrast(self) -> bool:
        return self.contrast_weight > 0.0


def _dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float) -> torch.Tensor:
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    intersection = (pred * target).sum(dim=-1)
    pred_sum = pred.sum(dim=-1)
    target_sum = target.sum(dim=-1)
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    return 1.0 - dice


def _masked_bce(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    if mask is None:
        return F.binary_cross_entropy(prediction, target)

    mask = mask.to(dtype=prediction.dtype)
    loss = F.binary_cross_entropy(prediction, target, reduction="none")
    loss = loss * mask
    loss = loss.reshape(loss.shape[0], -1)
    mask = mask.reshape(mask.shape[0], -1)
    denom = mask.sum(dim=-1).clamp_min(eps)
    reduced = loss.sum(dim=-1) / denom
    return reduced.mean()


def _region_mean(tensor: torch.Tensor, mask: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    masked = tensor * mask
    masked = masked.reshape(masked.shape[0], -1)
    mask = mask.reshape(mask.shape[0], -1)
    denom = mask.sum(dim=-1).clamp_min(eps)
    return masked.sum(dim=-1) / denom


def compute_pointer_losses(
    pointer_map: torch.Tensor,
    target_mask: torch.Tensor,
    config: PointerLossConfig,
) -> Dict[str, torch.Tensor]:
    pointer_map = pointer_map.to(dtype=torch.float32)
    target_mask = target_mask.to(dtype=torch.float32)

    pointer_map = pointer_map.clamp(0.0, 1.0)
    target_mask = target_mask.clamp(0.0, 1.0)

    losses: Dict[str, torch.Tensor] = {}

    if config.has_bce():
        losses["bce"] = F.binary_cross_entropy(pointer_map, target_mask, reduction="mean") * config.bce_weight

    if config.has_dice():
        dice = _dice_loss(pointer_map, target_mask, config.smooth).mean()
        losses["dice"] = dice * config.dice_weight

    if config.has_background():
        background_mask = (1.0 - target_mask).clamp(min=0.0, max=1.0)
        background_target = torch.zeros_like(pointer_map)
        bg_loss = _masked_bce(pointer_map, background_target, background_mask)
        losses["background"] = bg_loss * config.background_weight

    if config.has_contrast():
        foreground_mask = target_mask
        background_mask = (1.0 - target_mask).clamp(min=0.0, max=1.0)
        fg_mean = _region_mean(pointer_map, foreground_mask)
        bg_mean = _region_mean(pointer_map, background_mask)
        contrast_gap = fg_mean - bg_mean
        margin = torch.tensor(config.contrast_margin, dtype=contrast_gap.dtype, device=contrast_gap.device)
        contrast = torch.nn.functional.relu(margin - contrast_gap)
        losses["contrast"] = contrast.mean() * config.contrast_weight

    if not losses:
        raise ValueError("Pointer loss config disabled all terms; enable at least one of BCE or Dice.")

    return losses



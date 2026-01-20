from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(
                x1,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        out = torch.cat([x2, x1], dim=1)
        return self.conv(out)


class RefinerUNet(nn.Module):
    """A lightweight UNet that turns (image, gaussian hint) pairs into single-person masks."""

    def __init__(self, in_channels: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.up1 = Up(c * 4 + c * 2, c * 2)
        self.up2 = Up(c * 2 + c, c)
        self.outc = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        return self.outc(x)


@dataclass
class MaskRefinerConfig:
    in_channels: int = 4
    base_channels: int = 32
    weight_path: Optional[str] = None


def _normalize_state_dict(state: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "state_dict" in state:
        state = state["state_dict"]
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    return state


def load_mask_refiner(cfg: MaskRefinerConfig, device: torch.device) -> RefinerUNet:
    if not cfg.weight_path:
        raise ValueError("Mask refiner weight_path 未提供，无法加载 refiner。")
    weight_path = Path(cfg.weight_path)
    if not weight_path.exists():
        raise FileNotFoundError(f"未找到 mask refiner 权重：{weight_path}")
    model = RefinerUNet(in_channels=cfg.in_channels, base_channels=cfg.base_channels)
    state = torch.load(str(weight_path), map_location="cpu")
    state = _normalize_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[mask-refiner] warning: missing keys: {missing}")
    if unexpected:
        print(f"[mask-refiner] warning: unexpected keys: {unexpected}")
    model.to(device)
    model.eval()
    return model



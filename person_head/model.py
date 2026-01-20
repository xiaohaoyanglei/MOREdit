from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_gaussian_map(
    h: int,
    w: int,
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build a batch of gaussian maps at resolution (h,w).
    x,y are in pixel coordinates in [0,w) / [0,h), shape [B].
    Returns: [B,1,h,w] in [0,1].
    """
    xs = torch.arange(w, device=device, dtype=dtype)[None, None, None, :]  # [1,1,1,W]
    ys = torch.arange(h, device=device, dtype=dtype)[None, None, :, None]  # [1,1,H,1]
    x = x.to(device=device, dtype=dtype).view(-1, 1, 1, 1)
    y = y.to(device=device, dtype=dtype).view(-1, 1, 1, 1)
    dist2 = (xs - x) ** 2 + (ys - y) ** 2
    g = torch.exp(-dist2 / (2.0 * (sigma**2)))
    g = g / g.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return g.clamp(0.0, 1.0)


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(8, out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = ConvGNAct(in_ch, out_ch)
        self.conv2 = ConvGNAct(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ClickPersonHead(nn.Module):
    """
    Lightweight decoder head:
      input: EfficientSAM image embedding [B,256,64,64] + click gaussian [B,1,64,64]
      output: mask logits [B,1,1024,1024]
    """

    def __init__(self, in_embed_ch: int = 256, base_ch: int = 128) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvGNAct(in_embed_ch + 1, base_ch),
            ConvGNAct(base_ch, base_ch),
        )
        # 64 -> 128 -> 256 -> 512 -> 1024
        self.up1 = UpBlock(base_ch, base_ch // 2)  # 64->128
        self.up2 = UpBlock(base_ch // 2, base_ch // 4)  # 128->256
        self.up3 = UpBlock(base_ch // 4, base_ch // 8)  # 256->512
        self.up4 = UpBlock(base_ch // 8, base_ch // 16)  # 512->1024
        self.out = nn.Conv2d(base_ch // 16, 1, kernel_size=1)

    def forward(self, image_embed: torch.Tensor, click_map_64: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image_embed, click_map_64], dim=1)
        x = self.stem(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.out(x)


@dataclass
class EfficientSamFrozenEncoder:
    """
    Frozen EfficientSAM encoder wrapper.

    Exposes:
      - encode_image(image_tensor) -> embedding [B,256,64,64]

    Notes:
      - EfficientSAM preprocess resizes input to 1024x1024 (stretch) and normalizes with ImageNet mean/std.
      - You should resize GT masks to 1024x1024 in the same way for correct supervision.
    """

    model: nn.Module
    device: torch.device

    @torch.no_grad()
    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        image_tensor: [B,3,H,W] in [0,1]
        returns: [B,256,64,64]
        """
        image_tensor = image_tensor.to(self.device)
        # EfficientSAM exposes get_image_embeddings which includes preprocess (resize+normalize).
        embed = self.model.get_image_embeddings(image_tensor)
        # build_efficient_sam returns EfficientSam; get_image_embeddings returns image_encoder output tensor.
        return embed

    def train(self, mode: bool = True):
        # keep frozen
        self.model.eval()
        return self

    def eval(self):
        self.model.eval()
        return self



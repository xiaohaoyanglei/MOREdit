"""
轻量点引导 Refiner 训练脚本（不与 ai-toolkit 关联）。

思路：
- 输入：RGB（0~1） + 高斯点提示（1 通道），共 4 通道。
- 目标：单实例二值 mask。
- 点生成：从实例 mask 内随机采样一点（或质心），生成高斯提示，sigma 按短边比例设定。
- 模型：pointer_lora.inference.mask_refiner.RefinerUNet（in_channels=4, base_channels=32）。

使用示例：
python -m pointer_lora.refiner_train \
  --annotations /root/autodl-tmp/mhpv2_triples_en_val/annotations.json \
  --root /root/autodl-tmp/mhpv2_triples_en_val \
  --save-path /root/autodl-tmp/pointer_lora/output/refiner/refiner_unet_epoch2.pt \
  --epochs 2 --batch-size 8 --lr 1e-4 --sigma-ratio 0.05

默认数据期望是我们之前转好的 triples（mhpv2_triples_en_val）。若有自定义实例数据，
可自行整理成同样的 annotations.json 结构（image_path, mask_path 字段）。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from pointer_lora.inference.mask_refiner import RefinerUNet


def _load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "records" in data:
        data = data["records"]
    if not isinstance(data, list):
        raise ValueError("annotations 文件应为 list 或包含 records 字段")
    return data


def _normalize_path(path: str) -> str:
    path = path.replace("\\", "/").strip()
    if path.startswith("./"):
        path = path[2:]
    return path


def _sample_point_from_mask(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        # fallback 到中心
        return mask.shape[1] / 2.0, mask.shape[0] / 2.0
    idx = np.random.randint(0, len(xs))
    return float(xs[idx]), float(ys[idx])


def _mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return mask.shape[1] / 2.0, mask.shape[0] / 2.0
    return float(xs.mean()), float(ys.mean())


def _make_gaussian(h: int, w: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    g = np.exp(-dist_sq / (2.0 * sigma**2))
    g /= g.max().clip(min=1e-6)
    return g


@dataclass
class RefinerSample:
    image: torch.Tensor  # (3,H,W) in [0,1]
    mask: torch.Tensor  # (1,H,W)
    point: Tuple[float, float]


class PointRefinerDataset(Dataset):
    def __init__(
        self,
        annotations: str,
        root: str,
        image_dir: Optional[str] = None,
        mask_dir: Optional[str] = None,
        resolution: int = 640,
        sigma_ratio: float = 0.05,
        use_centroid_prob: float = 0.5,
        point_jitter_ratio: float = 0.0,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        records = _load_records(annotations)
        samples: List[RefinerSample] = []
        for rec in records:
            img_rel = _normalize_path(rec.get("image_path", ""))
            mask_rel = _normalize_path(rec.get("mask_path", ""))
            if not img_rel or not mask_rel:
                continue
            img_path = os.path.join(root, image_dir, img_rel) if image_dir else os.path.join(root, img_rel)
            mask_path = os.path.join(root, mask_dir, mask_rel) if mask_dir else os.path.join(root, mask_rel)
            if not (os.path.exists(img_path) and os.path.exists(mask_path)):
                continue
            samples.append(RefinerSample(image=img_path, mask=mask_path, point=(0.0, 0.0)))
        if max_samples:
            samples = samples[:max_samples]
        if not samples:
            raise ValueError("数据集中没有有效样本")
        self.samples = samples
        self.resolution = resolution
        self.sigma_ratio = sigma_ratio
        self.use_centroid_prob = use_centroid_prob
        self.point_jitter_ratio = max(0.0, float(point_jitter_ratio))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image = Image.open(sample.image).convert("RGB")
        mask_img = Image.open(sample.mask).convert("L")
        image_t = TF.to_tensor(image)  # [0,1]
        mask_t = TF.to_tensor(mask_img)  # [0,1]
        mask_bin = (mask_t > 0.5).float()

        _, h, w = image_t.shape
        if np.random.rand() < self.use_centroid_prob:
            cx, cy = _mask_centroid(mask_bin.squeeze(0).numpy())
        else:
            cx, cy = _sample_point_from_mask(mask_bin.squeeze(0).numpy())
        sigma = max(2.0, self.sigma_ratio * min(h, w))
        if self.point_jitter_ratio > 0.0:
            # Jitter the point to simulate pointer peak noise at inference.
            # jitter_radius ~= point_jitter_ratio * sigma, in pixels.
            r = self.point_jitter_ratio * sigma
            cx = float(np.clip(cx + np.random.uniform(-r, r), 0.0, w - 1.0))
            cy = float(np.clip(cy + np.random.uniform(-r, r), 0.0, h - 1.0))
        g = _make_gaussian(h, w, cx, cy, sigma)
        g_t = torch.from_numpy(g).unsqueeze(0)  # (1,H,W)

        # Letterbox to square (match inference _preprocess_image):
        # - resize to fit within target_size while preserving aspect ratio
        # - pad to (target_size, target_size)
        target_size = int(self.resolution)
        scale = target_size / max(h, w)
        resized_h = max(1, int(round(h * scale)))
        resized_w = max(1, int(round(w * scale)))

        img_batch = image_t.unsqueeze(0)
        mask_batch = mask_bin.unsqueeze(0)
        g_batch = g_t.unsqueeze(0)

        img_resized = F.interpolate(img_batch, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
        mask_resized = F.interpolate(mask_batch, size=(resized_h, resized_w), mode="nearest")
        g_resized = F.interpolate(g_batch, size=(resized_h, resized_w), mode="bilinear", align_corners=False)

        pad_top = (target_size - resized_h) // 2
        pad_bottom = target_size - resized_h - pad_top
        pad_left = (target_size - resized_w) // 2
        pad_right = target_size - resized_w - pad_left

        img_resized = F.pad(img_resized, (pad_left, pad_right, pad_top, pad_bottom))
        mask_resized = F.pad(mask_resized, (pad_left, pad_right, pad_top, pad_bottom))
        g_resized = F.pad(g_resized, (pad_left, pad_right, pad_top, pad_bottom))

        # 4 通道输入：RGB + G
        ref_in = torch.cat([img_resized, g_resized], dim=1).squeeze(0)  # (4,H,W)
        target = mask_resized.squeeze(0)  # (1,H,W)
        return {"ref_input": ref_in, "target": target}


def bce_dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(pred, target)
    prob = torch.sigmoid(pred)
    intersection = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - (2 * intersection + smooth) / (union + smooth)
    return bce + dice.mean()

def _to_u8(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().to(torch.float32).numpy()
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def save_preview(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    out_path: Path,
    max_items: int = 4,
) -> None:
    """
    Save a simple qualitative preview grid:
    [RGB | Gaussian | PredMask | GTMask] per row.
    """
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        ref_input = batch["ref_input"][:max_items].to(device)
        target = batch["target"][:max_items].to(device)
        logits = model(ref_input)
        prob = torch.sigmoid(logits)

    # ref_input: (B,4,H,W)
    rgb = ref_input[:, :3].clamp(0.0, 1.0)
    g = ref_input[:, 3:4].clamp(0.0, 1.0)
    pred = prob[:, :1].clamp(0.0, 1.0)
    gt = target[:, :1].clamp(0.0, 1.0)

    tiles: List[Image.Image] = []
    for i in range(rgb.size(0)):
        rgb_img = Image.fromarray(_to_u8(rgb[i].permute(1, 2, 0)))
        g_img = Image.fromarray(_to_u8(g[i, 0])).convert("RGB")
        pred_img = Image.fromarray(_to_u8(pred[i, 0])).convert("RGB")
        gt_img = Image.fromarray(_to_u8(gt[i, 0])).convert("RGB")
        row = Image.new("RGB", (rgb_img.width * 4, rgb_img.height))
        row.paste(rgb_img, (0, 0))
        row.paste(g_img, (rgb_img.width, 0))
        row.paste(pred_img, (rgb_img.width * 2, 0))
        row.paste(gt_img, (rgb_img.width * 3, 0))
        tiles.append(row)

    grid = Image.new("RGB", (tiles[0].width, tiles[0].height * len(tiles)))
    for r, row in enumerate(tiles):
        grid.paste(row, (0, r * row.height))
    grid.save(out_path)


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    global_step: int,
    preview_every: int = 0,
    preview_batch: Optional[Dict[str, torch.Tensor]] = None,
    preview_dir: Optional[Path] = None,
) -> Tuple[float, int]:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        global_step += 1
        ref_input = batch["ref_input"].to(device)
        target = batch["target"].to(device)
        logits = model(ref_input)
        loss = bce_dice_loss(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * ref_input.size(0)
        if preview_every and preview_batch is not None and preview_dir is not None and (global_step % preview_every == 0):
            save_preview(model, preview_batch, device, preview_dir / f"step_{global_step:07d}.png")
    return total_loss / len(loader.dataset), global_step


def evaluate(model, loader, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            ref_input = batch["ref_input"].to(device)
            target = batch["target"].to(device)
            logits = model(ref_input)
            loss = bce_dice_loss(logits, target)
            total_loss += loss.item() * ref_input.size(0)
    return total_loss / len(loader.dataset)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train point-guided lightweight refiner (RGB+Gaussian → mask)")
    ap.add_argument("--annotations", required=True, help="annotations.json 路径（包含 image_path, mask_path）")
    ap.add_argument("--root", required=True, help="数据根目录（images/masks 相对该目录）")
    ap.add_argument("--image-dir", default=None, help="可选：图片子目录（若 image_path 未含子路径，可指定如 images）")
    ap.add_argument("--mask-dir", default=None, help="可选：mask 子目录（若 mask_path 未含子路径，可指定如 masks）")
    ap.add_argument("--save-path", required=True, help="保存权重路径")
    ap.add_argument("--resolution", type=int, default=640, help="训练分辨率（平方）")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--sigma-ratio", type=float, default=0.05, help="高斯 sigma = ratio * min(H,W)")
    ap.add_argument("--use-centroid-prob", type=float, default=0.5, help="采样质心的概率（否则随机点）")
    ap.add_argument("--point-jitter-ratio", type=float, default=0.0, help="点抖动半径比例：jitter_radius = ratio * sigma（模拟 pointer peak 噪声）")
    ap.add_argument("--val-split", type=float, default=0.1, help="按样本随机切分验证集比例")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-samples", type=int, default=None, help="可选：最多使用多少条样本（用于快速试训）")
    ap.add_argument("--preview-every", type=int, default=0, help="每多少 step 导出一次可视化（0=关闭，建议 500~2000）")
    ap.add_argument("--preview-num", type=int, default=4, help="每次预览导出多少张样本")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = PointRefinerDataset(
        annotations=args.annotations,
        root=args.root,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        resolution=args.resolution,
        sigma_ratio=args.sigma_ratio,
        use_centroid_prob=args.use_centroid_prob,
        point_jitter_ratio=args.point_jitter_ratio,
        max_samples=args.max_samples,
    )
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RefinerUNet(in_channels=4, base_channels=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    preview_dir = save_path.parent / "previews"

    # fixed preview batch (from val set) for qualitative tracking
    preview_batch = None
    if args.preview_every and len(val_set) > 0:
        prev_loader = DataLoader(val_set, batch_size=args.preview_num, shuffle=False, num_workers=0)
        preview_batch = next(iter(prev_loader))

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            global_step=global_step,
            preview_every=args.preview_every,
            preview_batch=preview_batch,
            preview_dir=preview_dir,
        )
        val_loss = evaluate(model, val_loader, device)
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> saved best to {save_path}")

    print("done.")


if __name__ == "__main__":
    main()



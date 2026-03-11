"""Dataset utilities for pointer-style Qwen LoRA training."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _load_records(triples_path: str) -> List[Dict[str, Any]]:
    """Load annotation records from JSON/JSONL files."""

    if triples_path.endswith(".jsonl"):
        records: List[Dict[str, Any]] = []
        with open(triples_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    with open(triples_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
        if isinstance(data, dict) and "records" in data:
            data = data["records"]
        if not isinstance(data, list):
            raise ValueError(f"Unsupported annotation format in {triples_path}")
        return data


class PointerTripletDataset(Dataset):
    """Creates per-instance samples of image, target mask, and localization prompt."""

    def __init__(
        self,
        triples_path: str,
        root_dir: str,
        resolution: int = 640,
        pad_to_square: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.triples_path = triples_path
        self.root_dir = root_dir
        self.resolution = resolution
        self.pad_to_square = pad_to_square

        raw_records = _load_records(triples_path)
        if not raw_records:
            raise ValueError(f"No records found in {triples_path}")

        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        def _normalize_path(path: str) -> str:
            path = path.replace("\\", "/").strip()
            if path.startswith("./"):
                path = path[2:]
            prefix = "MPH2.0/converted/"
            if path.startswith(prefix):
                path = path[len(prefix) :]
            return path

        for rec in raw_records:
            image_rel = rec.get("image_path")
            mask_rel = rec.get("mask_path")
            prompt_loc = rec.get("prompt_loc")
            if image_rel is None or mask_rel is None or prompt_loc is None:
                continue

            image_rel = _normalize_path(image_rel)
            mask_rel = _normalize_path(mask_rel)

            image_abs = os.path.join(root_dir, image_rel)
            mask_abs = os.path.join(root_dir, mask_rel)
            if not os.path.exists(image_abs) or not os.path.exists(mask_abs):
                continue

            record = dict(rec)
            record["image_abs"] = image_abs
            record["mask_abs"] = mask_abs
            grouped[image_abs].append(record)

        samples: List[Dict[str, Any]] = []
        for image_abs, items in grouped.items():
            items.sort(key=lambda x: (x.get("position_idx", 0), x.get("mask_path")))
            for rec in items:
                rec = dict(rec)
                rec["group_records"] = items
                samples.append(rec)

        if max_samples is not None:
            samples = samples[:max_samples]

        if not samples:
            raise ValueError(f"No valid samples constructed from {triples_path}")

        self.samples = samples

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    @staticmethod
    def _resize_tensor(tensor: torch.Tensor, size: int, pad_to_square: bool, mode: str = "bilinear") -> torch.Tensor:
        original_shape = tensor.shape
        original_ndim = tensor.dim()

        if original_ndim == 2:
            batch_tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif original_ndim == 3:
            if original_shape[0] in (1, 3):
                batch_tensor = tensor.unsqueeze(0)
            else:
                batch_tensor = tensor.unsqueeze(1)
        elif original_ndim == 4:
            batch_tensor = tensor
        else:
            raise ValueError(f"Unsupported tensor shape for resize: {original_shape}")

        _, _, h, w = batch_tensor.shape
        if pad_to_square:
            max_hw = max(h, w)
            pad_h = max_hw - h
            pad_w = max_hw - w
            if pad_h or pad_w:
                batch_tensor = torch.nn.functional.pad(
                    batch_tensor,
                    (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
                    mode="constant",
                    value=0.0,
                )

        resized = torch.nn.functional.interpolate(
            batch_tensor,
            size=(size, size),
            mode=mode,
            align_corners=False if mode in ("bilinear", "bicubic") else None,
        )

        if original_ndim == 2:
            return resized.squeeze(0).squeeze(0)
        if original_ndim == 3 and original_shape[0] in (1, 3):
            return resized.squeeze(0)
        if original_ndim == 3:
            return resized.squeeze(1)
        return resized

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.samples[index]
        image = Image.open(record["image_abs"]).convert("RGB")
        target_mask = Image.open(record["mask_abs"]).convert("L")

        image_tensor = TF.to_tensor(image) * 2.0 - 1.0
        mask_tensor = TF.to_tensor(target_mask)
        mask_tensor = (mask_tensor > 0.5).float()

        all_masks: List[torch.Tensor] = []
        for item in record["group_records"]:
            mask_img = Image.open(item["mask_abs"]).convert("L")
            mask_tensor_all = TF.to_tensor(mask_img)
            mask_tensor_all = (mask_tensor_all > 0.5).float()
            all_masks.append(mask_tensor_all)
        all_masks_tensor = torch.stack(all_masks, dim=0)

        image_tensor = self._resize_tensor(image_tensor, self.resolution, self.pad_to_square, mode="bilinear")
        mask_tensor = self._resize_tensor(mask_tensor, self.resolution, self.pad_to_square, mode="nearest")

        sample = {
            "pixel_values": image_tensor,
            "target_mask": mask_tensor,
            "prompt_loc": record["prompt_loc"],
        }

        if "prompt_edit" in record:
            sample["prompt_edit"] = record["prompt_edit"]

        return sample


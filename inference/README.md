# Inference Guide (Qwen)

当前推理链路：

1. Pointer LoRA 生成 heatmap 与 peak。
2. 根据 `mask_mode` 生成 mask（`peak_region` / `refiner` / `gaussian` / `clickseg`）。
3. 使用 Qwen Edit 做整图黑底隔离编辑。
4. ORB 对齐后，用 `isolated_image` 作为隔离域背景兜底。
5. 将 `aligned_edit` 的前景 support inward choke `2px` 后，与 `isolated_image` 合并，再硬贴回原图。

## 快速运行

```bash
python -m MOREdit.inference.run_soft_mask \
  --job-config /workspace/MOREdit/inference/jobs/full_pipeline_isolate_test3.yaml
```

## 环境依赖

推荐直接安装：

```bash
pip install -r /workspace/MOREdit/requirements.txt
```

当前完整推理链路实际依赖至少包括：

- `torch`
- `torchvision`
- `diffusers`
- `transformers`
- `accelerate`
- `peft`
- `opencv-python`
- `Pillow`
- `PyYAML`
- `timm`（ClickSEG / `isegm` 依赖）

模型路径约定：

- Qwen Edit：`/workspace/models_weight/Qwen-Image-Edit-2511`
- ClickSEG checkpoint：`/workspace/models_weight/clickseg.pth`
- ControlNet（可选，仅 inpaint 路径）：`/workspace/models_weight/Qwen-Image-ControlNet-Inpainting`

## Who/What 解耦

- `pointer_prompt`：只负责定位谁（Who）。
- `edit_prompt`：只负责怎么改（What）。

建议两者分开写，避免语义互相干扰。

## 只跑 mask（不编辑）

把 job 里的 `edit_prompt` 设为 `null`，脚本会只导出：

- `pointer_heatmap_peak.png`
- `pointer_mask_refined.png`
- `pointer_peak.json`

## Qwen Inpaint + ControlNet

推荐配置：

- `edit_backend: qwen`
- `qwen_mode: inpaint`
- `qwen_controlnet_path: /workspace/models_weight/Qwen-Image-ControlNet-Inpainting`

模型下载：

```bash
hf download InstantX/Qwen-Image-ControlNet-Inpainting \
  --local-dir /workspace/models_weight/Qwen-Image-ControlNet-Inpainting
```

## 常用参数

- `mask_mode`: `peak_region | refiner | gaussian | clickseg`
- `mask_dilate`: 编辑前膨胀 mask
- `mask_feather`: 编辑前羽化 mask
- `isolate_edit`: 整图黑底隔离后走 Qwen Edit，再做 ORB 对齐与贴回
- `paste_choke`: 会叠加到当前默认的 `edit support choke=2px` 上
- `qwen_strength`: inpaint 强度（仅 inpaint 模式）

## ClickSEG 模式

当 `mask_mode=clickseg` 时需要：

- `--clickseg-checkpoint /workspace/models_weight/clickseg.pth`
- 可选 `--clickseg-use-prev-mask`

注意：代码会按 `isegm` 路径加载 ClickSEG 依赖，确保你的运行环境可 import 对应包。

## 输出文件

- `pointer_heatmap_peak.png`
- `pointer_mask_refined.png`
- `pointer_peak.json`
- `pointer_edit.png`（有编辑时）
- `pointer_isolated_image.png`
- `pointer_edited_isolated.png`
- `pointer_orb_aligned.png`
- `pointer_merged_isolated.png`
- `qwen_mask_used.png` / `qwen_image_resized.png`（Qwen 后端）

## 备注

- 当前主线是 `full-image isolate edit -> ORB -> merged isolated -> hard paste-back`。
- CLI 中仍保留了部分历史参数（与旧 Kontext 流程相关），但当前可用主线为 Qwen 后端。

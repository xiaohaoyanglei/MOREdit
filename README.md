# Pointer LoRA (Qwen)

本项目当前主线是 **Qwen 图像模型 + Pointer LoRA**：

- 训练：学习 `prompt_loc -> target mask` 的指针能力（Who）。
- 推理：先由 Pointer LoRA 产出热力图/峰值/mask，再交给 Qwen Inpaint/ControlNet 做编辑（What）。

## 当前状态

- 训练主干：Qwen（非 Flux/Kontext）。
- 推理后端：`edit_backend=qwen`（仅 Qwen）。
- 推荐编辑模式：`qwen_mode=inpaint` + Qwen ControlNet Inpainting。

## 目录说明

- `dataset.py`：读取三元数据集。
- `model.py`：加载 Qwen transformer/text encoder/VAE/scheduler。
- `lora.py`：向 Qwen joint attention 注入 LoRA。
- `pointer_recorder.py`：记录 pointer attention slice。
- `trainer.py`：LoRA 训练循环。
- `train.py`：训练入口。
- `inference/`：mask 生成与编辑推理。
- `eval_runs/`：pointing game 评测。

## 训练数据格式

`annotations.json` / `annotations.jsonl` 里每条至少包含：

- `image_path`
- `mask_path`
- `prompt_loc`

可选：

- `prompt_edit`
- `position_idx`

路径会相对 `data.root` 解析。

## 环境准备

```bash
pip install torch torchvision diffusers transformers pyyaml pillow tqdm opencv-python
```

模型目录至少要有：

- Qwen 主模型（示例：`/workspace/models_weight/Qwen-Image-Edit-2511`）
- ControlNet（可选但推荐，示例：`/workspace/models_weight/Qwen-Image-ControlNet-Inpainting`）

## 开始训练

先检查 `pointer_lora_config.yaml` 中这三项是否存在：

- `model.path`
- `data.annotations`
- `data.root`

启动训练：

```bash
python -m MOREdit.train --config /workspace/MOREdit/pointer_lora_config.yaml
```

输出目录（时间戳 run）包含：

- `weights/lora_step_XXXXXX.pt`
- `heatmaps/`
- `samples/`
- `run_metadata.json`

## 推理（Pointer -> Mask -> Edit）

推荐直接用 job YAML：

```bash
python -m MOREdit.inference.run_soft_mask \
  --job-config /workspace/MOREdit/inference/jobs/full_pipeline_isolate_test3.yaml
```

若你只想先验 LoRA mask（不跑编辑），把 `edit_prompt` 设为 `null`。

典型输出：

- `pointer_heatmap_peak.png`
- `pointer_mask_refined.png`
- `pointer_peak.json`
- `pointer_edit.png`（当提供 `edit_prompt` 时）
- `qwen_mask_used.png`、`qwen_image_resized.png`（Qwen inpaint）

## 下载 ControlNet（可选）

```bash
hf download InstantX/Qwen-Image-ControlNet-Inpainting \
  --local-dir /workspace/models_weight/Qwen-Image-ControlNet-Inpainting
```

## 评测

```bash
python -m MOREdit.eval \
  --config /workspace/MOREdit/pointer_lora_config.yaml \
  --lora-weights /path/to/lora_step_xxxxxx.pt \
  --annotations /path/to/annotations.json \
  --data-root /path/to/data_root \
  --per-bucket 300 \
  --compute-iou \
  --save-details
```

评测说明见 `eval_runs/README.md`。

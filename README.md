# Pointer LoRA Training for Qwen edit 2511

本目录提供一个轻量级的指针式 LoRA 训练流程，用于让 Qwen edit 模型学会通过序数描述精确聚焦到目标个体。

## 目录结构

- `dataset.py`：将三元组标注（原图 / mask / 序数 prompt）整理成模型输入。
- `lora.py`：为 cross-attention 的 `to_q / to_k / to_v / add_q_proj / add_k_proj / add_v_proj / to_add_out` 注入 LoRA 线性层。
- `model.py`：加载 Flux Kontext 的 transformer、CLIP/T5 编码器、VAE 和调度器，并暴露 `encode_prompts`、`encode_images`、`predict_noise` 等辅助函数。
- `pointer_recorder.py`：自定义 attention processor，采集 cross-attn 概率并缓存在 `PointerCache` 中。
- `scheduler.py`：复制官方 FlowMatch 调度策略，保证 timestep 采样一致。
- `losses.py`：BCE + Dice 指针监督。
- `trainer.py`：训练主循环、梯度更新、热力图保存。
- `train.py`：命令行入口。
- `pointer_lora_config.yaml`：示例配置，包含模型、数据、训练、LoRA、loss、测试图片信息。

## 环境要求

确保已激活你准备好的 Python 环境（例如 `/root/autodl-tmp/edit`）并安装基础依赖：

```bash
pip install torch torchvision diffusers transformers
```

Flux Kontext 权重需已解压在 `/root/autodl-tmp/FLUX1-Kontext-dev`。

## 训练

方法一：直接使用命令行参数（与 `pointer_lora_config.yaml` 内容一致）：

```bash
cd /root/autodl-tmp
python -m pointer_lora.train --config pointer_lora/pointer_lora_config.yaml
```

方法二：若希望在代码中读取 YAML，可自行加载 `pointer_lora/pointer_lora_config.yaml` 后构造 `PointerTrainerConfig`。

默认配置开启 `offload_text_encoders: true` 并使用 `bf16` 计算，让 CLIP/T5 常驻 CPU、transformer 以低精度运行，可显著降低显存占用；若显存充足，可在 YAML 或命令行将其覆盖为 `false` 并切换回 `float32/float16`。与此同时，`pointer` 小节定义了 token 聚合策略（`token_mode` / `token_filters`）以及热力图组合与温度参数，直接复用了 ai-toolkit 中的做法，确保只针对序数与主体相关的 token 聚焦；训练过程中会在 `output/heatmaps/` 目录生成两套热力图：`train_step_xxx` 来自当前 mini-batch，`test_step_xxx` 对应 `pointer_lora_config.yaml` 中的测试图。当 `heatmap_max_samples` 设为 `0` 或 `null` 时会保存 mini-batch 内的全部样本。

自 2025.11 起，每次启动训练都会在 `training.output_dir` 下创建一个带时间戳的子目录（例如 `output/20251126-153015`），其中包含：

- `run_metadata.json`：记录本次 run 的配置、设备与时间戳；
- `heatmaps/`：按 step 划分的 train/test 热力图 PNG；
- `samples/`：与热力图同步保存的 `.pt` 批次快照，内含 `pixel_values`、`pointer_map`、`target_mask` 与对应的 prompt，可供后处理或调试；
- `weights/`：每隔 `checkpoint_every` 步自动导出的 LoRA 权重（`lora_step_XXXXXX.pt`），训练结束会再保存一次以确保有最终版本。

### 指针抑制（步骤一）

`loss` 小节新增三项超参：

- `background_weight`：只在目标外部区域计算的 BCE，直接压制非目标激活；
- `contrast_weight` / `contrast_margin`：强制前景-背景平均激活差值至少满足 margin，否则施加惩罚；
- 这些额外 loss 默认关闭，可按需在 YAML 或 CLI（`--loss-background-weight` 等）中调整权重。

此外可以在 `pointer.token_filters` 中补充主体描述词（默认已包含 `person/man/woman/girl/boy`），让 token 选择更聚焦。

### 热力图超分（步骤二）

在 `pointer.superres` 中配置超分与锐化（例如 `factor: 2`, `sharpness: 0.3`）。训练/测试时会先把 cross-attn 栅格提升到更高的 latent 分辨率，再做 Laplacian-based 细节增强，最后映射到目标分辨率。CLI 对应参数为 `--pointer-superres-factor` 与 `--pointer-superres-sharpness`。

## 测试图片

示例测试图已经放在 `/root/autodl-tmp/test_images/`，其中：

- 路径：`/root/autodl-tmp/test_images/test3.png`
- Prompt：`The third person on the right`

可在训练完成后，用这对图文进行推理或可视化验证 pointer heatmap 的效果。

## 推理与编辑

训练完成后，可以使用 `inference/` 目录下的推理脚本进行实际的图像编辑。完整的推理流程包括：

1. **Pointer LoRA 生成 Mask**：通过训练好的 LoRA 权重，根据序数描述（如"右边第三个人"）生成精确的 attention heatmap，并转换为 mask。
2. **图像编辑**：使用生成的 mask 进行局部编辑，支持多种编辑后端。

### 快速开始

```bash
cd /root/autodl-tmp
python -m pointer_lora.inference.run_soft_mask \
  --config pointer_lora/pointer_lora_config.yaml \
  --lora-weights pointer_lora/output/20251201-141737/weights/lora_step_008000.pt \
  --image /root/autodl-tmp/test_images/test3.png \
  --pointer-prompt "the third person on the right" \
  --edit-prompt "Change the outfit to a red jacket" \
  --output-dir pointer_lora/output/inference_demo
```

### 编辑后端选择

#### 1. Kontext（默认）
基于 Flux Kontext 的 latent-space 编辑，适合风格迁移和整体调整：
```bash
--edit-backend kontext
```

#### 2. Qwen ControlNet Inpainting（推荐 ⭐）
使用 [InstantX/Qwen-Image-ControlNet-Inpainting](https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting) 进行精确的 mask-based 图像修复：

**优势**：
- 专门训练用于理解 mask 语义（6.5万步，1000万图像）
- 真正理解"编辑 mask 内区域"的意图，而非机械的硬 mask 排除
- 更精准的 inpainting 效果，减少伪影
- 更好地保持 mask 外区域不变

**快速使用**：
```bash
# 1. 下载 ControlNet 模型（约 2-3GB）
cd /root/autodl-tmp
huggingface-cli download InstantX/Qwen-Image-ControlNet-Inpainting \
  --local-dir Qwen-Image-ControlNet-Inpainting

# 2. 使用 ControlNet 进行推理
python -m pointer_lora.inference.run_soft_mask \
  --job-config pointer_lora/inference/jobs/test3_controlnet.yaml
```

**工作流程**：
```
Pointer LoRA 生成 Mask → ControlNet Inpainting 使用 Mask 编辑
```
- **Mask 100% 来自 Pointer LoRA**
- **ControlNet 只负责更智能的 inpainting**

详细文档请参考 [inference/README.md](inference/README.md)。

### 输出文件

推理完成后，在输出目录（带时间戳）中会生成：
- `pointer_heatmap.png` - Pointer LoRA 生成的原始热力图
- `pointer_mask_refined.png` - 精炼后的 mask
- `pointer_edit.png` - 最终的编辑结果
- `pointer_peak.json` - 热力图峰值坐标信息

如果使用 Qwen ControlNet 后端，还会额外生成：
- `qwen_mask_used.png` - 传递给 ControlNet 的 mask
- `qwen_image_resized.png` - ControlNet 输入的图像

## 保存与后续

当前脚本默认仅训练内存中的 LoRA 参数，权重会自动保存在 `output/<timestamp>/weights/` 目录下。另外，若要将指针 LoRA 与原 Kontext 模型合并，可以使用 `lora.py` 中注入的 `LoRALinear` 模块读取权重并手动写入原始线性层。

欢迎根据需求进一步扩展，包括：

- 调整 token 聚合策略（例如只聚焦序数词）。
- 集成其他图像编辑后端（SAM、GroundingDINO 等）。
- 添加 LoRA 权重导出到 safetensors。



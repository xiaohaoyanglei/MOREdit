# Pointer LoRA Inference（软掩码）

该目录提供“Pointer → SoftMask → Kontext”推理链路：

1. **Pointer LoRA**：输出热力图（注意力/热力），并缓存峰值点（argmax）。
2. **SoftMask 生成（默认：peak-region）**：以峰值为种子，在热力图上做“阈值候选 + 连通域扩张”，得到一个**不借助外部 SAM / Refiner** 的 soft mask。
3. **Kontext 编辑（推荐：reference latent mask）**：Kontext 内部会把 `image_latents` 当作强参考条件。为了让语义编辑真的发生（而不是只变亮/微调风格），需要在 **latent 空间把 mask 区域的参考 latents “遮掉/置噪”**，让模型在该区域更自由地改动；再配合 `latent_blend_alpha` 控制外溢。

## 运行示例（命令行参数）

```bash
cd /root/autodl-tmp
python -m pointer_lora.inference.run_soft_mask \
  --config pointer_lora/pointer_lora_config.yaml \
  --lora-weights /root/autodl-tmp/pointer_lora/output/20251201-141737/weights/lora_step_007000.pt \
  --image /root/autodl-tmp/test_images/test3.png \
  --prompt "Replace the clothing of the third person on the right with a red jacket" \
  --width 1024 --height 1024 --steps 24 \
  --latent-blend-alpha 0.6 \
  --reference-latent-mask \
  --reference-latent-mask-blur 8 \
  --reference-latent-mask-noise 1.0 \
  --output-dir pointer_lora/output/softmask_test
```

每次运行都会在 `--output-dir`（默认为 `pointer_lora/output/softmask_infer`）下自动新建一个时间戳子目录（如 `20251210-153045`），该目录内包含本次推理的全部产出：

- `pointer_heatmap.png / pointer_heatmap_overlay.png / pointer_heatmap.npy`：原始 pointer 热力图，仅用于排查定位问题。
- `pointer_mask_refined.png`：soft mask（默认 peak-region）。Kontext 依据它做注意力 gating。
- `pointer_peak.json`：峰值坐标（热力图坐标 + 原图归一化坐标）。
- `pointer_peak.json`：峰值坐标（热力图坐标 + 原图归一化坐标）。
- `pointer_edit.png`：若提示语包含编辑意图，会直接执行 Kontext 编辑；也可只取前面的掩码产物。

## 使用 YAML 启动

如果不想记住一堆参数，可以用 job YAML。示例配置在 `pointer_lora/inference/jobs/test3.yaml`：

```yaml
model:
  config: pointer_lora/pointer_lora_config.yaml
  lora_weights: /root/autodl-tmp/pointer_lora/output/20251201-141737/weights/lora_step_007000.pt
  refiner_weights: null

inference:
  image: /root/autodl-tmp/test_images/test3.png
  # 推荐解耦写法：
  # - pointer_prompt：只负责“指代/定位谁”
  # - edit_prompt：只负责“要改什么”，建议不要再写位置信息，避免与 mask guidance 产生冲突
  pointer_prompt: "the third person on the right"
  edit_prompt: "Replace the clothing of the selected person with a red jacket"
  width: 768
  height: 768
  steps: 20
  guidance_scale: 3.5
  seed: 42
  output_dir: pointer_lora/output/softmask_demo
  latent_blend_alpha: 0.6   # 0=关闭；建议 0.4~0.8
  reference_latent_mask: true
  reference_latent_mask_blur: 8.0
  reference_latent_mask_noise: 1.0
  # soft mask 生成策略（默认 peak_region）
  mask_mode: peak_region
  peak_region_threshold: 0.5
  peak_region_threshold_mode: rel   # rel=peak*ratio / abs=absolute
  peak_region_connectivity: 8       # 4 or 8
  peak_region_max_iters: 2048
```

直接用一条命令即可：

```bash
python -m pointer_lora.inference.run_soft_mask \
  --job-config pointer_lora/inference/jobs/test3.yaml
```

YAML 中的字段与 CLI 参数一一对应，可按需复制修改多个 job。

## Qwen ControlNet Inpainting（推荐）

除了默认的 Kontext 编辑后端，现在支持使用 **Qwen ControlNet Inpainting** 模型，这是专门为 mask-based 图像修复训练的 ControlNet 模型。

### 为什么使用 ControlNet Inpainting？

**当前问题**：Kontext 和基础 Qwen 模型在使用 mask 时采用"硬排除"策略：
- Mask 外的区域：完全不变
- Mask 内的区域：被强制修改
- 模型**不理解**"编辑 mask 内的内容"这个语义，只是机械地应用 mask

**ControlNet 优势**：
- 专门训练用于理解 mask 语义（6.5万步，1000万图像）
- 真正理解"编辑 mask 内的区域"的意图
- 更精准的 inpainting 效果，减少伪影
- 更好地保持 mask 外区域不变
- 支持复杂的编辑任务（物体替换、背景修改等）

### 下载 ControlNet 模型

```bash
cd /root/autodl-tmp

# 方法1: 使用 huggingface-cli（推荐，更快）
huggingface-cli download InstantX/Qwen-Image-ControlNet-Inpainting \
  --local-dir Qwen-Image-ControlNet-Inpainting

# 方法2: 使用 git-lfs
git lfs install
git clone https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting
```

模型大小约 2-3GB。

### 使用方法

#### 方式1: 使用示例配置（最简单）

```bash
python -m pointer_lora.inference.run_soft_mask \
  --job-config pointer_lora/inference/jobs/test3_controlnet.yaml
```

示例配置文件 `test3_controlnet.yaml` 已经配置好所有 ControlNet 相关参数。

#### 方式2: 命令行参数

```bash
python -m pointer_lora.inference.run_soft_mask \
  --config pointer_lora/pointer_lora_config.yaml \
  --lora-weights /path/to/lora_weights.pt \
  --image /path/to/image.png \
  --pointer-prompt "the first person from the right" \
  --edit-prompt "Change the outfit to a blue suit" \
  --edit-backend qwen \
  --qwen-mode inpaint \
  --qwen-controlnet-path /root/autodl-tmp/Qwen-Image-ControlNet-Inpainting \
  --steps 30 \
  --guidance-scale 4.0 \
  --output-dir output/controlnet_test
```

#### 方式3: 自动检测（默认行为）

如果你已经下载了 ControlNet 模型到默认路径，只需设置 `--edit-backend qwen --qwen-mode inpaint`，系统会自动使用 ControlNet：

```bash
python -m pointer_lora.inference.run_soft_mask \
  --job-config pointer_lora/inference/jobs/test3.yaml \
  --edit-backend qwen \
  --qwen-mode inpaint
```

### 环境变量配置

可以通过环境变量设置默认的 ControlNet 路径：

```bash
export QWEN_CONTROLNET_PATH=/root/autodl-tmp/Qwen-Image-ControlNet-Inpainting
```

### 关键参数说明

在 YAML 配置或命令行中：

```yaml
inference:
  # 必需：选择 qwen 后端
  edit_backend: qwen
  qwen_mode: inpaint  # 必须是 inpaint 才能使用 ControlNet

  # ControlNet 模型路径（可选，有默认值）
  qwen_controlnet_path: /root/autodl-tmp/Qwen-Image-ControlNet-Inpainting

  # Qwen inpaint 强度（0-1，越小越保留原图细节）
  qwen_strength: 0.8

  # ControlNet 推荐参数
  steps: 30              # 推荐 30-50 步
  guidance_scale: 4.0    # CFG scale
  true_cfg_scale: 4.0    # True CFG scale

  # 使用 ControlNet 时可以简化这些参数
  latent_blend_alpha: 0.0           # ControlNet 自己处理 mask，可关闭
  conditioning_masked_image: false  # 不需要额外的 conditioning
  use_inpaint_pipeline: false       # 使用 Qwen 而非 Kontext
```

### Prompt 建议

使用 ControlNet 时：
- **pointer_prompt**：描述要选择的对象位置（如 "the first person from the right"）
- **edit_prompt**：只描述**要改什么**，无需重复位置信息
  - ✅ 好："Change the outfit to a blue suit"
  - ❌ 差："Change the first person's outfit to a blue suit"（会与 mask 产生冲突）
- 使用**描述性 prompt**，而非指令性 prompt
  - ✅ 好："A person wearing a blue suit"
  - ❌ 差："Make the person wear a blue suit"

### 输出文件

使用 ControlNet 后，输出目录会包含：
- `pointer_heatmap.png` - Pointer LoRA 生成的热力图
- `pointer_mask_refined.png` - 精炼后的 mask
- `pointer_edit.png` - ControlNet inpainting 的最终结果
- `qwen_mask_used.png` - 传递给 ControlNet 的 mask
- `qwen_image_resized.png` - ControlNet 输入的图像

### 性能对比

| 特性 | Kontext (latent mask) | Qwen 基础 | Qwen ControlNet |
|------|----------------------|-----------|-----------------|
| Mask 理解 | 硬排除 | 硬排除 | 语义理解 ✓ |
| 训练目标 | 通用生成 | 通用编辑 | 专门 inpainting ✓ |
| 精度 | 中 | 中 | 高 ✓ |
| 伪影控制 | 中 | 中 | 优秀 ✓ |
| 推荐场景 | 风格迁移 | 全图编辑 | 局部精准编辑 ✓ |

## 注意事项

- Pointer 目前默认只输出 top-1 峰值；如需 top-K，可在 `PointerSoftMaskPipeline` 内扩展多次生成并做合并。
- `mask_mode=peak_region` 为**不借助外部模型**的方案（仅用 pointer 热力图）；如要回退到旧的 refiner 链路，可设置 `mask_mode=refiner` 并提供 `refiner_weights`。
- `latent_blend_alpha` 决定 mask 对编辑的"拉回"强度：越大越不容易串到其他人，但也越可能让编辑幅度变小（使用 ControlNet 时可设为 0）。
- Pointer/Refiner 的输入分辨率由 `data.resolution`（默认 640）决定，Kontext 编辑阶段会自动把 mask 还原到原图再缩放到目标宽高。
- 产出的 `pointer_mask_refined.npy` / `.png` 可直接作为后续局部编辑、合成或评测的单人实例掩码。
- **ControlNet 需要较新版本的 diffusers**：确保 `diffusers` 库支持 `QwenImageControlNetInpaintPipeline`。


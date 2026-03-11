# Pointer LoRA Eval 输出说明

该目录由 `python -m MOREdit.eval` 自动生成，用于保存 Pointing Game 评测结果。

每次评测会创建一个时间戳目录，例如：

```text
MOREdit/output/eval_runs/
└── 20260306-130000/
    ├── metrics.json
    ├── details.csv          # --save-details 时生成
    └── failures/            # 默认开启，命中失败样本可视化
```

## 快速开始

```bash
python -m MOREdit.eval \
  --config /workspace/MOREdit/pointer_lora_config.yaml \
  --lora-weights /path/to/lora_step_008000.pt \
  --annotations /path/to/annotations.json \
  --data-root /path/to/data_root \
  --per-bucket 300 \
  --compute-iou \
  --save-details
```

## 常用参数

- `--per-bucket`: 每个人数桶采样数量，`0` 表示全量。
- `--max-images`: 全局样本上限（smoke test 常用）。
- `--compute-iou`: 额外统计 IoU / Dice。
- `--heatmap-threshold`: 计算 IoU 时的二值阈值。
- `--save-details`: 保存逐样本结果。
- `--save-failures` / `--no-save-failures`: 是否导出失败可视化。
- `--output-dir`: 评测输出根目录。

## metrics.json

`metrics.json` 包含：

- `overall`：总体样本数、pointing accuracy、可选 IoU/Dice。
- `by_count`：按人数桶统计。
- `eval_config`：本次评测参数快照。
- `model_config`：模型与指针参数快照。

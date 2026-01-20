# Pointer LoRA 评测输出说明

该目录由 `python -m pointer_lora.eval` 自动创建，用于保存每次 **Pointing Game** 评测的结果。每次运行都会生成一个时间戳子目录，例如：

```
pointer_lora/eval_runs/
└── 20251202-153000/
    ├── metrics.json    # 指标汇总
    └── details.csv     # (可选) 每条样本的命中记录
```

## 快速开始

```bash
cd /root/autodl-tmp
python -m pointer_lora.eval \
  --config pointer_lora/pointer_lora_config.yaml \
  --lora-weights /root/autodl-tmp/pointer_lora/output/20251201-141737/weights/lora_step_007000.pt\
  --annotations /root/autodl-tmp/mhpv2_triples_en_val/annotations.json \
  --data-root /root/autodl-tmp/mhpv2_triples_en_val \
  --per-bucket 300 \
  --compute-iou \
  --save-details
```

## 常用可调参数（带中文注释）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--per-bucket` | 300 | **每种人数桶抽样多少张图像**，可快速控制评测规模，设为 0 表示使用全部。 |
| `--max-images` | None | **全局抽样上限**，当想做 smoke test 时限制总图像数量。 |
| `--seed` | 42 | **随机种子**，保证不同 run 之间的可重复性。 |
| `--compute-iou` | False | **勾选后额外计算 IoU/Dice**，否则只统计 pointing accuracy。 |
| `--heatmap-threshold` | 0.35 | **二值化热力图的阈值**，仅在 `--compute-iou` 时生效。 |
| `--save-details` | False | 写出 `details.csv`，记录每条样本的命中结果（bucket、坐标、命中与否）。 |
| `--save-failures` / `--no-save-failures` | 开启 | **命中失败时是否自动导出热力图 + mask**，默认开启，可用 `--no-save-failures` 关闭。 |
| `--output-dir` | `pointer_lora/output/eval_runs` | **评测记录保存位置**，可以根据需要改到其它磁盘路径。 |

> 以上参数都可直接在 CLI 中修改，便于根据不同 LoRA/数据集快速评估。

## metrics.json 内容

```json
{
  "metrics": {
    "overall": {"samples": 900, "pointing_acc": 0.66, "mean_iou": 0.48},
    "by_count": {
      "2": {"samples": 300, "pointing_acc": 0.72},
      "3": {"samples": 300, "pointing_acc": 0.63},
      "4": {"samples": 200, "pointing_acc": 0.59},
      "5+": {"samples": 100, "pointing_acc": 0.55}
    }
  },
  "eval_config": { ... },   # 本次评测使用的路径/采样设置
  "model_config": { ... },  # 指针相关参数快照
  "generated_at": "2025-12-02T15:30:00"
}
```

`details.csv`（在 `--save-details` 时生成）包含每条样本的 bucket、图像路径、prompt、是否命中以及热力图峰值坐标，可用于人工抽查或可视化。若开启 `--save-failures`，一张图里只要有任意目标命中失败，就会在 `eval_runs/<run_id>/failures/人数桶/图名/` 目录输出 **该图全部目标** 的热力图 PNG 与对应 mask 副本，方便对照整张图的指针表现。*** End Patch

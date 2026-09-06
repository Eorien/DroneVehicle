# DroneVehicle RGB-IR OBB

基于 YOLO11n-OBB 的 RGB + IR 四通道前期融合小目标检测。正式实验仅包括 `rgbir_baseline`、`rgbir_spd` 和 `rgbir_spd_deab`。

1. `rgbir_baseline`：四通道融合；
2. `rgbir_spd`：四通道融合 + SPD；
3. `rgbir_spd_deab`：四通道融合 + SPD + DEAB。

第 1、2 组用于评估 SPD，第 2、3 组用于评估 DEAB 在 SPD 基础上的增益；不再设置“仅 DEAB”正式实验。

## 环境

- Python 3.12
- PyTorch 2.8.0 + CUDA 12.8
- Ultralytics 8.4.140（固定源码：`third_party/ultralytics/`）

```bash
uv sync --locked
```

## 数据

原始 `data/` 保持只读。数据处理分为两步：

```bash
# XML 清洗并转换为 YOLO OBB
.venv/bin/python scripts/prepare_dronevehicle.py --data-root data --output-root data/cleaned

# 冻结 RGB-IR 严格交集
.venv/bin/python scripts/freeze_rgbir_labels.py --clean-root data/cleaned --output-root data/frozen_rgbir
```

交集规则：同类别一对一旋转 IoU ≥ 0.7，且一幅图内 RGB/IR 标签必须 100% 匹配；最终采用 RGB OBB 作为唯一监督标签。

| 划分 | 图像对 | 目标数 |
|---|---:|---:|
| train | 5,891 | 92,205 |
| val | 488 | 7,380 |
| test | 2,923 | 43,144 |

训练集类别分布：`car 75,661`、`truck 5,594`、`bus 5,072`、`van 2,202`、`Freight_car 3,676`。冻结子集 95% 的匹配框中心偏移不超过 3 像素。

## 四通道融合

- `RGBIRDataset` 按 frozen manifest 成对读取图像；
- RGB 转为正确的 RGB 顺序，IR 强制转为单通道；
- 拼接为 `(H, W, 4)` 后统一执行几何增强；
- 模型首层输入通道由 3 改为 4；
- RGB 预训练权重完整保留，IR 权重初始化为 RGB 权重均值；
- OBB Detect Head 和损失逻辑保持不变。

主要实现：`dronevehicle/rgbir_dataset.py`、`dronevehicle/rgbir_trainer.py`。

## 验证与正式实验进度

已通过真实数据和官方预训练权重验证：

- `(B, 4, 640, 640)` 输入、同步增强、OBB 标签读取和验证集矩形批处理；
- YOLO11n-OBB forward、loss、反向传播和 FP16 AMP；
- SPD 固定替换 Layer 3 的 P2/4→P3/8 下采样，并通过本地及 RTX 4090 `batch=64` 预检；
- DEAB 采用 DEConv + 通道/空间/像素注意力，紧接同一个 SPD 的 P3/8 输出；本地 CPU/CUDA、真实四通道 640 forward、loss/backward 和 AMP smoke test 均通过。

截至 2026-09-06：

- `rgbir_baseline` 已完成 200 epochs、验证集复评和冻结 test 集正式评估；
- `rgbir_spd` 已于 2026-09-06 18:01 启动 200 epochs 正式训练；
- `rgbir_spd_deab` 已完成本地实现与 smoke test，待 SPD 训练结束后进行 RTX 4090 `batch=64` 预检。

### `rgbir_baseline` 正式结果

| 数据集 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| val | 75.74% | 75.06% | 76.21% | 60.16% |
| test | 73.91% | 76.58% | 77.22% | 61.04% |

测试集逐类别 mAP@0.5:0.95：`car 81.80%`、`truck 58.57%`、`bus 76.34%`、`van 40.46%`、`Freight_car 48.01%`。当前主要短板是 `van`。

baseline 完成 200 epochs，best epoch 为 200，总耗时约 1.620 小时，训练日志峰值显存为 20.5GB。`best.pt` SHA-256：

```text
0e48f388ae8a1eca87e80d7e0c91ecfeacbfb0d2f43f51f842782f676d587062
```

本地归档位于：

```text
outputs/formal_experiments/rgbir_baseline/
```

## 运行

```bash
# 四通道数据、权重与 forward
uv run python -m scripts.smoke_test_rgbir

# baseline 调试训练（可选 AMP）
uv run python -m scripts.train_rgbir --smoke
uv run python -m scripts.train_rgbir --smoke --smoke-amp

# SPD 调试训练
uv run python -m scripts.train_rgbir --config configs/train/rgbir_spd.yaml --smoke --smoke-amp

# DEAB 专项与 SPD+DEAB 调试训练
uv run python -m scripts.smoke_test_deab --device cuda
uv run python -m scripts.train_rgbir --config configs/train/rgbir_spd_deab.yaml --smoke --smoke-amp

# 正式 baseline
uv run python -m scripts.train_rgbir --config configs/train/rgbir_baseline.yaml

# 正式 SPD
uv run python -m scripts.train_rgbir --config configs/train/rgbir_spd.yaml
```

权重位于被 Git 忽略的 `weights/`：`yolo11n-obb.pt` 用于训练，`yolo26n.pt` 仅用于 Ultralytics AMP 自检。正式实验产物位于被 Git 忽略的 `runs/` 或 `outputs/formal_experiments/`，不得提交权重。

完整启动门禁、配置/数据指纹和正式结果见 `docs/formal_experiment_record.md`。
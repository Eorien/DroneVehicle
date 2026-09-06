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

## 验证

已通过真实数据和官方预训练权重验证：

- `(B, 4, 640, 640)` 输入；本机已验证 `batch=16`、`workers=8`；
- 同步增强、OBB 标签读取和验证集矩形批处理；
- YOLO11n-OBB forward、loss、反向传播和 1 epoch 小数据训练；
- SPD 已固定替换 P2/4→P3/8 下采样层，并通过 640 输入、loss、backward、验证和 FP16 AMP；
- FP16 AMP 自检与训练。

Smoke test 指标不作为实验结果，正式 200 epochs 尚未运行。

正式三组实验统一配置 `batch=64`；本机 smoke test 可使用 `batch=16`。正式训练前需在 RTX 4090 上检查三种模型，若任一模型无法稳定使用 64，则三组统一降为 32。

## 运行

```bash
# 四通道数据、权重与 forward
uv run python -m scripts.smoke_test_rgbir

# 1 epoch 调试训练（可选 AMP）
uv run python -m scripts.train_rgbir --smoke
uv run python -m scripts.train_rgbir --smoke --smoke-amp

# SPD 调试训练
uv run python -m scripts.train_rgbir --config configs/train/rgbir_spd.yaml --smoke --smoke-amp

# 正式 baseline；默认读取 configs/train/rgbir_baseline.yaml
uv run python -m scripts.train_rgbir

# 正式 SPD
uv run python -m scripts.train_rgbir --config configs/train/rgbir_spd.yaml
```

权重位于被 Git 忽略的 `weights/`：`yolo11n-obb.pt` 用于训练，`yolo26n.pt` 仅用于 Ultralytics AMP 自检。当前 baseline 和 SPD 工程链路已完成，SPD+DEAB 尚未实现。
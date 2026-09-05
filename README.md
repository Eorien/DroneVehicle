# DroneVehicle RGB-IR OBB

基于 YOLO11-OBB 的 RGB + IR 四通道前期融合小目标检测项目。

## 环境

- Python 3.12
- PyTorch 2.8.0 + CUDA 12.8
- Ultralytics 8.4.140（本地 `third_party/ultralytics/`）

```bash
uv sync --locked
```

## 数据处理

原始 `data/` 始终只读，所有结果均通过脚本重新生成。

### 1. 清洗与格式转换

`scripts/prepare_dronevehicle.py` 完成以下工作：

- 校验 RGB、IR、两套 XML 的文件配对；
- 统一 5 类名称，将 `bndbox` 和四点多边形转换为 YOLO OBB；
- 修复越界和点顺序，移除未知、缺失框及零面积目标；
- 分别保留 RGB、IR 标签，避免在配准确认前错误合并。

```bash
.venv/bin/python scripts/prepare_dronevehicle.py \
  --data-root data \
  --output-root data/cleaned
```

清洗得到 28,439 对图像、953,082 个双模态标注；共移除 82 个无效目标。类别 ID 固定为：

```text
0 car    1 truck    2 bus    3 van    4 Freight_car
```

### 2. RGB-IR 严格交集

经过 300 对图像抽检和全量旋转框匹配，正式筛选规则确定为：

1. RGB 与 IR 目标类别相同；
2. 按旋转 IoU ≥ 0.7 一对一匹配；
3. 只保留两套标签均 100% 匹配的图像对，避免未匹配目标成为假负样本。

```bash
.venv/bin/python scripts/freeze_rgbir_labels.py \
  --clean-root data/cleaned \
  --output-root data/frozen_rgbir
```

冻结结果位于 `data/frozen_rgbir/`，采用 RGB OBB 作为唯一监督标签，并保存逐目标匹配记录、统计报告和 SHA-256 校验文件。

冻结子集内匹配框中心偏移的中位数为 0 像素，95% 不超过 3 像素，99% 不超过约 6.1 像素。严格交集规模为：

| 划分 | 图像对 | 目标数 |
|---|---:|---:|
| train | 5,891 | 92,205 |
| val | 488 | 7,380 |
| test | 2,923 | 43,144 |
| **总计** | **9,302** | **142,729** |

交集训练集类别分布：

| car | truck | bus | van | Freight_car |
|---:|---:|---:|---:|---:|
| 75,661 | 5,594 | 5,072 | 2,202 | 3,676 |

该规模足以支持 YOLO11n-OBB 的三组消融实验。数据仍明显偏向 `car`，为保持实验控制变量一致，暂不引入重采样或类别加权。冻结标签后续只作为输入使用，若规则改变必须通过脚本重新生成并重新校验。

## 四通道 Smoke Test

```bash
uv run python -m scripts.smoke_test_rgbir
```

该检查覆盖真实 RGB/IR 内容、四通道拼接、同步训练增强、多进程 DataLoader、三通道预训练权重迁移、IR 均值初始化、GPU 输入和 YOLO11n-OBB forward。正式参数可使用 `--batch 16 --workers 8 --train-samples 64` 复查。

本地权重均位于被 Git 忽略的 `weights/`：

- `yolo11n-obb.pt`：正式预训练权重，SHA-256 `b62898ebf38940ca4df323863e45ee9d84a1a46d5d11ebdde529fb33aa9f3a32`；
- `yolo26n.pt`：仅供 Ultralytics AMP 自检，SHA-256 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。

## Baseline 训练入口

```bash
# 1 epoch / 32 张图，仅验证训练链路
uv run python -m scripts.train_rgbir --smoke

# 同样的 smoke test，并启用正式实验所需的 FP16 AMP
uv run python -m scripts.train_rgbir --smoke --smoke-amp

# 正式配置：200 epochs / batch 16 / imgsz 640 / workers 8
uv run python -m scripts.train_rgbir
```

正式参数保存在 `configs/train/rgbir_baseline.yaml`。本地不自动运行正式训练；`--smoke` 产生的指标不作为实验结果。
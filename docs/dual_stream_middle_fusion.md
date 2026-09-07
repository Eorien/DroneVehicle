# RGB-IR 双流自适应中期融合

> 状态：本地实现与 smoke test 已完成；属于当前三组正式消融之后的探索阶段，尚未进行 RTX 4090 `batch=64` 预检或正式训练。

## 1. 目标与边界

第一版只研究融合方式，不同时引入其他结构：

- 输入仍为同步增强后的 `(B,4,H,W)`；
- 模型内拆分为 RGB `(B,3,H,W)` 和 IR `(B,1,H,W)`；
- 两条同构 YOLO11n Backbone 分别提取特征；
- 只在 P3/8、P4/16、P5/32 进行自适应中期融合；
- 融合后使用一套原始 YOLO11 Neck 和 OBB Head；
- 不使用 P2、SPD、DEAB、可变形对齐或额外注意力。

结构图：

- `docs/rgbir_dual_stream_adaptive_fusion.drawio`
- `docs/rgbir_dual_stream_adaptive_fusion.png`

## 2. 自适应融合

每个尺度输入两个 shape 完全一致的特征：

```text
F_rgb, F_ir ∈ (B,C,H,W)
```

使用拼接特征预测逐通道、逐位置门控：

```text
α = sigmoid(Conv1x1(concat(F_rgb, F_ir)))
F_fused = α · F_rgb + (1 - α) · F_ir
```

门控卷积权重和 bias 全部初始化为 0，因此初始 `α=0.5`，模型从 RGB/IR 等权平均开始学习，不预设某个模态更可靠。

## 3. 预训练权重迁移

官方 `yolo11n-obb.pt` 的层索引不能直接匹配双流 YAML，因此使用显式映射：

- 官方 Backbone Layer 0～10 → RGB Backbone Layer 2～12；
- 官方 Backbone Layer 0～10 → IR Backbone Layer 14～24；
- IR 第一层卷积 = 官方 RGB 第一层三个通道权重的均值；
- 官方 Neck/Head Layer 11～23 → 共享 Layer 28～40；
- 三个门控模块保持全零新参数；
- 五类别 OBB 分类输出因官方 checkpoint 为 15 类而保持新初始化。

当前迁移结果为 `775/787` 个目标状态项。未迁移的 12 项正好包括：

- 6 个门控卷积 weight/bias；
- 6 个类别数不同的三尺度分类输出 weight/bias。

所有 Backbone 状态必须完整迁移，否则初始化函数直接报错，不允许静默丢失。

## 4. 模型规模

| 模型 | Parameters | GFLOPs | 融合位置 |
|---|---:|---:|---|
| 四通道前期融合 baseline | 2,662,626 | 6.8 | 输入层 |
| 双流自适应中期融合 | 4,224,786 | 10.5 | P3/P4/P5 |

## 5. 本地验证

已通过：

- `RGBIRSplit` 精确拆分 RGB/IR；
- 门控初始输出严格等于两模态特征均值；
- 门控梯度非零、有限，Adam step 后参数更新；
- 改变 IR 输入不会影响融合前 RGB 分支，反之亦然；
- P3/P4/P5 两路 shape 对齐；
- 775/787 显式预训练迁移和遗漏原因检查；
- CPU/CUDA 专项 forward；
- 真实 RGB-IR `(2,4,640,640)` forward；
- 真实数据 640 AMP OBB loss/backward，6 个门控参数张量均有梯度；
- 32 张训练图、1 epoch FP16 AMP smoke training；
- checkpoint 保存、重载和最终验证；
- baseline Trainer 回归 smoke test；
- baseline、SPD、SPD+DEAB 的参数量/GFLOPs 回归不变。

Smoke test 指标没有实验意义。

## 6. 文件

```text
third_party/ultralytics/ultralytics/nn/modules/fusion.py
configs/models/yolo11n-obb-rgbir-dual.yaml
configs/train/rgbir_dual.yaml
scripts/smoke_test_dual_fusion.py
```

运行：

```bash
uv run python -m scripts.smoke_test_dual_fusion --device cuda

uv run python -m scripts.smoke_test_rgbir \
  --model configs/models/yolo11n-obb-rgbir-dual.yaml \
  --batch 2 --workers 2 --train-samples 8 --device cuda

uv run python -m scripts.train_rgbir \
  --config configs/train/rgbir_dual.yaml \
  --smoke --smoke-amp \
  --name rgbir-dual-smoke
```

## 7. 尚未完成

- RTX 4090 `batch=64, imgsz=640, AMP=True` backward 预检；
- 完整训练显存峰值确认；
- 门控在训练后对 RGB/IR 的实际偏好分析；
- 与前期融合 baseline 的正式对照训练。

如果 `batch=64` 不稳定，应为新的融合实验阶段确定统一 batch，并在相同 batch 下重跑该阶段基准；不直接混用当前第一阶段不同 batch 的结果。

# RGB-IR 双流 Hybrid Fusion

> 独立探索版本：在现有 `rgbir_dual` 双流中期融合基础上，仅增加一次浅层跨模态残差交互；不修改 `rgbir_dual`。

## 1. 结构

```text
RGB → RGB 浅层特征 ─┐
                   ├→ ShallowCrossModalInteraction
IR  → IR 浅层特征 ──┘
          ↓                  ↓
      RGB 增强特征        IR 增强特征
          ↓                  ↓
      RGB Backbone       IR Backbone
          └── P3/P4/P5 Adaptive Fusion → Shared Neck → OBB Head
```

实际插入位置：

```text
RGB layer 4 / IR layer 8 输出
shape: (B, 64, 160, 160)
stride: 4
阶段: P2/4 的 C3k2 输出之后、P3/8 stride-2 Conv 之前
```

两路已经完成独立浅层提取，交互后仍分别进入自己的深层 Backbone。P3/P4/P5 自适应融合、共享 Neck 和 OBB Head 与 `rgbir_dual` 保持同一方案。

## 2. 交互公式

```text
F_cat = concat(F_rgb, F_ir)
F_shared = Conv1x1(F_cat)
Δ_rgb = Proj_rgb(F_shared)
Δ_ir  = Proj_ir(F_shared)
F_rgb' = F_rgb + Δ_rgb
F_ir'  = F_ir + Δ_ir
```

`Proj_rgb` 和 `Proj_ir` 的 weight/bias 全部零初始化，因此启动时严格满足：

```text
F_rgb' = F_rgb
F_ir'  = F_ir
```

这使 Hybrid 模型在初始化时退化为当前 `rgbir_dual`，不会随机破坏已有预训练特征。

## 3. 独立配置

```text
configs/models/yolo11n-obb-rgbir-hybrid.yaml
configs/train/rgbir_hybrid.yaml
scripts/smoke_test_hybrid_fusion.py
```

现有文件保持不变：

```text
configs/models/yolo11n-obb-rgbir-dual.yaml
configs/train/rgbir_dual.yaml
```

## 4. 规模与迁移

| 模型 | Parameters | GFLOPs | 状态项 |
|---|---:|---:|---:|
| rgbir_dual | 4,224,786 | 10.5 | 787 |
| rgbir_hybrid | 4,241,362 | 11.3 | 793 |

Hybrid 比 `rgbir_dual` 新增 16,576 parameters，新增状态全部位于 `model.9` 的浅层交互模块。官方预训练映射：

- 775 个状态项显式迁移；
- 6 个原有 P3/P4/P5 gate 状态保持零初始化；
- 6 个五类 OBB 分类输出因官方 checkpoint 类别数不匹配而重新初始化；
- 6 个浅层交互状态保持新初始化，其中 `project_rgb` 和 `project_ir` 零初始化。

## 5. 本地验证

已通过：

- CPU/CUDA Hybrid 专项 smoke test；
- P2/4 插入位置 shape `(B,64,160,160)` 检查；
- Hybrid 与 `rgbir_dual` 的 787 个等价状态逐元素一致；
- 零扰动初始化下 Hybrid 与 `rgbir_dual` 输出逐元素一致；
- 浅层 projection 梯度、共享 Conv 梯度和 Adam step；
- 真实四通道 `640×640` forward；
- 真实数据 AMP OBB loss/backward；
- 32 张图、1 epoch AMP smoke training；
- baseline、dual 原有构建和 GFLOPs 回归检查。

本地 smoke 指标不作为实验结果。尚未进行 RTX 4090 预检和正式训练。

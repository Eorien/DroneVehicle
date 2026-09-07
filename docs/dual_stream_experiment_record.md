# 双流自适应中期融合完整训练记录

> 实验性质：第一阶段三组正式 SPD/DEAB 消融之外的第二阶段探索性完整训练。当前只评估纯双流 P3/P4/P5 中期融合，不加入 P2、SPD、DEAB 或特征对齐。

## 1. 实验定义

| 字段 | 固定值 |
|---|---|
| 实验 ID | `rgbir_dual` |
| 对照 | 第一阶段 `rgbir_baseline` |
| 唯一主要结构变化 | 四通道单 Backbone 改为 RGB/IR 双 Backbone，并在 P3/P4/P5 自适应融合 |
| 数据 | 同一冻结 RGB-IR train/val/test |
| 预训练 | 同一个 `yolo11n-obb.pt` |
| Neck / OBB Head | 保持原 YOLO11n-OBB 结构 |
| P2 / SPD / DEAB / 对齐 | 全部关闭 |

融合公式：

```text
α = sigmoid(Conv1x1(concat(F_rgb, F_ir)))
F_fused = α · F_rgb + (1 - α) · F_ir
```

三个门控的 weight/bias 初始化为 0，因此初始 `α=0.5`。

## 2. 统一训练配置

与 `configs/train/rgbir_baseline.yaml` 逐项比较，只有 `model` 和 `name` 不同：

```text
epochs=200
batch=64
imgsz=640
workers=8
patience=50
optimizer=Adam
lr0=0.01
momentum=0.937
weight_decay=0.0005
amp=True
seed=0
deterministic=True
```

现有 baseline 同样使用 `batch=64`。双流完整 1 epoch 压力测试已通过，因此本次不改变公共 batch，也不需要为该探索对照重跑 baseline。Trainer 重构后的 baseline 回归 smoke test 及参数/GFLOPs 检查均通过。

## 3. 代码、配置和数据指纹

| 项目 | SHA-256 / commit |
|---|---|
| 双流实现 commit | `f13c827cfe49ffd84055f5accede1cee963becaa` |
| 模型 YAML | `67b24e8186908e74fbdb576e1a52e1a1f3ed9eeb0e0829a86249fdb8ba34d083` |
| 训练 YAML | `bed0dcc2fe44139ebacd8546fdfddee9e754105c12a799dc5a784e1e6f524ed0` |
| 数据 YAML | `3d0fc28122a8d1d8aee03d15c895d740abf40d1fcc5517fafaeb03b774040b86` |
| frozen report | `c8887d31efcb4d6bf62890847e7fb64bb59fe6a5f2e0a42d07d50855b39756f9` |
| frozen checksums | `fd7eaf81719a76a401b97a34a25cc1e8a55b73b847539f2eda0c94376e698bd0` |
| `yolo11n-obb.pt` | `b62898ebf38940ca4df323863e45ee9d84a1a46d5d11ebdde529fb33aa9f3a32` |
| `yolo26n.pt` | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |

## 4. 模型与预训练迁移

| 项目 | 结果 |
|---|---:|
| Parameters | 4,224,786 |
| GFLOPs | 10.5 |
| 目标状态项 | 787 |
| 显式迁移 | 775 |
| 新门控状态 | 6 |
| 类别数不匹配的 OBB 分类输出 | 6 |

迁移规则：

- 官方 Backbone 复制到 RGB Backbone；
- 官方 Backbone 再复制到 IR Backbone，IR 第一层取 RGB 卷积权重通道均值；
- 官方 Neck/Head 显式映射到共享 Neck/Head；
- 6 个门控状态保持全零新初始化；
- 6 个 15 类→5 类不兼容分类输出明确跳过；
- 任一 Backbone 状态未迁移都会直接报错。

## 5. 本地验证

- [x] CPU/CUDA 双流专项 forward；
- [x] RGB/IR 分支隔离；
- [x] 初始融合严格等于两模态均值；
- [x] 门控梯度非零、有限且 Adam step 后更新；
- [x] 真实 `(2,4,640,640)` forward；
- [x] 真实数据 640 AMP OBB loss/backward；
- [x] 32 张图、1 epoch FP16 AMP smoke training；
- [x] checkpoint 保存、重载和验证；
- [x] baseline Trainer 回归 smoke test；
- [x] baseline/SPD/SPD+DEAB 参数和 GFLOPs 回归不变。

## 6. RTX 4090 预检

服务器环境：RTX 4090 24GB，Driver 595.71.05，PyTorch 2.8.0+cu128。

| 预检 | 结果 | 峰值显存 | 日志 SHA-256 |
|---|---|---:|---|
| 单 batch AMP backward | 通过 | 13.4GB | `44c9291701dd5a19fe8ae0dbdf6aa904b6895a04ae5f11b76491979892c7ab6e` |
| 完整 1 epoch、93 batches 压力测试 | 通过 | 16.5GB | `691dadc97bd6f266a83525841829cac81aac72ca9e62ac960822687b2b04af00` |

预检日志：

```text
runs/logs/rgbir_dual_batch64_precheck_20260907_154817.log
runs/logs/rgbir_dual_batch64_full_epoch_stress_20260907_154942.log
```

压力测试覆盖完整训练集、同步增强和 Mosaic，未发生 OOM/NaN。相对 24GB 仍有约 7.5GB 余量，允许以 `batch=64` 启动；完整 200 epochs 仍需监控动态峰值。

## 7. 启动门禁

- [x] 用户于 2026-09-07 明确批准提交代码并启动双流训练；
- [x] 本地实现、配置和 smoke test；
- [x] RTX 4090 单 batch backward；
- [x] RTX 4090 完整 1 epoch 压力测试；
- [ ] 当前 commit 推送到 GitHub；
- [ ] 服务器同步最终记录 commit，Git 工作树干净；
- [ ] 启动前再次校验 9,306 项冻结数据、权重、GPU 和磁盘；
- [ ] 启动后记录 `args.yaml`、日志和快照 SHA-256。

计划命令：

```bash
/root/.local/bin/uv run python -m scripts.train_rgbir \
  --config configs/train/rgbir_dual.yaml \
  --device 0 \
  --name rgbir_dual
```

预计输出：

```text
/root/autodl-tmp/DroneVehicle/runs/obb/runs/obb/rgbir_dual
```

## 8. 结果模板

| 字段 | 待填 |
|---|---|
| 开始/结束时间 | — |
| 完成 epoch / best epoch | — |
| 峰值显存 | — |
| Precision / Recall | — |
| mAP@0.5 / mAP@0.5:0.95 | — |
| `best.pt` / `last.pt` | — |
| `args.yaml` / `results.csv` | — |
| 三尺度门控统计 | — |

在第二阶段结构全部冻结之前，只依据 val 比较，不使用 test 选择 P2 或融合方案。

# 正式消融实验记录与启动核对表

> 当前状态：**允许按实验组分阶段启动**。每组只需完成自身实现、配置检查、RTX 4090 预检和用户批准；后续若公共 batch 改变，已完成组必须重跑。当前没有正式实验正在运行。

## 1. 固定实验矩阵

| 实验 ID | 结构 | 唯一结构变化 | 当前状态 |
|---|---|---|---|
| `rgbir_baseline` | RGB+IR 四通道融合 | 基准组 | 本地及 RTX 4090 预检通过 |
| `rgbir_spd` | 四通道融合 + SPD | Layer 3 的 P2/4→P3/8 下采样改为 SPDConv | 本地及 RTX 4090 预检通过 |
| `rgbir_spd_deab` | 四通道融合 + 同一个 SPD + DEAB | 相对 `rgbir_spd` 只增加 DEAB | 尚未实现 |

对比关系：

- `rgbir_baseline` vs `rgbir_spd`：衡量 SPD 增益；
- `rgbir_spd` vs `rgbir_spd_deab`：衡量 DEAB 在 SPD 基础上的增益；
- 不设置“仅 DEAB”正式实验。

## 2. 正式启动门禁

### 公共门禁

- [x] RGB/IR 配对、清洗、严格交集和标签冻结完成；
- [ ] 当前规则与记录表已提交并推送，服务器 commit 与本地/GitHub 一致；
- [ ] 启动当日再次确认服务器 Git 干净、GPU 空闲、磁盘充足及数据/权重哈希。

### `rgbir_baseline` 门禁

- [x] 四通道构建、forward、loss、backward 和 AMP smoke test 通过；
- [x] RTX 4090 `batch=64, imgsz=640, AMP=True` 单 batch 预检通过；
- [x] 用户已在 2026-09-06 明确批准优先分阶段启动四通道 baseline；
- [ ] 正式启动前补填第 8 节的最终 commit、哈希、命令和服务器快照。

### `rgbir_spd` 门禁

- [x] SPD 构建、forward、loss、backward 和 AMP smoke test 通过；
- [x] RTX 4090 `batch=64, imgsz=640, AMP=True` 单 batch 预检通过；
- [ ] 与 baseline 的最终配置一致性重新校验；
- [ ] 用户单独批准该组正式启动。

### `rgbir_spd_deab` 门禁

- [ ] SPD+DEAB 模块、模型 YAML 和训练配置完成；
- [ ] 与 SPD 的最终配置一致性校验；
- [ ] RTX 4090 `batch=64, imgsz=640, AMP=True` 预检；
- [ ] 用户单独批准该组正式启动。

各组只受自身门禁和公共门禁约束，不要求等待后续模型实现。若后续任一模型无法稳定使用 `batch=64`，三组必须统一改为 `batch=32`，已经完成的实验也必须重新训练。

## 3. 固定数据

数据根目录：`data -> /root/autodl-tmp/DroneVehicle-data`

筛选规则：同类别、一对一旋转 IoU ≥ 0.7，且每对图像的 RGB/IR 标签覆盖率均为 100%；最终使用 RGB OBB 作为唯一监督标签。

| 划分 | 图像对 | 目标数 | car | truck | bus | van | Freight_car |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 5,891 | 92,205 | 75,661 | 5,594 | 5,072 | 2,202 | 3,676 |
| val | 488 | 7,380 | 5,898 | 596 | 345 | 218 | 323 |
| test | 2,923 | 43,144 | 35,423 | 2,969 | 1,895 | 1,161 | 1,696 |
| **合计** | **9,302** | **142,729** | **116,982** | **9,159** | **7,312** | **3,581** | **5,695** |

配准统计：中心偏移 P50=0、P95≈3.0 px、P99≈6.08 px；87.2% 的冻结匹配框坐标完全一致。

### 数据指纹

| 文件 | SHA-256 |
|---|---|
| `data/cleaned/report.json` | `e506cc9ea4c4bead51608bcea3261673e52527e9b9670f5f6732ac7cec37ea9e` |
| `data/frozen_rgbir/report.json` | `c8887d31efcb4d6bf62890847e7fb64bb59fe6a5f2e0a42d07d50855b39756f9` |
| `data/frozen_rgbir/checksums.sha256` | `fd7eaf81719a76a401b97a34a25cc1e8a55b73b847539f2eda0c94376e698bd0` |
| `configs/data/dronevehicle_rgbir.yaml` | `3d0fc28122a8d1d8aee03d15c895d740abf40d1fcc5517fafaeb03b774040b86` |

正式启动前必须再次执行 `sha256sum -c data/frozen_rgbir/checksums.sha256`，要求 9,306 项全部通过。

## 4. 代码与权重指纹

| 项目 | 当前值 |
|---|---|
| GitHub | `https://github.com/Eorien/DroneVehicle.git` |
| 当前开发 commit | `5696550161a220a2b113ac920b9419f0757ad4ab` |
| 正式实验 commit | **待 SPD+DEAB 完成后冻结** |
| Ultralytics | `8.4.140` |
| Ultralytics upstream commit | `7401d284e77b58d20f3c59aa1d2fcbb496bb7a0e` |
| `yolo11n-obb.pt` SHA-256 | `b62898ebf38940ca4df323863e45ee9d84a1a46d5d11ebdde529fb33aa9f3a32` |
| `yolo26n.pt` SHA-256 | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |

`yolo11n-obb.pt` 是三组共同预训练权重；`yolo26n.pt` 仅供 Ultralytics AMP 自检，不参与训练。

当前配置哈希（正式启动前重新计算）：

| 配置 | SHA-256 |
|---|---|
| baseline 模型 YAML | `147a3bc56ae129d3760419f6c5d674f7d4f635ca9c5098aec90d1aeb376faea6` |
| SPD 模型 YAML | `3a27367d4bd1d8c02ad59719117cd0d3ca2c178810d41fa7e188bd70c0e0cecd` |
| baseline 训练 YAML | `a6055d742b226d54ec64008bdb00dc2d4c2ff53812b481277015961a04d18473` |
| SPD 训练 YAML | `27bea3c1231ae98e96d34deb3917f21cbd532ce66576dd592b785d9eb2a12deb` |
| SPD+DEAB 模型/训练 YAML | **待创建** |

## 5. 服务器环境

| 项目 | 当前值 |
|---|---|
| 工作目录 | `/root/autodl-tmp/DroneVehicle` |
| OS | Ubuntu 22.04.4 LTS |
| CPU 配额 | 12 核 |
| 内存配额 | 90GB |
| GPU | NVIDIA GeForce RTX 4090 24GB |
| NVIDIA Driver | `595.71.05` |
| uv | `0.11.7` |
| Python | `3.12.3` |
| PyTorch | `2.8.0+cu128` |
| TorchVision | `0.23.0+cu128` |
| cuDNN | `91002` |
| OpenCV | `5.0.0` |
| PyYAML | `6.0.3` |
| Ultralytics 来源 | 仓库内 `third_party/ultralytics/` |

环境位于项目 `.venv`。正式启动前记录 `nvidia-smi` 完整输出、磁盘剩余空间和 GPU 空闲状态。

## 6. 三组统一训练参数

### 显式配置

| 参数 | 固定值 |
|---|---:|
| epochs | 200 |
| batch | 64（若任一模型不稳定，三组统一降为 32） |
| imgsz | 640 |
| workers | 8 |
| patience | 50 |
| optimizer | Adam |
| lr0 | 0.01 |
| momentum | 0.937 |
| weight_decay | 0.0005 |
| AMP | True |
| seed | 0 |
| deterministic | True |
| cache | False |
| fraction | 1.0 |
| rect | False |
| multi_scale | 0.0 |
| HSV/BGR 增强 | 全部关闭 |

### 必须保持一致的有效默认项

| 参数 | 固定值 |
|---|---:|
| nbs / accumulate | 64 / 1（batch=64 时） |
| mosaic / close_mosaic | 1.0 / 10 |
| translate / scale | 0.1 / 0.5 |
| degrees / shear / perspective | 0.0 / 0.0 / 0.0 |
| fliplr / flipud | 0.5 / 0.0 |
| mixup / cutmix / copy_paste | 0.0 / 0.0 / 0.0 |
| lrf / cos_lr | 0.01 / False |
| warmup_epochs | 3.0 |
| warmup_momentum / warmup_bias_lr | 0.8 / 0.1 |
| box / cls / dfl / angle loss 权重 | 7.5 / 0.5 / 1.5 / 1.0 |
| max_det | 300 |

每次正式运行必须保存 Ultralytics 生成的完整 `args.yaml`，以它作为最终有效配置证据。

## 7. 模型结构与 4090 预检

| 模型 | Parameters | GFLOPs | 预训练迁移 | batch=64 AMP forward | 单 batch backward | 峰值显存 |
|---|---:|---:|---:|---|---|---:|
| baseline | 2,662,626 | 6.8 | 535/541 | 通过 | 通过 | 约 8.79GB |
| SPD | 2,773,218 | 8.2 | 529/541 | 通过 | 通过 | 约 8.79GB |
| SPD+DEAB | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

预检日期：2026-09-06。预检仅使用 64 张训练图、1 epoch 中的单个训练 batch，输出已删除，不属于正式实验结果。

## 8. 每次正式运行前必须记录

| 字段 | 记录值 |
|---|---|
| 实验 ID | `rgbir_baseline` / `rgbir_spd` / `rgbir_spd_deab` |
| 启动批准人及时间 | 待填 |
| 完整启动命令 | 待填 |
| Git commit | 待填 |
| `git status --short` | 必须为空 |
| 数据 report/checksums 哈希 | 待填 |
| 数据 9,306 项校验 | 必须全部通过 |
| 两个权重 SHA-256 | 待填 |
| 模型 YAML SHA-256 | 待填 |
| 训练 YAML SHA-256 | 待填 |
| 解析后的 `args.yaml` SHA-256 | 运行创建后补填 |
| GPU/驱动/PyTorch/cuDNN | 待填 |
| GPU 启动前显存与占用进程 | 必须空闲 |
| 磁盘剩余空间 | 待填 |
| Parameters / GFLOPs | 待填 |
| 预训练迁移数量及未迁移原因 | 待填 |
| 随机种子 | 必须为 0 |
| 是否从断点恢复 | 默认否；若是必须记录来源和原因 |

## 9. 每组正式实验结果记录模板

### 运行信息

| 字段 | 记录值 |
|---|---|
| 实验 ID | 待填 |
| run 目录 | 待填 |
| 开始/结束时间 | 待填 |
| 总耗时 | 待填 |
| 完成 epoch / best epoch | 待填 |
| 停止原因 | 正常完成 / early stop / 人工中断 / 异常 |
| 峰值显存 | 待填 |
| 平均单轮时间 | 待填 |
| 是否发生 OOM、NaN、重启或恢复 | 待填 |

### best.pt 验证集总体指标

| Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---:|---:|---:|---:|
| 待填 | 待填 | 待填 | 待填 |

### best.pt 测试集总体指标

| Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---:|---:|---:|---:|
| 待填 | 待填 | 待填 | 待填 |

### best.pt 测试集分类指标

| 类别 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| car | 待填 | 待填 | 待填 | 待填 |
| truck | 待填 | 待填 | 待填 | 待填 |
| bus | 待填 | 待填 | 待填 | 待填 |
| van | 待填 | 待填 | 待填 | 待填 |
| Freight_car | 待填 | 待填 | 待填 | 待填 |

### 训练状态与效率

| 字段 | 记录值 |
|---|---|
| best epoch 的 box/cls/dfl/angle loss | 待填 |
| 最终 epoch 的 box/cls/dfl/angle loss | 待填 |
| preprocess / inference / postprocess 速度 | 待填 |
| 实际 batch / accumulate | 待填 |
| 实际 workers | 待填 |
| 平均 GPU 利用率/温度（若采集） | 待填 |

### 产物

| 文件 | SHA-256 / 路径 |
|---|---|
| `best.pt` | 待填 |
| `last.pt` | 待填 |
| `args.yaml` | 待填 |
| `results.csv` | 待填 |
| 验证输出与混淆矩阵 | 待填 |
| 训练日志 | 待填 |

## 10. 三组测试集结果汇总

| 实验 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | ΔmAP@0.5:0.95 | Params | GFLOPs | 峰值显存 | 耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 待填 | 待填 | 待填 | 待填 | — | 2,662,626 | 6.8 | 待填 | 待填 |
| SPD | 待填 | 待填 | 待填 | 待填 | 相对 baseline | 2,773,218 | 8.2 | 待填 | 待填 |
| SPD+DEAB | 待填 | 待填 | 待填 | 待填 | 相对 SPD | 待填 | 待填 | 待填 | 待填 |

## 11. 异常与变更规则

- 不得在某一组训练中单独修改超参数；
- 不得把 smoke test 指标写入正式结果；
- 正式运行发生中断时，先记录时间、epoch、错误和 checkpoint，再决定是否恢复；
- 任何数据、代码、配置、权重或 batch 变化都会使已有对比失效，必须重新核对三组；
- `best.pt` 只能由验证集选择；测试集仅用于最终评价，不得据此调参或重新选择 checkpoint；
- 不覆盖已有 run 目录，每个正式实验只能对应一个明确目录；
- 不删除原始日志、`args.yaml`、`results.csv`、`best.pt` 或 `last.pt`；
- 不根据中途指标只调整某一个模型。

## 12. 用户分阶段批准

- [x] `rgbir_baseline`：用户于 2026-09-06 核对记录后明确批准优先启动；
- [ ] `rgbir_spd`：待该组启动前单独批准；
- [ ] `rgbir_spd_deab`：待实现、预检和启动前单独批准。

批准只对对应实验组有效。每组实际启动前仍必须完成公共门禁，并在第 8 节补齐最终运行快照。

# 正式消融实验记录与启动核对表

> 当前状态：`rgbir_baseline` 与 `rgbir_spd` 已完成正式训练、验证和 test 评估；`rgbir_spd_deab` 已于 2026-09-06 23:09 启动正式训练。

## 1. 固定实验矩阵

| 实验 ID | 结构 | 唯一结构变化 | 当前状态 |
|---|---|---|---|
| `rgbir_baseline` | RGB+IR 四通道融合 | 基准组 | 200 epochs、val、test 均已完成 |
| `rgbir_spd` | 四通道融合 + SPD | Layer 3 的 P2/4→P3/8 下采样改为 SPDConv | 200 epochs、val、test 均已完成 |
| `rgbir_spd_deab` | 四通道融合 + 同一个 SPD + DEAB | Layer 3 保持同一 SPD，紧接一个 P3/8 DEAB | 正式训练中（2026-09-06 23:09 启动） |

对比关系：

- `rgbir_baseline` vs `rgbir_spd`：衡量 SPD 增益；
- `rgbir_spd` vs `rgbir_spd_deab`：衡量 DEAB 在 SPD 基础上的增益；
- 不设置“仅 DEAB”正式实验。

## 2. 正式启动门禁

### 公共门禁

- [x] RGB/IR 配对、清洗、严格交集和标签冻结完成；
- [x] 当前规则与记录表已提交并推送，服务器 commit 与本地/GitHub 一致；
- [x] baseline 与 SPD 启动当日均已确认服务器 Git 干净、GPU 空闲、磁盘充足及数据/权重哈希。

### `rgbir_baseline` 门禁

- [x] 四通道构建、forward、loss、backward 和 AMP smoke test 通过；
- [x] RTX 4090 `batch=64, imgsz=640, AMP=True` 单 batch 预检通过；
- [x] 用户已在 2026-09-06 明确批准优先分阶段启动四通道 baseline；
- [x] 正式启动快照、最终 commit、哈希、命令和服务器状态均已记录。

### `rgbir_spd` 门禁

- [x] SPD 构建、forward、loss、backward 和 AMP smoke test 通过；
- [x] RTX 4090 `batch=64, imgsz=640, AMP=True` 单 batch 预检通过；
- [x] 与 baseline 的最终配置一致性重新校验，仅 `model` 和 `name` 不同；
- [x] 用户于 2026-09-06 明确批准该组正式启动。

### `rgbir_spd_deab` 门禁

- [x] SPD+DEAB 模块、模型 YAML 和训练配置完成；
- [x] 与 SPD 的配置和共享状态一致性校验通过，仅新增 `model.3.deab.*`；
- [x] RTX 4090 `batch=64, imgsz=640, AMP=True` 单 batch backward 预检通过，峰值 9.67GB；
- [x] 用户于 2026-09-06 在核对 SPD 结果和 OOM 风险后批准该组正式启动。

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
| 正式训练代码 commit | `bcdd0a44eadcb83d92488c7e6ad3512ccb5997b2` |
| 文档更新 | baseline 结果和 SPD 启动状态在本次本地更新中回填 |
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
| SPD+DEAB 模型 YAML | `0223979f7c09158a6074a28873adc112965c0d479710c713788fb34ddc8c946d` |
| SPD+DEAB 训练 YAML | `701eedac8171fd8770248984aed40e80591e3c1d9d9c5f087d40f5668cf258c9` |

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
| SPD+DEAB | 2,953,165 | 8.8 | 529/561 | 通过 | 通过 | 9.67GB |

预检日期：2026-09-06。预检仅使用 64 张训练图、1 epoch 中的单个训练 batch，输出已删除，不属于正式实验结果。

baseline 正式 200 epochs 训练日志中的实际峰值显存为 **20.5GB**；单 batch 预检值不能代表完整训练峰值。SPD 和后续 SPD+DEAB 必须继续监控密集 batch 的动态峰值。

SPD+DEAB 本地门禁已通过：DEConv 数学约束、CPU/CUDA forward/backward、SPD 的 541 个共享状态逐元素一致、真实四通道 `640×640` forward、AMP loss/backward，以及 1 epoch smoke training。新增 179,947 parameters、20 个状态张量，均位于 `model.3.deab.*`。

SPD+DEAB RTX 4090 预检日志：`runs/logs/rgbir_spd_deab_batch64_precheck_20260906_194619.log`，SHA-256=`e21cd0b7374c7037c956f00a270eee391f04f7308d638b3e279e79160e442d84`。相对 SPD 预检约增加 0.88GB；按 SPD 正式峰值 20.3GB 粗略估计，正式训练可能达到约 21～22GB，仍需全程监控。

## 8. 每次正式运行前必须记录

### 8.1 `rgbir_baseline` 启动记录

| 字段 | 记录值 |
|---|---|
| 批准与启动时间 | 不思议先生批准；2026-09-06 12:32:25 +08:00 启动 |
| 完整命令 | `/root/.local/bin/uv run python -m scripts.train_rgbir --config configs/train/rgbir_baseline.yaml --device 0 --name rgbir_baseline` |
| Git commit / status | `bcdd0a44eadcb83d92488c7e6ad3512ccb5997b2`；工作树干净 |
| 数据 report / checksums | `c8887d31efcb4d6bf62890847e7fb64bb59fe6a5f2e0a42d07d50855b39756f9` / `fd7eaf81719a76a401b97a34a25cc1e8a55b73b847539f2eda0c94376e698bd0`；9,306 项全部通过 |
| 权重 | `yolo11n-obb.pt b62898ebf38940ca4df323863e45ee9d84a1a46d5d11ebdde529fb33aa9f3a32`；`yolo26n.pt 9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| 模型 / 训练 YAML | `147a3bc56ae129d3760419f6c5d674f7d4f635ca9c5098aec90d1aeb376faea6` / `a6055d742b226d54ec64008bdb00dc2d4c2ff53812b481277015961a04d18473` |
| `args.yaml` | `dcf6f73772cc8aeabfaa49375a6ec18038e75cabb81332108a25db3569e410da` |
| 服务器快照 | RTX 4090 24GB、Driver 595.71.05、GPU 空闲、数据盘剩余 39GB |
| 模型 / 迁移 | 2,662,626 params、6.8 GFLOPs；535/541；IR=RGB 卷积权重均值 |
| seed / resume | `0` / 否 |

### 8.2 `rgbir_spd` 启动记录

| 字段 | 记录值 |
|---|---|
| 批准与启动时间 | 不思议先生批准；2026-09-06 18:01:43 +08:00 启动 |
| 完整命令 | `/root/.local/bin/uv run python -m scripts.train_rgbir --config configs/train/rgbir_spd.yaml --device 0 --name rgbir_spd` |
| Git commit / status | `bcdd0a44eadcb83d92488c7e6ad3512ccb5997b2`；工作树干净 |
| 配置一致性 | 与 baseline 仅 `model` 和 `name` 不同 |
| 数据 report / checksums | `c8887d31efcb4d6bf62890847e7fb64bb59fe6a5f2e0a42d07d50855b39756f9` / `fd7eaf81719a76a401b97a34a25cc1e8a55b73b847539f2eda0c94376e698bd0`；9,306 项全部通过 |
| 权重 | `yolo11n-obb.pt b62898ebf38940ca4df323863e45ee9d84a1a46d5d11ebdde529fb33aa9f3a32`；`yolo26n.pt 9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| 模型 / 训练 YAML | `3a27367d4bd1d8c02ad59719117cd0d3ca2c178810d41fa7e188bd70c0e0cecd` / `27bea3c1231ae98e96d34deb3917f21cbd532ce66576dd592b785d9eb2a12deb` |
| `args.yaml` | `02182648c240c1d7a7ff9d77d531de3c2e27fc6e68e91ff62c3f2638d1981ac0` |
| 服务器快照 | RTX 4090 24GB、Driver 595.71.05、GPU 空闲、数据盘剩余 39GB |
| 模型 / 迁移 | 2,773,218 params、8.2 GFLOPs；529/541；Layer 3 SPD 新参数未迁移；IR=RGB 卷积权重均值 |
| seed / resume | `0` / 否 |
| 启动快照 | `runs/logs/rgbir_spd_20260906_180143_launch.txt` |

### 8.3 `rgbir_spd_deab` 启动记录

| 字段 | 记录值 |
|---|---|
| 批准 / 启动 | 不思议先生于 2026-09-06 核对 SPD 结果和 OOM 风险后批准；2026-09-06 23:09:17 +08:00 启动 |
| 完整命令 | `/root/.local/bin/uv run python -m scripts.train_rgbir --config configs/train/rgbir_spd_deab.yaml --device 0 --name rgbir_spd_deab` |
| 训练代码 commit | `31eb000be586a4c3af7eba54cbcfb866dfc2c6ee` |
| 配置一致性 | 与 SPD 仅 `model` 和 `name` 不同；模型仅在 Layer 3 SPD 输出后增加 DEAB |
| 数据 report / checksums | `c8887d31efcb4d6bf62890847e7fb64bb59fe6a5f2e0a42d07d50855b39756f9` / `fd7eaf81719a76a401b97a34a25cc1e8a55b73b847539f2eda0c94376e698bd0` |
| 权重 | `yolo11n-obb.pt b62898ebf38940ca4df323863e45ee9d84a1a46d5d11ebdde529fb33aa9f3a32`；`yolo26n.pt 9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| 模型 / 训练 YAML | `0223979f7c09158a6074a28873adc112965c0d479710c713788fb34ddc8c946d` / `701eedac8171fd8770248984aed40e80591e3c1d9d9c5f087d40f5668cf258c9` |
| 模型 / 迁移 | 2,953,165 params、8.8 GFLOPs；529/561；IR=RGB 卷积权重均值 |
| RTX 4090 预检 | `batch=64` AMP backward 通过，峰值 9.67GB |
| 服务器状态 | repository commit `b0e13e052548876f09c0e83b37d65a36487de58d`、Git 干净、GPU 启动前空闲、数据盘剩余 39GB |
| seed / resume | `0` / 否 |
| `args.yaml` | `acd05986b8778750db1b3f1669cac44b3fbf093b70b349cd7df81a1b7a7b4b62` |
| 启动快照 | `runs/logs/rgbir_spd_deab_20260906_230917_launch.txt`；SHA-256=`9b219dcafa4c76a2dcce3d5eb55f85671f75fe822499c516d2340e7605935983` |

## 9. 正式实验结果

### 9.1 `rgbir_baseline`

#### 运行信息

| 字段 | 记录值 |
|---|---|
| run 目录 | `/root/autodl-tmp/DroneVehicle/runs/obb/runs/obb/rgbir_baseline` |
| 开始 / 结束 | 2026-09-06 12:32:25 / 14:10:31 +08:00 |
| 总耗时 | 1.620 小时（CSV 累计 5,833.22 秒） |
| 完成 epoch / best epoch | 200 / 200 |
| 停止原因 | 正常完成 |
| 训练日志峰值显存 | 20.5GB |
| 平均单轮时间 | 约 29.17 秒 |
| OOM / NaN / 恢复 | 均无；未从断点恢复 |

#### `best.pt` 验证集总体指标

| Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---:|---:|---:|---:|
| 0.75739 | 0.75059 | 0.76207 | 0.60160 |

#### `best.pt` 测试集总体指标

| Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---:|---:|---:|---:|
| 0.73906 | 0.76579 | 0.77223 | 0.61035 |

#### `best.pt` 测试集分类指标

| 类别 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| car | 0.93582 | 0.97112 | 0.98607 | 0.81800 |
| truck | 0.69328 | 0.80633 | 0.79463 | 0.58574 |
| bus | 0.91026 | 0.89920 | 0.92912 | 0.76338 |
| van | 0.55282 | 0.49096 | 0.50494 | 0.40456 |
| Freight_car | 0.60314 | 0.66132 | 0.64641 | 0.48008 |

#### 训练状态与效率

| 字段 | 记录值 |
|---|---|
| best/final epoch 的 box/cls/dfl/angle loss | `0.64329 / 0.40530 / 0.96035 / 0.00892` |
| test preprocess / inference / postprocess | `1.020 / 0.596 / 2.233 ms/image` |
| 实际 batch / accumulate / workers | `64 / 1 / 8` |
| 测试说明 | 首次独立评估包装器因未设置 `PYTHONPATH` 在导入阶段退出，未读取模型和数据；设置项目根目录后重试成功 |

#### 产物

| 文件 | SHA-256 / 路径 |
|---|---|
| `best.pt` | `0e48f388ae8a1eca87e80d7e0c91ecfeacbfb0d2f43f51f842782f676d587062` |
| `last.pt` | `a8cfaf5a832f2c59bcc9c8731936febb28ecfd22e425528f56ac2ecefd3451ab` |
| `args.yaml` | `dcf6f73772cc8aeabfaa49375a6ec18038e75cabb81332108a25db3569e410da` |
| `results.csv` | `e4c1309427aa4d829bab33f2295a8a1831e323f57b0ba10ed05437b905e75a02` |
| 训练日志 | `c8f73750f26832050f27be5d1ff951f866e311a54f612d56ed1c656446fd064f` |
| test 指标 | `test_metrics.json`：`078d1ed34a87c510c14c8911cad4cba1c6d015dc1dcd93ee80eef5a115061266` |
| test 日志 | `92fce89305697c3178e47a6f022e36e18f71d20a8adf2b0f7334adbdc0f6e12e` |
| 本地归档 | `outputs/formal_experiments/rgbir_baseline/`，47 个文件，完整校验通过 |

### 9.2 `rgbir_spd`

正常完成 200 epochs，best epoch=199，总耗时 1.551 小时，峰值显存 20.3GB，无 OOM/NaN/恢复。best.pt 的 val 指标为 P=`0.74731`、R=`0.76658`、mAP50=`0.76206`、mAP50-95=`0.60714`；test 指标为 P=`0.73300`、R=`0.77063`、mAP50=`0.76919`、mAP50-95=`0.61078`。

产物：`best.pt ff68f7e5...d555`、`last.pt 4190028f...da60`、`results.csv 6ac1bf2e...b46`、`test_metrics.json cc9dffee...5806`；已归档至 `outputs/formal_experiments/rgbir_spd/`，43 个文件校验通过。

### 9.3 `rgbir_spd_deab`

正式训练已于 2026-09-06 23:09:17 +08:00 启动，tmux=`rgbir_spd_deab`；结果待完成后回填。

## 10. 三组测试集结果汇总

| 实验 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | ΔmAP@0.5:0.95 | Params | GFLOPs | 峰值显存 | 耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.73906 | 0.76579 | 0.77223 | 0.61035 | — | 2,662,626 | 6.8 | 20.5GB | 1.620h |
| SPD | 0.73300 | 0.77063 | 0.76919 | 0.61078 | +0.00042 | 2,773,218 | 8.2 | 20.3GB | 1.551h |
| SPD+DEAB | 待填 | 待填 | 待填 | 待填 | 相对 SPD | 2,953,165 | 8.8 | 待填 | 待填 |

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
- [x] `rgbir_spd`：用户于 2026-09-06 核对 baseline 结果后明确批准并启动；
- [x] `rgbir_spd_deab`：用户于 2026-09-06 核对 SPD 结果、DEAB 本地测试和 OOM 风险后明确批准启动。

批准只对对应实验组有效。每组实际启动前仍必须完成公共门禁，并在第 8 节补齐最终运行快照。

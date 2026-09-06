# AGENTS

## 项目目标

本项目面向无人机交通场景，基于 **YOLO11-OBB** 逐步实现 **RGB-IR 四通道前期融合 + SPD + DEAB 小目标检测**。

当前只做 3 组正式实验：

1. `rgbir_baseline`：RGB+IR 四通道前期融合
2. `rgbir_spd`：四通道融合 + SPD
3. `rgbir_spd_deab`：四通道融合 + SPD + DEAB

目标是完成一个控制变量清晰的小型消融实验，不额外引入无关结构或训练策略。

---

## 基础设定

- 任务：OBB 旋转目标检测
- 输入：RGB 3 通道 + IR 1 通道，共 4 通道
- 基础模型：YOLO11-OBB
- 预训练权重：`yolo11n-obb.pt`
- 输入尺寸：`640`
- 类别：
  - car
  - truck
  - bus
  - van
  - Freight_car

RGB 与 IR 必须严格配对。所有旋转、裁剪、翻转、缩放等几何增强必须同步作用于 RGB、IR 和标签。

---

## 训练参考配置

正式实验默认保持一致：

```text
epochs = 200
batch = 64
imgsz = 640
workers = 8
patience = 50
optimizer = Adam
lr0 = 0.01
momentum = 0.937
weight_decay = 0.0005
AMP = True
seed = 0
deterministic = True
```

正式实验三组统一使用 `batch=64`。本机 smoke test 可使用 `batch=16` 并缩小数据集、epoch；不得把调试配置当作正式实验配置。允许分阶段训练：每组通过自身本地 smoke test 和 RTX 4090 `batch=64` 预检，并由用户核对该组实验记录后，即可启动该组正式训练。若后续任一模型无法稳定使用 64，三组统一降为 32，已完成的实验也必须重跑。

评价指标至少包括：

- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95

---

## Ultralytics 基线

本项目使用仓库内固定版本源码：

```text
third_party/ultralytics/
```

固定版本：

```text
Ultralytics: 8.4.140
Commit: 7401d284e77b58d20f3c59aa1d2fcbb496bb7a0e
```

规则：

- 不主动升级 Ultralytics
- 修改前先阅读本地源码
- 尽量最小化修改
- 优先通过 YAML / 配置控制 baseline、SPD、SPD+DEAB
- 不要复制三份完整模型代码
- 保持 OBB 检测逻辑不变

---

## 模块要求

### 四通道输入

YOLO11-OBB 需要从 3 通道输入改为 4 通道输入。

修改首层时：

- 尽量保留预训练权重
- 明确说明第 4 个 IR 通道如何初始化
- 不允许静默丢弃预训练参数

### SPD

用于减少下采样过程中小目标空间细节损失。

要求：

- 独立可开关
- 只修改预定位置
- SPD 实验与 baseline 的其他条件保持一致

### DEAB

用于增强小目标细节和有效响应。

要求：

- 只用于 `rgbir_spd_deab`，必须与 SPD 同时启用
- `rgbir_spd_deab` 与 `rgbir_spd` 之间只允许增加 DEAB
- 当前只实现一种明确插入方案，不额外派生多个 DEAB 版本

---

## 开发与训练流程

### 本地 WSL

本地是主要开发环境：

```text
RTX 5060
Python 3.12
uv 管理依赖
```

负责：

- 改代码
- 数据预处理
- RGB/IR 配对检查
- shape 检查
- forward 测试
- 小规模 smoke test
- Git 提交

### AutoDL

RTX 4090 只负责：

- 完整数据集训练
- 正式消融实验
- 验证
- 导出权重

#### SSH 连接

当前 WSL 的 `~/.ssh/config` 使用以下别名：

```sshconfig
Host autodl
    HostName connect.westc.seetacloud.com
    User root
    Port 52096
```

连接命令：

```bash
ssh autodl
# 等价于：ssh -p 52096 root@connect.westc.seetacloud.com
```

规则和环境说明：

- 认证密码不得写入 `AGENTS.md`、脚本、Git 或 shell 命令；连接时交互输入
- AutoDL 实例重启后 SSH 端口可能变化，连接失败时先到控制台确认并同步更新 `~/.ssh/config` 和本节
- 项目工作目录统一使用 `/root/autodl-tmp/DroneVehicle`
- 高速数据盘 `/root/autodl-tmp` 为 50GB，不同步原始 ZIP 和未入选图像
- 登录 shell 中 Miniconda 位于 `/root/miniconda3`；非交互命令应使用 `bash -lic` 或完整 Python 路径
- 服务器项目已部署到 `/root/autodl-tmp/DroneVehicle`，数据通过 `data -> ../DroneVehicle-data` 软链接接入
- 服务器使用 `uv 0.11.7` 和项目 `.venv`，环境已锁定为 Python 3.12、PyTorch 2.8.0+cu128、Ultralytics 8.4.140
- RTX 4090 上 baseline 与 SPD 的 `batch=64, imgsz=640, AMP=True` 单 batch 反向传播预检均通过，预检峰值约 8.79GB
- baseline 已完成 200 epochs、val 和 test：test mAP@0.5:0.95=`0.61035`，正式训练日志峰值显存 20.5GB，产物已同步至本地 `outputs/formal_experiments/rgbir_baseline/`
- SPD 已于 2026-09-06 18:01 启动正式训练，tmux 会话为 `rgbir_spd`；启动时 commit 为 `bcdd0a44eadcb83d92488c7e6ad3512ccb5997b2`
- SPD+DEAB 已完成本地实现与 smoke test：单个 DEAB 紧接 Layer 3 SPD 的 P3/8 输出，模型 2,953,165 parameters / 8.8 GFLOPs；待 RTX 4090 `batch=64` 预检
- 若 SPD+DEAB 无法稳定使用 `batch=64`，三组统一降为 32，baseline 与 SPD 必须重跑
推荐流程：

```text
本地修改/调试
→ git commit + push
→ AutoDL git pull
→ 正式训练
→ 下载 best.pt
```

服务器原则上不要直接改代码。

---

## 文件同步

Git 只管理：

- 代码
- 配置
- 脚本
- 文档
- `pyproject.toml`
- `uv.lock`

以下内容不要提交：

```text
data/
runs/
weights/
checkpoints/
outputs/
*.pt
*.pth
*.onnx
*.engine
.venv/
```

大文件、数据集、训练权重使用 `rsync` / SSH 同步。

---

## Codex 工作规则

在本仓库工作时：

1. 修改前先读相关源码，不猜 API 或文件位置。
2. 只做与当前任务直接相关的最小修改。
3. baseline / SPD / SPD+DEAB 必须严格控制变量。
4. 优先配置化，不复制整套模型。
5. 重点检查：
   - 4 通道输入
   - 首层卷积
   - SPD reshape / channel 变化
   - DEAB 插入位置
   - OBB Detect head
6. 未经用户核对该组实验记录并明确批准，不运行完整 200 epoch 训练；获批后允许按实验组分阶段启动。
   - 启动前必须读取并更新 `docs/formal_experiment_record.md`，记录该组 commit、配置/数据/权重哈希、服务器状态和批准信息。
7. 不自动下载大型数据集或权重。
8. 不伪造实验结果。
9. 每次修改后说明：
   - 改了哪些文件
   - 改了什么
   - 做了什么验证
   - 还有什么风险

---

# 学习笔记

D:\外接大脑\多模态\DroneVehicle.md
# DroneVehicle RGB-IR OBB

基于 YOLO11-OBB 的 RGB + IR 四通道前期融合小目标检测项目。

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
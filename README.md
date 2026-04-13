# Habitat-CBM

胶质瘤 IDH 突变状态预测的深度学习基线框架。基于多模态 MRI（T1/T1CE/T2/T2FLAIR/ADC/CBF）进行患者级别的二分类（野生型 vs 突变型），并提供基于 MONAI 的 2.5D 数据增强 pipeline。

## 快速开始

```bash
# 1. 环境安装
pip install -r requirements.txt

# 2. 数据划分（若需重新划分）
python srcs/data_split.py \
    --dataset-root ../dataset/images \
    --output-root ../dataset/splited_data \
    --train-ratio 0.7 --val-ratio 0.1 --test-ratio 0.2 \
    --link-mode copy

# 3. 运行基线训练
python srcs/baseline_ResNet18.py \
    --split-base-root ../dataset/splited_data \
    --modalities t1,t1ce,t2,t2flair,adc,cbf \
    --batch-size 32 --epochs 200 \
    --run-id exp_baseline
```

## 项目结构

```
repo/
├── srcs/
│   ├── data_split.py          # 患者级分层数据划分
│   ├── data_loader.py         # 2.5D 多模态数据加载器
│   ├── monai_augmentation.py  # MONAI 数据增强配置与 transform 构建器
│   └── baseline_ResNet18.py   # ResNet-18 训练/评估脚本
├── models/
│   └── resnet_18.py           # 适配多通道输入的 ResNet-18
├── requirements.txt           # 主环境依赖
└── requirements_pyradiomics.txt  # 影像组学环境（可选）

dataset/
├── images/                    # 原始 NIfTI 影像
│   ├── conventional/          # T1, T1CE, T2, T2FLAIR, VOI
│   └── functional/            # ADC, CBF, VOI
├── splited_data/              # 划分后的数据
│   ├── manifests/             # 划分清单 CSV
│   ├── train/val/test/        # 训练/验证/测试集
└── features/                  # 影像组学特征（可选）

results/
└── baseline_ResNet18/         # 训练输出
    ├── {run_id}/              # 单次实验结果
    │   ├── patient_predictions_*.csv
    │   ├── metrics_summary_*.csv
    │   ├── training_history_*.csv
    │   ├── run_summary_*.json
    │   └── training_log_*.txt
    └── checkpoints/           # 模型检查点
        └── {run_id}/
            ├── best.pt
            ├── last.pt
            └── epoch_*.pt
```

## 数据规范

### 输入目录结构

```
dataset_root/
├── idh.csv                    # 患者标签: columns=[patient, IDH, t1, t1ce, t2, t2flair]
├── conventional/
│   ├── mutant/{patient_id}/   # 包含 t1.nii.gz, t1ce.nii.gz, t2.nii.gz, t2flair.nii.gz, voi.nii.gz
│   └── wild_type/{patient_id}/
└── functional/
    ├── mutant/{patient_id}/   # 包含 adc.nii.gz, cbf.nii.gz, voi.nii.gz
    └── wild_type/{patient_id}/
```

### 当前数据划分（seed=42）

| 数据集 | 总数 | 野生型 | 突变型 |
|-------|------|--------|--------|
| Train | 97   | 64     | 33     |
| Val   | 13   | 7      | 6      |
| Test  | 28   | 18     | 10     |

## 核心模块

### 1. 数据加载 (`data_loader.py`)

```python
from srcs.data_loader import HabitatIDHBlockDataset

dataset = HabitatIDHBlockDataset(
    split_root="../dataset/splited_data/train",
    modalities=("t1", "t1ce", "t2", "t2flair", "adc", "cbf"),
    block_depth=5,              # 2.5D block 厚度（奇数）
    slice_axis=2,               # 切片轴: 0=X, 1=Y, 2=Z
    require_voi=True,           # 是否需要 VOI mask
    append_voi_mask=True,       # VOI 是否作为额外通道
    mask_background_with_voi=False,
    intensity_norm="zscore",    # zscore | minmax | none
    min_nonzero_voxels=16,
    transform=None,             # 可接 MONAI Compose / RandAffined 等 pipeline
)
# 返回: {"image": [C,H,W], "label": int, "patient_id": str, "slice_index": int}
```

### 2. 模型 (`models/resnet_18.py`)

```python
from models.resnet_18 import ResNet18Classifier

model = ResNet18Classifier(
    in_channels=35,    # 6模态*5层 + VOI*5层 = 35
    num_classes=2,
    pretrained=True,   # ImageNet 预训练权重
)
```

### 3. 训练脚本 (`baseline_ResNet18.py`)

关键参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--modalities` | t1,t1ce,t2,t2flair,adc,cbf | 输入模态列表 |
| `--block-depth` | 5 | 2.5D block 深度 |
| `--append-voi-mask` | true | VOI 作为额外通道 |
| `--mask-background-with-voi` | false | VOI 外区域置零 |
| `--use-monai-augmentation` | true | 训练集启用 MONAI 增强 |
| `--aug-rotate-deg` | 10.0 | 最大旋转角度（度） |
| `--aug-translate-px` | 8.0 | 最大平移像素 |
| `--aug-scale-range` | 0.1 | 最大缩放幅度 |
| `--aug-flip-prob` | 0.5 | 左右翻转概率 |
| `--pretrained` | true | ImageNet 预训练 |
| `--use-class-weights` | true | 类别不平衡加权 |
| `--early-stop-patience` | 5 | 早停耐心值 |
| `--early-stop-min-delta` | 0.001 | 早停最小提升阈值 |

## 典型用例

### 仅使用常规模态（无功能模态）
```bash
python srcs/baseline_ResNet18.py \
    --modalities t1,t1ce,t2,t2flair \
    --batch-size 16
```

### 不使用 VOI
```bash
python srcs/baseline_ResNet18.py \
    --require-voi false \
    --append-voi-mask false
```

### 自定义输入尺寸与数据增强
```bash
python srcs/baseline_ResNet18.py \
    --resize-height 224 \
    --resize-width 224 \
    --intensity-norm zscore
```

### 使用 MONAI 训练增强
```bash
python srcs/baseline_ResNet18.py \
    --use-monai-augmentation true \
    --aug-rotate-deg 10 \
    --aug-translate-px 8 \
    --aug-scale-range 0.1 \
    --aug-flip-prob 0.5 \
    --aug-intensity-scale-prob 0.3 \
    --aug-intensity-shift-prob 0.3
```

## 输出文件说明

| 文件 | 内容 |
|------|------|
| `patient_predictions_{split}.csv` | 患者级预测概率与标签 |
| `metrics_summary.csv` | AUC/ACC/SEN/SPE/F1 汇总 |
| `roc_raw_{split}.csv` | ROC 曲线原始数据点 |
| `confusion_matrix_{split}.csv` | 混淆矩阵 (TN/FP/FN/TP) |
| `training_history.csv` | 逐 epoch 训练指标 |
| `run_summary.json` | 完整实验配置与结果 |
| `training_log.txt` | 终端输出日志 |

## 依赖环境

- Python 3.10+
- PyTorch 2.2+
- monai, torchvision, nibabel, SimpleITK
- scikit-learn, numpy, pandas, tqdm

影像组学特征提取需额外创建 Python 3.10/3.11 环境（pyradiomics 与 Python 3.12 不兼容）。

## 设计原则

- **患者级别预测**: Block 级预测通过均值池化聚合为患者级概率
- **分层划分**: 保持突变型/野生型比例的数据划分
- **VOI 驱动**: 以功能性 VOI mask 定义感兴趣区域
- **模块化设计**: 数据加载、模型定义、训练流程分离，便于扩展

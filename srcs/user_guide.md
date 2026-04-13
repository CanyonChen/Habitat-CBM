# Habitat CBM 脚本用户指南

> 本指南详细说明 `habitat_CBM/repo/srcs/` 目录下所有脚本的功能、参数和使用方法。
> 适用于 IDH 突变状态预测的多模态 MRI 深度学习流程。

---

## 目录

1. [快速开始](#快速开始)
2. [脚本概览](#脚本概览)
3. [data_split.py - 数据划分](#data_splitpy---数据划分)
4. [data_loader.py - 数据加载](#data_loaderpy---数据加载)
5. [monai_augmentation.py - 数据增强](#monai_augmentationpy---数据增强)
6. [baseline_ResNet18.py - 基线训练](#baseline_resnet18py---基线训练)
7. [baseline_RadiomicsLR.py - 传统影像组学基线](#baseline_radiomicslrpy---传统影像组学基线)
8. [典型工作流程](#典型工作流程)
9. [数据流说明](#数据流说明)
10. [常见问题排查](#常见问题排查)

---

## 快速开始

### 环境要求

```bash
# 深度学习主环境
pip install -r habitat_CBM/repo/requirements.txt

# 传统影像组学专用环境（推荐单独创建 Python 3.10/3.11 环境）
pip install -r habitat_CBM/repo/requirements_pyradiomics.txt

# 主环境核心依赖
- torch >= 2.0
- monai >= 1.3
- nibabel >= 4.0
- numpy, scipy, scikit-learn
- matplotlib, tqdm, pyyaml

# Radiomics 额外依赖
- SimpleITK
- pyradiomics
- joblib
```

### 三步完成训练

```bash
# 1. 数据划分
python habitat_CBM/repo/srcs/data_split.py \
    --dataset-root /path/to/images \
    --output-root /path/to/splited_data \
    --train-ratio 0.7 --val-ratio 0.1 --test-ratio 0.2

# 2. 验证数据加载（可选）
python habitat_CBM/repo/srcs/data_loader.py \
    --split-root /path/to/splited_data/train \
    --block-depth 5

# 3. 训练 ResNet-18 基线模型
python habitat_CBM/repo/srcs/baseline_ResNet18.py \
    --split-base-root /path/to/splited_data \
    --output-root /path/to/results \
    --epochs 200 \
    --batch-size 32

# 4. 训练传统影像组学 + LASSO LR 基线模型
python habitat_CBM/repo/srcs/baseline_RadiomicsLR.py \
    --split-base-root /path/to/splited_data \
    --output-root /path/to/results/baseline_RadiomicsLR \
    --n-jobs 20 \
    --cv-folds 5
```

---

## 脚本概览

| 脚本 | 功能 | 主要用途 | 输入 | 输出 |
|------|------|----------|------|------|
| `data_split.py` | 患者级分层划分 | 将原始数据划分为训练/验证/测试集 | `images/` 目录 + `idh.csv` | `splited_data/` 目录 |
| `data_loader.py` | 2.5D 数据加载 | 读取多模态 MRI 并生成 2.5D blocks | `splited_data/` 目录 | PyTorch Dataset |
| `monai_augmentation.py` | 数据增强 | 提供训练时的数据增强变换 | Dataset samples | Augmented samples |
| `baseline_ResNet18.py` | 基线训练 | 训练 ResNet-18 进行 IDH 分类 | 划分后的数据 | 模型、预测、指标 |
| `baseline_RadiomicsLR.py` | 传统影像组学基线 | 提取 PyRadiomics 特征并训练 LASSO Logistic Regression | 划分后的 `conventional/` 数据 | 特征表、模型、预测、指标 |

---

## data_split.py - 数据划分

### 功能描述

将胶质瘤 IDH 数据集按患者级别进行分层划分，保持突变型(mutant)和野生型(wild_type)的类别比例。支持符号链接、复制或不创建文件夹三种模式。

### 预期输入目录结构

```
dataset_root/
├── idh.csv                    # 患者信息 CSV，必须包含 patient 和 IDH 列
├── conventional/              # 常规成像分支
│   ├── mutant/                # IDH 突变型患者文件夹
│   │   ├── 001/               # 患者文件夹
│   │   │   ├── t1.nii.gz
│   │   │   ├── t1ce.nii.gz
│   │   │   ├── t2.nii.gz
│   │   │   ├── t2flair.nii.gz
│   │   │   └── voi.nii.gz     # 可选的 VOI 掩模
│   │   └── ...
│   └── wild_type/             # IDH 野生型患者文件夹
└── functional/                # 功能性成像分支
    ├── mutant/
    │   ├── 001/
    │   │   ├── adc.nii.gz
    │   │   ├── cbf.nii.gz
    │   │   └── voi.nii.gz     # 优先使用 functional 下的 voi
    │   └── ...
    └── wild_type/
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset-root` | `Path` | **必填** | 原始数据根目录，包含 idh.csv 和分支目录 |
| `--output-root` | `Path` | **必填** | 划分结果输出目录 |
| `--train-ratio` | `float` | 0.7 | 训练集比例 |
| `--val-ratio` | `float` | 0.1 | 验证集比例 |
| `--test-ratio` | `float` | 0.2 | 测试集比例 |
| `--seed` | `int` | 42 | 随机种子，保证划分结果可复现 |
| `--csv-name` | `str` | "idh.csv" | 患者信息 CSV 文件名 |
| `--link-mode` | `str` | "copy" | 文件处理方式：`symlink`/`copy`/`none` |
| `--overwrite` | `flag` | False | 覆盖已存在的输出目录 |
| `--require-voi` | `bool` | True | 要求每个患者必须有 VOI 文件 |

### 输出目录结构

```
output_root/
├── manifests/                 # 划分清单文件
│   ├── split_assignments.csv  # 完整的患者划分分配表
│   ├── train_ids.txt          # 训练集患者 ID 列表
│   ├── val_ids.txt
│   ├── test_ids.txt
│   ├── train.csv              # 各子集的患者信息
│   ├── val.csv
│   └── test.csv
├── train/                     # 训练集数据
│   ├── conventional/
│   │   ├── mutant/
│   │   │   ├── 001/ -> /original/path/001
│   │   │   └── ...
│   │   └── wild_type/
│   └── functional/
├── val/                       # 验证集数据
└── test/                      # 测试集数据
```

### 运行示例

```bash
# 基本用法 - 默认 70/10/20 划分
python data_split.py \
    --dataset-root /path/to/images \
    --output-root /path/to/splited_data

# 自定义比例 80/10/10
python data_split.py \
    --dataset-root /path/to/images \
    --output-root /path/to/splited_data \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1

# 使用符号链接（节省磁盘空间）
python data_split.py \
    --dataset-root /path/to/images \
    --output-root /path/to/splited_data \
    --link-mode symlink

# 只生成分配清单，不创建文件夹
python data_split.py \
    --dataset-root /path/to/images \
    --output-root /path/to/splited_data \
    --link-mode none

# 指定不同随机种子
python data_split.py \
    --dataset-root /path/to/images \
    --output-root /path/to/splited_data \
    --seed 123

# 覆盖已存在的输出目录
python data_split.py \
    --dataset-root /path/to/images \
    --output-root /path/to/splited_data \
    --overwrite
```

### 注意事项

- **分层划分**: 自动保持 mutant/wild_type 比例在各个子集中一致
- **患者级别**: 同一患者不会出现在不同集合中
- **VOI 检查**: 默认要求每个患者必须有可识别的 functional/voi 文件
- **CSV 格式**: idh.csv 必须包含 `patient`（患者编号）和 `IDH`（0=wild_type, 1=mutant）列

---

## data_loader.py - 数据加载

### 功能描述

通用数据加载器，用于胶质瘤 IDH 多模态 MRI 的患者级扫描、2.5D 切片块构建。支持多模态输入、VOI 掩模处理、强度归一化等功能。

### 核心类: `HabitatIDHBlockDataset`

#### 主要参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `split_root` | `str/Path` | **必填** | 子集根目录，如 `splited_data/train` |
| `modalities` | `Sequence[str]` | 全部6模态 | 模态列表：`t1`, `t1ce`, `t2`, `t2flair`, `adc`, `cbf` |
| `require_voi` | `bool` | True | 是否要求每位患者都有 VOI 掩模 |
| `append_voi_mask` | `bool` | True | 将 VOI block 作为额外通道拼接 |
| `mask_background_with_voi` | `bool` | False | 用 VOI 将图像背景清零 |
| `block_depth` | `int` | 5 | 2.5D 块深度，必须是奇数 |
| `slice_axis` | `int` | 2 | 切片轴：0=X(冠状面), 1=Y(矢状面), 2=Z(横断面) |
| `intensity_norm` | `str` | "zscore" | 强度归一化：`zscore`/`minmax`/`none` |
| `min_nonzero_voxels` | `int` | 16 | 中心切片最少前景体素数 |
| `cache_volumes` | `bool` | True | 是否缓存体数据到内存 |
| `return_metadata` | `bool` | True | 是否返回患者元信息 |
| `transform` | `Callable` | None | 可选的数据增强变换 |

#### `__getitem__` 返回内容

```python
{
    "image": torch.FloatTensor,      # 形状 [C, H, W]
                                      # C = len(modalities) * block_depth
                                      # 若 append_voi_mask=True，则 + block_depth
    "label": torch.LongTensor,       # 0=wild_type, 1=mutant
    "patient_id": str,               # 如 "005"
    "slice_index": int,              # 中心切片索引
    "paths": Dict[str, str],         # 各模态文件路径
}
```

### 命令行参数（用于调试）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--split-root` | `Path` | **必填** | 要检查的数据子集目录 |
| `--block-depth` | `int` | 5 | 2.5D 切片块厚度 |
| `--require-voi` | `bool` | True | 是否要求 VOI |
| `--append-voi-mask` | `bool` | True | 是否拼接 VOI 通道 |
| `--mask-background-with-voi` | `bool` | False | 是否用 VOI 掩码背景 |
| `--slice-axis` | `int` | 2 | 切片轴 |
| `--intensity-norm` | `str` | "zscore" | 归一化方式 |
| `--min-nonzero-voxels` | `int` | 16 | 最小非零体素数 |
| `--cache-volumes` | `bool` | True | 是否缓存 |
| `--max-samples` | `int` | 3 | 打印多少个样本的调试信息 |

### 运行示例

```bash
# 检查 train 子集并打印统计信息
python data_loader.py \
    --split-root /path/to/splited_data/train \
    --require-voi true \
    --append-voi-mask true \
    --block-depth 5

# 检查 val 子集前 3 个样本的张量形状
python data_loader.py \
    --split-root /path/to/splited_data/val \
    --mask-background-with-voi true \
    --block-depth 5 \
    --max-samples 3

# 不缓存体数据（内存不足时）
python data_loader.py \
    --split-root /path/to/splited_data/test \
    --cache-volumes false
```

### 使用示例（Python API）

```python
from srcs.data_loader import HabitatIDHBlockDataset

# 创建数据集
dataset = HabitatIDHBlockDataset(
    split_root="/path/to/splited_data/train",
    modalities=["t1", "t1ce", "t2", "t2flair", "adc", "cbf"],
    require_voi=True,
    append_voi_mask=True,
    mask_background_with_voi=False,
    block_depth=5,
    slice_axis=2,
    intensity_norm="zscore",
    min_nonzero_voxels=16,
    cache_volumes=True,
)

# 获取样本
sample = dataset[0]
print(f"Image shape: {sample['image'].shape}")  # [35, H, W] (6*5 + 5)
print(f"Label: {sample['label']}")
print(f"Patient: {sample['patient_id']}")

# 查看统计信息
print(dataset.summary())
```

### 模态匹配规则

文件名匹配使用关键词（不区分大小写）：

| 模态 | 匹配关键词 |
|------|-----------|
| t1 | `_t1`, `t1.`, `t1_`, `t1wi`, `t1w` |
| t1ce | `t1ce`, `t1_ce`, `ce_t1`, `cet1`, `t1c`, `t1gd`, `t1_gd` |
| t2 | `_t2`, `t2.`, `t2_`, `t2wi`, `t2w` |
| t2flair | `t2flair`, `t2_flair`, `t2-flair`, `flair` |
| adc | `adc` |
| cbf | `cbf` |
| voi | `voi` |

> 注意：T1 和 T1CE、T2 和 T2FLAIR 有排除规则，避免误匹配。

---

## monai_augmentation.py - 数据增强

### 功能描述

基于 MONAI 的数据增强模块，为 2.5D 多模态 MRI blocks 提供训练时的数据增强。

### 核心类: `MonaiAugmentConfig`

#### 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | `bool` | True | 是否启用增强 |
| `affine_prob` | `float` | 0.5 | 应用仿射变换的概率 |
| `rotate_deg` | `float` | 10.0 | 最大旋转角度（度） |
| `translate_px` | `float` | 8.0 | 最大平移距离（像素） |
| `scale_range` | `float` | 0.1 | 最大缩放因子 |
| `flip_prob` | `float` | 0.5 | 左右翻转概率 |
| `intensity_scale_prob` | `float` | 0.3 | 强度缩放概率 |
| `intensity_scale` | `float` | 0.1 | 强度缩放因子 |
| `intensity_shift_prob` | `float` | 0.3 | 强度偏移概率 |
| `intensity_shift` | `float` | 0.1 | 强度偏移因子（标准差倍数） |

### 增强操作说明

| 操作 | 概率 | 参数 | 作用 |
|------|------|------|------|
| RandAffined | `affine_prob` | rotate/translate/scale | 空间仿射变换 |
| RandFlipd | `flip_prob` | spatial_axis=1 | 左右翻转 |
| RandScaleIntensityd | `intensity_scale_prob` | factors | 强度缩放 |
| RandStdShiftIntensityd | `intensity_shift_prob` | factors, nonzero=True | 强度偏移 |

### 使用示例

```python
from srcs.monai_augmentation import MonaiAugmentConfig, build_monai_block_transforms

# 创建配置
config = MonaiAugmentConfig(
    enabled=True,
    affine_prob=0.5,
    rotate_deg=15.0,
    translate_px=10.0,
    scale_range=0.1,
    flip_prob=0.5,
    intensity_scale_prob=0.3,
    intensity_scale=0.1,
    intensity_shift_prob=0.3,
    intensity_shift=0.1,
)

# 构建变换
transforms = build_monai_block_transforms(
    config=config,
    has_mask=True,  # 是否包含 mask（VOI）
)

train_transform = transforms["train"]  # 训练时使用增强
val_transform = transforms["val"]      # 验证时不使用增强
test_transform = transforms["test"]    # 测试时不使用增强
```

### 注意事项

- **空间一致性**: 仿射变换和翻转同时应用于 image 和 mask（如果有）
- **强度变换**: 仅应用于 image，不应用于 mask
- **插值模式**: image 使用 bilinear，mask 使用 nearest
- **验证/测试**: 始终不使用增强，只做类型转换

---

## baseline_ResNet18.py - 基线训练

### 功能描述

ResNet-18 基线训练脚本，用于患者级别的 IDH 突变状态预测。支持多模态 2.5D 输入、VOI 掩模处理、早停、类别权重等功能。

### 命令行参数

#### 路径相关

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--split-base-root` | `Path` | `data/splited_data` | 数据划分根目录 |
| `--output-root` | `Path` | `results/baseline_ResNet18` | 结果输出根目录 |
| `--checkpoint-root` | `Path` | `results/checkpoints` | checkpoint 保存目录 |
| `--run-id` | `str` | 时间戳 | 本次运行的标识符 |

#### 数据相关

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--modalities` | `str` | "t1,t1ce,t2,t2flair,adc,cbf" | 输入模态列表 |
| `--require-voi` | `bool` | True | 是否要求每位患者都有 VOI |
| `--append-voi-mask` | `bool` | True | 将 VOI block 作为额外通道 |
| `--mask-background-with-voi` | `bool` | False | 用 VOI 清零背景 |
| `--block-depth` | `int` | 5 | 2.5D block 深度 |
| `--slice-axis` | `int` | 2 | 切片轴 |
| `--intensity-norm` | `str` | "zscore" | 强度归一化方式 |
| `--min-nonzero-voxels` | `int` | 16 | 最小前景体素数 |
| `--cache-volumes` | `bool` | True | 是否缓存体数据 |

#### 训练相关

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--batch-size` | `int` | 32 | Batch size |
| `--epochs` | `int` | 200 | 最大训练轮数 |
| `--lr` | `float` | 5e-4 | 学习率 |
| `--weight-decay` | `float` | 1e-4 | 权重衰减 |
| `--num-workers` | `int` | 16 | DataLoader 工作进程数 |
| `--seed` | `int` | 42 | 随机种子 |
| `--pretrained` | `bool` | True | 使用 ImageNet 预训练权重 |
| `--resize-height` | `int` | 224 | 输入 resize 高度 |
| `--resize-width` | `int` | 224 | 输入 resize 宽度 |
| `--use-class-weights` | `bool` | True | 使用类别权重平衡 |
| `--early-stop-patience` | `int` | 5 | 早停耐心值 |
| `--early-stop-min-delta` | `float` | 0.001 | 判定改进的最小阈值 |
| `--save-train-predictions` | `bool` | True | 是否保存训练集预测 |
| `--device` | `str` | cuda/cpu | 计算设备 |
| `--threshold` | `float` | 0.5 | 患者级分类阈值 |
| `--log-to-file` | `bool` | True | 是否保存日志文件 |
| `--save-interval` | `int` | 50 | 定期保存 checkpoint 间隔 |

#### 数据增强相关

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use-monai-augmentation` | `bool` | True | 启用 MONAI 增强 |
| `--aug-affine-prob` | `float` | 0.5 | 仿射变换概率 |
| `--aug-rotate-deg` | `float` | 10.0 | 最大旋转角度 |
| `--aug-translate-px` | `float` | 8.0 | 最大平移像素 |
| `--aug-scale-range` | `float` | 0.1 | 最大缩放范围 |
| `--aug-flip-prob` | `float` | 0.5 | 翻转概率 |
| `--aug-intensity-scale-prob` | `float` | 0.3 | 强度缩放概率 |
| `--aug-intensity-scale` | `float` | 0.1 | 强度缩放因子 |
| `--aug-intensity-shift-prob` | `float` | 0.3 | 强度偏移概率 |
| `--aug-intensity-shift` | `float` | 0.1 | 强度偏移因子 |

### Checkpoint 保存策略

```
checkpoint_root/
└── {run_id}/
    ├── best.pt              # 验证集最优模型（覆盖更新）
    ├── last.pt              # 最新模型（每个 epoch 更新）
    ├── epoch_010.pt         # 周期性保存（可选）
    ├── epoch_020.pt
    └── ...
```

### 输出文件结构

```
output_root/
└── {run_id}/
    ├── configs/
    │   └── resnet18_base.yaml       # 运行配置
    ├── figures/
    │   ├── roc_curve_resnet18_test_{run_id}.png
    │   ├── confusion_matrix_resnet18_test_{run_id}.png
    │   ├── resnet18_roc.png         # 测试集 ROC 别名
    │   └── resnet18_cm.png          # 测试集混淆矩阵别名
    ├── patient_predictions_resnet18_train_{run_id}.csv
    ├── patient_predictions_resnet18_val_{run_id}.csv
    ├── patient_predictions_resnet18_test_{run_id}.csv
    ├── metrics_summary_resnet18_{run_id}.csv
    ├── metrics_resnet18_{run_id}.csv
    ├── roc_raw_resnet18_test_{run_id}.csv
    ├── confusion_matrix_resnet18_test_{run_id}.csv
    ├── training_history_resnet18_{run_id}.csv
    ├── train_log_resnet18_{run_id}.csv
    ├── train_log.csv
    ├── wrong_cases_resnet18_{run_id}.csv
    ├── run_summary_resnet18_{run_id}.json
    ├── run_config_resnet18_{run_id}.json
    └── training_log_resnet18_{run_id}.txt
```

### 运行示例

```bash
# 1. 使用默认参数运行
python baseline_ResNet18.py

# 2. 指定数据路径和 GPU
python baseline_ResNet18.py \
    --split-base-root /path/to/splited_data \
    --output-root /path/to/results \
    --checkpoint-root /path/to/checkpoints \
    --device cuda

# 3. 自定义训练参数
python baseline_ResNet18.py \
    --modalities t1,t1ce,t2,t2flair,adc,cbf \
    --batch-size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --run-id exp_custom

# 4. 不使用 VOI 训练
python baseline_ResNet18.py \
    --require-voi false \
    --append-voi-mask false \
    --mask-background-with-voi false

# 5. 使用 VOI 掩码背景并作为额外通道
python baseline_ResNet18.py \
    --require-voi true \
    --append-voi-mask true \
    --mask-background-with-voi true \
    --min-nonzero-voxels 32

# 6. 严格早停（需要至少 0.1% AUC 提升）
python baseline_ResNet18.py \
    --early-stop-patience 8 \
    --early-stop-min-delta 0.001

# 7. 保存所有 checkpoint 用于模型选择
python baseline_ResNet18.py \
    --save-interval 5 \
    --epochs 50

# 8. 最小化 checkpoint 节省空间
python baseline_ResNet18.py \
    --save-interval 0

# 9. 自定义数据增强
python baseline_ResNet18.py \
    --aug-rotate-deg 15.0 \
    --aug-translate-px 12.0 \
    --aug-flip-prob 0.5 \
    --aug-intensity-scale 0.15
```

### 评估指标说明

| 指标 | 说明 |
|------|------|
| AUC | ROC 曲线下面积 |
| ACC | 准确率 |
| SEN | 敏感性（召回率） |
| SPE | 特异性 |
| F1 | F1 分数 |
| Loss | 交叉熵损失 |

### 类别权重计算

当 `--use-class-weights=true` 时，权重计算公式：
```
weight = N / (n_classes × count)
```
其中：
- `N`: 总患者数
- `n_classes`: 类别数（2）
- `count`: 该类别患者数

> 注意：按 patient case 统计，而非按 block 统计。

---

## baseline_RadiomicsLR.py - 传统影像组学基线

### 功能描述

`baseline_RadiomicsLR.py` 用于构建患者级传统影像组学基线，完整流程为：

1. 从 `train / val / test` 的 `conventional/` 分支读取病例；
2. 仅使用 `T1 / T1CE / T2 / T2-FLAIR` 和 `conventional/voi`；
3. 用 PyRadiomics 提取 3D `Original` 图像的一阶统计、形状和纹理特征；
4. 在训练集内完成：
   - `median` 缺失值填补
   - `VarianceThreshold`
   - `StandardScaler`
   - `LASSO` 特征选择
   - `LASSO Logistic Regression` 分类
5. 导出患者级预测、ROC、混淆矩阵、错误病例、特征筛选表和运行摘要。

这个脚本的设计目标不是做“最复杂的传统模型”，而是做一个**规范、可复核、和主模型公平可比**的传统基线。

### 方法约束

- **只用四个常规序列**：`t1`、`t1ce`、`t2`、`t2flair`
- **只用全肿瘤 VOI**：固定使用 `conventional/` 分支下的 `voi`
- **特征类型固定**：`firstorder`、`shape`、`glcm`、`glrlm`、`glszm`、`gldm`、`ngtdm`
- **特征选择和分类器都用 LASSO**
- **超参数只在训练集内通过 k 折交叉验证确定**
- **验证集和测试集只做独立评估**

### 为什么 shape 特征只提取一次

`shape` 特征只依赖掩模本身，不依赖图像灰度。如果在四个模态上都提一次，会得到四份几乎完全相同的形状特征，既冗余，也会放大共线性问题。

因此脚本默认只在 `t1ce` 上提一次 shape 特征，并通过 `--shape-reference-modality` 显式记录这个选择。

### 为什么 GPU 不是主加速路径

- PyRadiomics 的特征提取主要依赖 `SimpleITK` 和 CPU；
- sklearn 的 `L1 LogisticRegressionCV` 也主要走 CPU；
- 因此这个脚本在 Linux 服务器上的主要加速方式是：
  - 提高 `--n-jobs`
  - 控制 `--itk-threads-per-worker`
  - 避免过度线程竞争

推荐在 25 vCPU 环境下优先尝试：

```bash
--n-jobs 16~20
--itk-threads-per-worker 1
```

### 核心参数

#### 路径相关

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--split-base-root` | `Path` | `data/splited_data` | 包含 `train/val/test` 的患者级划分目录 |
| `--output-root` | `Path` | `results/baseline_RadiomicsLR` | 结果输出根目录 |
| `--checkpoint-root` | `Path` | `results/baseline_RadiomicsLR/checkpoints` | 最终模型工件保存目录 |
| `--run-id` | `str` | 时间戳 | 本次运行标识 |

#### 输入与 radiomics 预处理

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--shape-reference-modality` | `str` | `t1ce` | shape 特征提取参考模态 |
| `--label-value` | `int` | 1 | VOI 掩模中的有效标签值 |
| `--resampled-spacing` | `tuple/none` | `1,1,1` | PyRadiomics 等体素重采样间距 |
| `--interpolator` | `str` | `sitkBSpline` | 影像重采样插值方式 |
| `--normalize` | `bool` | True | 是否启用 PyRadiomics 内部强度归一化 |
| `--normalize-scale` | `float` | 100 | normalizeScale |
| `--remove-outliers` | `float` | 3.0 | 离群值裁剪阈值 |
| `--bin-width` | `float` | 25.0 | 灰度离散化 binWidth |
| `--pad-distance` | `int` | 5 | padDistance |
| `--correct-mask-geometry` | `bool` | True | 掩模几何不一致时自动重采样到影像空间 |

#### LASSO 建模与交叉验证

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--cv-folds` | `int` | 5 | 训练集内分层 k 折数 |
| `--selector-c-grid` | `tuple` | 一组对数间隔值 | LASSO 特征选择的 C 搜索网格 |
| `--classifier-c-grid` | `tuple` | 一组对数间隔值 | LASSO 分类器的 C 搜索网格 |
| `--cv-scoring` | `str` | `roc_auc` | 交叉验证评分函数 |
| `--variance-threshold` | `float` | 0.0 | 方差过滤阈值 |
| `--max-iter` | `int` | 5000 | LASSO Logistic 最大迭代次数 |
| `--use-class-weights` | `bool` | True | 是否启用 `class_weight="balanced"` |
| `--threshold` | `float` | 0.5 | 患者级分类阈值 |
| `--feature-selection-fallback` | `bool` | True | 若最优 selector 全零，是否尝试更大 C 做稳定性兜底 |

#### 并行和输出

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--n-jobs` | `int` | 自动估计 | radiomics 提取和 CV 并行进程数 |
| `--itk-threads-per-worker` | `int` | 1 | 每个 worker 内部允许的 ITK 线程数 |
| `--save-train-predictions` | `bool` | True | 是否导出训练集患者级预测 |
| `--save-feature-table` | `bool` | True | 是否保存所有患者的原始 radiomics 特征表 |
| `--save-selected-feature-table` | `bool` | True | 是否保存最终选中特征子表 |
| `--log-to-file` | `bool` | True | 是否将终端输出写入日志文件 |
| `--seed` | `int` | 42 | 随机种子 |

### 输出目录结构

```text
output_root/
└── {run_id}/
    ├── configs/
    │   └── radiomics_lr_base.yaml
    ├── figures/
    │   ├── roc_curve_radiomics_lr_test_{run_id}.png
    │   ├── confusion_matrix_radiomics_lr_test_{run_id}.png
    │   ├── radiomics_lr_roc.png
    │   └── radiomics_lr_cm.png
    ├── radiomics_features_raw_radiomics_lr_{run_id}.csv
    ├── radiomics_features_selected_radiomics_lr_{run_id}.csv
    ├── patient_predictions_radiomics_lr_train_{run_id}.csv
    ├── patient_predictions_radiomics_lr_val_{run_id}.csv
    ├── patient_predictions_radiomics_lr_test_{run_id}.csv
    ├── patient_predictions_radiomics_lr.csv
    ├── roc_points_radiomics_lr_test_{run_id}.csv
    ├── roc_points_radiomics_lr.csv
    ├── confusion_matrix_radiomics_lr_test_{run_id}.csv
    ├── confusion_matrix_radiomics_lr.csv
    ├── metrics_summary_radiomics_lr_{run_id}.csv
    ├── metrics_radiomics_lr_{run_id}.csv
    ├── metrics_radiomics_lr.csv
    ├── selected_features_radiomics_lr_{run_id}.csv
    ├── selected_features_radiomics_lr.csv
    ├── wrong_cases_radiomics_lr_{run_id}.csv
    ├── wrong_cases_radiomics_lr.csv
    ├── scaler_stats_radiomics_lr.json
    ├── radiomics_lr_protocol.md
    ├── cv_results_radiomics_lr_{run_id}.csv
    ├── run_summary_radiomics_lr_{run_id}.json
    ├── run_config_radiomics_lr_{run_id}.json
    └── training_log_radiomics_lr_{run_id}.txt

checkpoint_root/
└── {run_id}/
    ├── best.joblib
    └── last.joblib
```

### 运行示例

```bash
# 1. 最常规的运行方式
python habitat_CBM/repo/srcs/baseline_RadiomicsLR.py \
    --split-base-root /path/to/splited_data \
    --output-root habitat_CBM/results/baseline_RadiomicsLR \
    --checkpoint-root habitat_CBM/results/baseline_RadiomicsLR/checkpoints \
    --cv-folds 5 \
    --n-jobs 20 \
    --seed 42 \
    --run-id baseline_radiomics_lr_seed42

# 2. 如果上游图像已经是等体素，关闭 radiomics 内部重采样
python habitat_CBM/repo/srcs/baseline_RadiomicsLR.py \
    --split-base-root /path/to/splited_data \
    --resampled-spacing none \
    --run-id baseline_radiomics_lr_no_resample

# 3. 收紧线程，避免服务器共享环境中的资源竞争
python habitat_CBM/repo/srcs/baseline_RadiomicsLR.py \
    --split-base-root /path/to/splited_data \
    --n-jobs 12 \
    --itk-threads-per-worker 1 \
    --run-id baseline_radiomics_lr_safe_threads
```

### 使用建议

- 这个基线的重点是**规范性**，不是复杂度。
- 不建议在第一版 baseline 中加入 wavelet、LoG、成百上千个派生特征。
- 不建议用测试集调 `C`、调阈值或调特征筛选规则。
- 如果 `selected_features_radiomics_lr.csv` 中保留特征非常少，不一定是坏事；LASSO 的作用就是在小样本高维场景中做强约束。

---

## 典型工作流程

### 完整实验流程

```bash
# Step 1: 数据划分
python habitat_CBM/repo/srcs/data_split.py \
    --dataset-root /path/to/images \
    --output-root habitat_CBM/data/splited_data \
    --train-ratio 0.7 \
    --val-ratio 0.1 \
    --test-ratio 0.2 \
    --seed 42 \
    --link-mode copy

# Step 2: 验证数据加载（调试）
for split in train val test; do
    python habitat_CBM/repo/srcs/data_loader.py \
        --split-root habitat_CBM/data/splited_data/$split \
        --block-depth 5 \
        --max-samples 2
done

# Step 3A: 训练 ResNet-18 深度学习基线
python habitat_CBM/repo/srcs/baseline_ResNet18.py \
    --split-base-root habitat_CBM/data/splited_data \
    --output-root habitat_CBM/results/baseline_ResNet18 \
    --checkpoint-root habitat_CBM/results/baseline_ResNet18/checkpoints \
    --modalities t1,t1ce,t2,t2flair,adc,cbf \
    --require-voi true \
    --append-voi-mask true \
    --block-depth 5 \
    --batch-size 32 \
    --epochs 200 \
    --lr 5e-4 \
    --seed 42 \
    --run-id baseline_seed42

# Step 3B: 训练传统影像组学 + LASSO LR 基线
python habitat_CBM/repo/srcs/baseline_RadiomicsLR.py \
    --split-base-root habitat_CBM/data/splited_data \
    --output-root habitat_CBM/results/baseline_RadiomicsLR \
    --checkpoint-root habitat_CBM/results/baseline_RadiomicsLR/checkpoints \
    --cv-folds 5 \
    --n-jobs 20 \
    --seed 42 \
    --run-id baseline_radiomics_lr_seed42

# Step 4: 查看结果
cat habitat_CBM/results/baseline_ResNet18/baseline_seed42/metrics_summary_resnet18_baseline_seed42.csv
cat habitat_CBM/results/baseline_RadiomicsLR/baseline_radiomics_lr_seed42/metrics_radiomics_lr.csv
```

### 多折交叉验证

```bash
SEEDS=(42 123 456 789 2024)

for SEED in "${SEEDS[@]}"; do
    echo "Running with seed $SEED..."
    
    # 数据划分
    python habitat_CBM/repo/srcs/data_split.py \
        --dataset-root /path/to/images \
        --output-root habitat_CBM/data/splits/seed_$SEED \
        --seed $SEED \
        --overwrite
    
    # 训练
    python habitat_CBM/repo/srcs/baseline_ResNet18.py \
        --split-base-root habitat_CBM/data/splits/seed_$SEED \
        --output-root habitat_CBM/results/cv_runs \
        --seed $SEED \
        --run-id cv_seed_$SEED
done

# 汇总所有运行的指标
cat habitat_CBM/results/cv_runs/metrics_resnet18_summary.csv
```

---

## 数据流说明

```
┌─────────────────────────────────────────────────────────────────┐
│                         原始数据                                 │
│  images/                                                         │
│  ├── idh.csv                                                     │
│  ├── conventional/                                               │
│  │   ├── mutant/001/, 002/, ...                                  │
│  │   └── wild_type/003/, 004/, ...                               │
│  └── functional/                                                 │
│      ├── mutant/001/, 002/, ...                                  │
│      └── wild_type/003/, 004/, ...                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ data_split.py
┌─────────────────────────────────────────────────────────────────┐
│                      划分后的数据                                │
│  splited_data/                                                   │
│  ├── manifests/                                                  │
│  │   ├── split_assignments.csv                                   │
│  │   ├── train_ids.txt, val_ids.txt, test_ids.txt               │
│  │   └── train.csv, val.csv, test.csv                           │
│  ├── train/                                                      │
│  ├── val/                                                        │
│  └── test/                                                       │
│      └── (conventional/functional 分支)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ data_loader.py
┌─────────────────────────────────────────────────────────────────┐
│                     PyTorch Dataset                              │
│  HabitatIDHBlockDataset                                          │
│  ├── 扫描目录 → PatientCase 索引                                │
│  ├── 构建 slice-level 样本索引                                  │
│  ├── __getitem__ 返回:                                          │
│  │   ├── image: [C, H, W] tensor                                │
│  │   ├── label: scalar tensor                                   │
│  │   ├── patient_id, slice_index                                │
│  │   └── paths: 文件路径字典                                    │
│  └── 可选: monai_augmentation.py 增强                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ baseline_ResNet18.py
┌─────────────────────────────────────────────────────────────────┐
│                      训练与评估                                  │
│  ResNet18Classifier                                              │
│  ├── DataLoader (train/val/test)                                │
│  ├── 训练循环 (AdamW, CrossEntropyLoss)                         │
│  ├── 早停 (Early Stopping)                                      │
│  ├── 验证集选择最佳模型                                         │
│  ├── Block-level → Patient-level 聚合                          │
│  │   └── 同一患者所有 block 概率取均值                          │
│  └── 导出结果:                                                   │
│      ├── checkpoints/best.pt, last.pt                           │
│      ├── patient_predictions_*.csv                              │
│      ├── metrics_summary_*.csv                                  │
│      ├── training_history_*.csv                                 │
│      ├── wrong_cases_*.csv                                      │
│      ├── ROC/Confusion Matrix PNG                               │
│      └── run_summary_*.json                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 常见问题排查

### 1. 数据划分问题

**Q: FileNotFoundError: Missing branch directory**
```
检查 images/ 目录下是否有 conventional/ 和 functional/ 子目录
```

**Q: ValueError: CSV missing required columns**
```
检查 idh.csv 是否包含 patient 和 IDH 列（大小写敏感）
```

**Q: Cannot find canonical VOI file**
```
检查 functional/<patient>/ 目录下是否有包含 "voi" 关键词的文件
或设置 --require-voi false
```

### 2. 数据加载问题

**Q: Shape mismatch within patient**
```
检查同一患者的不同模态文件尺寸是否一致
使用 nibabel 检查: nib.load(file).shape
```

**Q: Cannot find modality file for 't1ce'**
```
检查文件名是否包含 t1ce 关键词（见模态匹配规则）
避免 T1 和 T1CE 文件名混淆
```

**Q: No valid samples were constructed**
```
降低 --min-nonzero-voxels 值
检查 VOI 文件是否正确
检查 slice_axis 设置是否合理
```

### 3. 训练问题

**Q: CUDA out of memory**
```
减小 --batch-size
减小 --block-depth
设置 --cache-volumes false
```

**Q: Early stopping triggered immediately**
```
增大 --early-stop-patience
减小 --early-stop-min-delta
检查学习率是否过大/过小
```

**Q: Validation AUC is nan**
```
检查验证集是否同时包含正负样本
检查数据加载是否正确
```

### 4. 结果问题

**Q: 预测结果全为 0 或全为 1**
```
检查类别权重是否计算正确
检查数据归一化是否正常
尝试调整学习率
```

### 5. Radiomics 问题

**Q: PyRadiomics 报 mask / image geometry mismatch**
```
优先检查上游配准和导出是否正确
若只是 metadata 不一致，可保持 --correct-mask-geometry true
仍报错时，用 SimpleITK 检查 size / spacing / origin / direction
```

**Q: LASSO selector 选出 0 个特征**
```
先查看 radiomics_features_raw_*.csv 是否存在大量 NaN
检查 VOI 是否为空或过小
检查四个模态是否都真的完成了上游配准
必要时扩大 --selector-c-grid 的上界
也可以保留 --feature-selection-fallback true 作为稳定性兜底
```

**Q: 运行很慢**
```
Radiomics 提取本来就是 CPU 密集型
提高 --n-jobs，但建议保持 --itk-threads-per-worker 1
如果服务器多人共享，先从 --n-jobs 8~12 开始试
```

**Q: 为什么没有使用 GPU**
```
PyRadiomics 和 sklearn 的 L1 LogisticRegressionCV 都主要走 CPU
这个脚本的主要加速方式是 Linux 多进程并行，而不是 CUDA
```

---

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0 | 2024 | 初始版本，支持 ResNet-18 基线训练 |
| 1.1 | 2026-04-14 | 新增 `baseline_RadiomicsLR.py`，支持传统影像组学 + LASSO LR 基线 |

---

## 参考

- ResNet: "Deep Residual Learning for Image Recognition" (CVPR 2016)
- MONAI: https://monai.io/
- IDH 突变预测相关文献

---

> 本指南由 AI 助手根据源代码自动生成，如有疑问请参考源码中的详细注释。

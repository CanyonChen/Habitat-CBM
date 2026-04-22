# Habitat CBM 脚本用户指南

> 本指南详细说明 `habitat_CBM/repo/srcs/` 目录下所有脚本的功能、参数和使用方法。
> 适用于 IDH 突变状态预测的多模态 MRI 深度学习流程。

---

## 目录

1. [快速开始](#快速开始)
2. [脚本概览](#脚本概览)
3. [data_split.py - 数据划分](#data_splitpy---数据划分)
4. [data_loader.py - 数据加载](#data_loaderpy---数据加载)
5. [build_habitat.py - 多模态 Habitat 掩膜构建](#build_habitatpy---多模态-habitat-掩膜构建)
6. [monai_augmentation.py - 数据增强](#monai_augmentationpy---数据增强)
7. [baseline_ResNet18.py - 基线训练](#baseline_resnet18py---基线训练)
8. [baseline_RadiomicsLR.py - 传统影像组学基线](#baseline_radiomicslrpy---传统影像组学基线)
9. [train_habitat_CBM.py - Habitat-CBM 三阶段训练脚本](#train_habitat_cbmpy---habitat-cbm-三阶段训练脚本)
10. [eval_habitat_CBM.py - Habitat-CBM 推理与验证解耦工具](#eval_habitat_cbmpy---habitat-cbm-推理与验证解耦工具)
11. [intervene_habitat_cbm.py - 患者级概念干预工具](#intervene_habitat_cbmpy---患者级概念干预工具)
12. [典型工作流程](#典型工作流程)
13. [数据流说明](#数据流说明)
14. [常见问题排查](#常见问题排查)

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

### 五步完成 habitat 构建与基线训练

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

# 3. 构建 habitat masks（H1 / H2 / H3 / H1+2）
python habitat_CBM/repo/srcs/build_habitat.py \
    --dataset-root /path/to/images \
    --output-root habitat_CBM/dataset/habitat_masks \
    --device auto

# 4. 训练 ResNet-18 基线模型
python habitat_CBM/repo/srcs/baseline_ResNet18.py \
    --split-base-root /path/to/splited_data \
    --output-root /path/to/results \
    --epochs 200 \
    --batch-size 32

# 5. 训练传统影像组学 + LASSO LR 基线模型
python habitat_CBM/repo/srcs/baseline_RadiomicsLR.py \
    --split-base-root /path/to/splited_data \
    --output-root /path/to/results/baseline_RadiomicsLR \
    --n-jobs 20 \
    --cv-folds 5

# 6.（可选）Habitat-CBM 三阶段训练（训练结束会自动评估）
python habitat_CBM/repo/srcs/train_habitat_CBM.py \
    --config habitat_CBM/repo/srcs/args_train_habitat_CBM.json \
    --run-id run01 \
    --device cuda:0

# 7.（可选）Habitat-CBM 患者级概念干预
python habitat_CBM/repo/srcs/intervene_habitat_cbm.py \
    --checkpoint habitat_CBM/runs/03_habitat_cbm/run01/checkpoints/stage3_best.pt \
    --patient-predictions-csv habitat_CBM/results/03_habitat_cbm/run01/patient_predictions_habitat_cbm_run01.csv \
    --patient-concepts-csv habitat_CBM/results/03_habitat_cbm/run01/patient_concepts_habitat_cbm_run01.csv \
    --concept-scaler-json habitat_CBM/results/03_habitat_cbm/concept_scaler_stats.json \
    --output-dir habitat_CBM/results/05_intervention/run01 \
    --split test \
    --budgets 1,2,4,all
```

---

## 脚本概览

| 脚本 | 功能 | 主要用途 | 输入 | 输出 |
|------|------|----------|------|------|
| `data_split.py` | 患者级分层划分 | 将原始数据划分为训练/验证/测试集 | `images/` 目录 + `idh.csv` | `splited_data/` 目录 |
| `data_loader.py` | 2.5D 数据加载 | 读取多模态 MRI 并生成 2.5D blocks | `splited_data/` 目录 | PyTorch Dataset |
| `build_habitat.py` | Habitat 掩膜构建 | 用 `ADC + CBF + functional/voi` 生成 `H1/H2/H3/H1+2` | 预处理后的 `dataset/` | `dataset/habitat_masks/` + manifests |
| `monai_augmentation.py` | 数据增强 | 提供训练时的数据增强变换 | Dataset samples | Augmented samples |
| `baseline_ResNet18.py` | 基线训练 | 训练 ResNet-18 进行 IDH 分类 | 划分后的数据 | 模型、预测、指标 |
| `baseline_RadiomicsLR.py` | 传统影像组学基线 | 提取 PyRadiomics 特征并训练 LASSO Logistic Regression | 划分后的 `conventional/` 数据 | 特征表、模型、预测、指标 |
| `train_habitat_CBM.py` | Habitat-CBM 三阶段训练 | JSON 配置驱动训练 Stage1/2/3，并自动导出患者级评估 | split 数据 + concept labels/scaler | stage checkpoints、训练日志、患者级预测/概念表 |
| `eval_habitat_CBM.py` | Habitat-CBM 评估工具 | 独立重跑推理、患者级聚合、图表导出 | checkpoint + split 数据 + concept scaler | 预测/指标/概念误差/图表 |
| `intervene_habitat_cbm.py` | 患者级概念干预 | 离线修正预测概念并重新计算 `C->Y` | stage3 checkpoint + 患者级预测/概念 CSV | 干预前后指标、病例级纠正表 |

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

## build_habitat.py - 多模态 Habitat 掩膜构建

### 功能描述

`build_habitat.py` 用于把已经完成预处理的 `ADC / CBF / functional/voi` 转成论文同款的 habitat 掩膜。

脚本按**每位患者独立聚类**的方式完成：

1. 在 `functional/voi` 内提取 `ADC` 与 `CBF` 配对体素；
2. 分别做患者内 z-score 标准化；
3. 在二维空间执行 `K-means(k=3)`；
4. 将原始 cluster 映射为带生理语义的 `H1 / H2 / H3`；
5. 只持久化保存四个 mask：
   - `H1`
   - `H2`
   - `H3`
   - `H1+2`

其中 `H1+2` 是论文最终进入主线 radiomics 分析的最优 habitat。

> 注意：当前 `baseline_RadiomicsLR.py` 仍然默认读取 `conventional/voi` 做 whole-tumor radiomics。
> `build_habitat.py` 产物主要用于后续 habitat radiomics / Habitat-CBM 前置准备，不会自动被现有 baseline 脚本读取。

### 输入要求

脚本直接读取预处理后的 `dataset_root`，要求目录中至少存在：

```text
dataset_root/
├── conventional/
│   ├── mutant/<patient_id>/
│   └── wild_type/<patient_id>/
└── functional/
    ├── mutant/<patient_id>/
    │   ├── adc.nii.gz
    │   ├── cbf.nii.gz
    │   └── voi.nii.gz
    └── wild_type/<patient_id>/
```

强约束如下：

- habitat 聚类固定使用 `functional/<label>/<patient_id>/adc`
- habitat 聚类固定使用 `functional/<label>/<patient_id>/cbf`
- canonical VOI 固定使用 `functional/<label>/<patient_id>/voi`
- 三者必须 shape 一致，且 affine 一致

### 输出目录结构

默认输出目录为 `habitat_CBM/dataset/habitat_masks/`：

```text
habitat_CBM/dataset/habitat_masks/
├── mutant/
│   └── 005/
│       ├── h1.nii.gz
│       ├── h2.nii.gz
│       ├── h3.nii.gz
│       └── h12.nii.gz
├── wild_type/
│   └── 003/
│       ├── h1.nii.gz
│       ├── h2.nii.gz
│       ├── h3.nii.gz
│       └── h12.nii.gz
└── manifests/
    ├── habitat_centers.csv
    ├── habitat_volume_summary.csv
    ├── habitat_qc.csv
    ├── run_summary.json
    └── failed_cases.csv          # 仅在 --skip-errors true 且存在失败病例时生成
```

### GPU 加速说明

脚本优先使用 GPU，但加速重点在**体素级聚类**而不是整个 NIfTI I/O：

- CPU 负责：
  - 扫描目录
  - 读取/写出 NIfTI
  - 组织病例与 manifests
- GPU 负责（若 `torch + CUDA` 可用）：
  - VOI 内 `ADC/CBF` 张量标准化
  - K-means 距离计算、分配与中心更新

对你的服务器配置：

- `25 vCPU`
- `120 GB RAM`
- `Nvidia RTX Pro 6000 Blackwell 96 GB`

推荐优先用：

```bash
python habitat_CBM/repo/srcs/build_habitat.py \
    --dataset-root /path/to/images \
    --output-root habitat_CBM/dataset/habitat_masks \
    --device cuda:0 \
    --chunk-size 524288
```

如果单病例 VOI 很大、显存仍然充裕，可以进一步尝试把 `--chunk-size` 提到 `1048576`；默认值 `262144` 更稳健。

### 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset-root` | `Path` | `habitat_CBM/dataset` | 预处理后数据根目录 |
| `--output-root` | `Path` | `habitat_CBM/dataset/habitat_masks` | 四个 habitat mask 的输出根目录 |
| `--device` | `str` | `auto` | `auto/cpu/cuda/cuda:0` 等 |
| `--seed` | `int` | `42` | 随机种子 |
| `--n-clusters` | `int` | `3` | 论文固定为 3，不建议修改 |
| `--n-init` | `int` | `20` | K-means 初始化次数 |
| `--max-iter` | `int` | `100` | 最大迭代轮数 |
| `--tol` | `float` | `1e-4` | 收敛阈值 |
| `--chunk-size` | `int` | `262144` | torch/GPU 聚类分块大小 |
| `--eps` | `float` | `1e-8` | 数值稳定项 |
| `--overwrite-existing` | `bool` | `true` | 已存在患者输出目录时是否覆盖 |
| `--patient-ids` | `list[str]` | `None` | 仅处理指定患者，逗号分隔 |
| `--max-patients` | `int` | `0` | 仅处理前 N 位患者，调试用 |
| `--skip-errors` | `bool` | `false` | 单病例失败时是否跳过并继续 |
| `--allow-non-gzip-nii-gz` | `bool` | `true` | 当 `.nii.gz` 实际不是 gzip 时，回退按未压缩 NIfTI 读取 |
| `--save-run-config` | `bool` | `true` | 是否写入 `run_summary.json` |

### 运行示例

```bash
# 1. 默认自动选择设备（有 CUDA 就走 GPU）
python habitat_CBM/repo/srcs/build_habitat.py \
    --dataset-root /path/to/images \
    --output-root habitat_CBM/dataset/habitat_masks \
    --device auto

# 2. 指定 GPU，并放大 chunk 提高吞吐
python habitat_CBM/repo/srcs/build_habitat.py \
    --dataset-root /path/to/images \
    --device cuda:0 \
    --chunk-size 524288

# 3. 只调试 5 位患者
python habitat_CBM/repo/srcs/build_habitat.py \
    --dataset-root /path/to/images \
    --max-patients 5 \
    --device cpu

# 4. 仅处理指定病例，并在遇错时继续
python habitat_CBM/repo/srcs/build_habitat.py \
    --dataset-root /path/to/images \
    --patient-ids 003,005,013 \
    --skip-errors true

# 5. 严格检查数据（禁用 .nii.gz 非 gzip 容错）
python habitat_CBM/repo/srcs/build_habitat.py \
    --dataset-root /path/to/images \
    --allow-non-gzip-nii-gz false \
    --skip-errors true
```

### 质控建议

建议每次运行后至少检查三类 manifest：

- `habitat_centers.csv`
  - 看 H1 是否表现为低 ADC / 高 CBF
  - 看 H3 是否表现为高 ADC / 低 CBF
- `habitat_volume_summary.csv`
  - 看 `h1 + h2` 是否与 `h12` 体素数一致
- `habitat_qc.csv`
  - 看每位患者是否通过语义顺序检查和体积一致性检查

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

## train_habitat_CBM.py - Habitat-CBM 三阶段训练脚本

### 功能描述

`train_habitat_CBM.py` 是 Habitat-CBM 的主训练脚本。当前版本使用 `args_train_habitat_CBM.json` 作为主配置，完成 Stage1/Stage2/Stage3 三阶段训练、早停、分阶段 checkpoint 保存，并在训练结束后自动执行患者级评估导出。

它同时保留了一组训练工具函数，把以下逻辑从 `models/habitat_CBM.py` 中解耦出来：

1. 三阶段冻结策略（`stage1/stage2/stage3`）
2. 参数组构建（`encoder/concept_head/label_head`）
3. `MSE/MAE/Huber` 概念损失、`BCE/Focal BCE` 标签损失和 joint loss
4. stage 级别训练/验证循环、早停和日志导出

当前默认 Stage2 是患者级纯 oracle 校准：冻结 `encoder` 与 `concept_head`，只用真实标准化概念 `concept_true_std` 训练单层 `label_head`。默认关闭 label head Dropout 与 Stage2 概念噪声，并用 `val_label_loss` 选择 checkpoint；学习率调度器接收原始 `val_label_loss`，与 `ReduceLROnPlateau(mode="min")` 保持一致。只有当纯 oracle `C_true→Y` 链路稳定后，才建议把 `stages.stage2.concept_noise_mode` 改为 `residual` 做鲁棒性扫描。

当前默认 Stage3 是稳妥两步式。`stages.stage3.two_step_enabled=true` 时，训练顺序展开为 Stage1 -> Stage2 -> Stage3A -> Stage3B：Stage3A 冻结 `encoder/concept_head`，只用预测概念 `c_hat` 训练 `label_head`，默认 `lr_label_head=5e-4`，产物为 `stage3a_label_adapt_best.pt`；Stage3B 从 Stage3A 最优权重继续，冻结 `conv1/bn1/layer1/layer2/layer3`，只用极小学习率微调 `layer4/concept_head/label_head`，默认 `lr_encoder=5e-7`、`lr_concept_head=1e-6`、`lr_label_head=3e-6`。Stage3B 启用 concept guard，只有在 `val_auc` 改善且 `val_concept_loss <= Stage1 best val_concept_loss + concept_guard_max_delta` 时才保存最终 `stage3_best.pt`，默认 `concept_guard_max_delta=0.08`。

### 主要命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | `Path` | `args_train_habitat_CBM.json` | JSON 主配置文件 |
| `--run-id` | `str` | 配置或时间戳 | 当前训练 run 的标识 |
| `--output-root` | `Path` | 配置值 | 同时覆盖 runs/results 根目录 |
| `--checkpoint-root` | `Path` | 配置值 | checkpoint 输出目录 |
| `--device` | `str` | 配置值 | `cpu`、`cuda:0` 等 |
| `--epochs-stage1` | `int` | 配置值 | 覆盖 Stage1 最大 epoch |
| `--epochs-stage2` | `int` | 配置值 | 覆盖 Stage2 最大 epoch |
| `--epochs-stage3` | `int` | 配置值 | 覆盖 Stage3 最大 epoch |
| `--batch-size` | `int` | 配置值 | 覆盖全局 batch size |

### 核心工具接口

| 函数 | 作用 |
|------|------|
| `set_train_stage(model, stage)` | 设置阶段冻结策略 |
| `get_param_groups(model, lr_encoder, lr_concept_head, lr_label_head)` | 生成优化器参数组 |
| `compute_concept_loss(model, c_hat, c_true_std)` | 概念回归损失（MSE） |
| `compute_label_loss(y_logit, y_true)` | 标签二分类损失（BCEWithLogits） |
| `compute_joint_loss(...)` | 联合损失 |
| `stage1_train_step(...)` | Stage1（X->C）最小 step |
| `stage2_train_step(...)` | Stage2（C->Y）最小 step |
| `stage3_train_step(...)` | Stage3（X->C->Y）最小 step |

### 最小训练示例

```bash
# 使用默认 JSON 配置训练
python habitat_CBM/repo/srcs/train_habitat_CBM.py \
  --config habitat_CBM/repo/srcs/args_train_habitat_CBM.json \
  --run-id run01 \
  --device cuda:0

# 快速调试：减少三阶段 epoch
python habitat_CBM/repo/srcs/train_habitat_CBM.py \
  --config habitat_CBM/repo/srcs/args_train_habitat_CBM.json \
  --run-id debug01 \
  --device cuda:0 \
  --epochs-stage1 2 \
  --epochs-stage2 2 \
  --epochs-stage3 2
```

训练完成后重点产物：

```text
runs_root/<run_id>/
├── checkpoints/stage1_best.pt
├── checkpoints/stage2_best.pt
├── checkpoints/stage3a_label_adapt_best.pt
├── checkpoints/stage3_best.pt
├── stage1_concept_log.csv
├── stage2_label_head_log.csv
├── stage3a_label_adapt_log.csv
└── stage3_joint_log.csv

results_root/<run_id>/
├── patient_predictions_habitat_cbm_<run_id>.csv
├── patient_concepts_habitat_cbm_<run_id>.csv
├── metrics_habitat_cbm_<run_id>.csv
├── wrong_cases_habitat_cbm_<run_id>.csv
└── run_train_summary_habitat_cbm_<run_id>.json
```

---

## eval_habitat_CBM.py - Habitat-CBM 推理与验证解耦工具

### 功能描述

`eval_habitat_CBM.py` 承担推理/验证工具职责，和模型结构解耦。  
它提供三类能力：

1. 常规推理：`predict_batch`
2. 概念干预：`forward_with_intervention`
3. 患者级聚合：概率聚合与概念聚合

### 核心接口

| 函数 | 作用 |
|------|------|
| `predict_batch(model, x)` | 输出 `z/c_hat/y_logit/y_prob` |
| `prepare_intervention_order(...)` | 生成或校验概念替换顺序 |
| `forward_with_intervention(model, x, c_true_std, k, order=None)` | 执行 TTI 干预 |
| `aggregate_patient_probabilities(patient_ids, probs, topk=0)` | block->patient 概率聚合 |
| `aggregate_patient_concepts(patient_ids, concepts)` | block->patient 概念聚合 |

### 最小用法示例

```bash
# 独立重跑 test 集评估
python habitat_CBM/repo/srcs/eval_habitat_CBM.py \
  --config habitat_CBM/repo/srcs/args_train_habitat_CBM.json \
  --checkpoint habitat_CBM/runs/03_habitat_cbm/run01/checkpoints/stage3_best.pt \
  --run-id run01_eval \
  --output-dir habitat_CBM/results/03_habitat_cbm/run01_eval \
  --device cuda:0 \
  --splits test

# 同时评估 train/val/test，并使用固定阈值
python habitat_CBM/repo/srcs/eval_habitat_CBM.py \
  --config habitat_CBM/repo/srcs/args_train_habitat_CBM.json \
  --checkpoint habitat_CBM/runs/03_habitat_cbm/run01/checkpoints/stage3_best.pt \
  --run-id run01_eval_all \
  --output-dir habitat_CBM/results/03_habitat_cbm/run01_eval_all \
  --device cuda:0 \
  --threshold 0.5 \
  --splits train,val,test
```

### 干预策略说明

1. 当 `order=None` 时，默认按 `|c_pred - c_true_std|` 从大到小选概念。
2. `k` 为干预预算，`k > n_concepts` 会自动截断为 `n_concepts`。
3. `order` 若手动提供，必须是概念索引的合法排列。

---

## intervene_habitat_cbm.py - 患者级概念干预工具

### 功能描述

`intervene_habitat_cbm.py` 用于运行 Habitat-CBM 的患者级 TTI（test-time intervention）分析。

它模拟一个离线的“概念纠错”过程：

1. 读取训练后导出的患者级 IDH 预测结果；
2. 读取同一批患者的预测概念值和真实概念值；
3. 对每个患者，按标准化概念误差 `|c_pred_std - c_true_std|` 从大到小排序；
4. 在给定干预预算 `k` 下，把误差最大的前 `k` 个预测概念替换为真实概念；
5. 不再重跑图像编码器，只调用训练好的 `C->Y` 标签头重新计算 IDH 概率；
6. 对比干预前后 AUC、Accuracy、F1，以及误判病例纠正率。

这个脚本不是在线推理脚本，而是一个 post-hoc 患者级分析工具。它回答的问题是：如果测试时有专家能纠正模型最不准的概念，最终 IDH 预测是否会改善。

### 运行前置条件

必须先完成 `train_habitat_CBM.py` 的三阶段训练。训练结束后默认会自动导出以下文件：

```text
habitat_CBM/runs/03_habitat_cbm/<run_id>/checkpoints/
└── stage3_best.pt

habitat_CBM/results/03_habitat_cbm/<run_id>/
├── patient_predictions_habitat_cbm_<run_id>.csv
├── patient_concepts_habitat_cbm_<run_id>.csv
└── run_train_summary_habitat_cbm_<run_id>.json

habitat_CBM/results/03_habitat_cbm/
└── concept_scaler_stats.json
```

四个输入的作用如下：

| 输入 | 作用 |
|------|------|
| `stage3_best.pt` | 加载训练好的 `HabitatCBM`，主要使用其中的 `label_head` 执行 `C->Y` |
| `patient_predictions_*.csv` | 提供患者级 `prob_idh_mut`、`y_true`、split、run_id 等信息 |
| `patient_concepts_*.csv` | 提供患者级 `c*_pred_std`、`c*_true_std`、`c*_abs_error_std` |
| `concept_scaler_stats.json` | 校验概念维度；当 concept CSV 只有 raw 值时，用于 raw/std 转换 |

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint` | `Path` | **必填** | Stage3 最优 checkpoint，通常是 `stage3_best.pt` |
| `--patient-predictions-csv` | `Path` | **必填** | `eval_habitat_CBM.py` 或训练结束自动评估导出的患者级预测表 |
| `--patient-concepts-csv` | `Path` | **必填** | 同一次评估导出的患者级概念表 |
| `--concept-scaler-json` | `Path` | **必填** | 训练时使用的概念标准化统计文件 |
| `--output-dir` | `Path` | **必填** | 干预结果输出目录 |
| `--split` | `str` | `test` | 运行哪个 split：`train`、`val`、`test`、`all` |
| `--budgets` | `str` | `1,2,4,all` | 干预预算，逗号分隔；数字表示替换前 k 个概念，`all` 表示替换全部概念 |
| `--threshold` | `float` | `0.5` | 将概率转成 0/1 标签的阈值；建议与正式评估阈值一致 |
| `--low-confidence-margin` | `float` | `0.1` | 候选病例筛选边界：`abs(prob_before - threshold) < margin` 视为低置信度 |
| `--device` | `str` | `cpu` | 运行设备，如 `cpu`、`cuda:0` |
| `--in-channels` | `int` | `35` | checkpoint 缺少 `model_config` 时的 fallback 输入通道数 |
| `--n-concepts` | `int` | `8` | checkpoint 缺少 `model_config` 时的 fallback 概念数 |

`--in-channels` 和 `--n-concepts` 通常不用手动改。当前 `train_habitat_CBM.py` 保存的 checkpoint 已包含 `model_config`，脚本会优先使用 checkpoint 内的真实配置。

### 最常用运行方式

建议先设置 `RUN_ID`，再从训练 summary 中读取正式评估使用的阈值。训练脚本最终评估会用 val 集 Youden Index 选取阈值，并写入 `run_train_summary_*.json` 的 `threshold.eval_threshold_used`。

```bash
RUN_ID=run01

THRESHOLD=$(RUN_ID="${RUN_ID}" python - <<'PY'
import json
import os
from pathlib import Path

run_id = os.environ["RUN_ID"]
summary_path = Path(f"habitat_CBM/results/03_habitat_cbm/{run_id}/run_train_summary_habitat_cbm_{run_id}.json")
payload = json.loads(summary_path.read_text(encoding="utf-8"))
print(payload["threshold"]["eval_threshold_used"])
PY
)

python habitat_CBM/repo/srcs/intervene_habitat_cbm.py \
  --checkpoint habitat_CBM/runs/03_habitat_cbm/${RUN_ID}/checkpoints/stage3_best.pt \
  --patient-predictions-csv habitat_CBM/results/03_habitat_cbm/${RUN_ID}/patient_predictions_habitat_cbm_${RUN_ID}.csv \
  --patient-concepts-csv habitat_CBM/results/03_habitat_cbm/${RUN_ID}/patient_concepts_habitat_cbm_${RUN_ID}.csv \
  --concept-scaler-json habitat_CBM/results/03_habitat_cbm/concept_scaler_stats.json \
  --output-dir habitat_CBM/results/05_intervention/${RUN_ID} \
  --split test \
  --budgets 1,2,4,all \
  --threshold "${THRESHOLD}" \
  --low-confidence-margin 0.1 \
  --device cuda:0
```

如果不想读取动态阈值，也可以直接使用固定阈值：

```bash
python habitat_CBM/repo/srcs/intervene_habitat_cbm.py \
  --checkpoint habitat_CBM/runs/03_habitat_cbm/run01/checkpoints/stage3_best.pt \
  --patient-predictions-csv habitat_CBM/results/03_habitat_cbm/run01/patient_predictions_habitat_cbm_run01.csv \
  --patient-concepts-csv habitat_CBM/results/03_habitat_cbm/run01/patient_concepts_habitat_cbm_run01.csv \
  --concept-scaler-json habitat_CBM/results/03_habitat_cbm/concept_scaler_stats.json \
  --output-dir habitat_CBM/results/05_intervention/run01_thr050 \
  --split test \
  --budgets 1,2,4,all \
  --threshold 0.5
```

### 其他常用场景

```bash
# 1. 对 val 集做干预，用于方法调试
python habitat_CBM/repo/srcs/intervene_habitat_cbm.py \
  --checkpoint habitat_CBM/runs/03_habitat_cbm/run01/checkpoints/stage3_best.pt \
  --patient-predictions-csv habitat_CBM/results/03_habitat_cbm/run01/patient_predictions_habitat_cbm_run01.csv \
  --patient-concepts-csv habitat_CBM/results/03_habitat_cbm/run01/patient_concepts_habitat_cbm_run01.csv \
  --concept-scaler-json habitat_CBM/results/03_habitat_cbm/concept_scaler_stats.json \
  --output-dir habitat_CBM/results/05_intervention/run01_val \
  --split val \
  --budgets 1,2,4,all

# 2. 更细粒度的 budget 曲线
python habitat_CBM/repo/srcs/intervene_habitat_cbm.py \
  --checkpoint habitat_CBM/runs/03_habitat_cbm/run01/checkpoints/stage3_best.pt \
  --patient-predictions-csv habitat_CBM/results/03_habitat_cbm/run01/patient_predictions_habitat_cbm_run01.csv \
  --patient-concepts-csv habitat_CBM/results/03_habitat_cbm/run01/patient_concepts_habitat_cbm_run01.csv \
  --concept-scaler-json habitat_CBM/results/03_habitat_cbm/concept_scaler_stats.json \
  --output-dir habitat_CBM/results/05_intervention/run01_budget_curve \
  --split test \
  --budgets 1,2,3,4,5,6,7,8,all

# 3. CPU 环境运行
python habitat_CBM/repo/srcs/intervene_habitat_cbm.py \
  --checkpoint habitat_CBM/runs/03_habitat_cbm/run01/checkpoints/stage3_best.pt \
  --patient-predictions-csv habitat_CBM/results/03_habitat_cbm/run01/patient_predictions_habitat_cbm_run01.csv \
  --patient-concepts-csv habitat_CBM/results/03_habitat_cbm/run01/patient_concepts_habitat_cbm_run01.csv \
  --concept-scaler-json habitat_CBM/results/03_habitat_cbm/concept_scaler_stats.json \
  --output-dir habitat_CBM/results/05_intervention/run01_cpu \
  --split test \
  --device cpu
```

### 输出文件说明

运行完成后，`--output-dir` 下会生成：

```text
output_dir/
├── intervention_case_list.csv
├── intervention_per_case.csv
├── intervention_metrics_by_budget.csv
├── intervention_correction_summary.csv
└── intervention_summary.csv
```

各文件含义如下：

| 文件 | 粒度 | 内容 |
|------|------|------|
| `intervention_case_list.csv` | 患者 | 候选病例列表；候选 = 干预前误判或低置信度 |
| `intervention_per_case.csv` | 患者 x budget | 每个患者在每个 budget 下替换了哪些概念、干预前后概率/标签、是否纠正 |
| `intervention_metrics_by_budget.csv` | budget | 按 `all` 和 `candidate` 两种口径统计 AUC、Accuracy、F1 的干预前后变化 |
| `intervention_correction_summary.csv` | budget | 在候选误判病例中，有多少被干预纠正，以及纠正率 |
| `intervention_summary.csv` | run | 本次干预运行的 split、budget、阈值、输入文件路径等元信息 |

重点看两个文件：

1. `intervention_metrics_by_budget.csv`
   - `scope=all`：所有患者口径，适合汇报整体 TTI 性能提升；
   - `scope=candidate`：误判或低置信度病例口径，适合分析需要人工复核的病例；
   - `delta_auc/delta_acc/delta_f1` 大于 0 说明干预后对应指标提升。
2. `intervention_correction_summary.csv`
   - `correction_rate` 表示干预前误判病例中被纠正的比例；
   - 随 `budget_k` 增加，若纠正率明显上升，说明最终预测确实受概念瓶颈控制。

### 干预逻辑细节

对每个患者，脚本执行：

```python
order = argsort(abs(c_pred_std - c_true_std), descending=True)
replace_idx = order[:k]
c_after = c_pred_std.copy()
c_after[replace_idx] = c_true_std[replace_idx]
y_prob_after = sigmoid(model.forward_c_to_y(c_after))
```

因此当前实现是“误差优先”的 oracle TTI：

- 使用真实概念值来判断哪些概念预测误差最大；
- 替换的是标准化概念 `*_std`，和训练时 `label_head` 输入保持一致；
- 不重新计算图像特征，也不重新运行 `X->C`；
- 当前不支持 random order 或固定全局 order；如果需要复现 OAI 风格曲线，需要另外扩展排序策略。

### 结果解读建议

1. 如果 `all` 口径随 `budget_k` 增大持续提升，说明概念瓶颈对最终 IDH 分类有实际控制作用。
2. 如果 `candidate` 口径提升明显，但 `all` 口径提升不大，说明干预主要帮助边界样本或误判样本。
3. 如果概念误差被修正后指标仍无提升，可能说明：
   - `C->Y` 标签头没有充分利用这些概念；
   - 被修正的概念不是当前患者 IDH 判别的关键概念；
   - 概念本身与 IDH 标签的可解释关系较弱；
   - 使用的 `--threshold` 与正式评估阈值不一致，导致标签变化口径不一致。
4. 论文中建议把该实验称为 `patient-level oracle concept correction` 或 `error-prioritized TTI`，避免表述成真实部署时自动知道哪个概念错误。

### 常见错误

**Q: Missing concept rows for patients**
```
说明 patient_predictions CSV 和 patient_concepts CSV 不是同一次评估导出的，
或者 --split 选择后两个文件中的患者集合不一致。
```

**Q: Concept table must include either std columns or raw columns**
```
检查 patient_concepts CSV 是否包含 c0_true_std/c0_pred_std 等字段。
正常情况下 train_habitat_CBM.py 自动评估导出的 patient_concepts 文件已经包含这些列。
```

**Q: checkpoint 加载 shape mismatch**
```
通常是 checkpoint 与 patient_concepts 的概念数不一致。
确认 --checkpoint、--patient-concepts-csv、--concept-scaler-json 来自同一次训练配置。
```

**Q: 干预前 pred_before 和评估阶段 pred_label 不一致**
```
intervene_habitat_cbm.py 会用 --threshold 重新从 prob_idh_mut 计算 pred_before。
如果训练后正式评估使用的是 Youden 阈值，请把 run_train_summary_*.json 中的
threshold.eval_threshold_used 传给 --threshold。
```

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

# Step 3: 生成 habitat masks（为 habitat radiomics / Habitat-CBM 做准备）
python habitat_CBM/repo/srcs/build_habitat.py \
    --dataset-root /path/to/images \
    --output-root habitat_CBM/dataset/habitat_masks \
    --device cuda:0 \
    --chunk-size 524288

# Step 4A: 训练 ResNet-18 深度学习基线
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

# Step 4B: 训练传统影像组学 + LASSO LR 基线
python habitat_CBM/repo/srcs/baseline_RadiomicsLR.py \
    --split-base-root habitat_CBM/data/splited_data \
    --output-root habitat_CBM/results/baseline_RadiomicsLR \
    --checkpoint-root habitat_CBM/results/baseline_RadiomicsLR/checkpoints \
    --cv-folds 5 \
    --n-jobs 20 \
    --seed 42 \
    --run-id baseline_radiomics_lr_seed42

# Step 5: 查看结果
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
                 ┌────────────┴────────────┐
                 │                         │
                 ▼ build_habitat.py        ▼ data_split.py
┌─────────────────────────────────────────────────────────────────┐
│                    Habitat 掩膜输出                              │
│  habitat_masks/                                                  │
│  ├── mutant/<patient_id>/h1.nii.gz, h2.nii.gz, h3.nii.gz, h12.nii.gz │
│  ├── wild_type/<patient_id>/...                                 │
│  └── manifests/                                                 │
└─────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
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

### 4. Habitat 构建问题

**Q: build_habitat.py 提示 Missing runtime dependencies**
```
先安装:
pip install -r habitat_CBM/repo/requirements_pyradiomics.txt

若希望走 GPU，还需要按服务器 CUDA 版本安装匹配的 PyTorch 官方 wheel
```

**Q: build_habitat.py 没有用上 GPU**
```
先确认 python -c "import torch; print(torch.cuda.is_available())" 返回 True
再显式指定 --device cuda:0
若仍走 CPU，通常是当前环境安装的是 CPU-only PyTorch
```

**Q: ImageFileError: xxx.nii.gz is not a gzip file**
```
原因：
文件后缀是 .nii.gz，但文件内容不是 gzip（常见于 .nii 被误命名为 .nii.gz）。

当前脚本默认已开启容错：
--allow-non-gzip-nii-gz true
会自动回退为“按未压缩 NIfTI 读取”，并给出 RuntimeWarning。

建议：
1) 先让流程跑通（推荐）：
   python habitat_CBM/repo/srcs/build_habitat.py --dataset-root /path/to/images
2) 再做严格排查：
   python habitat_CBM/repo/srcs/build_habitat.py \
       --dataset-root /path/to/images \
       --allow-non-gzip-nii-gz false \
       --skip-errors true
   然后查看 habitat_masks/manifests/failed_cases.csv 中的失败病例。

长期修复（推荐在数据层面）：
- 把误命名文件改成 .nii，或重新 gzip 压缩为真正的 .nii.gz。
- 可用 gzip -t <file.nii.gz> 做快速校验（返回非 0 代表不是合法 gzip）。
```

**Q: habitat_qc.csv 里语义顺序检查失败**
```
先核对 ADC / CBF / VOI 是否真的在同一空间
再检查 VOI 是否过小、是否包含大量坏值
少量病例存在边界情况时，可先人工复核 overlay 和 cluster centers
```

**Q: 单病例运行很慢**
```
优先使用 --device cuda:0
在 96GB 显存环境下可尝试把 --chunk-size 提高到 524288 或 1048576
若病例很多，可配合 --patient-ids 或 --max-patients 先做小批量调试
```

### 5. 结果问题

**Q: 预测结果全为 0 或全为 1**
```
检查类别权重是否计算正确
检查数据归一化是否正常
尝试调整学习率
```

### 6. Radiomics 问题

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
| 1.2 | 2026-04-21 | 新增 `intervene_habitat_cbm.py` 患者级概念干预详细说明，并更新 Habitat-CBM 训练/评估命令 |

---

## 参考

- ResNet: "Deep Residual Learning for Image Recognition" (CVPR 2016)
- MONAI: https://monai.io/
- IDH 突变预测相关文献

---

> 本指南由 AI 助手根据源代码自动生成，如有疑问请参考源码中的详细注释。

---

## 2026-04-18 更新：Habitat-CBM 完整流程（JSON 配置驱动）

本次更新后，Habitat-CBM 不再仅是工具函数自检，已支持完整训练-评估-干预闭环。

### 新增脚本

1. `build_habitat_cbm_labels.py`：构建 `concept_labels.csv`、`concept_statistics.csv`、`concept_scaler_stats.json`
2. `eval_habitat_cbm_concepts.py`：概念层指标评估（`MAE/RMSE/R2/Pearson`）
3. `intervene_habitat_cbm.py`：患者级概念干预（`k=1,2,4,all`）

### 训练脚本行为更新

`train_habitat_CBM.py` 现已支持：

1. `--config`（默认读取 `args_train_habitat_CBM.json`）
2. CLI 覆盖：`--run-id`、`--output-root`、`--checkpoint-root`、`--device`、`--epochs-stage1/2/3`、`--batch-size`
3. 三阶段训练 + 早停 + 分阶段 best checkpoint
4. 训练结束自动患者级评估导出

### 数据加载器更新

`data_loader_habitat_CBM.py` 现已在 `HabitatIDHBlockDataset` 上直接扩展 concept 模式。

新增参数：

1. `--concept-label-csv`
2. `--concept-scaler-json`

开启后输出新增字段：

1. `concept_true_raw`（8 维）
2. `concept_true_std`（8 维）

### 一次跑通命令顺序

```bash
# 1) 构建 concept 资产
python habitat_CBM/repo/srcs/build_habitat_cbm_labels.py \
  --concept-proxy-csv habitat_CBM/results/02_habitat/concept_proxy_features_run01.csv \
  --output-label-csv habitat_CBM/results/02_habitat/concept_labels.csv \
  --output-stats-csv habitat_CBM/results/02_habitat/concept_statistics.csv \
  --output-scaler-json habitat_CBM/results/03_habitat_cbm/concept_scaler_stats.json

# 2) 三阶段训练（自动评估）
python habitat_CBM/repo/srcs/train_habitat_CBM.py \
  --config habitat_CBM/repo/srcs/args_train_habitat_CBM.json \
  --run-id run01 \
  --device cuda:0

# 3) 概念层评估
python habitat_CBM/repo/srcs/eval_habitat_cbm_concepts.py \
  --patient-concepts-csv habitat_CBM/results/03_habitat_cbm/run01/patient_concepts_habitat_cbm_run01.csv \
  --output-dir habitat_CBM/results/04_concept_eval/run01 \
  --split test \
  --scale raw

# 4) 概念干预
python habitat_CBM/repo/srcs/intervene_habitat_cbm.py \
  --checkpoint habitat_CBM/runs/03_habitat_cbm/run01/checkpoints/stage3_best.pt \
  --patient-predictions-csv habitat_CBM/results/03_habitat_cbm/run01/patient_predictions_habitat_cbm_run01.csv \
  --patient-concepts-csv habitat_CBM/results/03_habitat_cbm/run01/patient_concepts_habitat_cbm_run01.csv \
  --concept-scaler-json habitat_CBM/results/03_habitat_cbm/concept_scaler_stats.json \
  --output-dir habitat_CBM/results/05_intervention/run01 \
  --split test \
  --budgets 1,2,4,all
```

### 关键输出对照

1. 主任务：`patient_predictions_*`、`metrics_*`、`roc_points_*`、`confusion_matrix_*`、`wrong_cases_*`
2. 概念层：`patient_concepts_*`、`concept_metrics_*`、`concept_error_*`
3. 干预层：`intervention_case_list.csv`、`intervention_per_case.csv`、`intervention_metrics_by_budget.csv`、`intervention_correction_summary.csv`

### 环境说明

若当前环境缺少 `torch`，可先执行语法检查：

```bash
python -m py_compile habitat_CBM/repo/srcs/train_habitat_CBM.py
```

真实训练/推理/干预需在具备 `torch`、`monai`、`nibabel`、`scikit-learn` 的环境运行。

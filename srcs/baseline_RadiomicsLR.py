#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传统影像组学 + LASSO Logistic Regression 基线脚本。

本脚本面向胶质瘤 IDH 患者级二分类任务，完成以下端到端流程：

1. 从 `train / val / test` 三个患者级划分目录中读取病例；
2. 仅使用 `conventional/` 分支下的 `T1 / T1CE / T2 / T2-FLAIR` 以及全肿瘤 `VOI`；
3. 基于 PyRadiomics 提取 3D 原始影像的一阶统计、形状和纹理特征；
4. 单变量预筛选（SelectKBest + f_classif）将特征降维至合理规模，避免高维稀疏问题；
5. 使用训练集内的 LASSO (`L1` Logistic Regression CV) 同时完成特征选择与分类器训练；
6. 在训练集内通过 CV + Youden 指数自动选取最优分类阈值；
7. 在 train / val / test 上统一导出患者级预测、ROC、混淆矩阵、错误病例和汇总指标；
8. 保存完整的特征工程协议、标准化参数、筛选结果、模型工件和运行摘要。

===============================================================================
一、数据假设与范围
===============================================================================
1. 输入目录应为 `data_split.py` 生成后的划分目录，即：

   split_base_root/
   ├── train/
   │   └── conventional/
   │       ├── mutant/
   │       └── wild_type/
   ├── val/
   └── test/

2. 每位患者仅使用 `conventional/` 目录下的：
   - `t1`
   - `t1ce`
   - `t2`
   - `t2flair`
   - `voi`

3. 本脚本默认假设输入影像已经完成了上游预处理，例如：
   - 患者级入组过滤
   - 多模态配准
   - 统一空间
   - 基础重采样 / 标准化

   在此基础上，脚本还会执行 radiomics 侧常见处理：
   - 掩膜与影像几何一致性检查
   - 必要时将掩膜重采样到影像空间（最近邻）
   - 可选的 radiomics 侧等体素重采样
   - PyRadiomics 内部强度归一化和离群值裁剪

===============================================================================
二、方法设计（改进版）
===============================================================================
1. 特征提取：
   - 图像类型：仅 `Original`
   - 特征类别：`firstorder`、`shape`、`glcm`、`glrlm`、`glszm`、`gldm`、`ngtdm`
   - 形状特征仅从一个参考模态提取一次，避免四个模态重复写入完全相同的 shape 特征

2. 特征工程 Pipeline（含改进）：
   - 缺失值填补：训练集拟合 `median` imputer
   - 方差过滤：训练集拟合 `VarianceThreshold`（去常数特征）
   - 标准化：训练集拟合 `StandardScaler`
   - 【新增】单变量预筛选：训练集内 `SelectKBest(f_classif, k=univariate_k)`
     * 在 LASSO 之前将特征降至 k（默认 50），避免 n_samples << n_features 导致正则化失效
   - LASSO 特征选择 + 分类：训练集内单步 `L1 LogisticRegressionCV`
     * 【改进】C 搜索网格上限从 1000 降至 1.0，集中在中强度正则化区间
     * 两阶段 selector → classifier 合并为单步，消除双重拟合的冗余偏差

3. 阈值优化（新增）：
   - 在训练集内 CV 预测概率上通过 Youden 指数（SEN + SPE - 1 最大化）自动选取最优阈值
   - 将优化阈值同时应用于 val / test 集评估
   - 同时保存固定阈值 0.5 下的对比指标

4. 分类模型：
   - 最终分类器：训练集内 `L1 LogisticRegressionCV`（LASSO）
   - 类别不平衡：默认 `class_weight="balanced"`
   - 超参数选择：仅在训练集内通过 `k` 折交叉验证确定
   - 验证集只用于独立报告，不参与参数搜索

5. 计算加速：
   - PyRadiomics 提取默认走 CPU 并行，适合 Linux 多核环境
   - 每个 worker 默认只给 SimpleITK 分配 1 个线程，避免线程过度争抢
   - sklearn 的交叉验证支持多进程并行

===============================================================================
三、典型运行示例
===============================================================================
1. 使用默认参数运行：

   python habitat_CBM/repo/srcs/baseline_RadiomicsLR.py \
       --split-base-root /path/to/splited_data \
       --output-root /path/to/results/baseline_RadiomicsLR

2. 指定并行度与运行 ID：

   python habitat_CBM/repo/srcs/baseline_RadiomicsLR.py \
       --split-base-root /path/to/splited_data \
       --output-root /path/to/results/baseline_RadiomicsLR \
       --n-jobs 20 \
       --cv-folds 5 \
       --run-id baseline_radiomics_lr_seed42

3. 若上游影像已经是等体素，可关闭 radiomics 内部重采样：

   python habitat_CBM/repo/srcs/baseline_RadiomicsLR.py \
       --split-base-root /path/to/splited_data \
       --resampled-spacing none

4. 关闭单变量预筛选（保留全部特征送 LASSO）：

   python habitat_CBM/repo/srcs/baseline_RadiomicsLR.py \
       --split-base-root /path/to/splited_data \
       --univariate-k none

5. 使用固定阈值 0.5 而非 Youden 自动阈值：

   python habitat_CBM/repo/srcs/baseline_RadiomicsLR.py \
       --split-base-root /path/to/splited_data \
       --optimize-threshold false
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, TextIO, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml
from joblib import Parallel, delayed
from radiomics import featureextractor
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_NAME = "radiomics_lr"
DEFAULT_SEED = 42
DEFAULT_MODALITIES = ("t1", "t1ce", "t2", "t2flair")
DEFAULT_FEATURE_CLASSES = ("firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm")
DEFAULT_LABEL_TO_ID = {"wild_type": 0, "mutant": 1}
DEFAULT_SPLITS = ("train", "val", "test")
DEFAULT_ALLOWED_EXTENSIONS = (".nii", ".nii.gz", ".mha", ".mhd", ".nrrd")
DEFAULT_VOI_KEYWORDS = ("voi",)
DEFAULT_C_GRID = (
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    3e-1,
    1.0,
)

MODALITY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "t1": ("_t1", "t1.", "t1_", "t1wi", "t1w"),
    "t1ce": ("t1ce", "t1_ce", "ce_t1", "cet1", "t1c", "t1gd", "t1_gd"),
    "t2": ("_t2", "t2.", "t2_", "t2wi", "t2w"),
    "t2flair": ("t2flair", "t2_flair", "t2-flair", "flair"),
}
T1_EXCLUDE_KEYWORDS = MODALITY_KEYWORDS["t1ce"]
T2_EXCLUDE_KEYWORDS = MODALITY_KEYWORDS["t2flair"]

SITK_INTERPOLATORS: Dict[str, int] = {
    "sitknearestneighbor": sitk.sitkNearestNeighbor,
    "sitklinear": sitk.sitkLinear,
    "sitkbspline": sitk.sitkBSpline,
}

warnings.filterwarnings("ignore", category=ConvergenceWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PatientCase:
    """患者级样本信息。"""

    split: str
    patient_id: str
    label_name: str
    label_id: int
    modality_paths: Dict[str, Path]
    voi_path: Path


class TeeLogger:
    """同时将输出写入终端和日志文件。"""

    def __init__(self, filepath: Path, mode: str = "w") -> None:
        self.file: TextIO = open(filepath, mode, encoding="utf-8")
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, message: str) -> None:
        self.file.write(message)
        self.stdout.write(message)
        self.flush()

    def flush(self) -> None:
        self.file.flush()
        self.stdout.flush()

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "TeeLogger":
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.close()


def str2bool(value: str) -> bool:
    """将命令行字符串解析成布尔值。"""

    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_spacing(value: str) -> Optional[Tuple[float, float, float]]:
    """解析等体素重采样间距。"""

    text = value.strip().lower()
    if text in {"none", "null", "off"}:
        return None
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "resampled-spacing must be 'none' or three comma separated floats, e.g. 1,1,1"
        )
    try:
        spacing = tuple(float(item) for item in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid resampled-spacing value: {value}"
        ) from exc
    if any(item <= 0 for item in spacing):
        raise argparse.ArgumentTypeError("resampled-spacing values must all be positive.")
    return spacing


def parse_c_grid(value: str) -> Tuple[float, ...]:
    """解析 LASSO Logistic 回归的 C 搜索网格。"""

    parts = [item.strip() for item in value.split(",") if item.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("C grid cannot be empty.")
    try:
        grid = tuple(float(item) for item in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid C grid: {value}") from exc
    if any(item <= 0 for item in grid):
        raise argparse.ArgumentTypeError("All C values must be positive.")
    return grid


def parse_optional_int(value: str) -> Optional[int]:
    """解析可选整数参数。"""

    text = value.strip().lower()
    if text in {"none", "null", "off"}:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer value: {value}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Integer value must be positive.")
    return parsed


def build_argparser() -> argparse.ArgumentParser:
    """定义 CLI 参数。"""

    cpu_count = os.cpu_count() or 1
    default_jobs = max(1, min(cpu_count - 1 if cpu_count > 1 else 1, 20))

    parser = argparse.ArgumentParser(
        description="Train and evaluate the Radiomics + LASSO Logistic Regression IDH baseline."
    )
    parser.add_argument(
        "--split-base-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "splited_data",
        help="包含 train/、val/、test/ 的患者级划分目录。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "baseline_RadiomicsLR",
        help="结果输出根目录。",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "baseline_RadiomicsLR" / "checkpoints",
        help="模型工件保存目录，保存最终的 joblib 包。",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="本次运行 ID；若不指定，则使用时间戳。",
    )
    parser.add_argument(
        "--shape-reference-modality",
        type=str,
        choices=DEFAULT_MODALITIES,
        default="t1ce",
        help="形状特征只从该模态提取一次，默认 t1ce。",
    )
    parser.add_argument(
        "--label-value",
        type=int,
        default=1,
        help="VOI 掩模中的有效标签值，默认 1。",
    )
    parser.add_argument(
        "--resampled-spacing",
        type=parse_spacing,
        default=(1.0, 1.0, 1.0),
        help="PyRadiomics 内部等体素重采样间距，默认 1,1,1；可传 none 关闭。",
    )
    parser.add_argument(
        "--interpolator",
        type=str,
        choices=("sitkBSpline", "sitkLinear", "sitkNearestNeighbor"),
        default="sitkBSpline",
        help="影像重采样插值方式，默认 sitkBSpline。",
    )
    parser.add_argument(
        "--normalize",
        type=str2bool,
        default=True,
        help="是否启用 PyRadiomics 内部强度标准化，默认 true。",
    )
    parser.add_argument(
        "--normalize-scale",
        type=float,
        default=100.0,
        help="PyRadiomics normalizeScale，默认 100。",
    )
    parser.add_argument(
        "--remove-outliers",
        type=float,
        default=3.0,
        help="PyRadiomics removeOutliers，默认 3.0。",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=25.0,
        help="PyRadiomics 灰度离散化 binWidth，默认 25。",
    )
    parser.add_argument(
        "--pad-distance",
        type=int,
        default=5,
        help="PyRadiomics padDistance，默认 5。",
    )
    parser.add_argument(
        "--correct-mask-geometry",
        type=str2bool,
        default=True,
        help="若掩模与影像的几何信息不一致，是否自动将掩模重采样到影像空间，默认 true。",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=default_jobs,
        help="CPU 并行进程数，同时用于 radiomics 提取和 CV；默认根据 CPU 自动设置。",
    )
    parser.add_argument(
        "--itk-threads-per-worker",
        type=int,
        default=1,
        help="每个 radiomics worker 内部允许 SimpleITK 使用的线程数，默认 1。",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="训练集内用于超参数搜索的分层 k 折数，默认 5。",
    )
    parser.add_argument(
        "--c-grid",
        type=parse_c_grid,
        default=DEFAULT_C_GRID,
        help=(
            "LASSO Logistic 回归的 C 搜索网格，逗号分隔。"
            "C 越小正则化越强，建议上限不超过 1.0 以避免高维过拟合。"
        ),
    )
    parser.add_argument(
        "--cv-scoring",
        type=str,
        default="roc_auc",
        help="训练集内 CV 评分函数，默认 roc_auc。",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.0,
        help="方差过滤阈值，默认 0.0（去除常数特征）。",
    )
    parser.add_argument(
        "--univariate-k",
        type=parse_optional_int,
        default=50,
        help=(
            "单变量预筛选（SelectKBest + f_classif）保留的特征数，默认 50。"
            "传 none 可关闭此步骤（直接以全量特征送 LASSO）。"
            "建议保持 k << n_train_samples，以避免高维稀疏导致正则化失效。"
        ),
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5000,
        help="LASSO Logistic 回归的最大迭代次数，默认 5000。",
    )
    parser.add_argument(
        "--use-class-weights",
        type=str2bool,
        default=True,
        help="是否启用 class_weight='balanced'，默认 true。",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="患者级概率转标签的固定阈值，默认 0.5。当 --optimize-threshold=true 时此参数作为回退值。",
    )
    parser.add_argument(
        "--optimize-threshold",
        type=str2bool,
        default=True,
        help=(
            "是否通过训练集内 CV 预测概率的 Youden 指数自动选取最优分类阈值，默认 true。"
            "开启后 val/test 均使用优化阈值，同时保留固定阈值 0.5 的对比指标。"
        ),
    )
    parser.add_argument(
        "--feature-extractor-version-note",
        type=str,
        default="PyRadiomics original features only",
        help="写入 protocol 的版本说明字段。",
    )
    parser.add_argument(
        "--save-train-predictions",
        type=str2bool,
        default=True,
        help="是否导出训练集患者级预测，默认 true。",
    )
    parser.add_argument(
        "--save-feature-table",
        type=str2bool,
        default=True,
        help="是否保存所有患者的原始 radiomics 特征表，默认 true。",
    )
    parser.add_argument(
        "--save-selected-feature-table",
        type=str2bool,
        default=True,
        help="是否保存选中特征的患者级表格，默认 true。",
    )
    parser.add_argument(
        "--feature-selection-fallback",
        type=str2bool,
        default=True,
        help="若 CV 选出的 LASSO 系数全为 0，是否自动尝试更大的 C 以避免 0 特征崩溃，默认 true。",
    )
    parser.add_argument(
        "--log-to-file",
        type=str2bool,
        default=True,
        help="是否将终端输出同时写入日志文件，默认 true。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="随机种子，默认 42。",
    )
    return parser


def resolve_run_id(run_id: Optional[str]) -> str:
    """生成本次运行 ID。"""

    if run_id:
        return run_id
    return time.strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int) -> None:
    """固定随机种子。"""

    random.seed(seed)
    np.random.seed(seed)


def make_json_safe(value: object) -> object:
    """递归将 Path / NumPy 标量等对象转换为 JSON 安全类型。"""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def save_json(data: Mapping[str, object], path: Path) -> None:
    """保存 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(make_json_safe(dict(data)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_yaml(data: Mapping[str, object], path: Path) -> None:
    """保存 YAML。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            make_json_safe(dict(data)),
            f,
            allow_unicode=True,
            sort_keys=False,
        )


def save_rows(rows: Iterable[Mapping[str, object]], path: Path, fieldnames: Sequence[str]) -> None:
    """保存 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def normalize_name(name: str) -> str:
    """标准化文件名，用于关键词匹配。"""

    return name.lower().replace("-", "_").replace(" ", "_")


def is_supported_image(path: Path) -> bool:
    """判断是否为支持的医学图像文件。"""

    lower_name = path.name.lower()
    return any(lower_name.endswith(ext) for ext in DEFAULT_ALLOWED_EXTENSIONS)


def collect_branch_image_files(branch_dir: Path) -> List[Path]:
    """递归收集一个分支目录下的影像文件。"""

    return [
        path
        for path in branch_dir.rglob("*")
        if path.is_file() and is_supported_image(path)
    ]


def match_modality(path: Path, modality: str) -> bool:
    """根据文件名关键词匹配模态。"""

    name = normalize_name(path.name)
    keywords = MODALITY_KEYWORDS[modality]

    if modality == "t1" and any(item in name for item in T1_EXCLUDE_KEYWORDS):
        return False
    if modality == "t2" and any(item in name for item in T2_EXCLUDE_KEYWORDS):
        return False
    return any(keyword in name for keyword in keywords)


def match_voi(path: Path) -> bool:
    """根据文件名判断是否为 VOI。"""

    name = normalize_name(path.name)
    return any(keyword in name for keyword in DEFAULT_VOI_KEYWORDS)


def score_voi_candidate(path: Path) -> int:
    """VOI 候选文件打分，优先选命名更明确的掩模。"""

    name = normalize_name(path.name)
    if name.startswith("voi.") or name.startswith("voi_"):
        return 1000
    score = 0
    for rank, keyword in enumerate(DEFAULT_VOI_KEYWORDS):
        if keyword in name:
            score += 100 - rank
    return score


def discover_conventional_patient_dirs(split_root: Path) -> Dict[str, Dict[str, Path]]:
    """扫描单个 split 下 `conventional/` 分支的患者目录。"""

    conventional_root = split_root / "conventional"
    if not conventional_root.is_dir():
        raise FileNotFoundError(f"Missing conventional directory: {conventional_root}")

    patient_map: Dict[str, Dict[str, Path]] = {}
    for label_name in DEFAULT_LABEL_TO_ID:
        label_root = conventional_root / label_name
        if not label_root.is_dir():
            raise FileNotFoundError(f"Missing label directory: {label_root}")
        for patient_dir in sorted(label_root.iterdir()):
            if not patient_dir.is_dir():
                continue
            patient_id = patient_dir.name
            if patient_id in patient_map:
                raise ValueError(
                    f"Duplicate patient folder detected in split {split_root.name}: {patient_id}"
                )
            patient_map[patient_id] = {
                "label_name": label_name,
                "conventional": patient_dir,
            }
    return patient_map


def discover_modality_files(patient_dir: Path, modalities: Sequence[str]) -> Dict[str, Path]:
    """在单个患者目录下匹配四个常规模态。"""

    all_files = collect_branch_image_files(patient_dir)
    modality_paths: Dict[str, Path] = {}
    for modality in modalities:
        matches = [path for path in all_files if match_modality(path, modality)]
        if len(matches) == 0:
            raise FileNotFoundError(
                f"Cannot find modality '{modality}' under patient directory: {patient_dir}"
            )
        if len(matches) > 1:
            match_str = "\n".join(str(path) for path in matches[:10])
            raise ValueError(
                f"Found multiple candidate files for modality '{modality}' under {patient_dir}.\n"
                f"{match_str}"
            )
        modality_paths[modality] = matches[0]
    return modality_paths


def discover_conventional_voi_file(patient_dir: Path) -> Path:
    """仅在 conventional 分支下查找全肿瘤 VOI。"""

    matches = [path for path in collect_branch_image_files(patient_dir) if match_voi(path)]
    if len(matches) == 0:
        raise FileNotFoundError(f"Cannot find VOI file under patient directory: {patient_dir}")
    if len(matches) == 1:
        return matches[0]

    scored = sorted(matches, key=lambda path: score_voi_candidate(path), reverse=True)
    best_score = score_voi_candidate(scored[0])
    best_matches = [path for path in scored if score_voi_candidate(path) == best_score]
    if len(best_matches) > 1:
        match_str = "\n".join(str(path) for path in best_matches[:10])
        raise ValueError(
            f"Found multiple equally plausible VOI masks under {patient_dir}.\n{match_str}"
        )
    return best_matches[0]


def build_patient_cases(split_base_root: Path) -> Dict[str, List[PatientCase]]:
    """为 train / val / test 三个子集建立患者级索引。"""

    case_map: Dict[str, List[PatientCase]] = {}
    for split_name in DEFAULT_SPLITS:
        split_root = split_base_root / split_name
        if not split_root.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_root}")

        split_cases: List[PatientCase] = []
        patient_dir_map = discover_conventional_patient_dirs(split_root)
        for patient_id, entry in sorted(patient_dir_map.items()):
            label_name = str(entry["label_name"])
            patient_dir = entry["conventional"]
            split_cases.append(
                PatientCase(
                    split=split_name,
                    patient_id=patient_id,
                    label_name=label_name,
                    label_id=DEFAULT_LABEL_TO_ID[label_name],
                    modality_paths=discover_modality_files(patient_dir, DEFAULT_MODALITIES),
                    voi_path=discover_conventional_voi_file(patient_dir),
                )
            )
        case_map[split_name] = split_cases
    return case_map


def resolve_interpolator(name: str) -> int:
    """将字符串形式的插值器名称映射为 SimpleITK 常量。"""

    key = name.strip().lower()
    if key not in SITK_INTERPOLATORS:
        raise ValueError(f"Unsupported interpolator: {name}")
    return SITK_INTERPOLATORS[key]


def ensure_binary_mask(mask_image: sitk.Image, label_value: int) -> sitk.Image:
    """将原始掩模转换为二值掩模。"""

    mask = sitk.Cast(mask_image, sitk.sitkUInt16)
    return sitk.BinaryThreshold(
        mask,
        lowerThreshold=label_value,
        upperThreshold=label_value,
        insideValue=1,
        outsideValue=0,
    )


def maybe_resample_mask_to_image(
    image: sitk.Image,
    mask: sitk.Image,
    correct_geometry: bool,
) -> sitk.Image:
    """必要时将掩模对齐到影像空间。"""

    same_geometry = (
        image.GetSize() == mask.GetSize()
        and np.allclose(image.GetSpacing(), mask.GetSpacing())
        and np.allclose(image.GetOrigin(), mask.GetOrigin())
        and np.allclose(image.GetDirection(), mask.GetDirection())
    )
    if same_geometry:
        return mask
    if not correct_geometry:
        raise ValueError(
            "Image and mask geometry mismatch. "
            "Enable --correct-mask-geometry true or fix upstream preprocessing."
        )
    return sitk.Resample(
        mask,
        image,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


def build_radiomics_settings(args: argparse.Namespace) -> Dict[str, object]:
    """整理 PyRadiomics 配置。"""

    settings: Dict[str, object] = {
        "binWidth": args.bin_width,
        "normalize": args.normalize,
        "normalizeScale": args.normalize_scale,
        "removeOutliers": args.remove_outliers,
        "padDistance": args.pad_distance,
        "correctMask": False,
    }
    if args.resampled_spacing is not None:
        settings["resampledPixelSpacing"] = list(args.resampled_spacing)
        settings["interpolator"] = resolve_interpolator(args.interpolator)
    return settings


def create_radiomics_extractor(settings: Mapping[str, object], include_shape: bool) -> featureextractor.RadiomicsFeatureExtractor:
    """构建 PyRadiomics 特征提取器。"""

    extractor = featureextractor.RadiomicsFeatureExtractor(**dict(settings))
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("glcm")
    extractor.enableFeatureClassByName("glrlm")
    extractor.enableFeatureClassByName("glszm")
    extractor.enableFeatureClassByName("gldm")
    extractor.enableFeatureClassByName("ngtdm")
    if include_shape:
        extractor.enableFeatureClassByName("shape")
    return extractor


def sanitize_feature_value(value: object) -> float:
    """将 PyRadiomics 的输出转为浮点数，并清理 inf / nan。"""

    try:
        numeric = float(value)
    except Exception:
        return float("nan")
    if not np.isfinite(numeric):
        return float("nan")
    return numeric


def format_feature_name(modality: str, raw_key: str) -> str:
    """统一 radiomics 特征列名。"""

    if raw_key.startswith("original_shape_"):
        return f"shape_{raw_key.removeprefix('original_shape_')}"
    if not raw_key.startswith("original_"):
        return f"{modality}_{raw_key}"
    return f"{modality}_{raw_key.removeprefix('original_')}"


def extract_patient_features(
    case: PatientCase,
    settings: Mapping[str, object],
    shape_reference_modality: str,
    label_value: int,
    correct_mask_geometry: bool,
    itk_threads_per_worker: int,
) -> Dict[str, object]:
    """对单个患者提取 radiomics 特征。"""

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(max(1, itk_threads_per_worker))

    row: Dict[str, object] = {
        "patient_id": case.patient_id,
        "split": case.split,
        "y_true": case.label_id,
        "label_name": case.label_name,
        "voi_path": str(case.voi_path),
    }

    raw_mask = sitk.ReadImage(str(case.voi_path))
    raw_mask = ensure_binary_mask(raw_mask, label_value)

    for modality in DEFAULT_MODALITIES:
        image_path = case.modality_paths[modality]
        row[f"{modality}_path"] = str(image_path)

        image = sitk.ReadImage(str(image_path))
        mask = maybe_resample_mask_to_image(
            image=image,
            mask=raw_mask,
            correct_geometry=correct_mask_geometry,
        )

        mask_array = sitk.GetArrayViewFromImage(mask)
        if int(np.sum(mask_array > 0)) <= 0:
            raise ValueError(
                f"Empty VOI after alignment for patient {case.patient_id}, modality {modality}."
            )

        extractor = create_radiomics_extractor(
            settings=settings,
            include_shape=(modality == shape_reference_modality),
        )
        feature_dict = extractor.execute(image, mask, label=1)
        for raw_key, value in feature_dict.items():
            if not raw_key.startswith("original_"):
                continue
            if raw_key.startswith("original_shape_") and modality != shape_reference_modality:
                continue
            feature_name = format_feature_name(modality=modality, raw_key=raw_key)
            row[feature_name] = sanitize_feature_value(value)

    return row


def extract_all_features(
    case_map: Mapping[str, Sequence[PatientCase]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    """并行提取全部患者的 radiomics 特征。"""

    cases: List[PatientCase] = []
    for split_name in DEFAULT_SPLITS:
        cases.extend(case_map[split_name])

    settings = build_radiomics_settings(args)
    print(
        f"Radiomics extraction started: {len(cases)} patients | "
        f"n_jobs={args.n_jobs} | itk_threads_per_worker={args.itk_threads_per_worker}"
    )

    iterator = tqdm(cases, desc="extract-radiomics", total=len(cases))
    if args.n_jobs == 1:
        rows = [
            extract_patient_features(
                case=case,
                settings=settings,
                shape_reference_modality=args.shape_reference_modality,
                label_value=args.label_value,
                correct_mask_geometry=args.correct_mask_geometry,
                itk_threads_per_worker=args.itk_threads_per_worker,
            )
            for case in iterator
        ]
    else:
        rows = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=0)(
            delayed(extract_patient_features)(
                case=case,
                settings=settings,
                shape_reference_modality=args.shape_reference_modality,
                label_value=args.label_value,
                correct_mask_geometry=args.correct_mask_geometry,
                itk_threads_per_worker=args.itk_threads_per_worker,
            )
            for case in iterator
        )

    feature_df = pd.DataFrame(rows)
    feature_df = feature_df.sort_values(["split", "patient_id"]).reset_index(drop=True)
    return feature_df


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """在 split 中只有单一类别时返回 NaN。"""

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def compute_patient_metrics(patient_rows: List[Dict[str, object]]) -> Dict[str, float]:
    """计算患者级 AUC / ACC / SEN / SPE / F1。"""

    y_true = np.asarray([row["y_true"] for row in patient_rows], dtype=np.int64)
    y_prob = np.asarray([row["prob_idh_mut"] for row in patient_rows], dtype=np.float64)
    y_pred = np.asarray([row["pred_label"] for row in patient_rows], dtype=np.int64)

    auc = safe_auc(y_true, y_prob)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sen = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spe = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "auc": auc,
        "acc": acc,
        "sen": sen,
        "spe": spe,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def resolve_cv_folds(y_train: np.ndarray, requested_folds: int) -> int:
    """根据训练集最小类别样本数，自动修正 k 折数。"""

    bincount = np.bincount(y_train, minlength=2)
    positive_counts = bincount[bincount > 0]
    if positive_counts.size < 2:
        raise ValueError("Training split must contain both classes.")
    max_supported = int(np.min(positive_counts))
    folds = min(requested_folds, max_supported)
    if folds < 2:
        raise ValueError(
            f"Not enough samples for stratified CV: requested={requested_folds}, "
            f"max_supported={max_supported}"
        )
    return folds


def build_cv_object(y_train: np.ndarray, folds: int, seed: int) -> StratifiedKFold:
    """构建训练集内的 StratifiedKFold。"""

    actual_folds = resolve_cv_folds(y_train, folds)
    return StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=seed)




def extract_cv_score_table(
    model: LogisticRegressionCV,
    stage: str,
) -> List[Dict[str, object]]:
    """导出 LogisticRegressionCV 的逐 C 值交叉验证结果。"""

    if not model.scores_:
        return []

    class_key = sorted(model.scores_.keys())[-1]
    scores = np.asarray(model.scores_[class_key], dtype=np.float64)
    c_values = np.asarray(model.Cs_, dtype=np.float64)
    rows: List[Dict[str, object]] = []
    for idx, c_value in enumerate(c_values):
        score_column = scores[:, idx]
        rows.append(
            {
                "stage": stage,
                "c_value": float(c_value),
                "mean_score": float(np.mean(score_column)),
                "std_score": float(np.std(score_column)),
                "num_folds": int(score_column.shape[0]),
            }
        )
    return rows


def parse_feature_metadata(feature_name: str) -> Tuple[str, str, str]:
    """从列名解析模态、特征类别和简短名称。"""

    if feature_name.startswith("shape_"):
        return "shape", "shape", feature_name.removeprefix("shape_")

    parts = feature_name.split("_", 2)
    if len(parts) != 3:
        return "unknown", "unknown", feature_name
    return parts[0], parts[1], parts[2]


def fit_radiomics_lasso_pipeline(
    feature_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, object]:
    """在训练集上拟合完整的 radiomics + LASSO pipeline。

    改进点（相比原始两阶段版本）：
    - 新增单变量预筛选（SelectKBest + f_classif），在 LASSO 之前将特征降至
      ``args.univariate_k`` 个，避免 n_samples << n_features 导致正则化失效；
    - 将原来 selector → classifier 两步 LogisticRegressionCV 合并为单步，
      消除双重拟合的冗余偏差；
    - C 搜索网格上限收紧至 1.0，强制保留足够的 L1 惩罚力度；
    - 训练集内 CV 预测概率通过 Youden 指数优化分类阈值（当
      ``args.optimize_threshold=True`` 时）。
    """

    metadata_columns = {
        "patient_id",
        "split",
        "y_true",
        "label_name",
        "voi_path",
        "t1_path",
        "t1ce_path",
        "t2_path",
        "t2flair_path",
    }
    feature_columns = [col for col in feature_df.columns if col not in metadata_columns]
    feature_columns = sorted(feature_columns)
    if not feature_columns:
        raise RuntimeError("No radiomics feature columns were extracted.")

    df = feature_df.copy()
    df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    if train_df.empty:
        raise RuntimeError("Training split is empty.")

    x_train_raw = train_df[feature_columns].to_numpy(dtype=np.float64)
    y_train = train_df["y_true"].to_numpy(dtype=np.int64)

    # ── 步骤 1：缺失值填补 ────────────────────────────────────────────────────
    imputer = SimpleImputer(strategy="median")
    x_train_imputed = imputer.fit_transform(x_train_raw)

    # ── 步骤 2：方差过滤（去除常数特征）──────────────────────────────────────
    variance_selector = VarianceThreshold(threshold=args.variance_threshold)
    x_train_variance = variance_selector.fit_transform(x_train_imputed)
    variance_feature_names = [
        feature_name
        for feature_name, keep_flag in zip(feature_columns, variance_selector.get_support())
        if keep_flag
    ]
    if len(variance_feature_names) == 0:
        raise RuntimeError("All features were removed by VarianceThreshold.")

    # ── 步骤 3：标准化 ────────────────────────────────────────────────────────
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_variance)

    # ── 步骤 4（新增）：单变量预筛选 ──────────────────────────────────────────
    # 当 univariate_k 不为 None 时，使用 SelectKBest(f_classif) 预筛选特征。
    # 这一步将特征数压缩至远小于训练样本数，使后续 LASSO 正则化能真正发挥作用。
    univariate_k = getattr(args, "univariate_k", None)
    if univariate_k is not None:
        actual_k = min(univariate_k, x_train_scaled.shape[1])
        if actual_k < x_train_scaled.shape[1]:
            univariate_selector: Optional[SelectKBest] = SelectKBest(
                score_func=f_classif, k=actual_k
            )
            x_train_prescreened = univariate_selector.fit_transform(x_train_scaled, y_train)
            prescreened_feature_names = [
                feature_name
                for feature_name, keep_flag in zip(
                    variance_feature_names, univariate_selector.get_support()
                )
                if keep_flag
            ]
            print(
                f"Univariate prescreening: {len(variance_feature_names)} → "
                f"{len(prescreened_feature_names)} features (k={actual_k})"
            )
        else:
            # 特征数本就不超过 k，跳过预筛选
            univariate_selector = None
            x_train_prescreened = x_train_scaled
            prescreened_feature_names = variance_feature_names
            print(
                f"Univariate prescreening skipped: "
                f"n_features={x_train_scaled.shape[1]} <= k={actual_k}"
            )
    else:
        univariate_selector = None
        x_train_prescreened = x_train_scaled
        prescreened_feature_names = variance_feature_names
        print("Univariate prescreening disabled (--univariate-k none).")

    # ── 步骤 5：单步 LASSO LogisticRegressionCV（特征选择 + 分类合并）─────────
    # 原两阶段 selector → classifier 合并为单步，消除冗余偏差。
    # C 搜索网格上限 ≤ 1.0，保证 L1 惩罚有效压缩特征。
    class_weight = "balanced" if args.use_class_weights else None
    cv = build_cv_object(y_train=y_train, folds=args.cv_folds, seed=args.seed)

    c_grid = getattr(args, "c_grid", DEFAULT_C_GRID)
    classifier_cv = LogisticRegressionCV(
        Cs=np.asarray(c_grid, dtype=np.float64),
        cv=cv,
        penalty="l1",
        solver="saga",
        scoring=args.cv_scoring,
        class_weight=class_weight,
        n_jobs=args.n_jobs,
        max_iter=args.max_iter,
        fit_intercept=True,
        refit=True,
        random_state=args.seed,
    )
    classifier_cv.fit(x_train_prescreened, y_train)

    classifier_coef = classifier_cv.coef_.ravel()
    selected_mask_prescreened = np.abs(classifier_coef) > 1e-8

    if not np.any(selected_mask_prescreened):
        # 所有系数为 0，尝试最大 C 的单次拟合作为兜底
        warnings.warn(
            "LASSO selected zero features. Falling back to C=1.0 single fit.",
            RuntimeWarning,
            stacklevel=2,
        )
        from sklearn.linear_model import LogisticRegression as _LR

        _fallback = _LR(
            penalty="l1",
            solver="saga",
            C=1.0,
            class_weight=class_weight,
            max_iter=args.max_iter,
            fit_intercept=True,
            random_state=args.seed,
        )
        _fallback.fit(x_train_prescreened, y_train)
        selected_mask_prescreened = np.abs(_fallback.coef_.ravel()) > 1e-8
        # 用兜底模型的系数覆盖 classifier_cv，保证后续接口一致
        classifier_cv.coef_ = _fallback.coef_.copy()
        classifier_cv.intercept_ = _fallback.intercept_.copy()
        classifier_cv.C_ = np.asarray([1.0], dtype=np.float64)

    selected_feature_names = [
        feature_name
        for feature_name, keep_flag in zip(prescreened_feature_names, selected_mask_prescreened)
        if keep_flag
    ]
    if len(selected_feature_names) == 0:
        raise RuntimeError(
            "No selected features remain after LASSO. "
            "Try a larger C grid or enable --univariate-k none."
        )

    # ── 步骤 6（新增）：Youden 阈值优化 ──────────────────────────────────────
    # 在训练集内通过 CV 预测概率最大化 Youden 指数（SEN + SPE - 1），
    # 使分类阈值不依赖于固定的 0.5。
    optimize_threshold = getattr(args, "optimize_threshold", True)
    if optimize_threshold and len(np.unique(y_train)) == 2:
        # 用已拟合好的 prescreened 特征矩阵通过 cross_val_predict 获取 OOF 概率
        oof_probs = cross_val_predict(
            LogisticRegressionCV(
                Cs=np.asarray(c_grid, dtype=np.float64),
                cv=cv,
                penalty="l1",
                solver="saga",
                scoring=args.cv_scoring,
                class_weight=class_weight,
                n_jobs=args.n_jobs,
                max_iter=args.max_iter,
                fit_intercept=True,
                refit=True,
                random_state=args.seed,
            ),
            x_train_prescreened,
            y_train,
            cv=cv,
            method="predict_proba",
        )[:, 1]
        fpr_arr, tpr_arr, thr_arr = roc_curve(y_train, oof_probs)
        youden_scores = tpr_arr - fpr_arr
        best_idx = int(np.argmax(youden_scores))
        optimal_threshold = float(thr_arr[best_idx])
        print(
            f"Youden threshold optimization: "
            f"optimal_threshold={optimal_threshold:.4f} "
            f"(Youden={youden_scores[best_idx]:.4f}, "
            f"SEN={tpr_arr[best_idx]:.4f}, SPE={1 - fpr_arr[best_idx]:.4f})"
        )
    else:
        optimal_threshold = float(args.threshold)
        print(f"Threshold optimization disabled; using fixed threshold={optimal_threshold:.4f}")

    # ── 构建特征详情表 ────────────────────────────────────────────────────────
    # 将 selected_mask_prescreened 映射回原始 variance_feature_names 空间
    # 以便输出包含未被单变量预筛选也未被 LASSO 选中的特征的完整表格
    if univariate_selector is not None:
        univariate_support = univariate_selector.get_support()
        prescreened_idx_in_variance = [
            idx for idx, keep in enumerate(univariate_support) if keep
        ]
        # selected_mask_prescreened 的长度等于 prescreened_feature_names 的长度
        selected_mask_full = np.zeros(len(variance_feature_names), dtype=bool)
        for local_idx, (global_idx, keep) in enumerate(
            zip(prescreened_idx_in_variance, selected_mask_prescreened)
        ):
            selected_mask_full[global_idx] = bool(keep)
        prescreened_mask_full = np.zeros(len(variance_feature_names), dtype=bool)
        for global_idx in prescreened_idx_in_variance:
            prescreened_mask_full[global_idx] = True
    else:
        univariate_support = np.ones(len(variance_feature_names), dtype=bool)
        selected_mask_full = selected_mask_prescreened.copy()
        prescreened_mask_full = np.ones(len(variance_feature_names), dtype=bool)

    # 构建 classifier_coef_full（与 variance_feature_names 对齐）
    classifier_coef_full = np.zeros(len(variance_feature_names), dtype=np.float64)
    local_idx = 0
    for global_idx, prescreened in enumerate(prescreened_mask_full):
        if prescreened:
            classifier_coef_full[global_idx] = float(classifier_coef[local_idx])
            local_idx += 1

    selected_feature_rows: List[Dict[str, object]] = []
    selected_rank = 0
    for i, feature_name in enumerate(variance_feature_names):
        modality, feature_class, short_name = parse_feature_metadata(feature_name)
        in_prescreened = bool(prescreened_mask_full[i])
        in_selected = bool(selected_mask_full[i])
        if in_selected:
            coef_value = float(classifier_coef_full[i])
            rank_value = selected_rank + 1
            selected_rank += 1
        else:
            coef_value = float("nan")
            rank_value = ""
        selected_feature_rows.append(
            {
                "feature_name": feature_name,
                "modality": modality,
                "feature_class": feature_class,
                "feature_short_name": short_name,
                "passed_univariate_prescreen": int(in_prescreened),
                "selected_by_lasso": int(in_selected),
                "classifier_coefficient": coef_value,
                "classifier_abs_coefficient": abs(coef_value) if in_selected else float("nan"),
                "rank_in_final_model": rank_value,
            }
        )

    classifier_cv_rows = extract_cv_score_table(classifier_cv, stage="classification")

    return {
        "metadata_columns": sorted(metadata_columns),
        "feature_columns_raw": feature_columns,
        "feature_columns_after_variance": variance_feature_names,
        "prescreened_feature_names": prescreened_feature_names,
        "selected_feature_names": selected_feature_names,
        "imputer": imputer,
        "variance_selector": variance_selector,
        "univariate_selector": univariate_selector,
        "scaler": scaler,
        "classifier_cv": classifier_cv,
        "selected_mask_prescreened": selected_mask_prescreened,
        "selected_mask_full": selected_mask_full,
        "selected_feature_rows": selected_feature_rows,
        "selector_cv_rows": [],   # 已合并，保留键以兼容下游输出函数
        "classifier_cv_rows": classifier_cv_rows,
        "selector_fallback_info": {"fallback_used": False, "fallback_reason": "", "fallback_c": None},
        "cv_folds_actual": cv.n_splits,
        "class_weight": class_weight,
        "shape_reference_modality": args.shape_reference_modality,
        "num_raw_features": len(feature_columns),
        "num_features_after_variance": len(variance_feature_names),
        "num_prescreened_features": len(prescreened_feature_names),
        "num_selected_features": len(selected_feature_names),
        "optimal_threshold": optimal_threshold,
        "threshold_optimized": optimize_threshold,
    }


def transform_feature_matrix(
    raw_matrix: np.ndarray,
    pipeline: Mapping[str, object],
) -> np.ndarray:
    """将原始特征矩阵变换到最终分类器输入空间。

    变换顺序：impute → variance_filter → scale → univariate_prescreen → lasso_mask
    """

    x_imputed = pipeline["imputer"].transform(raw_matrix)
    x_variance = pipeline["variance_selector"].transform(x_imputed)
    x_scaled = pipeline["scaler"].transform(x_variance)

    univariate_selector = pipeline.get("univariate_selector")
    if univariate_selector is not None:
        x_prescreened = univariate_selector.transform(x_scaled)
    else:
        x_prescreened = x_scaled

    selected_mask = np.asarray(pipeline["selected_mask_prescreened"], dtype=bool)
    return x_prescreened[:, selected_mask]


def predict_split(
    split_df: pd.DataFrame,
    pipeline: Mapping[str, object],
    threshold: float,
    run_id: str,
    checkpoint_name: str,
) -> Dict[str, object]:
    """对单个 split 生成患者级预测和指标。

    同时计算 optimal_threshold 和固定 threshold=0.5 下的指标，便于对比。
    """

    feature_columns = pipeline["feature_columns_raw"]
    raw_matrix = split_df[feature_columns].to_numpy(dtype=np.float64)
    x_final = transform_feature_matrix(raw_matrix, pipeline)
    probs = pipeline["classifier_cv"].predict_proba(x_final)[:, 1]
    y_true = split_df["y_true"].to_numpy(dtype=np.int64)

    # 使用 optimal_threshold（可能来自 Youden 优化或固定值）
    optimal_threshold = float(pipeline.get("optimal_threshold", threshold))
    pred_labels = (probs >= optimal_threshold).astype(np.int64)

    patient_rows: List[Dict[str, object]] = []
    for patient_id, y_item, prob_item, pred_item, split_name in zip(
        split_df["patient_id"].tolist(),
        y_true.tolist(),
        probs.tolist(),
        pred_labels.tolist(),
        split_df["split"].tolist(),
    ):
        patient_rows.append(
            {
                "patient_id": str(patient_id),
                "split": str(split_name),
                "y_true": int(y_item),
                "prob_idh_mut": float(prob_item),
                "pred_label": int(pred_item),
                "run_id": run_id,
                "checkpoint_name": checkpoint_name,
            }
        )

    metrics = compute_patient_metrics(patient_rows)

    # 固定阈值 0.5 的对比指标
    pred_labels_fixed = (probs >= 0.5).astype(np.int64)
    fixed_rows_tmp = [
        {**row, "pred_label": int(pl)}
        for row, pl in zip(patient_rows, pred_labels_fixed)
    ]
    metrics_fixed05 = compute_patient_metrics(fixed_rows_tmp)

    return {
        "patient_rows": patient_rows,
        "metrics": metrics,
        "metrics_fixed_threshold_05": metrics_fixed05,
        "optimal_threshold": optimal_threshold,
        "num_patients": len(patient_rows),
        "num_features_before_selection": int(pipeline["num_raw_features"]),
        "num_features_after_selection": int(pipeline["num_selected_features"]),
        "classifier_best_c": float(np.asarray(pipeline["classifier_cv"].C_).ravel()[0]),
        # 向后兼容：保留 selector_best_c 键
        "selector_best_c": float(np.asarray(pipeline["classifier_cv"].C_).ravel()[0]),
    }


def save_feature_tables(
    feature_df: pd.DataFrame,
    output_dir: Path,
    run_id: str,
    pipeline: Mapping[str, object],
    save_raw: bool,
    save_selected: bool,
) -> None:
    """保存原始 radiomics 特征表和最终选中特征子表。"""

    if save_raw:
        raw_path = output_dir / f"radiomics_features_raw_{MODEL_NAME}_{run_id}.csv"
        feature_df.to_csv(raw_path, index=False)

    if save_selected:
        selected_columns = ["patient_id", "split", "y_true", "label_name"] + list(
            pipeline["selected_feature_names"]
        )
        # 过滤掉不存在于 feature_df 的列（防御性检查）
        available = [col for col in selected_columns if col in feature_df.columns]
        selected_path = output_dir / f"radiomics_features_selected_{MODEL_NAME}_{run_id}.csv"
        feature_df[available].to_csv(selected_path, index=False)


def save_model_artifacts(
    pipeline: Mapping[str, object],
    checkpoint_dir: Path,
    run_id: str,
) -> Path:
    """保存最终的 radiomics + LR 工件。"""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.joblib"
    last_path = checkpoint_dir / "last.joblib"

    bundle = {
        "model": MODEL_NAME,
        "run_id": run_id,
        "modalities": list(DEFAULT_MODALITIES),
        "shape_reference_modality": pipeline["shape_reference_modality"],
        "feature_columns_raw": list(pipeline["feature_columns_raw"]),
        "feature_columns_after_variance": list(pipeline["feature_columns_after_variance"]),
        "prescreened_feature_names": list(pipeline["prescreened_feature_names"]),
        "selected_feature_names": list(pipeline["selected_feature_names"]),
        "imputer": pipeline["imputer"],
        "variance_selector": pipeline["variance_selector"],
        "univariate_selector": pipeline["univariate_selector"],
        "scaler": pipeline["scaler"],
        "selected_mask_prescreened": np.asarray(pipeline["selected_mask_prescreened"], dtype=bool),
        "selected_mask_full": np.asarray(pipeline["selected_mask_full"], dtype=bool),
        "classifier_cv": pipeline["classifier_cv"],
        "optimal_threshold": float(pipeline["optimal_threshold"]),
        "threshold_optimized": bool(pipeline["threshold_optimized"]),
        "num_raw_features": int(pipeline["num_raw_features"]),
        "num_features_after_variance": int(pipeline["num_features_after_variance"]),
        "num_prescreened_features": int(pipeline["num_prescreened_features"]),
        "num_selected_features": int(pipeline["num_selected_features"]),
    }
    joblib.dump(bundle, best_path)
    shutil.copy2(best_path, last_path)
    return best_path


def export_prediction_tables(
    eval_outputs: Mapping[str, Mapping[str, object]],
    output_dir: Path,
    run_id: str,
    checkpoint_name: str,
    save_train_predictions: bool,
) -> None:
    """导出患者级预测表、ROC、混淆矩阵和指标汇总。"""

    metrics_rows: List[Dict[str, object]] = []
    for split_name, output in eval_outputs.items():
        if split_name == "train" and not save_train_predictions:
            continue

        patient_rows = list(output["patient_rows"])
        metrics = dict(output["metrics"])

        pred_path = output_dir / f"patient_predictions_{MODEL_NAME}_{split_name}_{run_id}.csv"
        save_rows(
            patient_rows,
            pred_path,
            fieldnames=[
                "patient_id",
                "split",
                "y_true",
                "prob_idh_mut",
                "pred_label",
                "run_id",
                "checkpoint_name",
            ],
        )

        y_true = np.asarray([row["y_true"] for row in patient_rows], dtype=np.int64)
        y_prob = np.asarray([row["prob_idh_mut"] for row in patient_rows], dtype=np.float64)
        if len(np.unique(y_true)) >= 2:
            fpr, tpr, threshold_arr = roc_curve(y_true, y_prob)
            roc_rows = [
                {
                    "fpr": float(fpr_item),
                    "tpr": float(tpr_item),
                    "threshold": float(thr_item),
                    "model": MODEL_NAME,
                    "run_id": run_id,
                }
                for fpr_item, tpr_item, thr_item in zip(fpr, tpr, threshold_arr)
            ]
            save_rows(
                roc_rows,
                output_dir / f"roc_points_{MODEL_NAME}_{split_name}_{run_id}.csv",
                fieldnames=["fpr", "tpr", "threshold", "model", "run_id"],
            )
            save_rows(
                roc_rows,
                output_dir / f"roc_raw_{MODEL_NAME}_{split_name}_{run_id}.csv",
                fieldnames=["fpr", "tpr", "threshold", "model", "run_id"],
            )
        save_rows(
            [
                {
                    "tn": metrics["tn"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "tp": metrics["tp"],
                    "model": MODEL_NAME,
                    "run_id": run_id,
                }
            ],
            output_dir / f"confusion_matrix_{MODEL_NAME}_{split_name}_{run_id}.csv",
            fieldnames=["tn", "fp", "fn", "tp", "model", "run_id"],
        )

        metrics_rows.append(
            {
                "split": split_name,
                "auc": metrics["auc"],
                "acc": metrics["acc"],
                "sen": metrics["sen"],
                "spe": metrics["spe"],
                "f1": metrics["f1"],
                "optimal_threshold": output.get("optimal_threshold", 0.5),
                "acc_fixed05": output.get("metrics_fixed_threshold_05", {}).get("acc", ""),
                "sen_fixed05": output.get("metrics_fixed_threshold_05", {}).get("sen", ""),
                "spe_fixed05": output.get("metrics_fixed_threshold_05", {}).get("spe", ""),
                "f1_fixed05": output.get("metrics_fixed_threshold_05", {}).get("f1", ""),
                "num_patients": output["num_patients"],
                "num_features_before_selection": output["num_features_before_selection"],
                "num_features_after_selection": output["num_features_after_selection"],
                "selector_best_c": output["selector_best_c"],
                "classifier_best_c": output["classifier_best_c"],
                "run_id": run_id,
                "checkpoint_name": checkpoint_name,
            }
        )

    metrics_fieldnames = [
        "split",
        "auc",
        "acc",
        "sen",
        "spe",
        "f1",
        "optimal_threshold",
        "acc_fixed05",
        "sen_fixed05",
        "spe_fixed05",
        "f1_fixed05",
        "num_patients",
        "num_features_before_selection",
        "num_features_after_selection",
        "selector_best_c",
        "classifier_best_c",
        "run_id",
        "checkpoint_name",
    ]
    save_rows(
        metrics_rows,
        output_dir / f"metrics_summary_{MODEL_NAME}_{run_id}.csv",
        fieldnames=metrics_fieldnames,
    )
    save_rows(
        metrics_rows,
        output_dir / f"metrics_{MODEL_NAME}_{run_id}.csv",
        fieldnames=metrics_fieldnames,
    )

    # 为时间线中的“单文件命名”提供 test 别名。
    test_pred = output_dir / f"patient_predictions_{MODEL_NAME}_test_{run_id}.csv"
    if test_pred.is_file():
        shutil.copy2(test_pred, output_dir / f"patient_predictions_{MODEL_NAME}.csv")
    test_roc = output_dir / f"roc_points_{MODEL_NAME}_test_{run_id}.csv"
    if test_roc.is_file():
        shutil.copy2(test_roc, output_dir / f"roc_points_{MODEL_NAME}.csv")
    test_cm = output_dir / f"confusion_matrix_{MODEL_NAME}_test_{run_id}.csv"
    if test_cm.is_file():
        shutil.copy2(test_cm, output_dir / f"confusion_matrix_{MODEL_NAME}.csv")
    shutil.copy2(
        output_dir / f"metrics_{MODEL_NAME}_{run_id}.csv",
        output_dir / f"metrics_{MODEL_NAME}.csv",
    )


def export_wrong_cases(
    eval_outputs: Mapping[str, Mapping[str, object]],
    output_dir: Path,
    run_id: str,
    threshold: float,
    save_train_predictions: bool,
) -> None:
    """导出误判病例。"""

    wrong_rows: List[Dict[str, object]] = []
    for split_name, output in eval_outputs.items():
        if split_name == "train" and not save_train_predictions:
            continue
        for row in output["patient_rows"]:
            if int(row["y_true"]) == int(row["pred_label"]):
                continue
            prob = float(row["prob_idh_mut"])
            wrong_rows.append(
                {
                    **row,
                    "error_type": "FP" if int(row["y_true"]) == 0 else "FN",
                    "prob_margin_to_threshold": abs(prob - threshold),
                    "is_low_confidence": int(abs(prob - threshold) < 0.1),
                    "num_patients_in_split": output["num_patients"],
                }
            )

    wrong_rows.sort(
        key=lambda item: (
            str(item["split"]),
            str(item["error_type"]),
            float(item["prob_margin_to_threshold"]),
            str(item["patient_id"]),
        )
    )
    fieldnames = [
        "patient_id",
        "split",
        "y_true",
        "prob_idh_mut",
        "pred_label",
        "run_id",
        "checkpoint_name",
        "error_type",
        "prob_margin_to_threshold",
        "is_low_confidence",
        "num_patients_in_split",
    ]
    save_rows(
        wrong_rows,
        output_dir / f"wrong_cases_{MODEL_NAME}_{run_id}.csv",
        fieldnames=fieldnames,
    )
    shutil.copy2(
        output_dir / f"wrong_cases_{MODEL_NAME}_{run_id}.csv",
        output_dir / f"wrong_cases_{MODEL_NAME}.csv",
    )


def plot_roc_curve(patient_rows: List[Dict[str, object]], path: Path, title: str) -> bool:
    """绘制患者级 ROC 曲线。"""

    y_true = np.asarray([row["y_true"] for row in patient_rows], dtype=np.int64)
    y_prob = np.asarray([row["prob_idh_mut"] for row in patient_rows], dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return False

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = safe_auc(y_true, y_prob)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=150)
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2.0, label=f"AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_confusion_matrix(metrics: Mapping[str, float], path: Path, title: str) -> None:
    """绘制 2x2 混淆矩阵。"""

    matrix = np.asarray(
        [
            [int(metrics["tn"]), int(metrics["fp"])],
            [int(metrics["fn"]), int(metrics["tp"])],
        ],
        dtype=np.int64,
    )
    labels = (("TN", "FP"), ("FN", "TP"))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 4.5), dpi=150)
    image = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_title(title)

    max_value = max(int(matrix.max()), 1)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = int(matrix[row_idx, col_idx])
            text_color = "white" if value > max_value / 2 else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{labels[row_idx][col_idx]}\n{value}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def export_evaluation_figures(
    eval_outputs: Mapping[str, Mapping[str, object]],
    output_dir: Path,
    run_id: str,
    save_train_predictions: bool,
) -> None:
    """导出 ROC 和混淆矩阵 PNG。"""

    figure_dir = output_dir / "figures"
    for split_name, output in eval_outputs.items():
        if split_name == "train" and not save_train_predictions:
            continue
        patient_rows = list(output["patient_rows"])
        metrics = dict(output["metrics"])

        roc_path = figure_dir / f"roc_curve_{MODEL_NAME}_{split_name}_{run_id}.png"
        cm_path = figure_dir / f"confusion_matrix_{MODEL_NAME}_{split_name}_{run_id}.png"
        wrote_roc = plot_roc_curve(
            patient_rows=patient_rows,
            path=roc_path,
            title=f"{MODEL_NAME.upper()} ROC ({split_name})",
        )
        plot_confusion_matrix(
            metrics=metrics,
            path=cm_path,
            title=f"{MODEL_NAME.upper()} Confusion Matrix ({split_name})",
        )

        if split_name == "test":
            if wrote_roc:
                plot_roc_curve(
                    patient_rows=patient_rows,
                    path=figure_dir / f"{MODEL_NAME}_roc.png",
                    title=f"{MODEL_NAME.upper()} ROC (test)",
                )
            plot_confusion_matrix(
                metrics=metrics,
                path=figure_dir / f"{MODEL_NAME}_cm.png",
                title=f"{MODEL_NAME.upper()} Confusion Matrix (test)",
            )


def export_selected_features(
    pipeline: Mapping[str, object],
    output_dir: Path,
    run_id: str,
) -> None:
    """导出特征筛选结果。"""

    rows = list(pipeline["selected_feature_rows"])
    fieldnames = [
        "feature_name",
        "modality",
        "feature_class",
        "feature_short_name",
        "passed_univariate_prescreen",
        "selected_by_lasso",
        "classifier_coefficient",
        "classifier_abs_coefficient",
        "rank_in_final_model",
    ]
    path = output_dir / f"selected_features_{MODEL_NAME}_{run_id}.csv"
    save_rows(rows, path, fieldnames=fieldnames)
    shutil.copy2(path, output_dir / f"selected_features_{MODEL_NAME}.csv")


def export_cv_tables(
    pipeline: Mapping[str, object],
    output_dir: Path,
    run_id: str,
) -> None:
    """导出 selector / classifier 的 CV 打分表。"""

    cv_rows = list(pipeline["selector_cv_rows"]) + list(pipeline["classifier_cv_rows"])
    if not cv_rows:
        return
    save_rows(
        cv_rows,
        output_dir / f"cv_results_{MODEL_NAME}_{run_id}.csv",
        fieldnames=["stage", "c_value", "mean_score", "std_score", "num_folds"],
    )


def export_scaler_stats(
    pipeline: Mapping[str, object],
    output_dir: Path,
) -> None:
    """导出训练集内的标准化参数和筛选摘要。"""

    feature_names = list(pipeline["feature_columns_after_variance"])
    scaler = pipeline["scaler"]
    imputer = pipeline["imputer"]
    classifier_cv = pipeline["classifier_cv"]
    classifier_best_c = float(np.asarray(classifier_cv.C_).ravel()[0])

    stats_payload = {
        "model": MODEL_NAME,
        "imputer_strategy": "median",
        "feature_names_after_variance": feature_names,
        "imputer_statistics": {
            feature_name: float(value)
            for feature_name, value in zip(feature_names, imputer.statistics_[pipeline["variance_selector"].get_support()])
        }
        if hasattr(pipeline["variance_selector"], "get_support")
        else {},
        "scaler_mean": {
            feature_name: float(value)
            for feature_name, value in zip(feature_names, scaler.mean_)
        },
        "scaler_scale": {
            feature_name: float(value)
            for feature_name, value in zip(feature_names, scaler.scale_)
        },
        "classifier_best_c": classifier_best_c,
        "selector_fallback_info": pipeline["selector_fallback_info"],
        "optimal_threshold": float(pipeline.get("optimal_threshold", 0.5)),
        "threshold_optimized": bool(pipeline.get("threshold_optimized", False)),
        "num_raw_features": int(pipeline["num_raw_features"]),
        "num_features_after_variance": int(pipeline["num_features_after_variance"]),
        "num_prescreened_features": int(pipeline["num_prescreened_features"]),
        "num_selected_features": int(pipeline["num_selected_features"]),
        "selected_feature_names": list(pipeline["selected_feature_names"]),
    }
    save_json(stats_payload, output_dir / f"scaler_stats_{MODEL_NAME}.json")


def export_protocol_markdown(
    args: argparse.Namespace,
    case_map: Mapping[str, Sequence[PatientCase]],
    pipeline: Mapping[str, object],
    output_dir: Path,
) -> None:
    """导出传统影像组学 baseline 的协议说明。"""

    classifier_best_c = float(np.asarray(pipeline["classifier_cv"].C_).ravel()[0])
    optimal_threshold = float(pipeline.get("optimal_threshold", args.threshold))
    threshold_optimized = bool(pipeline.get("threshold_optimized", False))
    univariate_k = getattr(args, "univariate_k", None)
    lines = [
        "# Radiomics + Logistic Regression 协议说明（改进版）",
        "",
        f"- 运行日期：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 模型标识：`{MODEL_NAME}`",
        f"- 输入模态：`{', '.join(DEFAULT_MODALITIES)}`",
        f"- VOI 来源：`conventional/` 分支下的全肿瘤 `VOI`",
        f"- 形状特征参考模态：`{args.shape_reference_modality}`",
        f"- 交叉验证折数：`{pipeline['cv_folds_actual']}`",
        f"- 最优 C（LASSO）：`{classifier_best_c}`",
        f"- 最优分类阈值：`{optimal_threshold:.4f}`（Youden 优化：`{threshold_optimized}`）",
        f"- 类别权重：`{pipeline['class_weight']}`",
        "",
        "## 1. 队列与划分",
        "",
        *[
            f"- `{split_name}`: {len(case_map[split_name])} 例"
            for split_name in DEFAULT_SPLITS
        ],
        "",
        "## 2. Radiomics 预处理",
        "",
        "- 假设输入影像已完成上游配准和空间对齐。",
        "- 当前脚本额外执行以下 radiomics 侧处理：",
        f"  - 掩模几何自动校正：`{args.correct_mask_geometry}`",
        f"  - 等体素重采样：`{args.resampled_spacing}`",
        f"  - 插值方式：`{args.interpolator}`",
        f"  - 强度归一化：`{args.normalize}`",
        f"  - normalizeScale：`{args.normalize_scale}`",
        f"  - removeOutliers：`{args.remove_outliers}`",
        f"  - binWidth：`{args.bin_width}`",
        f"  - padDistance：`{args.pad_distance}`",
        "",
        "## 3. 特征集合",
        "",
        "- 仅启用 `Original` 图像类型，不启用 wavelet / LoG / 自定义滤波特征。",
        "- 启用的 PyRadiomics 特征类：",
        *[f"  - `{feature_class}`" for feature_class in DEFAULT_FEATURE_CLASSES],
        "- 形状特征只提取一次，避免在多模态中重复写入完全相同的 shape 特征。",
        "",
        "## 4. 建模流程（改进版）",
        "",
        "- 训练集内 `median` 缺失值填补",
        f"- 训练集内 `VarianceThreshold(threshold={args.variance_threshold})`",
        "- 训练集内 `StandardScaler`",
        f"- 【新增】训练集内 `SelectKBest(f_classif, k={univariate_k})` 单变量预筛选"
        + ("（已禁用）" if univariate_k is None else ""),
        "- 训练集内单步 `L1 LogisticRegressionCV`（特征选择 + 分类合并）",
        f"  - C 搜索网格：`{list(getattr(args, 'c_grid', DEFAULT_C_GRID))}`（上限 ≤ 1.0）",
        "- 【新增】训练集内 CV OOF 概率 → Youden 指数自动阈值优化",
        "- 验证集和测试集仅做独立评估，不参与超参数搜索",
        "",
        "## 5. 结果摘要",
        "",
        f"- 原始 radiomics 特征数：`{pipeline['num_raw_features']}`",
        f"- 方差过滤后特征数：`{pipeline['num_features_after_variance']}`",
        f"- 单变量预筛选后特征数：`{pipeline['num_prescreened_features']}`",
        f"- LASSO 最终保留特征数：`{pipeline['num_selected_features']}`",
        f"- 特征选择 fallback：`{pipeline['selector_fallback_info']}`",
        "",
        "## 6. 工程补充说明",
        "",
        "- PyRadiomics 和 sklearn 的 LASSO Logistic 回归均以 CPU 为主，GPU 不作为主加速路径。",
        "- Linux 多核服务器建议优先提高 `--n-jobs`，同时保持 `--itk-threads-per-worker=1`，避免线程过度竞争。",
        f"- 版本说明：`{args.feature_extractor_version_note}`",
        "",
    ]
    protocol_path = output_dir / "radiomics_lr_protocol.md"
    protocol_path.write_text("\n".join(lines), encoding="utf-8")


def export_run_config_yaml(args: argparse.Namespace, output_dir: Path) -> None:
    """保存当前运行配置。"""

    config_payload = {
        "model": MODEL_NAME,
        "split_base_root": args.split_base_root,
        "output_root": args.output_root,
        "checkpoint_root": args.checkpoint_root,
        "input": {
            "modalities": list(DEFAULT_MODALITIES),
            "voi_source": "conventional/voi",
            "shape_reference_modality": args.shape_reference_modality,
            "label_value": args.label_value,
        },
        "radiomics": {
            "resampled_spacing": args.resampled_spacing,
            "interpolator": args.interpolator,
            "normalize": args.normalize,
            "normalize_scale": args.normalize_scale,
            "remove_outliers": args.remove_outliers,
            "bin_width": args.bin_width,
            "pad_distance": args.pad_distance,
            "correct_mask_geometry": args.correct_mask_geometry,
            "feature_classes": list(DEFAULT_FEATURE_CLASSES),
            "image_types": ["Original"],
        },
        "modeling": {
            "variance_threshold": args.variance_threshold,
            "c_grid": list(getattr(args, "c_grid", DEFAULT_C_GRID)),
            "cv_scoring": args.cv_scoring,
            "cv_folds": args.cv_folds,
            "max_iter": args.max_iter,
            "use_class_weights": args.use_class_weights,
            "univariate_k": getattr(args, "univariate_k", None),
            "optimize_threshold": getattr(args, "optimize_threshold", True),
            "threshold": args.threshold,
            "feature_selection_fallback": args.feature_selection_fallback,
        },
        "runtime": {
            "n_jobs": args.n_jobs,
            "itk_threads_per_worker": args.itk_threads_per_worker,
            "seed": args.seed,
        },
    }
    save_yaml(config_payload, output_dir / "configs" / "radiomics_lr_base.yaml")


def write_run_summary(
    args: argparse.Namespace,
    case_map: Mapping[str, Sequence[PatientCase]],
    pipeline: Mapping[str, object],
    checkpoint_path: Path,
    output_dir: Path,
    run_id: str,
    eval_outputs: Mapping[str, Mapping[str, object]],
) -> None:
    """写入运行摘要 JSON。"""

    summary = {
        "model": MODEL_NAME,
        "run_id": run_id,
        "seed": args.seed,
        "checkpoint_path": str(checkpoint_path),
        "modalities": list(DEFAULT_MODALITIES),
        "voi_source": "conventional/voi",
        "shape_reference_modality": args.shape_reference_modality,
        "radiomics_settings": build_radiomics_settings(args),
        "cv_folds_actual": int(pipeline["cv_folds_actual"]),
        "selector_fallback_info": pipeline["selector_fallback_info"],
        "feature_summary": {
            "num_raw_features": int(pipeline["num_raw_features"]),
            "num_features_after_variance": int(pipeline["num_features_after_variance"]),
            "num_selected_features": int(pipeline["num_selected_features"]),
            "selected_feature_names": list(pipeline["selected_feature_names"]),
        },
        "datasets": {
            split_name: {
                "num_patients": len(case_map[split_name]),
                "wild_type": int(sum(case.label_id == 0 for case in case_map[split_name])),
                "mutant": int(sum(case.label_id == 1 for case in case_map[split_name])),
            }
            for split_name in DEFAULT_SPLITS
        },
        "metrics": {
            split_name: {
                **dict(output["metrics"]),
                "num_patients": int(output["num_patients"]),
                "num_features_before_selection": int(output["num_features_before_selection"]),
                "num_features_after_selection": int(output["num_features_after_selection"]),
                "selector_best_c": float(output["selector_best_c"]),
                "classifier_best_c": float(output["classifier_best_c"]),
            }
            for split_name, output in eval_outputs.items()
        },
    }
    save_json(summary, output_dir / f"run_summary_{MODEL_NAME}_{run_id}.json")


def aggregate_metric_rows(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    """按 split 汇总多次运行的指标。"""

    if not rows:
        return []

    numeric_fields = (
        "auc",
        "acc",
        "sen",
        "spe",
        "f1",
        "num_patients",
        "num_features_before_selection",
        "num_features_after_selection",
        "selector_best_c",
        "classifier_best_c",
    )
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["split"])].append(row)

    summary_rows: List[Dict[str, object]] = []
    for split_name in sorted(grouped):
        split_rows = grouped[split_name]
        summary_row: Dict[str, object] = {
            "split": split_name,
            "n_runs": len(split_rows),
        }
        for field in numeric_fields:
            values = np.asarray(
                [
                    float(item[field])
                    for item in split_rows
                    if item.get(field, "") not in {"", "nan", "NaN"}
                ],
                dtype=np.float64,
            )
            if values.size == 0:
                summary_row[f"{field}_mean"] = float("nan")
                summary_row[f"{field}_std"] = float("nan")
            else:
                summary_row[f"{field}_mean"] = float(np.mean(values))
                summary_row[f"{field}_std"] = float(np.std(values))
        summary_rows.append(summary_row)
    return summary_rows


def update_cross_run_metric_summaries(output_root: Path) -> None:
    """扫描输出目录下所有运行结果，更新跨运行汇总。"""

    if not output_root.is_dir():
        return

    all_rows: List[Dict[str, str]] = []
    for run_dir in sorted(output_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "checkpoints":
            continue
        for metrics_path in run_dir.glob(f"metrics_summary_{MODEL_NAME}_*.csv"):
            with metrics_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["source_file"] = str(metrics_path.relative_to(output_root))
                    all_rows.append(row)

    if not all_rows:
        return

    all_rows.sort(key=lambda item: (str(item.get("run_id", "")), str(item.get("split", ""))))
    save_rows(
        all_rows,
        output_root / f"metrics_{MODEL_NAME}_all_runs.csv",
        fieldnames=[
            "split",
            "auc",
            "acc",
            "sen",
            "spe",
            "f1",
            "num_patients",
            "num_features_before_selection",
            "num_features_after_selection",
            "selector_best_c",
            "classifier_best_c",
            "run_id",
            "checkpoint_name",
            "source_file",
        ],
    )

    summary_rows = aggregate_metric_rows(all_rows)
    save_rows(
        summary_rows,
        output_root / f"metrics_{MODEL_NAME}_summary.csv",
        fieldnames=[
            "split",
            "n_runs",
            "auc_mean",
            "auc_std",
            "acc_mean",
            "acc_std",
            "sen_mean",
            "sen_std",
            "spe_mean",
            "spe_std",
            "f1_mean",
            "f1_std",
            "num_patients_mean",
            "num_patients_std",
            "num_features_before_selection_mean",
            "num_features_before_selection_std",
            "num_features_after_selection_mean",
            "num_features_after_selection_std",
            "selector_best_c_mean",
            "selector_best_c_std",
            "classifier_best_c_mean",
            "classifier_best_c_std",
        ],
    )


def print_case_summary(case_map: Mapping[str, Sequence[PatientCase]]) -> None:
    """打印病例划分摘要。"""

    print("Split summary:")
    for split_name in DEFAULT_SPLITS:
        split_cases = list(case_map[split_name])
        wild_count = sum(case.label_id == 0 for case in split_cases)
        mut_count = sum(case.label_id == 1 for case in split_cases)
        print(
            f"  - {split_name}: total={len(split_cases)} | "
            f"wild_type={wild_count} | mutant={mut_count}"
        )


def main() -> None:
    """主入口。"""

    args = build_argparser().parse_args()
    set_seed(args.seed)

    run_id = resolve_run_id(args.run_id)
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    export_run_config_yaml(args, output_dir)

    if args.log_to_file:
        log_path = output_dir / f"training_log_{MODEL_NAME}_{run_id}.txt"
        with TeeLogger(log_path, mode="w"):
            _main_pipeline(args=args, run_id=run_id, output_dir=output_dir)
    else:
        _main_pipeline(args=args, run_id=run_id, output_dir=output_dir)


def _main_pipeline(args: argparse.Namespace, run_id: str, output_dir: Path) -> None:
    """实际的 radiomics 基线流程。"""

    checkpoint_dir = args.checkpoint_root / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    case_map = build_patient_cases(args.split_base_root)
    print_case_summary(case_map)

    feature_df = extract_all_features(case_map=case_map, args=args)
    print(
        f"Radiomics extraction finished: {feature_df.shape[0]} patients | "
        f"{feature_df.shape[1]} total columns"
    )

    pipeline = fit_radiomics_lasso_pipeline(feature_df=feature_df, args=args)
    print(
        f"Feature summary | raw={pipeline['num_raw_features']} | "
        f"after_variance={pipeline['num_features_after_variance']} | "
        f"prescreened={pipeline['num_prescreened_features']} | "
        f"selected={pipeline['num_selected_features']}"
    )
    print(
        f"Best C (LASSO classifier) = {np.asarray(pipeline['classifier_cv'].C_).ravel()[0]} | "
        f"optimal_threshold = {pipeline['optimal_threshold']:.4f} "
        f"(Youden optimized: {pipeline['threshold_optimized']})"
    )

    save_feature_tables(
        feature_df=feature_df,
        output_dir=output_dir,
        run_id=run_id,
        pipeline=pipeline,
        save_raw=args.save_feature_table,
        save_selected=args.save_selected_feature_table,
    )
    checkpoint_path = save_model_artifacts(pipeline=pipeline, checkpoint_dir=checkpoint_dir, run_id=run_id)

    eval_outputs: Dict[str, Dict[str, object]] = {}
    splits_to_export = DEFAULT_SPLITS if args.save_train_predictions else ("val", "test")
    for split_name in splits_to_export:
        split_df = feature_df[feature_df["split"] == split_name].reset_index(drop=True)
        eval_outputs[split_name] = predict_split(
            split_df=split_df,
            pipeline=pipeline,
            threshold=args.threshold,
            run_id=run_id,
            checkpoint_name=checkpoint_path.name,
        )

    export_prediction_tables(
        eval_outputs=eval_outputs,
        output_dir=output_dir,
        run_id=run_id,
        checkpoint_name=checkpoint_path.name,
        save_train_predictions=args.save_train_predictions,
    )
    export_wrong_cases(
        eval_outputs=eval_outputs,
        output_dir=output_dir,
        run_id=run_id,
        threshold=args.threshold,
        save_train_predictions=args.save_train_predictions,
    )
    export_evaluation_figures(
        eval_outputs=eval_outputs,
        output_dir=output_dir,
        run_id=run_id,
        save_train_predictions=args.save_train_predictions,
    )
    export_selected_features(
        pipeline=pipeline,
        output_dir=output_dir,
        run_id=run_id,
    )
    export_cv_tables(
        pipeline=pipeline,
        output_dir=output_dir,
        run_id=run_id,
    )
    export_scaler_stats(
        pipeline=pipeline,
        output_dir=output_dir,
    )
    export_protocol_markdown(
        args=args,
        case_map=case_map,
        pipeline=pipeline,
        output_dir=output_dir,
    )
    write_run_summary(
        args=args,
        case_map=case_map,
        pipeline=pipeline,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        run_id=run_id,
        eval_outputs=eval_outputs,
    )
    save_json(vars(args), output_dir / f"run_config_{MODEL_NAME}_{run_id}.json")
    update_cross_run_metric_summaries(args.output_root)

    print("Run completed.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Outputs   : {output_dir}")


if __name__ == "__main__":
    main()

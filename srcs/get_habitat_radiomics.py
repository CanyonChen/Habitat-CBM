#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 Habitat 的 H12 掩膜内提取影像组学（Radiomics）特征。

===============================================================================
一、脚本定位
===============================================================================
本脚本是 `build_habitat.py` 与 concept/CBM 数据构建之间的“桥接脚本”：

1. `build_habitat.py` 负责生成 `h1/h2/h3/h12` 掩膜；
2. 第一轮提取 `H1/H2/H3` 生境内 radiomics（单表输出）；
3. 第二轮提取 `H1+2(H12)` 生境内 radiomics；
4. 第三轮计算论文概念代理特征（8 项）。

注意：
- 当前脚本只做“特征提取 + 质控 + 结果落盘”，不负责 LASSO / LR 建模。

===============================================================================
二、输入假设
===============================================================================
1. 已有 `train / val / test` 的患者级划分目录（split-base-root）；
2. 已有 habitat 掩膜目录（habitat-mask-root），且包含：
   `<label>/<patient_id>/h1|h2|h3|h12.nii.gz`
   或 `<split>/<label>/<patient_id>/h1|h2|h3|h12.nii.gz`
3. 所有图像都已完成上游预处理（同空间或可通过最近邻重采样掩膜对齐）。

===============================================================================
三、输出内容
===============================================================================
默认输出根目录：`habitat_CBM/results/radiomics/`

每次运行会创建一个 run 目录，包含：
- `radiomics_features_h123_raw_{run_id}.csv`：H1/H2/H3 radiomics 特征表
- `radiomics_features_h12_raw_{run_id}.csv`：H12 radiomics 特征表
- `concept_proxy_features_{run_id}.csv`：concept 代理特征表
- `run_summary_habitat_multi_roi_radiomics_{run_id}.json`：运行摘要
- `run_config_habitat_multi_roi_radiomics_{run_id}.json`：配置快照
- `configs/habitat_multi_roi_radiomics_base.yaml`：YAML 配置
- `habitat_multi_roi_radiomics_protocol.md`：方法协议说明
- `lab_timeline.md`：实验时间表快照（若源文件存在）
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
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, TextIO, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - 依赖可能在 --help 环境中缺失
    pd = None  # type: ignore[assignment]

try:
    import SimpleITK as sitk
except Exception:  # pragma: no cover - 依赖可能在 --help 环境中缺失
    sitk = None  # type: ignore[assignment]

try:
    import yaml
except Exception:  # pragma: no cover - 依赖可能在 --help 环境中缺失
    yaml = None  # type: ignore[assignment]

try:
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover - 依赖可能在 --help 环境中缺失
    Parallel = None  # type: ignore[assignment]
    delayed = None  # type: ignore[assignment]

try:
    from radiomics import featureextractor
except Exception:  # pragma: no cover - 依赖可能在 --help 环境中缺失
    featureextractor = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - 依赖可能在 --help 环境中缺失
    tqdm = None  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# 全局常量区
# -----------------------------------------------------------------------------

SCRIPT_NAME = "get_habitat_radiomics.py"
MODEL_NAME = "habitat_multi_roi_radiomics"
DEFAULT_SEED = 42

# 与 baseline 保持一致的标签定义，便于后续直接拼接下游分析。
DEFAULT_LABEL_TO_ID = {"wild_type": 0, "mutant": 1}

# 默认 split 顺序与 baseline 对齐。
DEFAULT_SPLITS = ("train", "val", "test")

# 默认模态设置：与 baseline 相同（便于做 whole-tumor vs H12 的公平对比）。
DEFAULT_MODALITIES = ("t1", "t1ce", "t2", "t2flair")

# 支持的医学图像后缀（沿用项目内脚本风格）。
DEFAULT_ALLOWED_EXTENSIONS = (".nii", ".nii.gz", ".mha", ".mhd", ".nrrd")

# PyRadiomics 的默认特征类别（与 baseline 一致）。
DEFAULT_FEATURE_CLASSES = (
    "firstorder",
    "shape",
    "glcm",
    "glrlm",
    "glszm",
    "gldm",
    "ngtdm",
)

ALL_HABITAT_ROIS = ("h1", "h2", "h3", "h12")
ROUND1_HABITAT_ROIS = ("h1", "h2", "h3")
ROUND2_HABITAT_ROIS = ("h12",)
CONCEPT_REQUIRED_MODALITIES = ("t1ce", "t2flair", "adc", "cbf")

# 模态关键词，兼容项目中出现过的常见命名方式。
MODALITY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "t1": ("_t1", "t1.", "t1_", "t1wi", "t1w"),
    "t1ce": ("t1ce", "t1_ce", "ce_t1", "cet1", "t1c", "t1gd", "t1_gd"),
    "t2": ("_t2", "t2.", "t2_", "t2wi", "t2w"),
    "t2flair": ("t2flair", "t2_flair", "t2-flair", "flair"),
    "adc": ("adc",),
    "cbf": ("cbf",),
}

# 避免 T1/T2 误匹配到增强或 FLAIR 文件。
T1_EXCLUDE_KEYWORDS = MODALITY_KEYWORDS["t1ce"]
T2_EXCLUDE_KEYWORDS = MODALITY_KEYWORDS["t2flair"]

# habitat 掩膜候选关键词：优先匹配标准命名，兼容历史命名。
HABITAT_MASK_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "h1": ("h1", "habitat1"),
    "h2": ("h2", "habitat2"),
    "h3": ("h3", "habitat3"),
    "h12": ("h12", "h1+2", "h1_2", "h1-2"),
}
SUBREGION_MASK_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "ce": ("ce_voi", "t1ce_voi", "enhance", "enhancing", "ce_mask"),
    "flair": ("flair_voi", "flair_mask", "edema", "oedema", "t2flair_voi"),
}
MASK_HINT_KEYWORDS = ("mask", "voi", "seg", "label", "roi")

# SimpleITK 插值器映射（与 baseline 风格一致）。
SITK_INTERPOLATORS: Dict[str, int] = {
    "sitknearestneighbor": sitk.sitkNearestNeighbor if sitk is not None else 1,
    "sitklinear": sitk.sitkLinear if sitk is not None else 2,
    "sitkbspline": sitk.sitkBSpline if sitk is not None else 3,
}


# -----------------------------------------------------------------------------
# 数据结构
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PatientCase:
    """单个患者的最小提取单元。"""

    split: str
    patient_id: str
    label_name: str
    label_id: int
    modality_paths: Dict[str, Path]
    habitat_paths: Dict[str, Path]
    conventional_dir: Path
    functional_dir: Path


@dataclass(frozen=True)
class CaseExtractionResult:
    """单病例执行结果，统一承载成功与失败状态。"""

    ok: bool
    feature_row: Dict[str, object]
    qc_row: Dict[str, object]
    error_row: Optional[Dict[str, object]]


# -----------------------------------------------------------------------------
# 输出日志工具
# -----------------------------------------------------------------------------


class TeeLogger:
    """把 stdout/stderr 同步写入终端与日志文件。"""

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


# -----------------------------------------------------------------------------
# 参数解析与通用工具
# -----------------------------------------------------------------------------


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # habitat_CBM/
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]  # codex/


def resolve_default_split_root() -> Path:
    """按优先级选择默认 split 根目录。"""

    candidates = [
        # PROJECT_ROOT / "data" / "splited_data",
        PROJECT_ROOT / "dataset" / "splited_data",
        # PROJECT_ROOT / "splits",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_default_habitat_root() -> Path:
    """按优先级选择默认 habitat 掩膜目录。"""

    candidates = [
        PROJECT_ROOT / "dataset" / "habitat_masks",
        # PROJECT_ROOT / "data" / "habitat_masks",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def str2bool(value: str) -> bool:
    """把命令行字符串解析成布尔值。"""

    text = value.strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_spacing(value: str) -> Optional[Tuple[float, float, float]]:
    """解析重采样 spacing，支持 `none` 或 `a,b,c`。"""

    text = value.strip().lower()
    if text in {"none", "null", "off"}:
        return None
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "resampled-spacing must be 'none' or three comma separated floats."
        )
    try:
        spacing = tuple(float(item) for item in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid resampled-spacing value: {value}"
        ) from exc
    if any(item <= 0 for item in spacing):
        raise argparse.ArgumentTypeError("resampled-spacing values must be positive.")
    return spacing


def parse_csv_items(value: str) -> Tuple[str, ...]:
    """解析逗号分隔列表，去重并保序。"""

    parts = [item.strip() for item in value.split(",") if item.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Input list cannot be empty.")
    seen = set()
    dedup: List[str] = []
    for part in parts:
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(key)
    return tuple(dedup)


def parse_modalities(value: str) -> Tuple[str, ...]:
    """解析模态列表并校验合法性。"""

    modalities = parse_csv_items(value)
    unsupported = [item for item in modalities if item not in MODALITY_KEYWORDS]
    if unsupported:
        raise argparse.ArgumentTypeError(
            f"Unsupported modalities: {unsupported}. "
            f"Supported: {sorted(MODALITY_KEYWORDS)}"
        )
    return modalities


def parse_splits(value: str) -> Tuple[str, ...]:
    """解析 split 名称列表。"""

    return parse_csv_items(value)


def resolve_run_id(run_id: Optional[str]) -> str:
    """运行 ID：用户指定优先，否则采用时间戳。"""

    if run_id:
        return run_id
    return time.strftime("%Y%m%d_%H%M%S")


def make_json_safe(value: object) -> object:
    """递归把 Path / NumPy 标量等对象转换为 JSON 可序列化类型。"""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def save_json(data: Mapping[str, object], path: Path) -> None:
    """保存 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(make_json_safe(dict(data)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_yaml(data: Mapping[str, object], path: Path) -> None:
    """保存 YAML 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            make_json_safe(dict(data)),
            f,
            allow_unicode=True,
            sort_keys=False,
        )


def save_rows(rows: Iterable[Mapping[str, object]], path: Path, fieldnames: Sequence[str]) -> None:
    """保存 CSV 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def set_seed(seed: int) -> None:
    """固定 Python 与 NumPy 的随机种子。"""

    random.seed(seed)
    np.random.seed(seed)


def ensure_required_dependencies() -> None:
    """在真正运行前检查关键依赖。

    这样做的目的：
    1. `--help` 在缺依赖时依旧可用；
    2. 真正执行时给出清晰的安装提示，而不是在中途抛出模糊错误。
    """

    missing: List[str] = []
    if pd is None:
        missing.append("pandas")
    if sitk is None:
        missing.append("SimpleITK")
    if yaml is None:
        missing.append("pyyaml")
    if Parallel is None or delayed is None:
        missing.append("joblib")
    if featureextractor is None:
        missing.append("pyradiomics")
    if tqdm is None:
        missing.append("tqdm")

    if missing:
        joined = ", ".join(missing)
        raise ImportError(
            "Missing runtime dependencies for get_habitat_radiomics.py: "
            f"{joined}. "
            "Please install `habitat_CBM/repo/requirements_pyradiomics.txt`."
        )


def build_argparser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    cpu_count = os.cpu_count() or 1
    default_jobs = max(1, min(cpu_count - 1 if cpu_count > 1 else 1, 20))

    parser = argparse.ArgumentParser(
        description=(
            "Extract habitat radiomics in three rounds: "
            "H1/H2/H3, H12, and concept proxy features."
        )
    )
    parser.add_argument(
        "--split-base-root",
        type=Path,
        default=resolve_default_split_root(),
        help="包含 train/val/test 的 split 根目录。",
    )
    parser.add_argument(
        "--habitat-mask-root",
        type=Path,
        default=resolve_default_habitat_root(),
        help=(
            "Habitat 掩膜根目录，内部应为 "
            "<label>/<patient_id>/h1|h2|h3|h12.nii.gz "
            "或 <split>/<label>/<patient_id>/h1|h2|h3|h12.nii.gz。"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "habitat_radiomics",
        help="结果输出根目录。默认 habitat_CBM/dataset/habitat_radiomics。",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="运行 ID；不传则使用时间戳。",
    )
    parser.add_argument(
        "--splits",
        type=parse_splits,
        default=DEFAULT_SPLITS,
        help="要处理的 split 列表，逗号分隔，默认 train,val,test。",
    )
    parser.add_argument(
        "--modalities",
        type=parse_modalities,
        default=DEFAULT_MODALITIES,
        help=(
            "第一/二轮 radiomics 提取模态列表，逗号分隔，默认 t1,t1ce,t2,t2flair。"
            "第三轮 concept 会额外自动读取 t1ce,t2flair,adc,cbf。"
        ),
    )
    parser.add_argument(
        "--shape-reference-modality",
        type=str,
        default="t1ce",
        help="shape 特征仅从该模态提取一次，默认 t1ce。",
    )
    parser.add_argument(
        "--h12-label-value",
        type=int,
        default=1,
        help="habitat 掩膜中视为前景的标签值，默认 1。",
    )
    parser.add_argument(
        "--resampled-spacing",
        type=parse_spacing,
        default=(1.0, 1.0, 1.0),
        help="PyRadiomics 内部等体素重采样，默认 1,1,1；可传 none 关闭。",
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
        help="PyRadiomics binWidth，默认 25。",
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
        help="掩膜与影像几何不一致时是否自动重采样掩膜，默认 true。",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=default_jobs,
        help="并行进程数，默认按 CPU 自动估计。",
    )
    parser.add_argument(
        "--itk-threads-per-worker",
        type=int,
        default=1,
        help="每个 worker 内部 SimpleITK 线程数，默认 1。",
    )
    parser.add_argument(
        "--tiny-h12-threshold",
        type=int,
        default=30,
        help="任一 habitat ROI 体素数小于该值时标记 tiny flag，默认 30。",
    )
    parser.add_argument(
        "--skip-errors",
        type=str2bool,
        default=False,
        help="遇到失败病例时是否跳过继续，默认 false。",
    )
    parser.add_argument(
        "--save-feature-table",
        type=str2bool,
        default=True,
        help="是否保存特征表 CSV，默认 true。",
    )
    parser.add_argument(
        "--save-qc-table",
        type=str2bool,
        default=True,
        help="是否保存质控表 CSV，默认 true。",
    )
    parser.add_argument(
        "--copy-lab-timeline",
        type=str2bool,
        default=True,
        help="是否复制 lab_timeline.md 到结果目录，默认 true。",
    )
    parser.add_argument(
        "--lab-timeline-path",
        type=Path,
        default=WORKSPACE_ROOT / "lab_timeline.md",
        help="lab_timeline.md 的源路径。",
    )
    parser.add_argument(
        "--log-to-file",
        type=str2bool,
        default=True,
        help="是否把终端输出同步写入日志文件，默认 true。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="随机种子，默认 42。",
    )
    return parser


# -----------------------------------------------------------------------------
# 文件发现与病例构建
# -----------------------------------------------------------------------------


def normalize_name(name: str) -> str:
    """统一文件名格式，降低命名差异造成的匹配问题。"""

    return name.lower().replace("-", "_").replace(" ", "_")


def is_supported_image(path: Path) -> bool:
    """判断文件后缀是否属于支持的医学影像格式。"""

    lower_name = path.name.lower()
    return any(lower_name.endswith(ext) for ext in DEFAULT_ALLOWED_EXTENSIONS)


def collect_branch_image_files(branch_dir: Path) -> List[Path]:
    """递归收集目录下所有医学图像文件。"""

    return [
        path
        for path in branch_dir.rglob("*")
        if path.is_file() and is_supported_image(path)
    ]


def match_modality(path: Path, modality: str) -> bool:
    """按关键词匹配目标模态。"""

    name = normalize_name(path.name)
    keywords = MODALITY_KEYWORDS[modality]

    if modality == "t1" and any(item in name for item in T1_EXCLUDE_KEYWORDS):
        return False
    if modality == "t2" and any(item in name for item in T2_EXCLUDE_KEYWORDS):
        return False
    return any(keyword in name for keyword in keywords)


def score_habitat_candidate(path: Path, roi: str) -> int:
    """给目标 habitat ROI 候选打分。"""

    if roi not in HABITAT_MASK_KEYWORDS:
        raise ValueError(f"Unsupported habitat roi: {roi}")

    name = normalize_name(path.name)

    if roi == "h12":
        if name.startswith("h12.") or name.startswith("h12_"):
            return 1000
    else:
        if name.startswith(f"{roi}.") or name.startswith(f"{roi}_"):
            return 1000
        # 避免 H1 误匹配到 H12。
        if roi == "h1" and any(token in name for token in HABITAT_MASK_KEYWORDS["h12"]):
            return 0

    score = 0
    for rank, keyword in enumerate(HABITAT_MASK_KEYWORDS[roi]):
        if keyword in name:
            score += 100 - rank
    return score


def discover_modality_file(patient_dir: Path, modality: str) -> Path:
    """在患者目录下发现唯一模态文件。"""

    all_files = collect_branch_image_files(patient_dir)
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
    return matches[0]


def discover_habitat_file(habitat_patient_dir: Path, roi: str) -> Path:
    """在 habitat 掩膜目录下发现唯一 ROI 文件。"""

    if roi not in ALL_HABITAT_ROIS:
        raise ValueError(f"Unsupported habitat roi: {roi}")
    if not habitat_patient_dir.is_dir():
        raise FileNotFoundError(f"Missing habitat patient directory: {habitat_patient_dir}")

    canonical = habitat_patient_dir / f"{roi}.nii.gz"
    if canonical.is_file():
        return canonical

    all_files = collect_branch_image_files(habitat_patient_dir)
    matches = [path for path in all_files if score_habitat_candidate(path=path, roi=roi) > 0]
    if len(matches) == 0:
        raise FileNotFoundError(f"Cannot find '{roi}' mask under: {habitat_patient_dir}")
    if len(matches) == 1:
        return matches[0]

    scored = sorted(
        matches,
        key=lambda path: score_habitat_candidate(path=path, roi=roi),
        reverse=True,
    )
    best_score = score_habitat_candidate(path=scored[0], roi=roi)
    best_matches = [path for path in scored if score_habitat_candidate(path=path, roi=roi) == best_score]
    if len(best_matches) > 1:
        match_str = "\n".join(str(path) for path in best_matches[:10])
        raise ValueError(
            f"Found multiple equally plausible '{roi}' masks under {habitat_patient_dir}.\n{match_str}"
        )
    return best_matches[0]


def resolve_habitat_patient_dir(
    habitat_mask_root: Path,
    split_name: str,
    label_name: str,
    patient_id: str,
) -> Path:
    """兼容两种 habitat 目录布局：
    1) <root>/<label>/<patient_id>
    2) <root>/<split>/<label>/<patient_id>
    """

    split_layout = habitat_mask_root / split_name / label_name / patient_id
    flat_layout = habitat_mask_root / label_name / patient_id
    if split_layout.is_dir():
        return split_layout
    if flat_layout.is_dir():
        return flat_layout
    raise FileNotFoundError(
        "Missing habitat patient directory. Tried:\n"
        f"- {split_layout}\n"
        f"- {flat_layout}"
    )


def modality_to_branch(modality: str) -> str:
    """根据模态推断其默认所在分支。"""

    if modality in {"adc", "cbf"}:
        return "functional"
    return "conventional"


def collect_anchor_patient_ids(split_root: Path, anchor_branch: str, label_name: str) -> List[str]:
    """收集某个 split+label 下 anchor 分支里的患者 ID。"""

    anchor_label_dir = split_root / anchor_branch / label_name
    if not anchor_label_dir.is_dir():
        raise FileNotFoundError(f"Missing label directory: {anchor_label_dir}")

    patient_ids: List[str] = []
    for path in sorted(anchor_label_dir.iterdir()):
        if path.is_dir():
            patient_ids.append(path.name)
    return patient_ids


def validate_split_roots(split_base_root: Path, splits: Sequence[str]) -> None:
    """确认 split 根目录结构至少包含所需 split 子目录。"""

    if not split_base_root.is_dir():
        raise FileNotFoundError(f"Split base root does not exist: {split_base_root}")
    for split_name in splits:
        split_dir = split_base_root / split_name
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")


def build_patient_cases(
    split_base_root: Path,
    habitat_mask_root: Path,
    splits: Sequence[str],
    modalities: Sequence[str],
) -> Dict[str, List[PatientCase]]:
    """建立 train/val/test 的病例索引。"""

    validate_split_roots(split_base_root=split_base_root, splits=splits)

    if not modalities:
        raise ValueError("At least one modality is required.")

    required_modalities = list(dict.fromkeys([*modalities, *CONCEPT_REQUIRED_MODALITIES]))

    # 使用首个模态对应分支作为患者锚点。
    anchor_branch = modality_to_branch(required_modalities[0])
    case_map: Dict[str, List[PatientCase]] = {split: [] for split in splits}

    for split_name in splits:
        split_root = split_base_root / split_name
        split_cases: List[PatientCase] = []

        for label_name, label_id in DEFAULT_LABEL_TO_ID.items():
            patient_ids = collect_anchor_patient_ids(
                split_root=split_root,
                anchor_branch=anchor_branch,
                label_name=label_name,
            )

            for patient_id in patient_ids:
                modality_paths: Dict[str, Path] = {}
                for modality in required_modalities:
                    branch = modality_to_branch(modality)
                    patient_dir = split_root / branch / label_name / patient_id
                    if not patient_dir.is_dir():
                        raise FileNotFoundError(
                            f"Missing patient directory for modality '{modality}': {patient_dir}"
                        )
                    modality_paths[modality] = discover_modality_file(
                        patient_dir=patient_dir,
                        modality=modality,
                    )

                habitat_patient_dir = resolve_habitat_patient_dir(
                    habitat_mask_root=habitat_mask_root,
                    split_name=split_name,
                    label_name=label_name,
                    patient_id=patient_id,
                )
                habitat_paths = {
                    roi: discover_habitat_file(habitat_patient_dir=habitat_patient_dir, roi=roi)
                    for roi in ALL_HABITAT_ROIS
                }

                split_cases.append(
                    PatientCase(
                        split=split_name,
                        patient_id=patient_id,
                        label_name=label_name,
                        label_id=label_id,
                        modality_paths=modality_paths,
                        habitat_paths=habitat_paths,
                        conventional_dir=split_root / "conventional" / label_name / patient_id,
                        functional_dir=split_root / "functional" / label_name / patient_id,
                    )
                )

        split_cases = sorted(split_cases, key=lambda item: item.patient_id)
        case_map[split_name] = split_cases

    return case_map


# -----------------------------------------------------------------------------
# Radiomics 配置与提取核心
# -----------------------------------------------------------------------------


def resolve_interpolator(name: str) -> int:
    """将字符串插值器名映射到 SimpleITK 常量。"""

    key = name.strip().lower()
    if key not in SITK_INTERPOLATORS:
        raise ValueError(f"Unsupported interpolator: {name}")
    return SITK_INTERPOLATORS[key]


def ensure_binary_mask(mask_image: sitk.Image, label_value: int) -> sitk.Image:
    """把掩膜转为 0/1 二值图。"""

    mask = sitk.Cast(mask_image, sitk.sitkUInt16)
    return sitk.BinaryThreshold(
        mask,
        lowerThreshold=label_value,
        upperThreshold=label_value,
        insideValue=1,
        outsideValue=0,
    )


def ensure_positive_mask(mask_image: sitk.Image) -> sitk.Image:
    """把任意非零标签掩膜转为 0/1。"""

    mask = sitk.Cast(mask_image, sitk.sitkUInt16)
    return sitk.BinaryThreshold(
        mask,
        lowerThreshold=1,
        upperThreshold=65535,
        insideValue=1,
        outsideValue=0,
    )


def maybe_resample_mask_to_image(
    image: sitk.Image,
    mask: sitk.Image,
    correct_geometry: bool,
) -> sitk.Image:
    """必要时把掩膜重采样到目标影像空间。"""

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
            "Image and H12 mask geometry mismatch. "
            "Enable --correct-mask-geometry true or fix upstream preprocessing."
        )

    # 掩膜重采样必须使用最近邻，避免标签边界被平滑污染。
    return sitk.Resample(
        mask,
        image,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


def build_radiomics_settings(args: argparse.Namespace) -> Dict[str, object]:
    """整理 PyRadiomics 参数字典。"""

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


def create_radiomics_extractor(
    settings: Mapping[str, object],
    include_shape: bool,
) -> featureextractor.RadiomicsFeatureExtractor:
    """创建并配置 PyRadiomics 提取器。"""

    extractor = featureextractor.RadiomicsFeatureExtractor(**dict(settings))
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.disableAllFeatures()

    # 与 baseline 保持一致：只启用原始图像上的基础特征组。
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
    """将 PyRadiomics 输出转为干净浮点数。"""

    try:
        numeric = float(value)
    except Exception:
        return float("nan")
    if not np.isfinite(numeric):
        return float("nan")
    return numeric


def format_feature_name(modality: str, raw_key: str) -> str:
    """统一特征列名风格。"""

    if raw_key.startswith("original_shape_"):
        return f"shape_{raw_key.removeprefix('original_shape_')}"
    if raw_key.startswith("original_"):
        return f"{modality}_{raw_key.removeprefix('original_')}"
    return f"{modality}_{raw_key}"


def add_roi_prefix(roi: str, feature_name: str) -> str:
    """按约定把 ROI 前缀加到特征名上。"""

    return f"{roi}_{feature_name}"


def build_union_mask(reference_mask: sitk.Image, masks: Sequence[sitk.Image]) -> sitk.Image:
    """合并多个二值掩膜（并集）。"""

    if not masks:
        raise ValueError("At least one mask is required for union.")

    union_array = np.zeros_like(
        np.asarray(sitk.GetArrayViewFromImage(reference_mask), dtype=np.uint8),
        dtype=np.uint8,
    )
    for mask in masks:
        union_array = np.maximum(
            union_array,
            np.asarray(sitk.GetArrayViewFromImage(mask) > 0, dtype=np.uint8),
        )
    union_mask = sitk.GetImageFromArray(union_array)
    union_mask.CopyInformation(reference_mask)
    return sitk.Cast(union_mask, sitk.sitkUInt8)


def extract_values_in_aligned_mask(image: sitk.Image, mask: sitk.Image) -> np.ndarray:
    """返回影像在掩膜内的体素值。"""

    image_array = np.asarray(sitk.GetArrayViewFromImage(image), dtype=np.float32)
    mask_array = np.asarray(sitk.GetArrayViewFromImage(mask) > 0, dtype=np.uint8)
    return image_array[mask_array > 0]


def count_positive_voxels(mask: sitk.Image) -> int:
    """统计二值掩膜前景体素数。"""

    return int(np.sum(np.asarray(sitk.GetArrayViewFromImage(mask) > 0, dtype=np.uint8)))


def mask_physical_volume_cm3(mask: sitk.Image) -> float:
    """根据掩膜前景体素数和 spacing 计算物理体积，单位 cm^3。"""

    voxel_count = count_positive_voxels(mask)
    spacing = np.asarray(mask.GetSpacing(), dtype=np.float64)
    voxel_volume_mm3 = float(np.prod(spacing))
    if not np.isfinite(voxel_volume_mm3) or voxel_volume_mm3 <= 0:
        raise ValueError(f"Invalid mask spacing for physical volume: {tuple(mask.GetSpacing())}")
    return float(voxel_count * voxel_volume_mm3 / 1000.0)


def safe_percentile(values: np.ndarray, q: float) -> float:
    """安全计算百分位，空输入时返回 NaN。"""

    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def score_optional_subregion_mask(path: Path, region: str) -> int:
    """给 CE/FLAIR 子区掩膜候选打分。"""

    if region not in SUBREGION_MASK_KEYWORDS:
        raise ValueError(f"Unsupported region: {region}")

    name = normalize_name(path.name)
    if not any(hint in name for hint in MASK_HINT_KEYWORDS):
        return 0

    score = 0
    for rank, keyword in enumerate(SUBREGION_MASK_KEYWORDS[region]):
        if keyword in name:
            score += 100 - rank
    return score


def discover_optional_subregion_mask(patient_dir: Path, region: str) -> Optional[Path]:
    """在患者目录下尝试发现 CE/FLAIR 子区掩膜。"""

    if not patient_dir.is_dir():
        return None

    all_files = collect_branch_image_files(patient_dir)
    matches = [
        path
        for path in all_files
        if score_optional_subregion_mask(path=path, region=region) > 0
    ]
    if not matches:
        return None

    scored = sorted(
        matches,
        key=lambda path: score_optional_subregion_mask(path=path, region=region),
        reverse=True,
    )
    best_score = score_optional_subregion_mask(path=scored[0], region=region)
    best_matches = [
        path
        for path in scored
        if score_optional_subregion_mask(path=path, region=region) == best_score
    ]
    if len(best_matches) > 1:
        # 可选输入不阻断流程，留给上层使用 fallback。
        return None
    return best_matches[0]


def estimate_high_intensity_volume(
    image: sitk.Image,
    mask: sitk.Image,
    quantile: float = 75.0,
) -> int:
    """在掩膜内用高强度分位阈值估计子区体积（体素数）。"""

    values = extract_values_in_aligned_mask(image=image, mask=mask)
    if values.size == 0:
        return 0
    threshold = float(np.percentile(values, quantile))
    return int(np.sum(values >= threshold))


def extract_patient_features(
    case: PatientCase,
    rois: Sequence[str],
    modalities: Sequence[str],
    settings: Mapping[str, object],
    shape_reference_modality: str,
    mask_label_value: int,
    correct_mask_geometry: bool,
    itk_threads_per_worker: int,
    tiny_voxel_threshold: int,
) -> CaseExtractionResult:
    """对单个患者执行多 ROI radiomics 提取。"""

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(max(1, itk_threads_per_worker))

    feature_row: Dict[str, object] = {
        "patient_id": case.patient_id,
        "split": case.split,
        "y_true": case.label_id,
        "label_name": case.label_name,
    }
    for modality in modalities:
        feature_row[f"{modality}_path"] = str(case.modality_paths[modality])
    for roi in rois:
        feature_row[f"{roi}_path"] = str(case.habitat_paths[roi])

    qc_row: Dict[str, object] = {
        "patient_id": case.patient_id,
        "split": case.split,
        "label_name": case.label_name,
        "y_true": case.label_id,
        "rois": ",".join(rois),
        "modalities": ",".join(modalities),
        "shape_reference_modality": shape_reference_modality,
        "status": "success",
        "error": "",
    }

    try:
        for roi in rois:
            roi_path = case.habitat_paths[roi]
            raw_mask = sitk.ReadImage(str(roi_path))
            binary_mask = ensure_binary_mask(mask_image=raw_mask, label_value=mask_label_value)
            raw_voxels = int(np.sum(np.asarray(sitk.GetArrayViewFromImage(binary_mask) > 0, dtype=np.uint8)))
            qc_row[f"{roi}_voxels_raw"] = raw_voxels
            qc_row[f"{roi}_tiny_flag"] = int(raw_voxels < tiny_voxel_threshold)
            if raw_voxels <= 0:
                raise ValueError(f"Empty '{roi}' mask for patient {case.patient_id}.")

            aligned_voxel_counts: List[int] = []
            geometry_mismatch_count = 0

            for modality in modalities:
                image_path = case.modality_paths[modality]
                image = sitk.ReadImage(str(image_path))

                same_geometry_before = (
                    image.GetSize() == binary_mask.GetSize()
                    and np.allclose(image.GetSpacing(), binary_mask.GetSpacing())
                    and np.allclose(image.GetOrigin(), binary_mask.GetOrigin())
                    and np.allclose(image.GetDirection(), binary_mask.GetDirection())
                )
                if not same_geometry_before:
                    geometry_mismatch_count += 1

                aligned_mask = maybe_resample_mask_to_image(
                    image=image,
                    mask=binary_mask,
                    correct_geometry=correct_mask_geometry,
                )
                aligned_voxels = int(
                    np.sum(np.asarray(sitk.GetArrayViewFromImage(aligned_mask) > 0, dtype=np.uint8))
                )
                aligned_voxel_counts.append(aligned_voxels)
                if aligned_voxels <= 0:
                    raise ValueError(
                        f"Empty '{roi}' after alignment for patient {case.patient_id}, modality {modality}."
                    )

                extractor = create_radiomics_extractor(
                    settings=settings,
                    include_shape=(modality == shape_reference_modality),
                )
                feature_dict = extractor.execute(image, aligned_mask, label=1)
                for raw_key, value in feature_dict.items():
                    if not raw_key.startswith("original_"):
                        continue
                    if raw_key.startswith("original_shape_") and modality != shape_reference_modality:
                        continue
                    base_feature_name = format_feature_name(modality=modality, raw_key=raw_key)
                    feature_name = add_roi_prefix(roi=roi, feature_name=base_feature_name)
                    feature_row[feature_name] = sanitize_feature_value(value)

            qc_row[f"{roi}_geometry_mismatch_modalities"] = int(geometry_mismatch_count)
            qc_row[f"{roi}_voxels_min_aligned"] = int(np.min(aligned_voxel_counts))
            qc_row[f"{roi}_voxels_max_aligned"] = int(np.max(aligned_voxel_counts))
            qc_row[f"{roi}_voxels_mean_aligned"] = float(np.mean(aligned_voxel_counts))

        return CaseExtractionResult(ok=True, feature_row=feature_row, qc_row=qc_row, error_row=None)

    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        qc_row["status"] = "failed"
        qc_row["error"] = error_message
        error_row = {
            "patient_id": case.patient_id,
            "split": case.split,
            "label_name": case.label_name,
            "y_true": case.label_id,
            "error": error_message,
            "traceback": traceback.format_exc(limit=5),
        }
        for roi in rois:
            error_row[f"{roi}_path"] = str(case.habitat_paths[roi])
        return CaseExtractionResult(ok=False, feature_row=feature_row, qc_row=qc_row, error_row=error_row)


def extract_patient_concept_proxies(
    case: PatientCase,
    settings: Mapping[str, object],
    mask_label_value: int,
    correct_mask_geometry: bool,
    itk_threads_per_worker: int,
) -> CaseExtractionResult:
    """提取单病例的 8 个 concept 代理特征。"""

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(max(1, itk_threads_per_worker))

    feature_row: Dict[str, object] = {
        "patient_id": case.patient_id,
        "split": case.split,
        "y_true": case.label_id,
        "label_name": case.label_name,
        "h1_path": str(case.habitat_paths["h1"]),
        "h2_path": str(case.habitat_paths["h2"]),
        "h3_path": str(case.habitat_paths["h3"]),
        "h12_path": str(case.habitat_paths["h12"]),
        "t1ce_path": str(case.modality_paths["t1ce"]),
        "t2flair_path": str(case.modality_paths["t2flair"]),
        "adc_path": str(case.modality_paths["adc"]),
        "cbf_path": str(case.modality_paths["cbf"]),
    }
    qc_row: Dict[str, object] = {
        "patient_id": case.patient_id,
        "split": case.split,
        "label_name": case.label_name,
        "status": "success",
        "error": "",
    }

    try:
        h1_mask = ensure_binary_mask(sitk.ReadImage(str(case.habitat_paths["h1"])), mask_label_value)
        h2_mask = ensure_binary_mask(sitk.ReadImage(str(case.habitat_paths["h2"])), mask_label_value)
        h3_mask = ensure_binary_mask(sitk.ReadImage(str(case.habitat_paths["h3"])), mask_label_value)
        h12_mask = ensure_binary_mask(sitk.ReadImage(str(case.habitat_paths["h12"])), mask_label_value)

        h23_mask = build_union_mask(reference_mask=h2_mask, masks=[h2_mask, h3_mask])
        whole_tumor_mask = build_union_mask(reference_mask=h1_mask, masks=[h1_mask, h2_mask, h3_mask])

        h1_voxels = count_positive_voxels(h1_mask)
        h2_voxels = count_positive_voxels(h2_mask)
        h3_voxels = count_positive_voxels(h3_mask)
        whole_voxels = count_positive_voxels(whole_tumor_mask)

        feature_row["h1_volume_voxels"] = h1_voxels
        feature_row["h2_volume_voxels"] = h2_voxels
        feature_row["h3_volume_voxels"] = h3_voxels
        feature_row["whole_tumor_volume_voxels"] = whole_voxels

        if whole_voxels <= 0:
            raise ValueError(f"Empty whole-tumor mask for patient {case.patient_id}.")

        t1ce_image = sitk.ReadImage(str(case.modality_paths["t1ce"]))
        t2flair_image = sitk.ReadImage(str(case.modality_paths["t2flair"]))
        adc_image = sitk.ReadImage(str(case.modality_paths["adc"]))
        cbf_image = sitk.ReadImage(str(case.modality_paths["cbf"]))

        h1_t1ce_mask = maybe_resample_mask_to_image(t1ce_image, h1_mask, correct_mask_geometry)
        h23_t1ce_mask = maybe_resample_mask_to_image(t1ce_image, h23_mask, correct_mask_geometry)
        whole_t1ce_mask = maybe_resample_mask_to_image(t1ce_image, whole_tumor_mask, correct_mask_geometry)
        whole_t2flair_mask = maybe_resample_mask_to_image(t2flair_image, whole_tumor_mask, correct_mask_geometry)
        h12_adc_mask = maybe_resample_mask_to_image(adc_image, h12_mask, correct_mask_geometry)
        h1_cbf_mask = maybe_resample_mask_to_image(cbf_image, h1_mask, correct_mask_geometry)

        c1_values = extract_values_in_aligned_mask(image=t1ce_image, mask=h1_t1ce_mask)
        c2_values = extract_values_in_aligned_mask(image=t1ce_image, mask=h23_t1ce_mask)
        c5_values = extract_values_in_aligned_mask(image=adc_image, mask=h12_adc_mask)
        c6_values = extract_values_in_aligned_mask(image=cbf_image, mask=h1_cbf_mask)

        if c1_values.size <= 0 or c2_values.size <= 0 or c5_values.size <= 0 or c6_values.size <= 0:
            raise ValueError(f"Empty aligned ROI while computing concepts for patient {case.patient_id}.")

        feature_row["c1_h1_t1ce_firstorder_mean"] = float(np.mean(c1_values))
        feature_row["c2_h23_t1ce_firstorder_mean"] = float(np.mean(c2_values))
        feature_row["c5_h12_adc_10percentile"] = safe_percentile(c5_values, 10.0)
        feature_row["c6_h1_cbf_95percentile"] = safe_percentile(c6_values, 95.0)
        feature_row["c7_h1_volume_ratio"] = float(h1_voxels / whole_voxels)
        feature_row["c8_h2_volume_ratio"] = float(h2_voxels / whole_voxels)

        shape_extractor = create_radiomics_extractor(
            settings=settings,
            include_shape=True,
        )
        shape_dict = shape_extractor.execute(t1ce_image, whole_t1ce_mask, label=1)
        feature_row["c3_whole_tumor_shape_sphericity"] = sanitize_feature_value(
            shape_dict.get("original_shape_Sphericity", float("nan"))
        )

        c4_voxels = count_positive_voxels(whole_t2flair_mask)
        if c4_voxels <= 0:
            raise ValueError(f"Empty T2-FLAIR VOI while computing C4 for patient {case.patient_id}.")
        c4_volume_cm3 = mask_physical_volume_cm3(whole_t2flair_mask)
        feature_row["c4_estimation_method"] = "t2flair_voi_physical_volume"
        feature_row["c4_t2flair_voi_volume_voxels"] = int(c4_voxels)
        feature_row["c4_t2flair_voi_volume_cm3"] = c4_volume_cm3
        feature_row["c4_t2flair_voi_volume_log1p_cm3"] = float(np.log1p(c4_volume_cm3))

        return CaseExtractionResult(ok=True, feature_row=feature_row, qc_row=qc_row, error_row=None)

    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        qc_row["status"] = "failed"
        qc_row["error"] = error_message
        error_row = {
            "patient_id": case.patient_id,
            "split": case.split,
            "label_name": case.label_name,
            "y_true": case.label_id,
            "error": error_message,
            "traceback": traceback.format_exc(limit=5),
        }
        return CaseExtractionResult(ok=False, feature_row=feature_row, qc_row=qc_row, error_row=error_row)


def extract_all_features(
    case_map: Mapping[str, Sequence[PatientCase]],
    args: argparse.Namespace,
    rois: Sequence[str],
    modalities: Sequence[str],
    progress_desc: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """并行提取指定 ROI 组合的 radiomics 特征。"""

    all_cases: List[PatientCase] = []
    for split_name in args.splits:
        all_cases.extend(case_map[split_name])
    if not all_cases:
        raise RuntimeError("No patient cases found for extraction.")

    settings = build_radiomics_settings(args)
    print(
        f"Radiomics extraction started ({progress_desc}): {len(all_cases)} patients | "
        f"n_jobs={args.n_jobs} | itk_threads_per_worker={args.itk_threads_per_worker}"
    )

    iterator = tqdm(all_cases, desc=progress_desc, total=len(all_cases)) if tqdm is not None else all_cases
    if args.n_jobs == 1:
        results = [
            extract_patient_features(
                case=case,
                rois=rois,
                modalities=modalities,
                settings=settings,
                shape_reference_modality=args.shape_reference_modality,
                mask_label_value=args.h12_label_value,
                correct_mask_geometry=args.correct_mask_geometry,
                itk_threads_per_worker=args.itk_threads_per_worker,
                tiny_voxel_threshold=args.tiny_h12_threshold,
            )
            for case in iterator
        ]
    else:
        results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=0)(
            delayed(extract_patient_features)(
                case=case,
                rois=rois,
                modalities=modalities,
                settings=settings,
                shape_reference_modality=args.shape_reference_modality,
                mask_label_value=args.h12_label_value,
                correct_mask_geometry=args.correct_mask_geometry,
                itk_threads_per_worker=args.itk_threads_per_worker,
                tiny_voxel_threshold=args.tiny_h12_threshold,
            )
            for case in iterator
        )

    feature_rows = [result.feature_row for result in results if result.ok]
    qc_rows = [result.qc_row for result in results]
    error_rows = [result.error_row for result in results if (not result.ok and result.error_row is not None)]

    feature_df = pd.DataFrame(feature_rows)
    if not feature_df.empty:
        feature_df = feature_df.sort_values(["split", "patient_id"]).reset_index(drop=True)

    qc_df = pd.DataFrame(qc_rows)
    if not qc_df.empty:
        qc_df = qc_df.sort_values(["split", "patient_id"]).reset_index(drop=True)

    failed_df = pd.DataFrame(error_rows)
    if not failed_df.empty:
        failed_df = failed_df.sort_values(["split", "patient_id"]).reset_index(drop=True)
    return feature_df, qc_df, failed_df


def extract_all_concept_proxies(
    case_map: Mapping[str, Sequence[PatientCase]],
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """并行提取全部病例的 concept 代理特征。"""

    all_cases: List[PatientCase] = []
    for split_name in args.splits:
        all_cases.extend(case_map[split_name])
    if not all_cases:
        raise RuntimeError("No patient cases found for concept extraction.")

    settings = build_radiomics_settings(args)
    print(
        f"Concept proxy extraction started: {len(all_cases)} patients | "
        f"n_jobs={args.n_jobs} | itk_threads_per_worker={args.itk_threads_per_worker}"
    )
    iterator = tqdm(all_cases, desc="extract-concept-proxies", total=len(all_cases)) if tqdm is not None else all_cases
    if args.n_jobs == 1:
        results = [
            extract_patient_concept_proxies(
                case=case,
                settings=settings,
                mask_label_value=args.h12_label_value,
                correct_mask_geometry=args.correct_mask_geometry,
                itk_threads_per_worker=args.itk_threads_per_worker,
            )
            for case in iterator
        ]
    else:
        results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=0)(
            delayed(extract_patient_concept_proxies)(
                case=case,
                settings=settings,
                mask_label_value=args.h12_label_value,
                correct_mask_geometry=args.correct_mask_geometry,
                itk_threads_per_worker=args.itk_threads_per_worker,
            )
            for case in iterator
        )

    feature_rows = [result.feature_row for result in results if result.ok]
    qc_rows = [result.qc_row for result in results]
    error_rows = [result.error_row for result in results if (not result.ok and result.error_row is not None)]

    feature_df = pd.DataFrame(feature_rows)
    if not feature_df.empty:
        feature_df = feature_df.sort_values(["split", "patient_id"]).reset_index(drop=True)

    qc_df = pd.DataFrame(qc_rows)
    if not qc_df.empty:
        qc_df = qc_df.sort_values(["split", "patient_id"]).reset_index(drop=True)

    failed_df = pd.DataFrame(error_rows)
    if not failed_df.empty:
        failed_df = failed_df.sort_values(["split", "patient_id"]).reset_index(drop=True)
    return feature_df, qc_df, failed_df


# -----------------------------------------------------------------------------
# 输出与汇总
# -----------------------------------------------------------------------------


def print_case_summary(case_map: Mapping[str, Sequence[PatientCase]], splits: Sequence[str]) -> None:
    """打印 split 级病例统计。"""

    print("Split summary:")
    for split_name in splits:
        split_cases = list(case_map[split_name])
        wild_count = sum(case.label_id == 0 for case in split_cases)
        mut_count = sum(case.label_id == 1 for case in split_cases)
        print(
            f"  - {split_name}: total={len(split_cases)} | "
            f"wild_type={wild_count} | mutant={mut_count}"
        )


def copy_lab_timeline_if_needed(
    output_dir: Path,
    copy_enabled: bool,
    lab_timeline_path: Path,
) -> Dict[str, object]:
    """按配置复制 lab_timeline.md 到输出目录。"""

    status = {
        "copy_enabled": bool(copy_enabled),
        "source_path": str(lab_timeline_path),
        "copied": False,
        "target_path": "",
        "note": "",
    }

    if not copy_enabled:
        status["note"] = "copy disabled by --copy-lab-timeline false"
        return status

    if not lab_timeline_path.is_file():
        status["note"] = f"source file not found: {lab_timeline_path}"
        return status

    target_path = output_dir / "lab_timeline.md"
    shutil.copy2(lab_timeline_path, target_path)
    status["copied"] = True
    status["target_path"] = str(target_path)
    status["note"] = "copied successfully"
    return status


def count_radiomics_feature_columns(feature_df: pd.DataFrame) -> int:
    """统计表中 radiomics 特征列数量（排除元信息列）。"""

    if feature_df.empty:
        return 0
    metadata_cols = {"patient_id", "split", "y_true", "label_name"}
    count = 0
    for col in feature_df.columns:
        if col in metadata_cols or col.endswith("_path"):
            continue
        count += 1
    return int(count)


def export_protocol_markdown(
    args: argparse.Namespace,
    output_dir: Path,
    case_map: Mapping[str, Sequence[PatientCase]],
    h123_feature_df: pd.DataFrame,
    h123_failed_df: pd.DataFrame,
    h12_feature_df: pd.DataFrame,
    h12_failed_df: pd.DataFrame,
    concept_feature_df: pd.DataFrame,
    concept_failed_df: pd.DataFrame,
) -> Path:
    """导出多轮提取协议说明。"""

    lines = [
        "# Habitat Radiomics + Concept Proxy 协议说明",
        "",
        f"- 运行时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 脚本：`{SCRIPT_NAME}`",
        f"- 模型标识：`{MODEL_NAME}`",
        f"- split 根目录：`{args.split_base_root}`",
        f"- habitat 掩膜根目录：`{args.habitat_mask_root}`",
        f"- 输出根目录：`{args.output_root}`",
        "",
        "## 1. 提取轮次",
        "",
        "- 第一轮：H1/H2/H3 生境内 radiomics，输出单表。",
        "- 第二轮：H1+2（H12）生境内 radiomics，输出单表。",
        "- 第三轮：8 个 concept 代理特征提取与计算。",
        "",
        "## 2. 命名规则",
        "",
        "- 特征命名：`{roi}_{modality}_{feature}`。",
        "- 跨四模态共享特征（如 shape）：`{roi}_{feature}`。",
        "",
        "## 3. 数据规模",
        "",
        *[
            f"- `{split_name}`: {len(case_map[split_name])} 例"
            for split_name in args.splits
        ],
        f"- H123 成功/失败：`{int(h123_feature_df.shape[0])}` / `{int(h123_failed_df.shape[0])}`",
        f"- H12 成功/失败：`{int(h12_feature_df.shape[0])}` / `{int(h12_failed_df.shape[0])}`",
        f"- Concept 成功/失败：`{int(concept_feature_df.shape[0])}` / `{int(concept_failed_df.shape[0])}`",
        "",
        "## 4. 输出文件",
        "",
        "- radiomics_features_h123_raw_{run_id}.csv",
        "- h123_extraction_qc_{run_id}.csv",
        "- failed_cases_h123_{run_id}.csv（如存在）",
        "- radiomics_features_h12_raw_{run_id}.csv",
        "- h12_extraction_qc_{run_id}.csv",
        "- failed_cases_h12_{run_id}.csv（如存在）",
        "- concept_proxy_features_{run_id}.csv",
        "- concept_proxy_qc_{run_id}.csv",
        "- failed_cases_concept_proxy_{run_id}.csv（如存在）",
        "- run_summary_habitat_multi_roi_radiomics_{run_id}.json",
        "- run_config_habitat_multi_roi_radiomics_{run_id}.json",
        "- configs/habitat_multi_roi_radiomics_base.yaml",
        "",
    ]

    protocol_path = output_dir / "habitat_multi_roi_radiomics_protocol.md"
    protocol_path.write_text("\n".join(lines), encoding="utf-8")
    return protocol_path


def export_run_config_yaml(args: argparse.Namespace, output_dir: Path) -> Path:
    """导出 YAML 版本运行配置。"""

    payload = {
        "script": SCRIPT_NAME,
        "model": MODEL_NAME,
        "io": {
            "split_base_root": args.split_base_root,
            "habitat_mask_root": args.habitat_mask_root,
            "output_root": args.output_root,
            "splits": list(args.splits),
        },
        "rounds": {
            "round1": {
                "rois": list(ROUND1_HABITAT_ROIS),
                "modalities": list(args.modalities),
                "output": "radiomics_features_h123_raw_{run_id}.csv",
            },
            "round2": {
                "rois": list(ROUND2_HABITAT_ROIS),
                "modalities": list(args.modalities),
                "output": "radiomics_features_h12_raw_{run_id}.csv",
            },
            "round3": {
                "type": "concept_proxy",
                "required_modalities": list(CONCEPT_REQUIRED_MODALITIES),
                "output": "concept_proxy_features_{run_id}.csv",
            },
        },
        "naming": {
            "feature": "{roi}_{modality}_{feature}",
            "shared_feature": "{roi}_{feature}",
        },
        "roi": {
            "mask_label_value": args.h12_label_value,
            "tiny_voxel_threshold": args.tiny_h12_threshold,
        },
        "radiomics": {
            "image_types": ["Original"],
            "feature_classes": list(DEFAULT_FEATURE_CLASSES),
            "shape_reference_modality": args.shape_reference_modality,
            "resampled_spacing": args.resampled_spacing,
            "interpolator": args.interpolator,
            "normalize": args.normalize,
            "normalize_scale": args.normalize_scale,
            "remove_outliers": args.remove_outliers,
            "bin_width": args.bin_width,
            "pad_distance": args.pad_distance,
            "correct_mask_geometry": args.correct_mask_geometry,
        },
        "runtime": {
            "n_jobs": args.n_jobs,
            "itk_threads_per_worker": args.itk_threads_per_worker,
            "skip_errors": args.skip_errors,
            "seed": args.seed,
        },
    }
    yaml_path = output_dir / "configs" / "habitat_multi_roi_radiomics_base.yaml"
    save_yaml(payload, yaml_path)
    return yaml_path


def write_run_summary(
    args: argparse.Namespace,
    run_id: str,
    output_dir: Path,
    case_map: Mapping[str, Sequence[PatientCase]],
    h123_feature_df: pd.DataFrame,
    h123_qc_df: pd.DataFrame,
    h123_failed_df: pd.DataFrame,
    h12_feature_df: pd.DataFrame,
    h12_qc_df: pd.DataFrame,
    h12_failed_df: pd.DataFrame,
    concept_feature_df: pd.DataFrame,
    concept_qc_df: pd.DataFrame,
    concept_failed_df: pd.DataFrame,
    elapsed_seconds: float,
    timeline_copy_status: Mapping[str, object],
) -> Path:
    """写出运行摘要 JSON。"""

    summary = {
        "script": SCRIPT_NAME,
        "model": MODEL_NAME,
        "run_id": run_id,
        "elapsed_seconds": float(elapsed_seconds),
        "seed": int(args.seed),
        "inputs": {
            "split_base_root": str(args.split_base_root),
            "habitat_mask_root": str(args.habitat_mask_root),
            "splits": list(args.splits),
            "modalities": list(args.modalities),
        },
        "radiomics_settings": build_radiomics_settings(args),
        "runtime": {
            "n_jobs": int(args.n_jobs),
            "itk_threads_per_worker": int(args.itk_threads_per_worker),
            "skip_errors": bool(args.skip_errors),
        },
        "datasets": {
            split_name: {
                "num_patients": len(case_map[split_name]),
                "wild_type": int(sum(case.label_id == 0 for case in case_map[split_name])),
                "mutant": int(sum(case.label_id == 1 for case in case_map[split_name])),
            }
            for split_name in args.splits
        },
        "rounds": {
            "h123_radiomics": {
                "num_success": int(h123_feature_df.shape[0]),
                "num_failed": int(h123_failed_df.shape[0]),
                "num_qc_rows": int(h123_qc_df.shape[0]),
                "num_radiomics_feature_columns": count_radiomics_feature_columns(h123_feature_df),
            },
            "h12_radiomics": {
                "num_success": int(h12_feature_df.shape[0]),
                "num_failed": int(h12_failed_df.shape[0]),
                "num_qc_rows": int(h12_qc_df.shape[0]),
                "num_radiomics_feature_columns": count_radiomics_feature_columns(h12_feature_df),
            },
            "concept_proxy": {
                "num_success": int(concept_feature_df.shape[0]),
                "num_failed": int(concept_failed_df.shape[0]),
                "num_qc_rows": int(concept_qc_df.shape[0]),
                "num_columns": int(concept_feature_df.shape[1]) if not concept_feature_df.empty else 0,
            },
        },
        "timeline_copy": dict(timeline_copy_status),
        "output_dir": str(output_dir),
    }
    summary_path = output_dir / f"run_summary_habitat_multi_roi_radiomics_{run_id}.json"
    save_json(summary, summary_path)
    return summary_path


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------


def _main_pipeline(args: argparse.Namespace, run_id: str, output_dir: Path) -> None:
    """实际执行主流程。"""

    start_time = time.time()

    if args.shape_reference_modality not in args.modalities:
        raise ValueError(
            f"--shape-reference-modality '{args.shape_reference_modality}' "
            f"is not in --modalities {args.modalities}."
        )

    case_map = build_patient_cases(
        split_base_root=args.split_base_root,
        habitat_mask_root=args.habitat_mask_root,
        splits=args.splits,
        modalities=args.modalities,
    )

    print_case_summary(case_map=case_map, splits=args.splits)

    h123_feature_df, h123_qc_df, h123_failed_df = extract_all_features(
        case_map=case_map,
        args=args,
        rois=ROUND1_HABITAT_ROIS,
        modalities=args.modalities,
        progress_desc="extract-h123-radiomics",
    )
    print(
        f"H123 extraction finished: success={h123_feature_df.shape[0]} | "
        f"failed={h123_failed_df.shape[0]} | qc_rows={h123_qc_df.shape[0]}"
    )
    if (not args.skip_errors) and (not h123_failed_df.empty):
        preview = h123_failed_df[["patient_id", "split", "error"]].head(10)
        raise RuntimeError(
            "H123 round failed cases detected and --skip-errors is false.\n"
            f"{preview.to_string(index=False)}"
        )

    h12_feature_df, h12_qc_df, h12_failed_df = extract_all_features(
        case_map=case_map,
        args=args,
        rois=ROUND2_HABITAT_ROIS,
        modalities=args.modalities,
        progress_desc="extract-h12-radiomics",
    )
    print(
        f"H12 extraction finished: success={h12_feature_df.shape[0]} | "
        f"failed={h12_failed_df.shape[0]} | qc_rows={h12_qc_df.shape[0]}"
    )
    if (not args.skip_errors) and (not h12_failed_df.empty):
        preview = h12_failed_df[["patient_id", "split", "error"]].head(10)
        raise RuntimeError(
            "H12 round failed cases detected and --skip-errors is false.\n"
            f"{preview.to_string(index=False)}"
        )

    concept_feature_df, concept_qc_df, concept_failed_df = extract_all_concept_proxies(
        case_map=case_map,
        args=args,
    )
    print(
        f"Concept proxy extraction finished: success={concept_feature_df.shape[0]} | "
        f"failed={concept_failed_df.shape[0]} | qc_rows={concept_qc_df.shape[0]}"
    )
    if (not args.skip_errors) and (not concept_failed_df.empty):
        preview = concept_failed_df[["patient_id", "split", "error"]].head(10)
        raise RuntimeError(
            "Concept proxy round failed cases detected and --skip-errors is false.\n"
            f"{preview.to_string(index=False)}"
        )

    h123_features_path = output_dir / f"radiomics_features_h123_raw_{run_id}.csv"
    h123_qc_path = output_dir / f"h123_extraction_qc_{run_id}.csv"
    h123_failed_path = output_dir / f"failed_cases_h123_{run_id}.csv"
    h12_features_path = output_dir / f"radiomics_features_h12_raw_{run_id}.csv"
    h12_qc_path = output_dir / f"h12_extraction_qc_{run_id}.csv"
    h12_failed_path = output_dir / f"failed_cases_h12_{run_id}.csv"
    concept_features_path = output_dir / f"concept_proxy_features_{run_id}.csv"
    concept_qc_path = output_dir / f"concept_proxy_qc_{run_id}.csv"
    concept_failed_path = output_dir / f"failed_cases_concept_proxy_{run_id}.csv"

    if args.save_feature_table:
        h123_feature_df.to_csv(h123_features_path, index=False)
        h12_feature_df.to_csv(h12_features_path, index=False)
        concept_feature_df.to_csv(concept_features_path, index=False)
    if args.save_qc_table:
        h123_qc_df.to_csv(h123_qc_path, index=False)
        h12_qc_df.to_csv(h12_qc_path, index=False)
        concept_qc_df.to_csv(concept_qc_path, index=False)
    if not h123_failed_df.empty:
        h123_failed_df.to_csv(h123_failed_path, index=False)
    if not h12_failed_df.empty:
        h12_failed_df.to_csv(h12_failed_path, index=False)
    if not concept_failed_df.empty:
        concept_failed_df.to_csv(concept_failed_path, index=False)

    # 导出配置与协议文档。
    yaml_path = export_run_config_yaml(args=args, output_dir=output_dir)
    protocol_path = export_protocol_markdown(
        args=args,
        output_dir=output_dir,
        case_map=case_map,
        h123_feature_df=h123_feature_df,
        h123_failed_df=h123_failed_df,
        h12_feature_df=h12_feature_df,
        h12_failed_df=h12_failed_df,
        concept_feature_df=concept_feature_df,
        concept_failed_df=concept_failed_df,
    )
    run_config_json_path = output_dir / f"run_config_habitat_multi_roi_radiomics_{run_id}.json"
    save_json(vars(args), run_config_json_path)

    # 复制 lab_timeline.md 到本次结果目录。
    timeline_copy_status = copy_lab_timeline_if_needed(
        output_dir=output_dir,
        copy_enabled=args.copy_lab_timeline,
        lab_timeline_path=args.lab_timeline_path,
    )

    elapsed_seconds = time.time() - start_time
    summary_path = write_run_summary(
        args=args,
        run_id=run_id,
        output_dir=output_dir,
        case_map=case_map,
        h123_feature_df=h123_feature_df,
        h123_qc_df=h123_qc_df,
        h123_failed_df=h123_failed_df,
        h12_feature_df=h12_feature_df,
        h12_qc_df=h12_qc_df,
        h12_failed_df=h12_failed_df,
        concept_feature_df=concept_feature_df,
        concept_qc_df=concept_qc_df,
        concept_failed_df=concept_failed_df,
        elapsed_seconds=elapsed_seconds,
        timeline_copy_status=timeline_copy_status,
    )

    # 维护 output_root 级别的 latest 软别名（复制一份，便于快速查看）。
    # 这样可以同时满足“每次运行有独立目录”和“固定路径快速读取”。
    if args.save_feature_table and h123_features_path.is_file():
        shutil.copy2(h123_features_path, args.output_root / "radiomics_features_h123_raw_latest.csv")
    if args.save_feature_table and h12_features_path.is_file():
        shutil.copy2(h12_features_path, args.output_root / "radiomics_features_h12_raw_latest.csv")
    if args.save_feature_table and concept_features_path.is_file():
        shutil.copy2(concept_features_path, args.output_root / "concept_proxy_features_latest.csv")
    if args.save_qc_table and h123_qc_path.is_file():
        shutil.copy2(h123_qc_path, args.output_root / "h123_extraction_qc_latest.csv")
    if args.save_qc_table and h12_qc_path.is_file():
        shutil.copy2(h12_qc_path, args.output_root / "h12_extraction_qc_latest.csv")
    if args.save_qc_table and concept_qc_path.is_file():
        shutil.copy2(concept_qc_path, args.output_root / "concept_proxy_qc_latest.csv")
    if h123_failed_path.is_file():
        shutil.copy2(h123_failed_path, args.output_root / "failed_cases_h123_latest.csv")
    if h12_failed_path.is_file():
        shutil.copy2(h12_failed_path, args.output_root / "failed_cases_h12_latest.csv")
    if concept_failed_path.is_file():
        shutil.copy2(concept_failed_path, args.output_root / "failed_cases_concept_proxy_latest.csv")
    shutil.copy2(summary_path, args.output_root / "run_summary_habitat_multi_roi_radiomics_latest.json")
    shutil.copy2(run_config_json_path, args.output_root / "run_config_habitat_multi_roi_radiomics_latest.json")
    shutil.copy2(protocol_path, args.output_root / "habitat_multi_roi_radiomics_protocol_latest.md")
    shutil.copy2(yaml_path, args.output_root / "habitat_multi_roi_radiomics_base_latest.yaml")
    if timeline_copy_status.get("copied"):
        timeline_dst = output_dir / "lab_timeline.md"
        if timeline_dst.is_file():
            shutil.copy2(timeline_dst, args.output_root / "lab_timeline.md")

    print("Run completed.")
    print(f"Output directory : {output_dir}")
    if args.save_feature_table:
        print(f"H123 features    : {h123_features_path}")
        print(f"H12 features     : {h12_features_path}")
        print(f"Concept features : {concept_features_path}")
    if args.save_qc_table:
        print(f"H123 QC          : {h123_qc_path}")
        print(f"H12 QC           : {h12_qc_path}")
        print(f"Concept QC       : {concept_qc_path}")
    if h123_failed_path.is_file():
        print(f"H123 failed      : {h123_failed_path}")
    if h12_failed_path.is_file():
        print(f"H12 failed       : {h12_failed_path}")
    if concept_failed_path.is_file():
        print(f"Concept failed   : {concept_failed_path}")
    print(f"Run summary      : {summary_path}")
    print(f"Run config (json): {run_config_json_path}")
    print(f"Run config (yaml): {yaml_path}")
    print(f"Protocol         : {protocol_path}")
    if timeline_copy_status.get("copied"):
        print(f"Timeline copied  : {timeline_copy_status.get('target_path')}")
    else:
        print(f"Timeline note    : {timeline_copy_status.get('note')}")


def main() -> None:
    """程序入口。"""

    args = build_argparser().parse_args()
    ensure_required_dependencies()
    set_seed(args.seed)

    run_id = resolve_run_id(args.run_id)
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    args.split_base_root = args.split_base_root.resolve()
    args.habitat_mask_root = args.habitat_mask_root.resolve()
    args.output_root = args.output_root.resolve()
    args.lab_timeline_path = args.lab_timeline_path.resolve()

    if args.log_to_file:
        log_path = output_dir / f"extraction_log_habitat_multi_roi_radiomics_{run_id}.txt"
        with TeeLogger(log_path, mode="w"):
            _main_pipeline(args=args, run_id=run_id, output_dir=output_dir)
    else:
        _main_pipeline(args=args, run_id=run_id, output_dir=output_dir)


if __name__ == "__main__":
    main()

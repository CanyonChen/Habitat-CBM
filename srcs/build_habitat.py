#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态生理 MRI Habitat 掩膜构建脚本。

===============================================================================
一、脚本目标
===============================================================================
本脚本用于根据 `ADC + CBF + functional/voi` 自动构建论文

`Interpretable Habitat Radiomics from Multimodal Physiological MRI for Grading
and IDH Mutation Status Prediction of Adult-type Diffuse Gliomas`

中定义的多模态 habitat 掩膜，并按当前项目协议输出每位患者的四个核心 mask：

1. `H1`   : 高灌注 + 高细胞密度   = 高 CBF + 低 ADC
2. `H2`   : 低灌注 + 高细胞密度   = 低 CBF + 低 ADC
3. `H3`   : 低灌注 + 低细胞密度   = 低 CBF + 高 ADC
4. `H1+2` : 论文最终主线 radiomics 建模使用的最优 habitat

===============================================================================
二、输入目录格式
===============================================================================
脚本支持两种输入布局，其中默认推荐直接使用 `data_split.py` 的输出根目录：

1. split 布局（推荐）：

dataset_root/
├── train/
│   ├── conventional/
│   └── functional/
├── val/
│   ├── conventional/
│   └── functional/
└── test/
    ├── conventional/
    └── functional/

2. 旧版布局（兼容）：

dataset_root/
├── idh.csv
├── conventional/
│   ├── mutant/
│   │   ├── 005/
│   │   └── ...
│   └── wild_type/
│       ├── 003/
│       └── ...
└── functional/
    ├── mutant/
    │   ├── 005/
    │   │   ├── adc.nii.gz
    │   │   ├── cbf.nii.gz
    │   │   └── voi.nii.gz
    │   └── ...
    └── wild_type/
        ├── 003/
        └── ...

按当前项目协议：
- habitat 聚类只使用 `functional/<label>/<patient_id>/adc`
- habitat 聚类只使用 `functional/<label>/<patient_id>/cbf`
- VOI 一律使用 `functional/<label>/<patient_id>/voi`

===============================================================================
三、输出目录格式
===============================================================================
默认输出到：

habitat_CBM/results/02_habitat/habitat_masks/

当输入为 split 布局时，输出按 split 分层组织：

habitat_masks/
├── train/
│   ├── mutant/<patient_id>/h1|h2|h3|h12.nii.gz
│   └── wild_type/<patient_id>/h1|h2|h3|h12.nii.gz
├── val/
│   ├── mutant/<patient_id>/h1|h2|h3|h12.nii.gz
│   └── wild_type/<patient_id>/h1|h2|h3|h12.nii.gz
├── test/
│   ├── mutant/<patient_id>/h1|h2|h3|h12.nii.gz
│   └── wild_type/<patient_id>/h1|h2|h3|h12.nii.gz
└── manifests/
    ├── habitat_centers.csv
    ├── habitat_volume_summary.csv
    ├── habitat_qc.csv
    └── run_summary.json

===============================================================================
四、GPU 加速策略
===============================================================================
若服务器安装了 PyTorch 且可用 CUDA，本脚本会默认：

1. 将 VOI 内的 `ADC/CBF` 体素送入 GPU；
2. 在 GPU 上完成：
   - z-score 标准化
   - K-means 聚类
   - 聚类中心语义映射所需的中心统计

由于医学影像 NIfTI 文件读取仍主要依赖 CPU / 磁盘 I/O，
因此“整个流程”的 GPU 加速并非 100%，但耗时最大的体素级聚类部分会尽量使用 GPU。

若没有可用 GPU，则自动退回 CPU：
- 若安装了 PyTorch，则使用 PyTorch CPU 版本；
- 否则使用 sklearn 的 CPU KMeans。

===============================================================================
五、实现假设
===============================================================================
1. 用户已确认输入图像都完成预处理；
2. `ADC / CBF / VOI` 已处于共同空间；
3. `VOI` 内不存在大面积 NaN / Inf；
4. 论文方法的可执行落地版本采用“每位患者独立聚类”。

===============================================================================
六、推荐运行示例
===============================================================================
1. 自动优先使用 GPU：

python habitat_CBM/repo/srcs/build_habitat.py \
    --dataset-root /path/to/splited_data \
    --output-root /path/to/habitat_CBM/results/02_habitat/habitat_masks \
    --device auto

2. 强制使用 GPU：

python habitat_CBM/repo/srcs/build_habitat.py \
    --dataset-root /path/to/splited_data \
    --device cuda:0

3. 强制使用 CPU：

python habitat_CBM/repo/srcs/build_habitat.py \
    --dataset-root /path/to/splited_data \
    --device cpu
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import shutil
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

try:
    import nibabel as nib
except Exception:  # pragma: no cover - 运行环境可能尚未安装 nibabel
    nib = None  # type: ignore[assignment]

try:
    import pandas as pd
except Exception:  # pragma: no cover - 运行环境可能尚未安装 pandas
    pd = None  # type: ignore[assignment]

try:
    import torch
except Exception:  # pragma: no cover - 运行环境可能没有 torch
    torch = None  # type: ignore[assignment]

try:
    from sklearn.cluster import KMeans as SklearnKMeans
except Exception:  # pragma: no cover - 运行环境可能没有 sklearn
    SklearnKMeans = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # habitat_CBM/


def resolve_default_dataset_root() -> Path:
    """按优先级推断默认输入目录。

    优先使用 data_split.py 的输出目录（train/val/test 在其下），
    若不存在则回退到旧版数据根目录。
    """

    candidates = [
        # PROJECT_ROOT / "data" / "splited_data",
        PROJECT_ROOT / "dataset" / "splited_data",
        # PROJECT_ROOT / "data" / "images",
        # PROJECT_ROOT / "dataset" / "images",
        # PROJECT_ROOT / "dataset",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_DATASET_ROOT = resolve_default_dataset_root()
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "habitat_masks"

BRANCHES = ("conventional", "functional")
LABEL_TO_ID = {"wild_type": 0, "mutant": 1}
KNOWN_SPLITS = ("train", "val", "test")
ALLOWED_EXTENSIONS = (".nii", ".nii.gz", ".mha", ".mhd", ".nrrd")
DEFAULT_VOI_KEYWORDS = ("voi",)

# 为了与现有仓库的文件发现规则保持一致，这里沿用较宽松的关键词匹配策略。
MODALITY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "adc": ("adc",),
    "cbf": ("cbf",),
}

# 语义模板：论文中的三类 habitat 在 z-score 空间里的目标方向。
# 说明：
# - H1: 低 ADC + 高 CBF
# - H2: 低 ADC + 低 CBF
# - H3: 高 ADC + 低 CBF
SEMANTIC_TEMPLATES = np.asarray(
    [
        [-1.0, +1.0],  # H1
        [-1.0, -1.0],  # H2
        [+1.0, -1.0],  # H3
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class PatientCase:
    """患者级 habitat 构建所需的最小信息。"""

    split_name: str
    patient_id: str
    label_name: str
    label_id: int
    adc_path: Path
    cbf_path: Path
    voi_path: Path


@dataclass
class ClusterResult:
    """单位患者聚类输出。

    为了后续构建 mask、输出汇总表和做质控，这里一次性保存：
    - VOI 内每个体素的 mapped habitat label
    - raw / mapped center
    - inertia / 迭代轮数 / 后端信息
    """

    mapped_labels_flat: np.ndarray
    raw_labels_flat: np.ndarray
    centers_z_raw: np.ndarray
    centers_z_mapped: np.ndarray
    adc_centers_raw: np.ndarray
    cbf_centers_raw: np.ndarray
    adc_centers_mapped: np.ndarray
    cbf_centers_mapped: np.ndarray
    best_cost: float
    inertia: float
    num_iterations: int
    backend: str


def ensure_required_dependencies() -> None:
    """在真正开始处理病例前，统一检查运行时依赖。

    这样可以做到两件事：
    1. `python build_habitat.py --help` 在缺少运行依赖时仍可正常查看参数；
    2. 真正运行时给出比 `ModuleNotFoundError` 更明确的安装提示。
    """

    missing_packages: List[str] = []
    if nib is None:
        missing_packages.append("nibabel")
    if pd is None:
        missing_packages.append("pandas")
    if torch is None and SklearnKMeans is None:
        missing_packages.append("torch or scikit-learn")

    if missing_packages:
        joined = ", ".join(missing_packages)
        raise ImportError(
            "Missing runtime dependencies for build_habitat.py: "
            f"{joined}. "
            "Please install `habitat_CBM/repo/requirements_pyradiomics.txt`. "
            "If you want GPU acceleration, also install a CUDA-enabled PyTorch "
            "wheel that matches your server environment."
        )


def str2bool(value: str) -> bool:
    """将命令行中的字符串解析为布尔值。"""

    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def normalize_name(name: str) -> str:
    """统一文件名字符串风格，降低命名差异带来的匹配困难。"""

    return name.lower().replace("-", "_").replace(" ", "_")


def is_supported_image(path: Path) -> bool:
    """判断文件是否为脚本支持的医学图像格式。"""

    lower_name = path.name.lower()
    return any(lower_name.endswith(ext) for ext in ALLOWED_EXTENSIONS)


def collect_branch_image_files(branch_dir: Path) -> List[Path]:
    """递归收集某个患者分支目录下的所有医学图像文件。"""

    return [
        path
        for path in branch_dir.rglob("*")
        if path.is_file() and is_supported_image(path)
    ]


def match_modality(path: Path, modality: str) -> bool:
    """根据文件名关键词匹配某个模态。

    这里仅用于 `adc` 和 `cbf`，因此规则保持简洁。
    """

    name = normalize_name(path.name)
    return any(keyword in name for keyword in MODALITY_KEYWORDS[modality])


def match_voi(path: Path) -> bool:
    """判断某个文件名是否可能是 VOI 掩模。"""

    name = normalize_name(path.name)
    return any(keyword in name for keyword in DEFAULT_VOI_KEYWORDS)


def score_voi_candidate(path: Path) -> int:
    """为 VOI 候选文件打分，优先选择命名更明确的掩模文件。"""

    name = normalize_name(path.name)
    if name.startswith("voi.") or name.startswith("voi_"):
        return 1000
    score = 0
    for rank, keyword in enumerate(DEFAULT_VOI_KEYWORDS):
        if keyword in name:
            score += 100 - rank
    return score


def discover_patient_dirs(case_root: Path) -> Dict[str, Dict[str, Path]]:
    """扫描数据根目录，收集所有患者在两个分支下的患者文件夹路径。

    返回结构：
    {
        "005": {
            "label_name": "mutant",
            "conventional": Path(...),
            "functional": Path(...),
        },
        ...
    }
    """

    patient_map: Dict[str, Dict[str, Path]] = {}

    for branch in BRANCHES:
        branch_root = case_root / branch
        if not branch_root.is_dir():
            raise FileNotFoundError(f"Missing branch directory: {branch_root}")

        for label_name in LABEL_TO_ID:
            label_root = branch_root / label_name
            if not label_root.is_dir():
                raise FileNotFoundError(f"Missing label directory: {label_root}")

            for patient_dir in sorted(label_root.iterdir()):
                if not patient_dir.is_dir():
                    continue

                patient_id = patient_dir.name
                patient_entry = patient_map.setdefault(patient_id, {"label_name": label_name})

                if patient_entry["label_name"] != label_name:
                    raise ValueError(
                        f"Patient {patient_id} appears under conflicting labels: "
                        f"{patient_entry['label_name']} vs {label_name}"
                    )
                patient_entry[branch] = patient_dir

    for patient_id, entry in patient_map.items():
        for branch in BRANCHES:
            if branch not in entry:
                raise FileNotFoundError(
                    f"Patient {patient_id} is missing branch directory: {branch}"
                )

    return patient_map


def resolve_case_roots(dataset_root: Path) -> List[Tuple[str, Path]]:
    """解析病例扫描根目录。

    支持两种输入布局：
    1. split 布局（推荐）：dataset_root/train|val|test/{conventional,functional}/...
    2. 旧版布局：dataset_root/{conventional,functional}/...
    """

    split_roots: List[Tuple[str, Path]] = []
    for split_name in KNOWN_SPLITS:
        candidate = dataset_root / split_name
        if candidate.is_dir():
            split_roots.append((split_name, candidate))

    if split_roots:
        for split_name, split_root in split_roots:
            for branch in BRANCHES:
                branch_root = split_root / branch
                if not branch_root.is_dir():
                    raise FileNotFoundError(
                        f"Split root detected but missing branch directory: {branch_root} "
                        f"(split={split_name})"
                    )
        return split_roots

    # 回退到旧版单根目录布局。
    for branch in BRANCHES:
        branch_root = dataset_root / branch
        if not branch_root.is_dir():
            raise FileNotFoundError(
                f"Missing branch directory: {branch_root}. "
                "Expected either split layout (train/val/test) or legacy layout "
                "(conventional + functional under dataset-root)."
            )
    return [("all", dataset_root)]


def discover_modality_file(functional_dir: Path, modality: str) -> Path:
    """在 `functional/<label>/<patient_id>/` 下发现唯一的 `adc` 或 `cbf` 文件。"""

    all_files = collect_branch_image_files(functional_dir)
    matches = [path for path in all_files if match_modality(path, modality)]
    if len(matches) == 0:
        raise FileNotFoundError(
            f"Cannot find modality file for '{modality}' under {functional_dir}"
        )
    if len(matches) > 1:
        match_str = "\n".join(str(path) for path in matches[:10])
        raise ValueError(
            f"Found multiple candidate files for modality '{modality}' under {functional_dir}.\n"
            f"{match_str}"
        )
    return matches[0]


def discover_voi_file(functional_dir: Path) -> Path:
    """在 functional 分支下找到唯一 VOI 文件。"""

    all_files = collect_branch_image_files(functional_dir)
    matches = [path for path in all_files if match_voi(path)]
    if len(matches) == 0:
        raise FileNotFoundError(f"Cannot find VOI file under {functional_dir}")
    if len(matches) == 1:
        return matches[0]

    scored = sorted(matches, key=lambda path: score_voi_candidate(path), reverse=True)
    best_score = score_voi_candidate(scored[0])
    best_matches = [path for path in scored if score_voi_candidate(path) == best_score]
    if len(best_matches) > 1:
        match_str = "\n".join(str(path) for path in best_matches[:10])
        raise ValueError(
            f"Found multiple equally plausible VOI files under {functional_dir}.\n"
            f"{match_str}"
        )
    return best_matches[0]


def build_patient_cases(dataset_root: Path) -> List[PatientCase]:
    """扫描数据根目录并构造全部 PatientCase。"""

    cases: List[PatientCase] = []

    case_roots = resolve_case_roots(dataset_root)
    for split_name, case_root in case_roots:
        patient_dirs = discover_patient_dirs(case_root)

        for patient_id in sorted(patient_dirs):
            entry = patient_dirs[patient_id]
            label_name = str(entry["label_name"])
            functional_dir = Path(entry["functional"])

            cases.append(
                PatientCase(
                    split_name=split_name,
                    patient_id=patient_id,
                    label_name=label_name,
                    label_id=LABEL_TO_ID[label_name],
                    adc_path=discover_modality_file(functional_dir, "adc"),
                    cbf_path=discover_modality_file(functional_dir, "cbf"),
                    voi_path=discover_voi_file(functional_dir),
                )
            )

    return cases


def _maybe_load_uncompressed_nifti_from_nii_gz(
    path: Path,
    original_error: Exception,
) -> Optional[nib.spatialimages.SpatialImage]:
    """尝试把“后缀为 .nii.gz 但实际未 gzip 压缩”的文件当作裸 NIfTI 读取。"""

    lower_name = path.name.lower()
    if not lower_name.endswith(".nii.gz"):
        return None
    if "not a gzip file" not in str(original_error).lower():
        return None

    # 某些数据在导出时会把 .nii 文件误命名为 .nii.gz。
    # nib.load 会按 gzip 解码，从而抛出 "not a gzip file"。
    # 这里回退到按裸 NIfTI 字节解析，保证流程可继续。
    raw_bytes = path.read_bytes()
    image = nib.Nifti1Image.from_bytes(raw_bytes)
    warnings.warn(
        f"Detected non-gzip payload with '.nii.gz' suffix, loaded as uncompressed NIfTI: {path}",
        RuntimeWarning,
        stacklevel=2,
    )
    return image


def load_nifti_image(
    path: Path,
    allow_non_gzip_nii_gz: bool = True,
) -> nib.spatialimages.SpatialImage:
    """读取医学影像文件并返回 nibabel image 对象。"""

    try:
        return nib.load(str(path))
    except Exception as exc:
        if not allow_non_gzip_nii_gz:
            raise
        recovered_image = _maybe_load_uncompressed_nifti_from_nii_gz(path, exc)
        if recovered_image is not None:
            return recovered_image
        raise


def load_nifti_array(path: Path) -> np.ndarray:
    """读取影像并转为 float32 NumPy 数组。"""

    image = load_nifti_image(path)
    array = image.get_fdata(dtype=np.float32)
    return np.asarray(array, dtype=np.float32)


def assert_same_geometry(reference_img: nib.spatialimages.SpatialImage, other_img: nib.spatialimages.SpatialImage, name: str) -> None:
    """严格检查两幅图像是否在同一空间。

    因为用户已明确说明“所有图像都已经完成预处理”，
    所以这里采用严格一致策略：如果 geometry 不一致，直接报错。
    """

    if reference_img.shape != other_img.shape:
        raise ValueError(
            f"Geometry mismatch for {name}: shape {reference_img.shape} vs {other_img.shape}"
        )

    ref_affine = np.asarray(reference_img.affine, dtype=np.float64)
    oth_affine = np.asarray(other_img.affine, dtype=np.float64)
    if not np.allclose(ref_affine, oth_affine, atol=1e-4):
        raise ValueError(f"Geometry mismatch for {name}: affine matrices differ.")


def zscore_numpy(values: np.ndarray, eps: float) -> Tuple[np.ndarray, float, float]:
    """对一维向量做 z-score，并返回标准化结果、均值和标准差。"""

    mean = float(values.mean())
    std = float(values.std())
    if std < eps:
        return np.zeros_like(values, dtype=np.float32), mean, std
    return ((values - mean) / std).astype(np.float32), mean, std


def set_random_seed(seed: int) -> None:
    """统一设置 Python / NumPy / Torch 的随机种子。"""

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> Tuple[str, Optional["torch.device"]]:
    """解析脚本设备参数。

    返回：
    - backend_device_name: 供日志打印的人类可读字符串
    - torch_device: 若可用则返回 torch.device，否则返回 None
    """

    if device_arg == "auto":
        if torch is not None and torch.cuda.is_available():
            return "cuda", torch.device("cuda")
        if torch is not None:
            return "cpu", torch.device("cpu")
        return "cpu", None

    if device_arg.startswith("cuda"):
        if torch is None:
            raise RuntimeError("CUDA was requested, but PyTorch is not installed.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        return device_arg, torch.device(device_arg)

    if device_arg == "cpu":
        if torch is not None:
            return "cpu", torch.device("cpu")
        return "cpu", None

    raise ValueError(f"Unsupported device: {device_arg}")


def initialize_centers_kmeanspp_torch(
    x: "torch.Tensor",
    n_clusters: int,
    generator: "torch.Generator",
) -> "torch.Tensor":
    """使用 k-means++ 初始化聚类中心。

    之所以自己实现，而不是直接依赖 sklearn / cuML：
    - 这样可以统一 CPU / GPU 两条路径；
    - 在 GPU 上也能直接用 torch 完成初始化。
    """

    n_samples = x.shape[0]
    if n_samples < n_clusters:
        raise ValueError(
            f"Number of voxels ({n_samples}) is smaller than n_clusters ({n_clusters})."
        )

    first_idx = int(torch.randint(0, n_samples, (1,), generator=generator, device=x.device).item())
    centers = [x[first_idx]]

    closest_dist_sq = ((x - centers[0]) ** 2).sum(dim=1)
    for _ in range(1, n_clusters):
        dist_sum = float(closest_dist_sq.sum().item())
        if dist_sum <= 0:
            # 所有点都重合时退化为随机抽样。
            next_idx = int(
                torch.randint(0, n_samples, (1,), generator=generator, device=x.device).item()
            )
        else:
            probs = closest_dist_sq / closest_dist_sq.sum()
            next_idx = int(torch.multinomial(probs, 1, generator=generator).item())
        centers.append(x[next_idx])
        new_dist_sq = ((x - centers[-1]) ** 2).sum(dim=1)
        closest_dist_sq = torch.minimum(closest_dist_sq, new_dist_sq)

    return torch.stack(centers, dim=0)


def assign_clusters_torch(
    x: "torch.Tensor",
    centers: "torch.Tensor",
    chunk_size: int,
) -> Tuple["torch.Tensor", "torch.Tensor", float]:
    """分块计算每个体素到各中心的距离，并返回聚类标签。

    这里采用分块而不是一次性构造完整 `[N, K, 2]` 大张量，
    是为了在大 VOI 情况下更稳地控制显存。
    """

    label_chunks: List["torch.Tensor"] = []
    min_dist_chunks: List["torch.Tensor"] = []
    inertia = 0.0

    for start in range(0, x.shape[0], chunk_size):
        end = min(start + chunk_size, x.shape[0])
        chunk = x[start:end]
        dist_sq = ((chunk[:, None, :] - centers[None, :, :]) ** 2).sum(dim=2)
        min_dist_sq, labels = torch.min(dist_sq, dim=1)
        label_chunks.append(labels)
        min_dist_chunks.append(min_dist_sq)
        inertia += float(min_dist_sq.sum().item())

    return torch.cat(label_chunks, dim=0), torch.cat(min_dist_chunks, dim=0), inertia


def run_kmeans_torch(
    x_np: np.ndarray,
    n_clusters: int,
    n_init: int,
    max_iter: int,
    tol: float,
    seed: int,
    device: "torch.device",
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray, float, int, str]:
    """使用 PyTorch 在 CPU 或 GPU 上运行 K-means。

    返回：
    - labels_np: 每个体素的 raw cluster id
    - centers_np: raw cluster centers（z-score 空间）
    - best_inertia
    - best_num_iterations
    - backend name
    """

    if torch is None:
        raise RuntimeError("PyTorch is required for run_kmeans_torch().")

    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    best_labels: Optional["torch.Tensor"] = None
    best_centers: Optional["torch.Tensor"] = None
    best_inertia: Optional[float] = None
    best_num_iterations = 0

    with torch.no_grad():
        for init_idx in range(n_init):
            # 为了让多次初始化彼此不同，这里对 seed 做轻微偏移。
            generator.manual_seed(seed + init_idx)
            centers = initialize_centers_kmeanspp_torch(x, n_clusters=n_clusters, generator=generator)

            num_iterations = 0
            for iteration in range(max_iter):
                num_iterations = iteration + 1
                labels, min_dist_sq, inertia = assign_clusters_torch(
                    x=x,
                    centers=centers,
                    chunk_size=chunk_size,
                )

                new_centers = torch.empty_like(centers)
                for cluster_id in range(n_clusters):
                    member_mask = labels == cluster_id
                    if bool(member_mask.any().item()):
                        new_centers[cluster_id] = x[member_mask].mean(dim=0)
                    else:
                        # 空簇是 K-means 常见退化情况。
                        # 这里将其重置为当前“最难解释”的体素，即距离最近中心最远的点。
                        farthest_idx = int(torch.argmax(min_dist_sq).item())
                        new_centers[cluster_id] = x[farthest_idx]

                shift = float(torch.norm(new_centers - centers, dim=1).max().item())
                centers = new_centers
                if shift <= tol:
                    break

            # 最终中心确定后，再做一次完整赋值，得到可比较的 inertia。
            labels, _, inertia = assign_clusters_torch(
                x=x,
                centers=centers,
                chunk_size=chunk_size,
            )

            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.clone()
                best_centers = centers.clone()
                best_num_iterations = num_iterations

    if best_inertia is None or best_labels is None or best_centers is None:
        raise RuntimeError("K-means failed to produce a valid result.")

    return (
        best_labels.detach().cpu().numpy().astype(np.int64),
        best_centers.detach().cpu().numpy().astype(np.float32),
        float(best_inertia),
        int(best_num_iterations),
        f"torch-{device.type}",
    )


def run_kmeans_sklearn(
    x_np: np.ndarray,
    n_clusters: int,
    n_init: int,
    max_iter: int,
    tol: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, float, int, str]:
    """使用 sklearn 在 CPU 上运行 K-means。"""

    if SklearnKMeans is None:
        raise RuntimeError(
            "Neither GPU torch-kmeans nor sklearn KMeans is available in this environment."
        )

    model = SklearnKMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=seed,
        algorithm="lloyd",
    )
    labels = model.fit_predict(x_np)
    centers = np.asarray(model.cluster_centers_, dtype=np.float32)
    inertia = float(model.inertia_)
    num_iterations = int(model.n_iter_)
    return labels.astype(np.int64), centers, inertia, num_iterations, "sklearn-cpu"


def map_clusters_to_habitats(
    labels_raw: np.ndarray,
    centers_z_raw: np.ndarray,
    adc_values: np.ndarray,
    cbf_values: np.ndarray,
) -> ClusterResult:
    """将 raw cluster id 映射为有生理意义的 H1 / H2 / H3。

    映射原则：
    - 比较 raw centers 与三类语义模板的欧氏距离；
    - 穷举 3! = 6 种匹配方式；
    - 选择总距离最小的一组映射。
    """

    if centers_z_raw.shape != (3, 2):
        raise ValueError(f"Expected centers shape (3, 2), got {centers_z_raw.shape}")

    adc_centers_raw = np.zeros(3, dtype=np.float32)
    cbf_centers_raw = np.zeros(3, dtype=np.float32)
    for cluster_id in range(3):
        member_mask = labels_raw == cluster_id
        if not np.any(member_mask):
            raise ValueError(f"Cluster {cluster_id} is empty after K-means.")
        adc_centers_raw[cluster_id] = float(adc_values[member_mask].mean())
        cbf_centers_raw[cluster_id] = float(cbf_values[member_mask].mean())

    best_perm: Optional[Tuple[int, int, int]] = None
    best_cost: Optional[float] = None

    for perm in itertools.permutations([0, 1, 2]):
        total_cost = 0.0
        for habitat_idx, cluster_idx in enumerate(perm):
            total_cost += float(
                np.linalg.norm(centers_z_raw[cluster_idx] - SEMANTIC_TEMPLATES[habitat_idx])
            )
        if best_cost is None or total_cost < best_cost:
            best_cost = total_cost
            best_perm = perm

    if best_perm is None or best_cost is None:
        raise RuntimeError("Failed to resolve semantic cluster mapping.")

    # best_perm[0] 表示哪个 raw cluster 被映射成 H1；
    # best_perm[1] 表示哪个 raw cluster 被映射成 H2；
    # best_perm[2] 表示哪个 raw cluster 被映射成 H3。
    raw_to_mapped = {
        best_perm[0]: 1,
        best_perm[1]: 2,
        best_perm[2]: 3,
    }
    mapped_to_raw = {
        1: best_perm[0],
        2: best_perm[1],
        3: best_perm[2],
    }

    mapped_labels_flat = np.vectorize(raw_to_mapped.get)(labels_raw).astype(np.uint8)

    centers_z_mapped = np.asarray(
        [centers_z_raw[mapped_to_raw[idx]] for idx in (1, 2, 3)],
        dtype=np.float32,
    )
    adc_centers_mapped = np.asarray(
        [adc_centers_raw[mapped_to_raw[idx]] for idx in (1, 2, 3)],
        dtype=np.float32,
    )
    cbf_centers_mapped = np.asarray(
        [cbf_centers_raw[mapped_to_raw[idx]] for idx in (1, 2, 3)],
        dtype=np.float32,
    )

    return ClusterResult(
        mapped_labels_flat=mapped_labels_flat,
        raw_labels_flat=labels_raw.astype(np.int64),
        centers_z_raw=centers_z_raw.astype(np.float32),
        centers_z_mapped=centers_z_mapped,
        adc_centers_raw=adc_centers_raw,
        cbf_centers_raw=cbf_centers_raw,
        adc_centers_mapped=adc_centers_mapped,
        cbf_centers_mapped=cbf_centers_mapped,
        best_cost=float(best_cost),
        inertia=float("nan"),  # 后续在构建阶段补齐
        num_iterations=0,      # 后续在构建阶段补齐
        backend="",
    )


def save_mask(mask: np.ndarray, reference_img: nib.spatialimages.SpatialImage, path: Path) -> None:
    """保存二值 mask 到指定路径。

    这里显式复制 affine 和 header，保证输出与参考图像处于相同空间。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference_img.header.copy()
    header.set_data_dtype(np.uint8)
    image = nib.Nifti1Image(mask.astype(np.uint8), affine=reference_img.affine, header=header)
    nib.save(image, str(path))


def maybe_remove_patient_dir(patient_output_dir: Path, overwrite_existing: bool) -> None:
    """根据配置决定是否先清空已有患者输出目录。"""

    if overwrite_existing and patient_output_dir.exists():
        shutil.rmtree(patient_output_dir)


def parse_patient_ids(raw_value: str) -> List[str]:
    """将逗号分隔的 patient id 字符串解析成列表。"""

    ids = [item.strip() for item in raw_value.split(",") if item.strip()]
    return sorted(set(ids))


def build_argument_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="根据 ADC + CBF + functional/voi 构建 H1/H2/H3/H1+2 habitat 掩膜。"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=(
            "输入数据根目录。优先推荐 data_split.py 的输出根目录（内部包含 train/val/test）；"
            "也兼容旧版 conventional/ + functional/ 直连布局。"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Habitat 输出根目录。若输入是 split 布局，则输出按 split 分层："
            "<output-root>/<split>/<label>/<patient_id>/h1|h2|h3|h12.nii.gz。"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="计算设备：auto/cpu/cuda/cuda:0 等；默认 auto。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认 42。",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=3,
        help="聚类数，论文固定为 3；默认 3。",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=20,
        help="K-means 初始化次数；默认 20。",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="K-means 最大迭代轮数；默认 100。",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="K-means 迭代停止阈值；默认 1e-4。",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=262144,
        help="GPU/torch 聚类时的分块大小；默认 262144。",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="z-score 与数值稳定性使用的最小 epsilon；默认 1e-8。",
    )
    parser.add_argument(
        "--overwrite-existing",
        type=str2bool,
        default=True,
        help="若患者输出目录已存在，是否先删除再重建；默认 true。",
    )
    parser.add_argument(
        "--patient-ids",
        type=parse_patient_ids,
        default=None,
        help="可选：仅处理指定患者，逗号分隔，例如 003,005,013。",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=0,
        help="可选：仅处理前 N 位患者，用于调试；默认 0 表示全量。",
    )
    parser.add_argument(
        "--skip-errors",
        type=str2bool,
        default=False,
        help="遇到单病例错误时是否跳过并继续；默认 false。",
    )
    parser.add_argument(
        "--allow-non-gzip-nii-gz",
        type=str2bool,
        default=True,
        help=(
            "当文件名为 .nii.gz 但内容不是 gzip 时，是否回退为未压缩 NIfTI 读取；"
            "默认 true。建议同时在数据层面修复该文件。"
        ),
    )
    parser.add_argument(
        "--save-run-config",
        type=str2bool,
        default=True,
        help="是否保存本次运行配置到 manifests/run_summary.json；默认 true。",
    )
    return parser


def main() -> None:
    """脚本主入口。"""

    parser = build_argument_parser()
    args = parser.parse_args()

    ensure_required_dependencies()

    if args.n_clusters != 3:
        raise ValueError("This script currently only supports the paper-defined K=3 setting.")

    set_random_seed(args.seed)

    start_time = time.time()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    manifests_dir = output_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    backend_device_name, torch_device = resolve_device(args.device)
    if torch is not None and torch_device is not None and torch_device.type == "cuda":
        # 这些设置不会改变算法逻辑，只是尽量榨出 GPU 的吞吐。
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    cases = build_patient_cases(dataset_root)
    if args.patient_ids:
        allow_set = set(args.patient_ids)
        cases = [case for case in cases if case.patient_id in allow_set]
    if args.max_patients > 0:
        cases = cases[: args.max_patients]

    if len(cases) == 0:
        raise RuntimeError("No patients matched the current filters.")

    split_counts: Dict[str, int] = {}
    for case in cases:
        split_counts[case.split_name] = split_counts.get(case.split_name, 0) + 1

    print("=" * 80)
    print("Build Habitat Masks")
    print("=" * 80)
    print(f"Dataset root      : {dataset_root}")
    print(f"Output root       : {output_root}")
    print(f"Patients          : {len(cases)}")
    print(f"Device            : {backend_device_name}")
    print(f"n_clusters        : {args.n_clusters}")
    print(f"n_init            : {args.n_init}")
    print(f"max_iter          : {args.max_iter}")
    print(f"allow_non_gzip    : {args.allow_non_gzip_nii_gz}")
    print(f"overwrite_existing: {args.overwrite_existing}")
    print(f"splits            : {split_counts}")
    if torch is not None and torch_device is not None and torch_device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(torch_device)
        print(f"GPU               : {gpu_name}")
    print("=" * 80)

    centers_rows: List[Dict[str, object]] = []
    volume_rows: List[Dict[str, object]] = []
    qc_rows: List[Dict[str, object]] = []
    failed_rows: List[Dict[str, object]] = []

    for case in tqdm(cases, desc="build-habitat", total=len(cases)):
        try:
            adc_img = load_nifti_image(
                case.adc_path,
                allow_non_gzip_nii_gz=args.allow_non_gzip_nii_gz,
            )
            cbf_img = load_nifti_image(
                case.cbf_path,
                allow_non_gzip_nii_gz=args.allow_non_gzip_nii_gz,
            )
            voi_img = load_nifti_image(
                case.voi_path,
                allow_non_gzip_nii_gz=args.allow_non_gzip_nii_gz,
            )

            assert_same_geometry(adc_img, cbf_img, name=f"{case.patient_id}: adc vs cbf")
            assert_same_geometry(adc_img, voi_img, name=f"{case.patient_id}: adc vs voi")

            adc = np.asarray(adc_img.get_fdata(dtype=np.float32), dtype=np.float32)
            cbf = np.asarray(cbf_img.get_fdata(dtype=np.float32), dtype=np.float32)
            voi = np.asarray(voi_img.get_fdata(dtype=np.float32), dtype=np.float32) > 0

            if not np.any(voi):
                raise ValueError(f"Patient {case.patient_id} has an empty VOI.")

            adc_vals = adc[voi]
            cbf_vals = cbf[voi]

            # 只要 VOI 内存在 NaN / Inf，就直接报错。
            # 原因很简单：这是体素级聚类，异常值会直接改变 cluster center。
            if not np.all(np.isfinite(adc_vals)):
                raise ValueError(f"Patient {case.patient_id} has non-finite ADC values inside VOI.")
            if not np.all(np.isfinite(cbf_vals)):
                raise ValueError(f"Patient {case.patient_id} has non-finite CBF values inside VOI.")

            adc_z, adc_mean, adc_std = zscore_numpy(adc_vals, eps=args.eps)
            cbf_z, cbf_mean, cbf_std = zscore_numpy(cbf_vals, eps=args.eps)
            x = np.stack([adc_z, cbf_z], axis=1).astype(np.float32)

            if torch is not None and torch_device is not None:
                labels_raw, centers_z_raw, inertia, num_iterations, backend_name = run_kmeans_torch(
                    x_np=x,
                    n_clusters=args.n_clusters,
                    n_init=args.n_init,
                    max_iter=args.max_iter,
                    tol=args.tol,
                    seed=args.seed,
                    device=torch_device,
                    chunk_size=args.chunk_size,
                )
            else:
                labels_raw, centers_z_raw, inertia, num_iterations, backend_name = run_kmeans_sklearn(
                    x_np=x,
                    n_clusters=args.n_clusters,
                    n_init=args.n_init,
                    max_iter=args.max_iter,
                    tol=args.tol,
                    seed=args.seed,
                )

            cluster_result = map_clusters_to_habitats(
                labels_raw=labels_raw,
                centers_z_raw=centers_z_raw,
                adc_values=adc_vals,
                cbf_values=cbf_vals,
            )
            cluster_result.inertia = inertia
            cluster_result.num_iterations = num_iterations
            cluster_result.backend = backend_name

            habitat3 = np.zeros_like(adc, dtype=np.uint8)
            habitat3[voi] = cluster_result.mapped_labels_flat

            h1 = (habitat3 == 1).astype(np.uint8)
            h2 = (habitat3 == 2).astype(np.uint8)
            h3 = (habitat3 == 3).astype(np.uint8)
            h12 = ((habitat3 == 1) | (habitat3 == 2)).astype(np.uint8)

            if case.split_name == "all":
                patient_output_dir = output_root / case.label_name / case.patient_id
            else:
                patient_output_dir = output_root / case.split_name / case.label_name / case.patient_id
            maybe_remove_patient_dir(patient_output_dir, overwrite_existing=args.overwrite_existing)
            save_mask(h1, reference_img=adc_img, path=patient_output_dir / "h1.nii.gz")
            save_mask(h2, reference_img=adc_img, path=patient_output_dir / "h2.nii.gz")
            save_mask(h3, reference_img=adc_img, path=patient_output_dir / "h3.nii.gz")
            save_mask(h12, reference_img=adc_img, path=patient_output_dir / "h12.nii.gz")

            # 体积统计直接按体素数计算。
            tumor_volume = int(np.count_nonzero(voi))
            h1_volume = int(np.count_nonzero(h1))
            h2_volume = int(np.count_nonzero(h2))
            h3_volume = int(np.count_nonzero(h3))
            h12_volume = int(np.count_nonzero(h12))

            # 语义映射后的中心顺序固定为 H1/H2/H3。
            for habitat_id, habitat_name in enumerate(("H1", "H2", "H3"), start=1):
                centers_rows.append(
                    {
                        "patient_id": case.patient_id,
                        "split": case.split_name,
                        "label_name": case.label_name,
                        "cluster_mapped": habitat_name,
                        "adc_center_raw": float(cluster_result.adc_centers_mapped[habitat_id - 1]),
                        "cbf_center_raw": float(cluster_result.cbf_centers_mapped[habitat_id - 1]),
                        "adc_center_z": float(cluster_result.centers_z_mapped[habitat_id - 1, 0]),
                        "cbf_center_z": float(cluster_result.centers_z_mapped[habitat_id - 1, 1]),
                    }
                )

            volume_rows.append(
                {
                    "patient_id": case.patient_id,
                    "split": case.split_name,
                    "label_name": case.label_name,
                    "tumor_volume": tumor_volume,
                    "h1_volume": h1_volume,
                    "h2_volume": h2_volume,
                    "h3_volume": h3_volume,
                    "h12_volume": h12_volume,
                    "h1_ratio": float(h1_volume / tumor_volume),
                    "h2_ratio": float(h2_volume / tumor_volume),
                    "h3_ratio": float(h3_volume / tumor_volume),
                    "h12_ratio": float(h12_volume / tumor_volume),
                }
            )

            semantic_order_ok = bool(
                (cluster_result.cbf_centers_mapped[0] >= cluster_result.cbf_centers_mapped[1])
                and (cluster_result.cbf_centers_mapped[0] >= cluster_result.cbf_centers_mapped[2])
                and (cluster_result.adc_centers_mapped[2] >= cluster_result.adc_centers_mapped[0])
                and (cluster_result.adc_centers_mapped[2] >= cluster_result.adc_centers_mapped[1])
            )

            tiny_cluster_flag = bool(
                min(h1_volume, h2_volume, h3_volume) < max(1, math.ceil(tumor_volume * 0.01))
            )
            empty_cluster_flag = bool(min(h1_volume, h2_volume, h3_volume) == 0)
            mask_sum_consistent = bool((h1_volume + h2_volume + h3_volume) == tumor_volume)
            h12_consistent = bool(h12_volume == (h1_volume + h2_volume))

            qc_rows.append(
                {
                    "patient_id": case.patient_id,
                    "split": case.split_name,
                    "label_name": case.label_name,
                    "backend": cluster_result.backend,
                    "voi_nonzero_voxels": tumor_volume,
                    "adc_nonzero_voxels_in_voi": int(np.count_nonzero(np.abs(adc_vals) > args.eps)),
                    "cbf_nonzero_voxels_in_voi": int(np.count_nonzero(np.abs(cbf_vals) > args.eps)),
                    "adc_mean_in_voi": float(adc_mean),
                    "adc_std_in_voi": float(adc_std),
                    "cbf_mean_in_voi": float(cbf_mean),
                    "cbf_std_in_voi": float(cbf_std),
                    "adc_center_h1": float(cluster_result.adc_centers_mapped[0]),
                    "adc_center_h2": float(cluster_result.adc_centers_mapped[1]),
                    "adc_center_h3": float(cluster_result.adc_centers_mapped[2]),
                    "cbf_center_h1": float(cluster_result.cbf_centers_mapped[0]),
                    "cbf_center_h2": float(cluster_result.cbf_centers_mapped[1]),
                    "cbf_center_h3": float(cluster_result.cbf_centers_mapped[2]),
                    "semantic_cost": float(cluster_result.best_cost),
                    "inertia": float(cluster_result.inertia),
                    "num_iterations": int(cluster_result.num_iterations),
                    "semantic_order_ok": semantic_order_ok,
                    "empty_cluster_flag": empty_cluster_flag,
                    "tiny_cluster_flag": tiny_cluster_flag,
                    "mask_sum_consistent": mask_sum_consistent,
                    "h12_consistent": h12_consistent,
                    "qc_note": "",
                }
            )

        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            failed_rows.append(
                {
                    "patient_id": case.patient_id,
                    "split": case.split_name,
                    "label_name": case.label_name,
                    "error": error_message,
                }
            )
            if args.skip_errors:
                print(f"[WARN] Skip patient {case.patient_id}: {error_message}")
                continue
            raise

    pd.DataFrame(centers_rows).to_csv(manifests_dir / "habitat_centers.csv", index=False)
    pd.DataFrame(volume_rows).to_csv(manifests_dir / "habitat_volume_summary.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(manifests_dir / "habitat_qc.csv", index=False)
    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(manifests_dir / "failed_cases.csv", index=False)

    elapsed_seconds = time.time() - start_time
    run_summary = {
        "script": "build_habitat.py",
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "split_counts": split_counts,
        "num_patients_requested": len(cases),
        "num_patients_succeeded": len(volume_rows),
        "num_patients_failed": len(failed_rows),
        "device": backend_device_name,
        "torch_available": bool(torch is not None),
        "cuda_available": bool(torch is not None and torch.cuda.is_available()),
        "seed": int(args.seed),
        "n_clusters": int(args.n_clusters),
        "n_init": int(args.n_init),
        "max_iter": int(args.max_iter),
        "tol": float(args.tol),
        "chunk_size": int(args.chunk_size),
        "elapsed_seconds": float(elapsed_seconds),
        "args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
    }
    if args.save_run_config:
        with open(manifests_dir / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Build Habitat Finished")
    print("=" * 80)
    print(f"Succeeded patients : {len(volume_rows)}")
    print(f"Failed patients    : {len(failed_rows)}")
    print(f"Elapsed seconds    : {elapsed_seconds:.2f}")
    print(f"Mask output root   : {output_root}")
    print(f"Manifest directory : {manifests_dir}")
    if failed_rows:
        print(f"Failed manifest    : {manifests_dir / 'failed_cases.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    main()

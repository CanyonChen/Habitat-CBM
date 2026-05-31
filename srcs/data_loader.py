#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用数据加载器：用于胶质瘤 IDH 多模态 MRI 的患者级扫描、2.5D 切片块构建与调试。

===============================================================================
一、适用目录格式
===============================================================================
本加载器默认读取 `data_split.py` 生成后的划分目录，即：

splited_data/
├─ train/
│  ├─ conventional/
│  │  ├─ mutant/
│  │  └─ wild_type/
│  └─ functional/
│     ├─ mutant/
│     └─ wild_type/
├─ val/
└─ test/

每位患者在 `conventional/<label>/<patient_id>/` 与
`functional/<label>/<patient_id>/` 下各有一个同名患者文件夹。

按当前数据协议：
- `conventional/<label>/<patient_id>/` 下保存 `t1`、`t1ce`、`t2`、`t2flair` 和 `voi`
- `functional/<label>/<patient_id>/` 下保存 `adc`、`cbf` 和 `voi`
- 若两个分支下都存在 `voi`，下游一律以 `functional/` 分支下的 `voi` 为准

===============================================================================
二、核心类 HabitatIDHBlockDataset 的主要参数说明
===============================================================================
1. split_root
   - 含义：某一个数据子集的根目录，例如 `.../splited_data/train`
   - 要求：目录下必须包含 `conventional/` 与 `functional/`

2. modalities
   - 含义：需要加载的模态名列表
   - 默认：`("t1", "t1ce", "t2", "t2flair", "adc", "cbf")`
   - 输出时会按照该顺序拼接通道

3. require_voi
   - 含义：是否要求每位患者都存在 VOI 掩模
   - 默认：`True`
   - 若为 `True`，缺失 VOI 的患者会直接报错

4. append_voi_mask
   - 含义：是否将 VOI 掩模 block 作为额外输入通道拼接到 `image`
   - 默认：`True`
   - 若开启，则最终通道数会额外增加 `block_depth`

5. mask_background_with_voi
   - 含义：是否用 VOI 掩模将模态图像的背景清零
   - 默认：`False`
   - 该选项不会改变通道数，只改变像素值

6. block_depth
   - 含义：2.5D 切片块厚度，必须是奇数
   - 例如：`5` 表示以中心切片为锚点，取前后各 2 张切片

7. slice_axis
   - 含义：沿哪个轴抽取 2D 切片
   - 默认：`2`
   - 对于常见的 `(H, W, D)` 体数据，`2` 表示沿深度方向逐层取片

8. intensity_norm
   - 含义：单病例、单模态的强度归一化方式
   - 可选：`zscore`、`minmax`、`none`
   - 默认：`zscore`

9. min_nonzero_voxels
   - 含义：一个中心切片至少要有多少个前景体素才会被纳入样本索引
   - 当 `require_voi=True` 时，这里的前景指 VOI 中的非零体素
   - 当 `require_voi=False` 时，退化为参考模态中的非零体素
   - 默认：`16`

10. cache_volumes
   - 含义：是否在内存中缓存患者体数据，避免同一患者重复从磁盘读取
   - 默认：`True`
   - 说明：适合 debug 和中小规模实验；若内存不足，可关闭

11. return_metadata
   - 含义：是否在 `__getitem__` 中额外返回患者与切片元信息
   - 默认：`True`

===============================================================================
三、__getitem__ 返回内容
===============================================================================
默认返回一个字典：

- `image`:
  torch.FloatTensor，形状为 `[C, H, W]`
  若 `append_voi_mask=False`，则 `C = len(modalities) * block_depth`
  若 `append_voi_mask=True`，则 `C = (len(modalities) + 1) * block_depth`
- `label`:
  torch.LongTensor，患者级 IDH 标签，`0=wild_type`，`1=mutant`
- `patient_id`:
  三位患者编号字符串，例如 `005`
- `slice_index`:
  当前样本对应的中心切片索引
- `paths`:
  该患者六模态文件路径以及实际使用的 `functional/voi` 路径，便于调试

===============================================================================
四、CLI 命令行运行示例
===============================================================================
1. 检查 train 子集能否正常扫描，并打印数据集统计信息：

python srcs/data_loader.py \
  --split-root /root/autodl-tmp/habitat_CBM/data/splited_data/train \
  --require-voi true \
  --append-voi-mask true \
  --block-depth 5 \
  --slice-axis 2 \
  --intensity-norm zscore

2. 检查 val 子集前 3 个样本的张量形状：

python srcs/data_loader.py \
  --split-root /root/autodl-tmp/habitat_CBM/data/splited_data/val \
  --mask-background-with-voi true \
  --block-depth 5 \
  --max-samples 3

3. 若只想构建索引、不缓存体数据：

python srcs/data_loader.py \
  --split-root /root/autodl-tmp/habitat_CBM/data/splited_data/test \
  --cache-volumes false
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

SEED = 42
BRANCHES = ("conventional", "functional")
VOI_SOURCE_BRANCH = "functional"
LABEL_TO_ID = {"wild_type": 0, "mutant": 1}
DEFAULT_MODALITIES = ("t1", "t1ce", "t2", "t2flair", "adc", "cbf")
ALLOWED_EXTENSIONS = (".nii", ".nii.gz", ".mha", ".mhd", ".nrrd")
DEFAULT_VOI_KEYWORDS = ("voi",)

# 模态关键词用于从文件名中匹配真实影像文件。
# 之所以做成映射，是因为不同中心或导出脚本的命名并不完全一致。
MODALITY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "t1": ("_t1", "t1.", "t1_", "t1wi", "t1w"),
    "t1ce": ("t1ce", "t1_ce", "ce_t1", "cet1", "t1c", "t1gd", "t1_gd"),
    "t2": ("_t2", "t2.", "t2_", "t2wi", "t2w"),
    "t2flair": ("t2flair", "t2_flair", "t2-flair", "flair"),
    "adc": ("adc",),
    "cbf": ("cbf",),
}

# T1 容易和 T1CE 在文件名上混淆，因此单独列出冲突关键词。
T1_EXCLUDE_KEYWORDS = MODALITY_KEYWORDS["t1ce"]
T2_EXCLUDE_KEYWORDS = MODALITY_KEYWORDS["t2flair"]


@dataclass(frozen=True)
class PatientCase:
    """患者级样本信息。

    一个 PatientCase 对应一位患者，保存其标签、六模态文件路径，
    以及实际用于建模的 `functional/voi` 文件路径。
    """

    patient_id: str
    label_name: str
    label_id: int
    modality_paths: Dict[str, Path]
    voi_path: Optional[Path] = None


@dataclass(frozen=True)
class SliceSample:
    """2.5D 样本索引项。

    每一项表示某位患者的一个中心切片位置，真正取数发生在 __getitem__。
    """

    patient_id: str
    slice_index: int


def str2bool(value: str) -> bool:
    """将命令行字符串解析为布尔值。"""

    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def normalize_name(name: str) -> str:
    """统一文件名格式，降低不同命名风格带来的匹配困难。"""

    return name.lower().replace("-", "_").replace(" ", "_")


def is_supported_image(path: Path) -> bool:
    """判断文件扩展名是否为当前加载器支持的医学影像格式。"""

    lower_name = path.name.lower()
    return any(lower_name.endswith(ext) for ext in ALLOWED_EXTENSIONS)


def collect_patient_image_files(patient_dirs: Dict[str, Path]) -> List[Path]:
    """收集一个患者目录下所有可能的医学影像文件。"""

    all_files: List[Path] = []
    for branch in BRANCHES:
        branch_dir = patient_dirs[branch]
        all_files.extend(
            [
                path
                for path in branch_dir.rglob("*")
                if path.is_file() and is_supported_image(path)
            ]
        )
    return all_files


def collect_branch_image_files(branch_dir: Path) -> List[Path]:
    """收集单个分支目录下所有可能的医学影像文件。"""

    return [
        path
        for path in branch_dir.rglob("*")
        if path.is_file() and is_supported_image(path)
    ]


def match_modality(path: Path, modality: str) -> bool:
    """根据文件名判断某个文件是否属于指定模态。

    注意：
    - `t1` 与 `t1ce` 的文件名很容易互相误匹配，因此这里做了显式排除。
    - 本函数只依赖文件名关键词，不依赖目录名。
    """

    name = normalize_name(path.name)
    keywords = MODALITY_KEYWORDS[modality]

    if modality == "t1" and any(ex_keyword in name for ex_keyword in T1_EXCLUDE_KEYWORDS):
        return False

    if modality == "t2" and any(ex_keyword in name for ex_keyword in T2_EXCLUDE_KEYWORDS):
        return False

    return any(keyword in name for keyword in keywords)


def match_voi(path: Path) -> bool:
    """根据文件名判断某个文件是否可能是 VOI 掩模。"""

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


def discover_patient_dirs(split_root: Path) -> Dict[str, Dict[str, Path]]:
    """扫描一个子集目录，收集所有患者在两个分支下的患者文件夹路径。

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
        branch_root = split_root / branch
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

                # 同一患者若在两个标签目录下冲突，直接报错。
                if patient_entry["label_name"] != label_name:
                    raise ValueError(
                        f"Patient {patient_id} appears under conflicting labels: "
                        f"{patient_entry['label_name']} vs {label_name}"
                    )
                patient_entry[branch] = patient_dir

    # 确保每位患者同时拥有 conventional 和 functional 两个分支。
    for patient_id, entry in patient_map.items():
        for branch in BRANCHES:
            if branch not in entry:
                raise FileNotFoundError(
                    f"Patient {patient_id} is missing branch directory: {branch}"
                )

    return patient_map


def discover_modality_files(
    patient_dirs: Dict[str, Path],
    modalities: Sequence[str],
    all_files: Optional[Sequence[Path]] = None,
) -> Dict[str, Path]:
    """在患者目录下为每个模态找到唯一文件。

    搜索策略：
    - 同时扫描 `conventional/患者目录` 与 `functional/患者目录`
    - 递归搜索所有支持的影像文件
    - 按文件名关键词进行模态匹配
    - 每个模态必须且只能匹配到一个文件
    """

    if all_files is None:
        all_files = collect_patient_image_files(patient_dirs)

    modality_paths: Dict[str, Path] = {}
    for modality in modalities:
        matches = [path for path in all_files if match_modality(path, modality)]
        if len(matches) == 0:
            raise FileNotFoundError(
                f"Cannot find modality file for '{modality}' under patient folders: "
                f"{patient_dirs['conventional']} and {patient_dirs['functional']}"
            )
        if len(matches) > 1:
            match_str = "\n".join(str(path) for path in matches[:10])
            raise ValueError(
                f"Found multiple candidate files for modality '{modality}'. "
                f"Please refine file names or keyword rules.\n{match_str}"
            )
        modality_paths[modality] = matches[0]

    return modality_paths


def discover_voi_file(patient_dirs: Dict[str, Path]) -> Path:
    """
    在患者目录下找到实际用于建模的唯一 VOI 掩模文件。

    当前协议明确规定：若 `conventional/` 与 `functional/` 都存在 `voi`，
    一律使用 `functional/` 分支下的 `voi`。
    """

    functional_dir = patient_dirs[VOI_SOURCE_BRANCH]
    all_files = collect_branch_image_files(functional_dir)
    matches = [path for path in all_files if match_voi(path)]
    if len(matches) == 0:
        raise FileNotFoundError(
            "Cannot find VOI mask file under the functional branch: "
            f"{functional_dir}"
        )
    if len(matches) == 1:
        return matches[0]

    scored = sorted(matches, key=lambda path: score_voi_candidate(path), reverse=True)
    best_score = score_voi_candidate(scored[0])
    best_matches = [path for path in scored if score_voi_candidate(path) == best_score]
    if len(best_matches) > 1:
        match_str = "\n".join(str(path) for path in best_matches[:10])
        raise ValueError(
            f"Found multiple equally plausible VOI files under {functional_dir}. "
            "Please keep only one canonical functional/voi file or refine the rules.\n"
            f"{match_str}"
        )
    return best_matches[0]


def load_nifti_array(path: Path) -> np.ndarray:
    """读取单个医学影像文件并返回 float32 数组。

    这里统一转为 `float32`，以便后续归一化与 PyTorch 张量化。
    兼容处理：扩展名为 .gz 但实际未压缩的 NIfTI 文件。
    """
    import io

    path = Path(path)

    if path.suffix == ".gz":
        with open(path, "rb") as _f:
            magic = _f.read(2)
        if magic != b"\x1f\x8b":
            # 文件是未压缩的 NIfTI 但扩展名为 .gz，用 BytesIO 绕过 nibabel 的扩展名检测
            with open(path, "rb") as _f:
                raw = _f.read()
            fobj = io.BytesIO(raw)
            fh = nib.FileHolder(fileobj=fobj)
            image = nib.Nifti1Image.from_file_map({"header": fh, "image": fh})
            array = image.get_fdata(dtype=np.float32)
            return np.asarray(array, dtype=np.float32)

    image = nib.load(str(path))
    array = image.get_fdata(dtype=np.float32)
    return np.asarray(array, dtype=np.float32)


def load_voi_array(path: Path) -> np.ndarray:
    """读取 VOI 掩模并转为 float32 二值数组。"""

    array = load_nifti_array(path)
    return (array > 0).astype(np.float32)


def normalize_volume(volume: np.ndarray, mode: str) -> np.ndarray:
    """对单病例、单模态体数据进行归一化。"""

    if mode == "none":
        return volume

    # 只基于非零区域估计统计量，尽量避免大面积背景拉低有效信号。
    mask = np.abs(volume) > 1e-8
    if not np.any(mask):
        return volume

    roi = volume[mask]
    if mode == "zscore":
        mean = float(roi.mean())
        std = float(roi.std())
        if std < 1e-8:
            return volume - mean
        return (volume - mean) / std

    if mode == "minmax":
        vmin = float(roi.min())
        vmax = float(roi.max())
        if vmax - vmin < 1e-8:
            return volume - vmin
        return (volume - vmin) / (vmax - vmin)

    raise ValueError(f"Unsupported intensity_norm mode: {mode}")


def extract_slice_2d(volume: np.ndarray, index: int, axis: int) -> np.ndarray:
    """按指定轴提取单张 2D 切片。"""

    if axis == 0:
        return volume[index, :, :]
    if axis == 1:
        return volume[:, index, :]
    if axis == 2:
        return volume[:, :, index]
    raise ValueError(f"Unsupported slice axis: {axis}")


def build_slice_block(volume: np.ndarray, center_index: int, axis: int, block_depth: int) -> np.ndarray:
    """围绕中心切片构建 2.5D 切片块。

    边界处理策略：
    - 若中心切片靠近边界，则采用“复制边界切片”的方式补足厚度。
    """

    radius = block_depth // 2
    max_index = volume.shape[axis] - 1
    slices: List[np.ndarray] = []

    for offset in range(-radius, radius + 1):
        index = min(max(center_index + offset, 0), max_index)
        slices.append(extract_slice_2d(volume, index, axis))

    return np.stack(slices, axis=0)


def ensure_tensor(data: object, dtype: torch.dtype) -> torch.Tensor:
    """Convert MONAI / NumPy outputs into a torch tensor with the target dtype."""

    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype)
    return torch.as_tensor(data, dtype=dtype)


class HabitatIDHBlockDataset(Dataset):
    """胶质瘤 IDH 通用 2.5D 数据加载器。

    设计目标：
    1. 先按患者扫描目录并构建六模态文件索引；
    2. 再按中心切片展开为 block 级样本；
    3. 在 __getitem__ 中返回多模态 2.5D 张量与患者级标签。
    """

    def __init__(
        self,
        split_root: str | Path,
        modalities: Sequence[str] = DEFAULT_MODALITIES,
        require_voi: bool = True,
        append_voi_mask: bool = True,
        mask_background_with_voi: bool = False,
        block_depth: int = 5,
        slice_axis: int = 2,
        intensity_norm: str = "zscore",
        min_nonzero_voxels: int = 16,
        cache_volumes: bool = True,
        return_metadata: bool = True,
        transform: Optional[Callable[[Dict[str, object]], Dict[str, object]]] = None,
    ) -> None:
        if block_depth % 2 == 0:
            raise ValueError("block_depth must be an odd number, e.g. 3, 5, or 7.")
        if slice_axis not in {0, 1, 2}:
            raise ValueError("slice_axis must be one of {0, 1, 2}.")

        self.split_root = Path(split_root)
        self.modalities = tuple(modalities)
        self.require_voi = require_voi or append_voi_mask or mask_background_with_voi
        self.append_voi_mask = append_voi_mask
        self.mask_background_with_voi = mask_background_with_voi
        self.block_depth = block_depth
        self.slice_axis = slice_axis
        self.intensity_norm = intensity_norm
        self.min_nonzero_voxels = min_nonzero_voxels
        self.cache_volumes = cache_volumes
        self.return_metadata = return_metadata
        self.transform = transform

        # 体数据缓存：key 为 patient_id，value 为 {modality: volume_array}
        self._volume_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._voi_cache: Dict[str, np.ndarray] = {}

        self.patient_cases = self._build_patient_cases()
        self.sample_index = self._build_sample_index()

    def _build_patient_cases(self) -> Dict[str, PatientCase]:
        """扫描划分目录并建立患者级索引。"""

        patient_dir_map = discover_patient_dirs(self.split_root)
        patient_cases: Dict[str, PatientCase] = {}

        for patient_id, entry in sorted(patient_dir_map.items()):
            label_name = str(entry["label_name"])
            all_files = collect_patient_image_files(entry)
            modality_paths = discover_modality_files(entry, self.modalities, all_files=all_files)
            voi_path = discover_voi_file(entry) if self.require_voi else None
            patient_cases[patient_id] = PatientCase(
                patient_id=patient_id,
                label_name=label_name,
                label_id=LABEL_TO_ID[label_name],
                modality_paths=modality_paths,
                voi_path=voi_path,
            )

        return patient_cases

    def _get_patient_volumes(self, patient_id: str) -> Dict[str, np.ndarray]:
        """获取某位患者的全部模态体数据。

        如果开启缓存，则同一患者只在第一次访问时从磁盘读取。
        """

        if self.cache_volumes and patient_id in self._volume_cache:
            return self._volume_cache[patient_id]

        case = self.patient_cases[patient_id]
        volumes: Dict[str, np.ndarray] = {}
        reference_shape: Optional[Tuple[int, int, int]] = None

        for modality, path in case.modality_paths.items():
            volume = load_nifti_array(path)
            volume = normalize_volume(volume, self.intensity_norm)

            if volume.ndim != 3:
                raise ValueError(
                    f"Modality '{modality}' of patient {patient_id} is not 3D: "
                    f"shape={volume.shape}, path={path}"
                )

            if reference_shape is None:
                reference_shape = volume.shape
            elif volume.shape != reference_shape:
                raise ValueError(
                    f"Shape mismatch within patient {patient_id}: "
                    f"expected {reference_shape}, got {volume.shape} for {modality}"
                )

            volumes[modality] = volume

        if self.cache_volumes:
            self._volume_cache[patient_id] = volumes
        return volumes

    def _get_patient_voi(self, patient_id: str) -> np.ndarray:
        """获取某位患者的 VOI 掩模。"""

        if not self.require_voi:
            raise RuntimeError("VOI is disabled for this dataset instance.")

        if self.cache_volumes and patient_id in self._voi_cache:
            return self._voi_cache[patient_id]

        case = self.patient_cases[patient_id]
        if case.voi_path is None:
            raise FileNotFoundError(f"Missing VOI path for patient {patient_id}.")

        voi = load_voi_array(case.voi_path)
        if voi.ndim != 3:
            raise ValueError(
                f"VOI of patient {patient_id} is not 3D: "
                f"shape={voi.shape}, path={case.voi_path}"
            )

        if self.cache_volumes:
            self._voi_cache[patient_id] = voi
        return voi

    def _build_sample_index(self) -> List[SliceSample]:
        """根据患者体数据构建 block 级样本索引。

        若启用 VOI，则以 VOI 中心切片的前景体素数决定该切片是否保留；
        否则退化为以第一个模态的非零体素数决定是否保留。
        """

        sample_index: List[SliceSample] = []
        reference_modality = self.modalities[0]

        for patient_id in sorted(self.patient_cases):
            if self.require_voi:
                reference_volume = self._get_patient_voi(patient_id)
            else:
                reference_path = self.patient_cases[patient_id].modality_paths[reference_modality]
                reference_volume = normalize_volume(
                    load_nifti_array(reference_path),
                    self.intensity_norm,
                )
            num_slices = reference_volume.shape[self.slice_axis]

            for slice_index in range(num_slices):
                slice_2d = extract_slice_2d(reference_volume, slice_index, self.slice_axis)
                nonzero_count = int(np.count_nonzero(np.abs(slice_2d) > 1e-8))
                if nonzero_count < self.min_nonzero_voxels:
                    continue
                sample_index.append(SliceSample(patient_id=patient_id, slice_index=slice_index))

        if not sample_index:
            raise ValueError(
                f"No valid samples were constructed from split_root={self.split_root}. "
                f"Please check modality files, slice_axis, or min_nonzero_voxels."
            )

        return sample_index

    def __len__(self) -> int:
        """返回 block 级样本总数。"""

        return len(self.sample_index)

    def __getitem__(self, index: int) -> Dict[str, object]:
        """返回一个 2.5D block 样本。

        输出 `image` 的通道顺序为：
        [t1_block, t1ce_block, t2_block, t2flair_block, adc_block, cbf_block, voi_block?]
        其中每个块各自贡献 `block_depth` 个通道；
        若 `append_voi_mask=False`，则最后不会拼接 `voi_block`。
        """

        sample = self.sample_index[index]
        patient_id = sample.patient_id
        slice_index = sample.slice_index

        case = self.patient_cases[patient_id]
        volumes = self._get_patient_volumes(patient_id)
        voi_volume = self._get_patient_voi(patient_id) if self.require_voi else None
        voi_block = None
        if voi_volume is not None:
            voi_block = build_slice_block(
                volume=voi_volume,
                center_index=slice_index,
                axis=self.slice_axis,
                block_depth=self.block_depth,
            )

        modality_blocks: List[np.ndarray] = []
        for modality in self.modalities:
            block = build_slice_block(
                volume=volumes[modality],
                center_index=slice_index,
                axis=self.slice_axis,
                block_depth=self.block_depth,
            )
            if voi_block is not None:
                if block.shape != voi_block.shape:
                    raise ValueError(
                        f"VOI shape mismatch for patient {patient_id}: "
                        f"image_block={block.shape}, voi_block={voi_block.shape}"
                    )
            if voi_block is not None and self.mask_background_with_voi:
                block = block * voi_block
            modality_blocks.append(block)

        sample_dict: Dict[str, object] = {
            "image": np.concatenate(modality_blocks, axis=0).astype(np.float32),
            "label": case.label_id,
        }
        if voi_block is not None:
            sample_dict["mask"] = voi_block.astype(np.float32)

        if self.transform is not None:
            sample_dict = dict(self.transform(sample_dict))

        image_tensor = ensure_tensor(sample_dict["image"], dtype=torch.float32)
        if image_tensor.ndim != 3:
            raise ValueError(
                f"Expected image tensor with shape [C, H, W], got {tuple(image_tensor.shape)} "
                f"for patient {patient_id}, slice {slice_index}."
            )

        if voi_block is not None and self.append_voi_mask:
            if "mask" not in sample_dict:
                raise KeyError(
                    "Mask is required for append_voi_mask=True, but transform output has no 'mask' key."
                )
            mask_tensor = ensure_tensor(sample_dict["mask"], dtype=torch.float32)
            if mask_tensor.ndim != 3:
                raise ValueError(
                    f"Expected mask tensor with shape [C, H, W], got {tuple(mask_tensor.shape)} "
                    f"for patient {patient_id}, slice {slice_index}."
                )
            if image_tensor.shape[1:] != mask_tensor.shape[1:]:
                raise ValueError(
                    f"Image/mask spatial shape mismatch after transform for patient {patient_id}: "
                    f"image={tuple(image_tensor.shape)}, mask={tuple(mask_tensor.shape)}"
                )
            image_tensor = torch.cat([image_tensor, mask_tensor], dim=0)

        label_tensor = ensure_tensor(sample_dict["label"], dtype=torch.long)
        if label_tensor.ndim != 0:
            label_tensor = label_tensor.reshape(()).to(dtype=torch.long)

        output: Dict[str, object] = {
            "image": image_tensor,
            "label": label_tensor,
        }

        if self.return_metadata:
            output.update(
                {
                    "patient_id": patient_id,
                    "slice_index": slice_index,
                    "paths": {
                        **{key: str(value) for key, value in case.modality_paths.items()},
                        **({"voi": str(case.voi_path)} if case.voi_path is not None else {}),
                    },
                }
            )

        return output

    def summary(self) -> Dict[str, object]:
        """返回当前数据集的简要统计信息。"""

        patient_count = len(self.patient_cases)
        mutant_count = sum(case.label_id == 1 for case in self.patient_cases.values())
        wild_count = patient_count - mutant_count

        return {
            "split_root": str(self.split_root),
            "patient_count": patient_count,
            "sample_count": len(self.sample_index),
            "modalities": list(self.modalities),
            "voi_source_branch": VOI_SOURCE_BRANCH,
            "require_voi": self.require_voi,
            "append_voi_mask": self.append_voi_mask,
            "mask_background_with_voi": self.mask_background_with_voi,
            "block_depth": self.block_depth,
            "slice_axis": self.slice_axis,
            "intensity_norm": self.intensity_norm,
            "min_nonzero_voxels": self.min_nonzero_voxels,
            "transform_enabled": self.transform is not None,
            "mutant_count": mutant_count,
            "wild_type_count": wild_count,
        }


def build_argparser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="通用 2.5D MRI 数据加载器，用于扫描一个数据子集并输出调试信息。"
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        required=True,
        help="某一个子集目录，例如 /path/to/splited_data/train",
    )
    parser.add_argument(
        "--block-depth",
        type=int,
        default=5,
        help="2.5D 切片块厚度，必须为奇数，默认 5。",
    )
    parser.add_argument(
        "--require-voi",
        type=str2bool,
        default=True,
        help="是否要求每位患者都有可识别的 functional/voi，默认 true。",
    )
    parser.add_argument(
        "--append-voi-mask",
        type=str2bool,
        default=True,
        help="是否把 functional/voi 的 block 作为额外输入通道拼接，默认 true。",
    )
    parser.add_argument(
        "--mask-background-with-voi",
        type=str2bool,
        default=False,
        help="是否用 functional/voi 将图像背景清零，默认 false。",
    )
    parser.add_argument(
        "--slice-axis",
        type=int,
        default=2,
        help="沿哪个轴取切片，默认 2。",
    )
    parser.add_argument(
        "--intensity-norm",
        choices=("zscore", "minmax", "none"),
        default="zscore",
        help="强度归一化方式，默认 zscore。",
    )
    parser.add_argument(
        "--min-nonzero-voxels",
        type=int,
        default=16,
        help="中心切片至少需要多少个非零体素才会被纳入索引，默认 16。",
    )
    parser.add_argument(
        "--cache-volumes",
        type=str2bool,
        default=True,
        help="是否缓存患者体数据，默认 true。",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3,
        help="最多打印多少个样本的调试信息，默认 3。",
    )
    return parser


def format_summary(summary: Dict[str, object]) -> str:
    """将 summary() 输出格式化为更易读的文本。"""

    lines = [
        "数据集摘要：",
        f"  split_root          : {summary['split_root']}",
        f"  patient_count       : {summary['patient_count']}",
        f"  sample_count        : {summary['sample_count']}",
        f"  modalities          : {summary['modalities']}",
        f"  voi_source_branch   : {summary['voi_source_branch']}",
        f"  require_voi         : {summary['require_voi']}",
        f"  append_voi_mask     : {summary['append_voi_mask']}",
        f"  mask_bg_with_voi    : {summary['mask_background_with_voi']}",
        f"  block_depth         : {summary['block_depth']}",
        f"  slice_axis          : {summary['slice_axis']}",
        f"  intensity_norm      : {summary['intensity_norm']}",
        f"  min_nonzero_voxels  : {summary['min_nonzero_voxels']}",
        f"  mutant_count        : {summary['mutant_count']}",
        f"  wild_type_count     : {summary['wild_type_count']}",
    ]
    return "\n".join(lines)


def main() -> None:
    """命令行入口：用于快速检查数据加载器是否工作正常。"""

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    args = build_argparser().parse_args()
    dataset = HabitatIDHBlockDataset(
        split_root=args.split_root,
        block_depth=args.block_depth,
        require_voi=args.require_voi,
        append_voi_mask=args.append_voi_mask,
        mask_background_with_voi=args.mask_background_with_voi,
        slice_axis=args.slice_axis,
        intensity_norm=args.intensity_norm,
        min_nonzero_voxels=args.min_nonzero_voxels,
        cache_volumes=args.cache_volumes,
        return_metadata=True,
    )

    print(format_summary(dataset.summary()))
    print("")
    print("样本调试信息：")

    max_samples = min(args.max_samples, len(dataset))
    for idx in range(max_samples):
        sample = dataset[idx]
        image = sample["image"]
        label = sample["label"]
        print(
            f"[{idx}] patient_id={sample['patient_id']}, "
            f"slice_index={sample['slice_index']}, "
            f"image_shape={tuple(image.shape)}, "
            f"label={int(label)}"
        )


if __name__ == "__main__":
    main()
